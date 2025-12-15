#!/usr/bin/env python3
"""
Re-optimización de Thresholds con Restricción de Trades Mínimos

Encuentra el threshold óptimo que:
1. Genera al menos MIN_TRADES operaciones
2. Maximiza Sharpe Ratio (o PnL)

Usage:
    python scripts/optimize_threshold.py --symbol BTCUSDT --timeframe 1h --min-trades 20
"""
import sys
from pathlib import Path
import click
import json
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.advanced_models.predictor import AdvancedPredictor
from ml.advanced_models.dataset import load_sequence_dataset, AdvancedDatasetConfig
from utils.logger import setup_logger

logger = setup_logger("optimize_threshold")
MODEL_DIR = REPO_ROOT / "models" / "advanced"


def _symbol_key(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").replace("-", "").upper()


def simulate_trades(predictions, returns, threshold, min_confidence_diff=0.1):
    """Simula trades con threshold dado."""
    trades = []
    equity = 1.0
    equity_curve = [equity]
    
    for i in range(len(predictions)):
        class_probs = predictions[i]
        long_prob = class_probs[1]
        short_prob = class_probs[2]
        
        diff = abs(long_prob - short_prob)
        
        # Señal Long
        if long_prob > threshold and diff > min_confidence_diff:
            ret = returns[i]
            equity *= (1 + ret)
            trades.append({'side': 'LONG', 'return': ret})
        
        # Señal Short  
        elif short_prob > threshold and diff > min_confidence_diff:
            ret = -returns[i]  # Invertir rendimiento para short
            equity *= (1 + ret)
            trades.append({'side': 'SHORT', 'return': ret})
        
        equity_curve.append(equity)
    
    if not trades:
        return {'n_trades': 0, 'pnl': 0.0, 'sharpe': 0.0, 'win_rate': 0.0, 'max_dd': 0.0}
    
    trade_returns = np.array([t['return'] for t in trades])
    pnl = (equity - 1.0) * 100
    sharpe = (trade_returns.mean() / (trade_returns.std() + 1e-8)) * np.sqrt(252)
    win_rate = (trade_returns > 0).sum() / len(trade_returns)
    
    # Max Drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100
    
    return {
        'n_trades': len(trades),
        'pnl': pnl,
        'sharpe': sharpe,
        'win_rate': win_rate * 100,
        'max_dd': max_dd
    }


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--min-trades", default=20, help="Mínimo de trades requeridos")
@click.option("--metric", default="sharpe", type=click.Choice(["sharpe", "pnl"]), help="Métrica a optimizar")
@click.option("--min-diff", default=0.1, help="Diferencia mínima de confianza entre clases")
def main(symbol: str, timeframe: str, min_trades: int, metric: str, min_diff: float):
    """Re-optimiza threshold con restricción de mínimo de trades."""
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZANDO THRESHOLD: {symbol} {timeframe}")
    print(f"Restricción: Mínimo {min_trades} trades")
    print(f"Métrica: {metric.upper()}")
    print(f"Min Diff: {min_diff}")
    print(f"{'='*80}\n")
    
    # Cargar modelo
    symbol_key = _symbol_key(symbol)
    model_path = MODEL_DIR / symbol_key / timeframe
    
    if not model_path.exists():
        logger.error(f"Modelo no encontrado: {model_path}")
        return
    
    # Cargar metadata
    meta_path = model_path / "meta.json"
    meta = json.loads(meta_path.read_text())
    
    # Cargar predictor
    predictor = AdvancedPredictor(
        model_path=model_path,
        scaler_path=model_path / "scaler.pkl",
        meta_path=model_path / "meta.json"
    )
    
    # Cargar datos de test (últimos 15%)
    config = AdvancedDatasetConfig(
        symbol=meta['symbol'],
        timeframe=meta['timeframe'],
        sequence_length=meta['sequence_length'],
        prediction_horizon=meta['prediction_horizon'],
        target_return=meta['target_return'],
        max_history_days=1000,
    )
    
    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)
    
    # Scale features manually since we bypass predictor pipeline
    # Note: Scaler expects selected features ONLY if feature selection was done before scaling
    # But in training: Selection -> Scaling. So Scaler expects 32 features.
    # We need to apply selection THEN scaling?
    
    # Let's check predictor logic.
    # Predictor loads scaler.
    # Predictor loads selector.
    
    # We should use predictor's artifacts.
    # But predictor stores them per fold in ensemble_pipelines.
    # For single model, it's in predictor.scaler (if loaded globally) or pipeline['scaler'].
    
    # Since we iterate pipelines, we should do it inside the loop?
    # Or can we assume global scaler?
    # ETH 1h has global scaler.pkl.
    
    # Correct order:
    # 1. Select features (if selector exists)
    # 2. Scale features (if scaler exists)
    
    # But wait, load_sequence_dataset returns ALL features.
    
    # Let's do it inside the loop to be safe and compatible with ensembles.
    
    # Usar últimos 15% como test
    n_samples = len(features)
    test_start = int(n_samples * 0.85)
    
    X_test = features[test_start:]
    y_test_reg = regression_targets[test_start:]
    
    print(f"Datos de test: {len(X_test)} muestras\n")
    
    # Obtener predicciones
    print("Generando predicciones...")
    predictions = []
    
    for i in range(config.sequence_length, len(X_test)):
        window = X_test[i-config.sequence_length:i]
        # Predict with ensemble
        probs_sum = np.zeros(3)
        
        for pipeline in predictor.ensemble_pipelines:
            model = pipeline['model']
            selector = pipeline['selector']
            scaler = pipeline['scaler']
            
            # 1. Apply feature selection if available
            window_input = window
            if selector:
                try:
                    window_input = selector.transform(window)
                except Exception as e:
                    if hasattr(selector, 'support_'):
                        window_input = window[:, selector.support_]
                    else:
                        raise e
            
            # 2. Apply scaling if available
            if scaler:
                window_input = scaler.transform(window_input)
            
            # Convert to tensor
            
            # Convert to tensor
            window_tensor = torch.FloatTensor(window_input).unsqueeze(0).to(predictor.device)
            
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(window_tensor).cpu().numpy()[0]
                else:
                    # Fallback for DeepTemporalNet or models returning dict
                    outputs = model(window_tensor)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    elif isinstance(outputs, torch.Tensor):
                        logits = outputs
                    else:
                        # Assume tuple or other format, take first element as logits
                        logits = outputs[0]
                        
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                probs_sum += probs
        
        avg_probs = probs_sum / len(predictor.ensemble_pipelines)
        predictions.append(avg_probs)
    
    predictions = np.array(predictions)
    returns = y_test_reg[config.sequence_length:]
    
    print(f"Predicciones generadas: {len(predictions)}\n")
    
    # Buscar threshold óptimo
    print("Buscando threshold óptimo...\n")
    
    thresholds = np.arange(0.35, 0.75, 0.01)
    results = []
    
    for thr in thresholds:
        metrics = simulate_trades(predictions, returns, thr, min_confidence_diff=min_diff)
        
        if metrics['n_trades'] >= min_trades:
            results.append({
                'threshold': thr,
                **metrics
            })
    
    if not results:
        print(f"❌ No se encontró threshold que genere >={min_trades} trades")
        print("   Intenta reducir min_trades o revisar el modelo")
        return
    
    # Ordenar por métrica
    if metric == "sharpe":
        results.sort(key=lambda x: x['sharpe'], reverse=True)
    else:
        results.sort(key=lambda x: x['pnl'], reverse=True)
    
    best = results[0]
    
    print(f"{'='*80}")
    print("MEJOR CONFIGURACIÓN")
    print(f"{'='*80}\n")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Trades: {best['n_trades']}")
    print(f"PnL: {best['pnl']:.2f}%")
    print(f"Sharpe: {best['sharpe']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Max Drawdown: {best['max_dd']:.2f}%\n")
    
    # Top 5
    print("Top 5 Configuraciones:\n")
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. Thr={res['threshold']:.2f} | Trades={res['n_trades']} | "
              f"PnL={res['pnl']:.1f}% | Sharpe={res['sharpe']:.2f}")
    
    # Guardar threshold óptimo
    threshold_path = model_path / "optimal_threshold.json"
    
    # Convert numpy types to python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    threshold_data = {
        'threshold': float(best['threshold']),
        'min_trades_constraint': int(min_trades),
        'optimized_for': metric,
        'backtest_metrics': {k: convert_types(v) for k, v in best.items()}
    }
    
    threshold_path.write_text(json.dumps(threshold_data, indent=2))
    print(f"\n✅ Threshold guardado en: {threshold_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
