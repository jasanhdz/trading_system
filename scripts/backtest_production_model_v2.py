"""
Backtesting pipeline para modelos de producción.
Pipeline EXACTAMENTE igual al entrenamiento para garantizar reproducibilidad.
"""
import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Añadir directorio raíz al path
sys.path.append(os.getcwd())

from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.nn_pattern.features import build_feature_frame
from data.storage.database_manager import db_manager

def load_model_artifacts(symbol, timeframe, fold=5):
    """Carga modelo, scaler, selector y configuración."""
    base_path = Path(f"models/advanced/{symbol}/{timeframe}")
    
    # Cargar configuración
    with open(base_path / "production_training_results.json", "r") as f:
        results = json.load(f)
        
    config = results["config"]
    model_config = results["model_config"]
    selected_features = results["selected_features"]
    
    # Cargar scaler y selector
    scaler = joblib.load(base_path / "scaler.pkl")
    selector = joblib.load(base_path / "feature_selector.pkl")
    
    # Reconstruir modelo
    model = DeepTemporalNet(
        input_dim=len(selected_features),
        hidden_dim=model_config["hidden_dim"],
        lstm_layers=model_config["lstm_layers"],
        dense_dims=model_config["dense_dims"],
        dropout=model_config["dropout"],
        use_attention=model_config["use_attention"],
        bidirectional=model_config["bidirectional"],
        num_classes=model_config["num_classes"],
        use_regression=model_config["use_regression"],
        num_attention_heads=model_config["num_attention_heads"]
    )
    
    # Cargar pesos (SIEMPRE en CPU para backtest)
    model_path = base_path / f"best_model_fold{fold}.pt"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, scaler, selector, config, selected_features

def prepare_backtest_data(symbol, timeframe, config, selector, scaler, limit=10000):
    """
    Prepara datos para backtest usando el MISMO pipeline que entrenamiento.
    
    Returns:
        sequences: (n_samples, sequence_length, n_features)
        indices: Timestamps correspondientes a cada secuencia
        df_aligned: DataFrame con precios alineados a las secuencias
    """
    # 1. Cargar datos brutos (probar variantes de símbolo)
    variants = [symbol]
    if "/" not in symbol:
        variants.append(symbol.replace("USDT", "/USDT"))
    else:
        variants.append(symbol.replace("/", ""))
        
    df = pd.DataFrame()
    for s in variants:
        df = db_manager.get_ohlcv_data(s, timeframe, limit=limit)
        if not df.empty:
            print(f"✅ Datos encontrados para {s}: {len(df)} velas")
            break
            
    if df.empty:
        raise ValueError(f"No se encontraron datos para {symbol} (probado: {variants})")
    
    # 2. Calcular features (MISMO pipeline que entrenamiento)
    print("⚙️  Calculando features...")
    features_df, all_feature_names = build_feature_frame(df)
    
    # CRÍTICO: Reordenar columnas según all_feature_names (orden usado en entrenamiento)
    # Esto asegura que selector.transform use las columnas correctas
    features_df = features_df[all_feature_names]
    
    print(f"   ✅ Features calculadas: {len(all_feature_names)}")
    print(f"      Primeras 5: {all_feature_names[:5]}")
    
    # 3. Alinear DataFrame de precios con features
    df_aligned = df.loc[features_df.index].copy()
    
    # 4. Aplicar Feature Selector (de 99 a 50 features)
    print(f"🎯 Aplicando selector: {len(all_feature_names)} → {selector.n_features} features")
    X_all = features_df.values.astype(np.float32)
    X_selected = selector.transform(X_all)
    
    # 5. Aplicar Scaler
    print("📊 Aplicando scaler...")
    X_scaled = scaler.transform(X_selected)
    
    # Validación: Verificar distribución de features
    print(f"   Features escaladas - Media: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    if abs(X_scaled.mean()) > 0.1 or abs(X_scaled.std() - 1.0) > 0.2:
        print("   ⚠️  WARNING: Distribución de features anormal")
    
    # 6. Crear secuencias (MISMO que SequenceDataset)
    sequence_length = config["sequence_length"]
    print(f"🔗 Creando secuencias (length={sequence_length})...")
    
    sequences = []
    indices = []
    
    # SequenceDataset usa valid_indices desde sequence_length-1
    for i in range(sequence_length - 1, len(X_scaled)):
        seq = X_scaled[i - sequence_length + 1:i + 1]  # Lookback window
        sequences.append(seq)
        indices.append(df_aligned.index[i])
    
    sequences = np.array(sequences, dtype=np.float32)
    
    print(f"✅ {len(sequences)} secuencias creadas")
    
    return sequences, indices, df_aligned

def run_backtest(
    symbol="BTCUSDT",
    timeframe="15m",
    initial_capital=10000,
    commission=0.0004,
    confidence_threshold=0.40,
    use_trend_filter=True,
    max_trades=None
):
    """
    Ejecuta backtest vectorial con protecciones institucionales.
    """
    print(f"\n{'='*70}")
    print(f"🚀 BACKTEST INSTITUCIONAL: {symbol} {timeframe}")
    print(f"{'='*70}\n")
    
    # 1. Cargar modelo y artefactos
    print("📦 Cargando modelo...")
    try:
        model, scaler, selector, config, selected_features = load_model_artifacts(symbol, timeframe)
        print(f"✅ Modelo cargado (Fold 5)")
        print(f"   Features: {len(selected_features)}")
        print(f"   Sequence length: {config['sequence_length']}")
        print(f"   Prediction horizon: {config['prediction_horizon']}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None
    
    # 2. Preparar datos
    print(f"\n📊 Preparando datos de backtest...")
    try:
        sequences, indices, df_aligned = prepare_backtest_data(
            symbol, timeframe, config, selector, scaler, limit=10000
        )
    except Exception as e:
        print(f"❌ Error preparando datos: {e}")
        return None
    
    # 3. Inferencia
    print(f"\n🔮 Generando predicciones...")
    model.eval()
    X_tensor = torch.FloatTensor(sequences)
    
    predictions = []
    probabilities = []
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Diagnóstico de predicciones
    print(f"\n🔍 DIAGNÓSTICO DE PREDICCIONES:")
    pred_counts = pd.Series(predictions).value_counts().to_dict()
    print(f"   Distribución: {pred_counts}")
    probs_neutral = [p[0] for p in probabilities]
    probs_long = [p[1] for p in probabilities]
    probs_short = [p[2] for p in probabilities]
    print(f"   Prob. Neutral - Max: {max(probs_neutral):.4f}, Media: {np.mean(probs_neutral):.4f}")
    print(f"   Prob. Long    - Max: {max(probs_long):.4f}, Media: {np.mean(probs_long):.4f}")
    print(f"   Prob. Short   - Max: {max(probs_short):.4f}, Media: {np.mean(probs_short):.4f}")
    
    # 4. Crear DataFrame de resultados
    results_df = pd.DataFrame(index=indices)
    results_df['close'] = df_aligned.loc[indices, 'close']
    results_df['prediction'] = predictions
    results_df['prob_neutral'] = probs_neutral
    results_df['prob_long'] = probs_long
    results_df['prob_short'] = probs_short
    
    # Filtro de tendencia (opcional)
    if use_trend_filter:
        results_df['ema_200'] = df_aligned.loc[indices, 'close'].ewm(span=200, adjust=False).mean()
    
    # 5. Simulación de Trading
    print(f"\n💰 Simulando trading...")
    print(f"   Capital inicial: ${initial_capital:,.2f}")
    print(f"   Comisión: {commission*100:.2f}%")
    print(f"   Umbral de confianza: {confidence_threshold:.2f}")
    print(f"   Filtro de tendencia: {'Sí (EMA 200)' if use_trend_filter else 'No'}")
    
    position = 0  # 0: Flat, 1: Long, -1: Short
    equity = initial_capital
    equity_curve = [equity]
    trades = []
    
    entry_price = 0
    entry_time = None
    trade_count = 0
    
    for i in range(1, len(results_df)):
        if max_trades and trade_count >= max_trades:
            break
            
        current_bar = results_df.iloc[i]
        timestamp = results_df.index[i]
        price = current_bar['close']
        
        pred = current_bar['prediction']
        prob_long = current_bar['prob_long']
        prob_short = current_bar['prob_short']
        
        # Señal de entrada
        signal = 0
        if pred == 1 and prob_long > confidence_threshold:
            if use_trend_filter:
                if price > current_bar['ema_200']:
                    signal = 1
            else:
                signal = 1
        elif pred == 2 and prob_short > confidence_threshold:
            if use_trend_filter:
                if price < current_bar['ema_200']:
                    signal = -1
            else:
                signal = -1
        
        # Ejecución
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                entry_time = timestamp
                equity -= equity * commission
                trade_count += 1
            elif signal == -1:
                position = -1
                entry_price = price
                entry_time = timestamp
                equity -= equity * commission
                trade_count += 1
                
        elif position == 1:
            if signal == -1 or pred == 0:
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl)
                equity -= equity * commission
                trades.append({
                    'type': 'Long',
                    'entry': entry_time,
                    'exit': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_pct': pnl * 100,
                    'equity_after': equity
                })
                
                position = 0
                if signal == -1:
                    position = -1
                    entry_price = price
                    entry_time = timestamp
                    equity -= equity * commission
                    trade_count += 1
                    
        elif position == -1:
            if signal == 1 or pred == 0:
                pnl = (entry_price - price) / entry_price
                equity *= (1 + pnl)
                equity -= equity * commission
                trades.append({
                    'type': 'Short',
                    'entry': entry_time,
                    'exit': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_pct': pnl * 100,
                    'equity_after': equity
                })
                
                position = 0
                if signal == 1:
                    position = 1
                    entry_price = price
                    entry_time = timestamp
                    equity -= equity * commission
                    trade_count += 1
                    
        equity_curve.append(equity)
    
    # 6. Métricas finales
    total_return = (equity - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS BACKTEST: {symbol} {timeframe}")
    print(f"{'='*70}")
    print(f"Capital Inicial:  ${initial_capital:,.2f}")
    print(f"Capital Final:    ${equity:,.2f}")
    print(f"Retorno Total:    {total_return:+.2f}%")
    print(f"Total Trades:     {num_trades}")
    
    if num_trades > 0:
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / num_trades * 100
        
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Avg Win:          {avg_win:+.2f}%")
        print(f"Avg Loss:         {avg_loss:+.2f}%")
        print(f"Avg PnL:          {avg_pnl:+.2f}%")
        
        if wins and losses:
            profit_factor = sum([t['pnl_pct'] for t in wins]) / abs(sum([t['pnl_pct'] for t in losses]))
            print(f"Profit Factor:    {profit_factor:.2f}")
        
        # Guardar trades
        trades_df = pd.DataFrame(trades)
        output_file = f"backtest_trades_{symbol}_{timeframe}_v2.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n📝 Trades guardados: {output_file}")
        
        # Gráfico
        plt.figure(figsize=(14, 7))
        plt.plot(equity_curve, linewidth=2)
        plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Capital Inicial')
        plt.title(f"Equity Curve - {symbol} {timeframe} (Fold 5)", fontsize=14, fontweight='bold')
        plt.ylabel("Capital ($)", fontsize=12)
        plt.xlabel("Tiempo (índice)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_file = f"backtest_equity_{symbol}_{timeframe}_v2.png"
        plt.savefig(plot_file, dpi=150)
        print(f"📈 Gráfico guardado: {plot_file}")
    else:
        print("\n⚠️  No se ejecutaron trades")
    
    print(f"{'='*70}\n")
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'initial_capital': initial_capital,
        'final_capital': equity,
        'return_pct': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate if num_trades > 0 else 0,
        'trades': trades
    }

if __name__ == "__main__":
    # Backtest institucional con validaciones
    results = run_backtest(
        symbol="ETHUSDT",
        timeframe="15m",
        initial_capital=10000,
        commission=0.0004,
        confidence_threshold=0.40,  # Ajustar según el modelo
        use_trend_filter=True,
        max_trades=None  # Sin límite
    )
