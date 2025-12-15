#!/usr/bin/env python3
"""
Calibración de Probabilidades para Modelos ML de Trading

Este script toma un modelo entrenado y calibra sus probabilidades usando
Isotonic Regression o Platt Scaling para mejorar la confiabilidad.

Usage:
    python scripts/calibrate_model.py --symbol BTCUSDT --timeframe 1h
"""
import sys
from pathlib import Path
import click
import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.advanced_models.predictor import AdvancedPredictor
from ml.advanced_models.dataset import load_sequence_dataset, AdvancedDatasetConfig
from utils.logger import setup_logger

logger = setup_logger("calibrate_model")

MODEL_DIR = REPO_ROOT / "models" / "advanced"


def _symbol_key(symbol: str) -> str:
    """Create filesystem-safe symbol key."""
    return symbol.replace("/", "").replace(":", "").replace("-", "").upper()


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--method", default="isotonic", type=click.Choice(["isotonic", "platt"]), help="Calibration method")
def main(symbol: str, timeframe: str, method: str):
    """Calibrate model probabilities for better reliability."""
    
    print(f"\n{'='*80}")
    print(f"CALIBRANDO MODELO: {symbol} {timeframe}")
    print(f"Método: {method}")
    print(f"{'='*80}\n")
    
    # Cargar modelo
    symbol_key = _symbol_key(symbol)
    model_path = MODEL_DIR / symbol_key / timeframe
    
    if not model_path.exists():
        logger.error(f"Modelo no encontrado en: {model_path}")
        return
    
    # Cargar predictor
    predictor = AdvancedPredictor(
        model_path=model_path,
        scaler_path=model_path / "scaler.pkl",
        meta_path=model_path / "meta.json"
    )
    
    # Cargar datos de validación (últimos 20% de los datos)
    meta_path = model_path / "meta.json"
    import json
    meta = json.loads(meta_path.read_text())
    
    config = AdvancedDatasetConfig(
        symbol=meta['symbol'],
        timeframe=meta['timeframe'],
        sequence_length=meta['sequence_length'],
        prediction_horizon=meta['prediction_horizon'],
        target_return=meta['target_return'],
        max_history_days=300,  # Subset para calibración
    )
    
    # Cargar datos
    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)
    
    # Usar últimos 20% para calibración
    n_samples = len(features)
    cal_start = int(n_samples * 0.8)
    
    X_cal = features[cal_start:]
    y_cal = class_labels[cal_start:]
    
    print(f"Datos de calibración: {len(X_cal)} muestras\n")
    
    # Obtener probabilidades sin calibrar
    print("Obteniendo probabilidades originales...")
    raw_probs = []
    
    for i in range(len(X_cal)):
        # Usar predictor para obtener probabilidades
        window = X_cal[max(0, i-config.sequence_length):i+1]
        if len(window) < config.sequence_length:
            continue
            
        pred = predictor.predict(window[-config.sequence_length:])
        raw_probs.append(pred['class_probs'])
    
    raw_probs = np.array(raw_probs)
    y_cal = y_cal[-len(raw_probs):]  # Ajustar longitud
    
    print(f"Probabilidades obtenidas: {raw_probs.shape}\n")
    
    # Calibrar cada clase
    print(f"Calibrando con {method}...")
    calibrators = []
    
    for class_idx in range(3):
        y_binary = (y_cal == class_idx).astype(int)
        probs_class = raw_probs[:, class_idx]
        
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs_class, y_binary)
        else:  # platt
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
            calibrator.fit(probs_class.reshape(-1, 1), y_binary)
        
        calibrators.append(calibrator)
        print(f"  Clase {class_idx}: Calibrado")
    
    # Guardar calibradores
    calibrator_path = model_path / f"calibrator_{method}.pkl"
    joblib.dump(calibrators, calibrator_path)
    print(f"\n✅ Calibradores guardados en: {calibrator_path}")
    
    # Evaluar mejora
    print(f"\n{'='*80}")
    print("COMPARACIÓN PRE/POST CALIBRACIÓN")
    print(f"{'='*80}\n")
    
    # Probabilidades calibradas
    cal_probs = np.zeros_like(raw_probs)
    for class_idx, calibrator in enumerate(calibrators):
        if method == "isotonic":
            cal_probs[:, class_idx] = calibrator.transform(raw_probs[:, class_idx])
        else:
            cal_probs[:, class_idx] = calibrator.predict_proba(raw_probs[:, class_idx].reshape(-1, 1))[:, 1]
    
    # Normalizar
    cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
    
    # Mostrar diferencias
    for class_idx, class_name in enumerate(['Neutral', 'Long', 'Short']):
        y_binary = (y_cal == class_idx).astype(int)
        
        # Brier score (lower is better)
        brier_raw = np.mean((raw_probs[:, class_idx] - y_binary) ** 2)
        brier_cal = np.mean((cal_probs[:, class_idx] - y_binary) ** 2)
        
        print(f"{class_name}:")
        print(f"  Brier Score RAW: {brier_raw:.4f}")
        print(f"  Brier Score CAL: {brier_cal:.4f}")
        print(f"  Mejora: {((brier_raw - brier_cal) / brier_raw * 100):.2f}%\n")
    
    print(f"{'='*80}")
    print("CALIBRACIÓN COMPLETA")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
