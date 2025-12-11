#!/usr/bin/env python3
"""
Script de validación End-to-End para el AdvancedPredictor.
Carga el modelo entrenado y realiza inferencia sobre datos recientes para verificar
la integridad del pipeline (carga de artefactos, alineación de features, ejecución).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import logging
import pandas as pd
import torch
from ml.advanced_models.predictor import AdvancedPredictor
from data.storage.database_manager import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger("predictor_validator")

def validate_predictor(symbol="BTCUSDT", timeframe="5m"):
    logger.info(f"🔍 Iniciando validación End-to-End para {symbol} {timeframe}")
    
    # 1. Localizar artefactos
    model_dir = REPO_ROOT / "models" / "advanced" / symbol / timeframe
    if not model_dir.exists():
        logger.error(f"❌ Directorio de modelo no encontrado: {model_dir}")
        return False
        
    logger.info(f"📂 Directorio de modelo: {model_dir}")
    
    # 2. Inicializar Predictor
    try:
        predictor = AdvancedPredictor(
            model_path=model_dir,
            scaler_path=model_dir, # Ahora es un directorio
            meta_path=model_dir / "meta.json",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("✅ AdvancedPredictor inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error al inicializar Predictor: {e}")
        return False

    # 3. Cargar datos recientes para prueba
    logger.info("📊 Cargando datos recientes para inferencia...")
    db = DatabaseManager()
    # Cargar suficientes datos para cubrir sequence_length + un poco más
    required_len = predictor.sequence_length + 100
    
    # Usamos una consulta directa o el método get_ohlcv_data
    # Asumimos que hay datos en la DB. Si no, esto fallará.
    df = db.get_ohlcv_data(symbol, timeframe, limit=required_len)
    
    if df.empty or len(df) < predictor.sequence_length:
        logger.error(f"❌ Datos insuficientes en DB. Se necesitan {predictor.sequence_length}, se obtuvieron {len(df)}")
        return False
        
    logger.info(f"✓ Datos cargados: {len(df)} velas")

    # 4. Prueba de predicción simple (última secuencia)
    logger.info("🧪 Ejecutando predicción simple (última vela)...")
    try:
        prediction = predictor.predict(df)
        logger.info(f"✅ Predicción exitosa: {prediction}")
    except Exception as e:
        logger.error(f"❌ Error en predicción simple: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Prueba de predicción por lotes (Batch)
    logger.info("🧪 Ejecutando predicción por lotes (últimas 50 velas)...")
    try:
        # Pasamos el DF completo, predict_batch se encarga del sliding window
        batch_predictions = predictor.predict_batch(df)
        
        probs = batch_predictions['probabilities']
        preds = batch_predictions['predictions']
        
        logger.info(f"✅ Predicción por lotes exitosa. Shape: {probs.shape}")
        logger.info(f"   Muestra de predicciones: {preds[:10]}")
        
        if 'predicted_returns' in batch_predictions:
             logger.info(f"   Muestra de retornos predichos: {batch_predictions['predicted_returns'][:5]}")
             
    except Exception as e:
        logger.error(f"❌ Error en predicción por lotes: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "="*50)
    logger.info("🎉 VALIDACIÓN END-TO-END COMPLETADA CON ÉXITO")
    logger.info("="*50)
    return True

if __name__ == "__main__":
    validate_predictor()
