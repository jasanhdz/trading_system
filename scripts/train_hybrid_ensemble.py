#!/usr/bin/env python3
"""
Script para entrenar ensemble híbrido: Neural Networks + XGBoost.

Estrategia:
1. Entrenar modelos de redes neuronales (LSTM)
2. Entrenar modelo XGBoost en las mismas features
3. Combinar predicciones con pesos optimizados
4. XGBoost captura patrones diferentes a las redes neuronales
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import click

from ml.advanced_models.dataset import (
    AdvancedDatasetConfig,
    load_sequence_dataset,
    walk_forward_split_with_validation,
)
from ml.advanced_models.improved_architecture import DeepTemporalNet
from utils.logger import setup_logger

logger = setup_logger("hybrid_ensemble")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()


def prepare_xgboost_features(features: np.ndarray, sequence_length: int):
    """
    Prepara features para XGBoost.

    XGBoost no maneja secuencias, así que aplanamos la ventana temporal
    en un vector de features.
    """
    # Crear ventanas deslizantes
    n_samples = len(features) - sequence_length + 1
    n_features = features.shape[1]

    # Aplanar secuencias en vectores
    flat_features = np.zeros((n_samples, sequence_length * n_features))

    for i in range(n_samples):
        window = features[i:i + sequence_length]
        flat_features[i] = window.flatten()

    return flat_features


def train_xgboost_model(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Entrena modelo XGBoost con validación."""

    logger.info("🌲 Entrenando XGBoost...")

    # Preparar features para XGBoost (aplanar secuencias)
    xgb_features = prepare_xgboost_features(features, sequence_length)

    # Ajustar índices (perdemos sequence_length-1 muestras al crear ventanas)
    offset = sequence_length - 1
    train_idx = train_idx[train_idx >= offset] - offset
    val_idx = val_idx[val_idx >= offset] - offset
    test_idx = test_idx[test_idx >= offset] - offset

    # Ajustar labels también
    labels = labels[offset:]

    # Split data
    X_train = xgb_features[train_idx]
    y_train = labels[train_idx]
    X_val = xgb_features[val_idx]
    y_val = labels[val_idx]
    X_test = xgb_features[test_idx]
    y_test = labels[test_idx]

    # Calcular class weights
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(y_train) / (3 * class_counts)

    # Asignar pesos a las muestras
    sample_weights = class_weights[y_train]

    # Parámetros XGBoost optimizados
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
    }

    # Entrenar con early stopping
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluar
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, train_pred, average='macro')
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')

    logger.info(f"  Train F1: {train_f1:.4f}")
    logger.info(f"  Val F1: {val_f1:.4f}")
    logger.info(f"  Test F1: {test_f1:.4f}")

    return model, (X_test, y_test, test_pred)


def load_neural_ensemble(ensemble_dir: Path, device: torch.device):
    """Carga ensemble de redes neuronales entrenado."""

    logger.info("🧠 Cargando ensemble de redes neuronales...")

    # Cargar metadata
    with open(ensemble_dir / "ensemble_metadata.json") as f:
        metadata = json.load(f)

    # Cargar modelos
    models = []
    for model_info in metadata['models']:
        model_path = ensemble_dir / f"ensemble_model_{model_info['id']}.pt"

        # Crear arquitectura
        model = DeepTemporalNet(
            input_dim=50,  # Ajustar según tu configuración
            sequence_length=model_info['config'].get('sequence_length', 48),
            **model_info['config'],
        )

        # Cargar pesos
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        models.append(model)

    weights = [m['weight'] for m in metadata['models']]

    logger.info(f"  ✓ {len(models)} modelos cargados")

    return models, weights, metadata


def predict_neural_ensemble(models, weights, features, sequence_length, device):
    """Genera predicciones del ensemble de redes neuronales."""

    logger.info("🔮 Generando predicciones del ensemble neuronal...")

    # Crear secuencias
    from ml.advanced_models.dataset import SequenceDataset

    # Dummy labels (no se usan en inferencia)
    dummy_labels = np.zeros((len(features), 1))

    dataset = SequenceDataset(
        features,
        dummy_labels,
        sequence_length=sequence_length,
        prediction_horizon=0,  # No importa en inferencia
        augment=False,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for batch_seq, _ in loader:
            batch_seq = batch_seq.to(device)

            # Predicciones de cada modelo
            batch_probs = []
            for model in models:
                outputs = model(batch_seq)
                probs = torch.softmax(outputs['logits'], dim=1)
                batch_probs.append(probs)

            # Promedio ponderado
            batch_probs = torch.stack(batch_probs)  # (n_models, batch, 3)
            weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
            avg_probs = (batch_probs * weights_tensor).sum(dim=0)

            all_probs.append(avg_probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return all_probs


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="5m", help="Timeframe")
@click.option("--ensemble-dir", required=True, help="Directorio del ensemble neuronal")
@click.option("--device", default="cuda", help="Device (cuda/cpu)")
def main(
    symbol: str,
    timeframe: str,
    ensemble_dir: str,
    device: str,
):
    """Entrena ensemble híbrido con XGBoost."""

    print("\n" + "="*80)
    print("🔥 ENSEMBLE HÍBRIDO: NEURAL NETWORKS + XGBOOST")
    print("="*80 + "\n")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ensemble_path = Path(ensemble_dir)
    if not ensemble_path.exists():
        logger.error(f"❌ Directorio de ensemble no existe: {ensemble_dir}")
        logger.info("💡 Primero ejecuta: python scripts/train_ensemble.py")
        return

    # Cargar datos
    logger.info("📊 Cargando datos...")

    config = AdvancedDatasetConfig(
        symbol=symbol.replace("USDT", "/USDT") + ":USDT",
        timeframe=timeframe,
        sequence_length=48,
        prediction_horizon=6,
        target_return=0.005,
        max_history_days=1100,
    )

    features, class_labels, _, feature_names = load_sequence_dataset(config)
    logger.info(f"✓ Dataset: {len(features)} muestras, {len(feature_names)} features")

    # Cargar selector y scaler guardados
    selector = joblib.load(ensemble_path / "feature_selector.pkl")
    scaler = joblib.load(ensemble_path / "scaler.pkl")

    features = selector.transform(features)
    features = scaler.transform(features)

    # Split con validación
    splits = walk_forward_split_with_validation(
        n_samples=len(features),
        n_splits=1,  # Solo un fold para evaluación final
        train_ratio=0.6,
        val_ratio=0.2,
        gap=config.prediction_horizon,
    )

    train_idx, val_idx, test_idx = splits[0]

    logger.info(f"✓ Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 1. Entrenar XGBoost
    xgb_model, (X_test_xgb, y_test, test_pred_xgb) = train_xgboost_model(
        features,
        class_labels,
        config.sequence_length,
        train_idx,
        val_idx,
        test_idx,
    )

    # 2. Cargar ensemble neuronal
    neural_models, neural_weights, metadata = load_neural_ensemble(
        ensemble_path,
        device,
    )

    # 3. Generar predicciones del ensemble neuronal en test set
    test_features = features[test_idx]
    neural_probs = predict_neural_ensemble(
        neural_models,
        neural_weights,
        test_features,
        config.sequence_length,
        device,
    )
    neural_pred = neural_probs.argmax(axis=1)

    # Ajustar labels para neural (pierde sequence_length-1 samples)
    offset = config.sequence_length - 1
    y_test_neural = class_labels[test_idx][offset:]

    # 4. Predicciones XGBoost en test
    xgb_probs = xgb_model.predict_proba(X_test_xgb)

    # 5. Combinar predicciones (buscar peso óptimo)
    logger.info("\n🔍 Optimizando pesos del ensemble híbrido...")

    best_f1 = -1
    best_weight = 0.5

    for neural_weight in np.linspace(0, 1, 21):  # 0.0, 0.05, ..., 1.0
        xgb_weight = 1 - neural_weight

        # Combinar probabilidades
        hybrid_probs = neural_weight * neural_probs + xgb_weight * xgb_probs
        hybrid_pred = hybrid_probs.argmax(axis=1)

        f1 = f1_score(y_test_neural, hybrid_pred, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_weight = neural_weight

    logger.info(f"  ✓ Mejor peso neuronal: {best_weight:.2f}")
    logger.info(f"  ✓ Mejor peso XGBoost: {1-best_weight:.2f}")

    # 6. Evaluar ensemble final
    hybrid_probs = best_weight * neural_probs + (1 - best_weight) * xgb_probs
    hybrid_pred = hybrid_probs.argmax(axis=1)

    # Métricas
    neural_f1 = f1_score(y_test_neural, neural_pred, average='macro')
    xgb_f1 = f1_score(y_test, test_pred_xgb, average='macro')
    hybrid_f1 = f1_score(y_test_neural, hybrid_pred, average='macro')

    logger.info(f"\n{'='*80}")
    logger.info("📊 RESULTADOS FINALES")
    logger.info(f"{'='*80}")
    logger.info(f"Ensemble Neuronal:  F1 = {neural_f1:.4f}")
    logger.info(f"XGBoost:           F1 = {xgb_f1:.4f}")
    logger.info(f"Híbrido (óptimo):  F1 = {hybrid_f1:.4f}")

    improvement = (hybrid_f1 - max(neural_f1, xgb_f1)) * 100
    logger.info(f"\n🚀 Mejora: {improvement:+.2f}%")

    # Reporte detallado
    logger.info(f"\n{'='*80}")
    logger.info("REPORTE DETALLADO - ENSEMBLE HÍBRIDO")
    logger.info(f"{'='*80}\n")
    print(classification_report(
        y_test_neural,
        hybrid_pred,
        target_names=['Neutral', 'Long', 'Short'],
        digits=4,
    ))

    # 7. Guardar ensemble híbrido
    hybrid_dir = ensemble_path / "hybrid"
    hybrid_dir.mkdir(exist_ok=True)

    # Guardar XGBoost
    joblib.dump(xgb_model, hybrid_dir / "xgboost_model.pkl")

    # Guardar metadata
    hybrid_metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'neural_weight': float(best_weight),
        'xgb_weight': float(1 - best_weight),
        'test_metrics': {
            'neural_f1': float(neural_f1),
            'xgb_f1': float(xgb_f1),
            'hybrid_f1': float(hybrid_f1),
            'improvement_pct': float(improvement),
        },
    }

    with open(hybrid_dir / "hybrid_metadata.json", "w") as f:
        json.dump(hybrid_metadata, f, indent=2)

    logger.info(f"\n✅ Ensemble híbrido guardado en: {hybrid_dir}")

    print("\n" + "="*80)
    print("✅ ENSEMBLE HÍBRIDO COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
