#!/usr/bin/env python3
"""
Script para entrenar ensemble de múltiples modelos.

Estrategia:
1. Entrenar 5 modelos con diferentes configuraciones y seeds
2. Guardar cada modelo individualmente
3. Crear predictor de ensemble que promedia las predicciones
4. Evaluar ensemble vs modelos individuales
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import click

from ml.advanced_models.dataset import (
    AdvancedDatasetConfig,
    SequenceDataset,
    load_sequence_dataset,
    walk_forward_split_with_validation,
)
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.temporal_model import MultiTaskLoss, EnsembleModel
from scripts.train_production_ready import (
    FocalLoss,
    CosineWarmupScheduler,
    evaluate_model,
)
from utils.logger import setup_logger

logger = setup_logger("ensemble_trainer")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()


def set_seed(seed: int):
    """Establece seed para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_single_model(
    features: np.ndarray,
    class_labels: np.ndarray,
    model_config: dict,
    training_config: dict,
    device: torch.device,
    seed: int,
    model_id: int,
    save_dir: Path,
):
    """Entrena un modelo individual del ensemble."""

    set_seed(seed)

    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 Entrenando Modelo {model_id} (seed={seed})")
    logger.info(f"{'='*70}")

    config = training_config['config']

    # Walk-forward split con validación
    splits = walk_forward_split_with_validation(
        n_samples=len(features),
        n_splits=3,  # Menos folds para ensemble (más rápido)
        train_ratio=0.6,
        val_ratio=0.2,
        gap=config.prediction_horizon,
    )

    best_models = []
    best_val_f1 = -float('inf')

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, 1):
        logger.info(f"  Fold {fold_idx}/{len(splits)}")

        # Split data
        train_features = features[train_idx]
        train_labels = class_labels[train_idx]
        val_features = features[val_idx]
        val_labels = class_labels[val_idx]

        # Datasets
        train_dataset = SequenceDataset(
            train_features,
            train_labels.reshape(-1, 1),
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=True,
            augmentation_noise=0.01,
        )

        val_dataset = SequenceDataset(
            val_features,
            val_labels.reshape(-1, 1),
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Modelo
        model = DeepTemporalNet(
            input_dim=features.shape[1],
            sequence_length=config.sequence_length,
            **model_config,
        ).to(device)

        # Loss
        classification_criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=3)
        criterion = MultiTaskLoss(
            class_weights=None,
            classification_weight=1.0,
            regression_weight=0.2,
        )
        criterion.classification_criterion = classification_criterion

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
        )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=training_config['warmup_epochs'],
            max_epochs=training_config['epochs'],
        )

        # Training loop
        epochs_without_improvement = 0
        fold_best_val_f1 = -float('inf')

        for epoch in range(training_config['epochs']):
            model.train()
            train_loss = 0.0

            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(device)
                batch_labels = batch_labels.squeeze(-1).to(device)

                optimizer.zero_grad()
                outputs = model(batch_seq)
                loss, _ = criterion(
                    outputs['logits'],
                    batch_labels,
                    outputs.get('regression'),
                    None,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validación
            val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)
            scheduler.step(epoch)

            if val_metrics['macro_f1'] > fold_best_val_f1:
                fold_best_val_f1 = val_metrics['macro_f1']
                epochs_without_improvement = 0

                # Guardar mejor modelo del fold
                model_copy = DeepTemporalNet(
                    input_dim=features.shape[1],
                    sequence_length=config.sequence_length,
                    **model_config,
                )
                model_copy.load_state_dict(model.state_dict())
                best_models.append((fold_best_val_f1, model_copy))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= training_config['patience']:
                break

        logger.info(f"    ✓ Fold {fold_idx} Val F1: {fold_best_val_f1:.4f}")

        if fold_best_val_f1 > best_val_f1:
            best_val_f1 = fold_best_val_f1

    # Guardar el mejor modelo
    best_models.sort(key=lambda x: x[0], reverse=True)
    best_model = best_models[0][1]

    model_path = save_dir / f"ensemble_model_{model_id}.pt"
    torch.save(best_model.state_dict(), model_path)

    logger.info(f"  ✅ Modelo {model_id} guardado (Val F1: {best_val_f1:.4f})")

    return best_model, best_val_f1


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="5m", help="Timeframe")
@click.option("--n-models", default=5, help="Número de modelos en ensemble")
@click.option("--epochs", default=150, help="Training epochs por modelo")
@click.option("--device", default="cuda", help="Device (cuda/cpu)")
def main(
    symbol: str,
    timeframe: str,
    n_models: int,
    epochs: int,
    device: str,
):
    """Entrena ensemble de modelos."""

    print("\n" + "="*80)
    print("🎯 ENTRENAMIENTO DE ENSEMBLE")
    print("="*80 + "\n")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Usando device: {device}")

    # Configuraciones para diferentes modelos del ensemble
    ensemble_configs = [
        {
            'name': 'deep_192_seq48',
            'seed': 42,
            'model_config': {
                'hidden_dim': 192,
                'lstm_layers': 3,
                'dense_dims': (384, 256, 128),
                'dropout': 0.35,
                'use_attention': True,
                'bidirectional': True,
                'num_classes': 3,
                'use_regression': True,
                'num_attention_heads': 8,
            },
            'sequence_length': 48,
            'prediction_horizon': 6,
        },
        {
            'name': 'deep_128_seq48',
            'seed': 123,
            'model_config': {
                'hidden_dim': 128,
                'lstm_layers': 3,
                'dense_dims': (256, 128, 64),
                'dropout': 0.3,
                'use_attention': True,
                'bidirectional': True,
                'num_classes': 3,
                'use_regression': True,
                'num_attention_heads': 4,
            },
            'sequence_length': 48,
            'prediction_horizon': 6,
        },
        {
            'name': 'deep_192_seq64',
            'seed': 456,
            'model_config': {
                'hidden_dim': 192,
                'lstm_layers': 2,
                'dense_dims': (384, 256, 128),
                'dropout': 0.35,
                'use_attention': True,
                'bidirectional': True,
                'num_classes': 3,
                'use_regression': True,
                'num_attention_heads': 8,
            },
            'sequence_length': 64,
            'prediction_horizon': 6,
        },
        {
            'name': 'deep_160_seq48',
            'seed': 789,
            'model_config': {
                'hidden_dim': 160,
                'lstm_layers': 3,
                'dense_dims': (320, 256, 128),
                'dropout': 0.32,
                'use_attention': True,
                'bidirectional': True,
                'num_classes': 3,
                'use_regression': True,
                'num_attention_heads': 8,
            },
            'sequence_length': 48,
            'prediction_horizon': 6,
        },
        {
            'name': 'deep_192_seq96',
            'seed': 999,
            'model_config': {
                'hidden_dim': 192,
                'lstm_layers': 2,
                'dense_dims': (384, 256, 128),
                'dropout': 0.35,
                'use_attention': True,
                'bidirectional': True,
                'num_classes': 3,
                'use_regression': True,
                'num_attention_heads': 8,
            },
            'sequence_length': 96,
            'prediction_horizon': 6,
        },
    ]

    # Usar solo n_models
    ensemble_configs = ensemble_configs[:n_models]

    logger.info(f"📊 Entrenando {len(ensemble_configs)} modelos")

    # Directorio de guardado
    save_dir = MODEL_DIR / symbol / timeframe / "ensemble"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos UNA VEZ (usar la config del primer modelo)
    logger.info("📊 Cargando datos...")
    config = AdvancedDatasetConfig(
        symbol=symbol.replace("USDT", "/USDT") + ":USDT",
        timeframe=timeframe,
        sequence_length=48,  # Usar el más común
        prediction_horizon=6,
        target_return=0.005,
        max_history_days=1100,
        use_feature_selection=True,
        n_features_to_select=50,
        feature_selection_method="mutual_info",
    )

    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)

    logger.info(f"✓ Dataset: {len(features)} muestras, {len(feature_names)} features")

    # Feature selection y scaling
    from ml.advanced_models.dataset import FeatureSelector

    selector = FeatureSelector(
        method="mutual_info",
        n_features=50,
    )

    sample_idx = np.random.choice(len(features), min(10000, len(features)), replace=False)
    selected_features = selector.fit(features[sample_idx], class_labels[sample_idx], feature_names)
    features = selector.transform(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Guardar selector y scaler
    import joblib
    joblib.dump(selector, save_dir / "feature_selector.pkl")
    joblib.dump(scaler, save_dir / "scaler.pkl")

    # Configuración de entrenamiento
    training_config = {
        'config': config,
        'epochs': epochs,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'patience': 20,
    }

    # Entrenar cada modelo
    models = []
    model_scores = []

    for i, ens_config in enumerate(ensemble_configs, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Modelo {i}/{len(ensemble_configs)}: {ens_config['name']}")
        logger.info(f"{'#'*80}")

        # Actualizar sequence_length en config temporal
        temp_config = AdvancedDatasetConfig(
            symbol=config.symbol,
            timeframe=config.timeframe,
            sequence_length=ens_config['sequence_length'],
            prediction_horizon=ens_config['prediction_horizon'],
            target_return=config.target_return,
            max_history_days=config.max_history_days,
        )

        training_config['config'] = temp_config

        model, val_f1 = train_single_model(
            features=features,
            class_labels=class_labels,
            model_config=ens_config['model_config'],
            training_config=training_config,
            device=device,
            seed=ens_config['seed'],
            model_id=i,
            save_dir=save_dir,
        )

        models.append(model)
        model_scores.append(val_f1)

    # Crear ensemble
    logger.info(f"\n{'='*80}")
    logger.info("🎯 CREANDO ENSEMBLE")
    logger.info(f"{'='*80}")

    # Pesos basados en rendimiento (mejor modelo tiene más peso)
    weights = np.array(model_scores)
    weights = weights / weights.sum()

    logger.info(f"Pesos del ensemble:")
    for i, (w, score) in enumerate(zip(weights, model_scores), 1):
        logger.info(f"  Modelo {i}: weight={w:.3f}, val_f1={score:.4f}")

    # Guardar ensemble
    ensemble = EnsembleModel(models, weights.tolist())

    # Guardar metadata
    ensemble_metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'n_models': len(models),
        'models': [
            {
                'id': i,
                'name': cfg['name'],
                'val_f1': float(score),
                'weight': float(w),
                'config': cfg['model_config'],
            }
            for i, (cfg, score, w) in enumerate(zip(ensemble_configs, model_scores, weights), 1)
        ],
        'avg_val_f1': float(np.mean(model_scores)),
        'best_val_f1': float(np.max(model_scores)),
    }

    with open(save_dir / "ensemble_metadata.json", "w") as f:
        json.dump(ensemble_metadata, f, indent=2)

    logger.info(f"\n✅ Ensemble guardado en: {save_dir}")
    logger.info(f"📊 Promedio Val F1: {np.mean(model_scores):.4f}")
    logger.info(f"🏆 Mejor individual: {np.max(model_scores):.4f}")

    print("\n" + "="*80)
    print("✅ ENSEMBLE COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
