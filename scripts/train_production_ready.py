#!/usr/bin/env python3
"""
Script de entrenamiento mejorado para producción.

Mejoras implementadas:
1. Features de régimen de mercado (99 features totales)
2. Arquitectura más profunda y poderosa
3. Split con validación (train/val/test)
4. Focal Loss para manejar desbalanceo
5. Mejores métricas de evaluación
6. Extensión de días de entrenamiento
7. Configuración optimizada para producción
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import torch
import torch.nn as nn
import numpy as np

# Mixed precision - compatible con ROCm y CUDA
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import click

from ml.advanced_models.dataset import (
    AdvancedDatasetConfig,
    SequenceDataset,
    load_sequence_dataset,
    walk_forward_split_with_validation,
)
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.temporal_model import MultiTaskLoss
from utils.logger import setup_logger

logger = setup_logger("production_trainer")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalanceo de clases.

    Mejor que class weighting porque reduce la importancia de
    ejemplos fáciles y se enfoca en ejemplos difíciles.
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CosineWarmupScheduler:
    """Learning rate scheduler con warmup y cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale


def evaluate_model(model, dataloader, device, criterion):
    """Evalúa el modelo y retorna métricas detalladas."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_seq, batch_labels in dataloader:
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.squeeze(-1).to(device)

            outputs = model(batch_seq)
            loss, _ = criterion(
                outputs['logits'],
                batch_labels,
                outputs.get('regression'),
                None,
            )

            total_loss += loss.item()

            probs = torch.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calcular métricas
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Por clase
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'neutral_f1': per_class_f1[0],
        'long_f1': per_class_f1[1],
        'short_f1': per_class_f1[2],
        'neutral_precision': per_class_precision[0],
        'long_precision': per_class_precision[1],
        'short_precision': per_class_precision[2],
        'neutral_recall': per_class_recall[0],
        'long_recall': per_class_recall[1],
        'short_recall': per_class_recall[2],
    }

    return metrics, all_preds, all_labels


def train_production_model(
    config: AdvancedDatasetConfig,
    model_config: dict,
    device: str = "cuda",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 15,
    patience: int = 30,
    gradient_clip: float = 1.0,
    use_amp: bool = True,
    use_focal_loss: bool = True,
    save_dir: Path = None,
):
    """Entrena modelo con todas las mejoras para producción."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Detectar ROCm y desactivar AMP si es necesario
    is_rocm = bool(getattr(torch.version, "hip", None))

    if is_rocm:
        if use_amp:
            logger.warning("⚠️  ROCm detectado. Desactivando Mixed Precision (AMP)")
            logger.warning("    (Mixed precision puede causar problemas en algunas versiones de ROCm)")
            use_amp = False
        else:
            logger.info("ℹ️  ROCm detectado. Configuración optimizada para AMD GPUs")

    logger.info(f"🚀 Usando device: {device}")
    logger.info(f"✓ Modelo: DeepTemporalNet (arquitectura profunda optimizada)")
    if use_amp:
        logger.info("✓ Mixed Precision (AMP): Activado")
    else:
        logger.info("✓ Mixed Precision (AMP): Desactivado")

    # Cargar datos
    logger.info("📊 Cargando datos...")
    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)

    logger.info(f"✓ Dataset: {len(features)} muestras, {len(feature_names)} features")
    logger.info(f"✓ Features incluyen régimen de mercado: {len(feature_names)} totales")

    # Feature selection
    from ml.advanced_models.dataset import FeatureSelector

    if config.use_feature_selection and len(feature_names) > config.n_features_to_select:
        logger.info(f"🎯 Seleccionando top {config.n_features_to_select} features...")
        selector = FeatureSelector(
            method=config.feature_selection_method,
            n_features=config.n_features_to_select,
        )

        sample_idx = np.random.choice(len(features), min(10000, len(features)), replace=False)
        selected_features = selector.fit(features[sample_idx], class_labels[sample_idx], feature_names)
        features = selector.transform(features)
        feature_names = selected_features

        logger.info(f"✓ Features seleccionadas: {len(feature_names)}")

        # Guardar selector
        if save_dir:
            import joblib
            joblib.dump(selector, save_dir / "feature_selector.pkl")

    # Escalar features
    logger.info("⚙️  Escalando features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if save_dir:
        import joblib
        joblib.dump(scaler, save_dir / "scaler.pkl")

    # Walk-forward splits CON VALIDACIÓN
    logger.info("📈 Creando walk-forward splits con validación...")
    splits = walk_forward_split_with_validation(
        n_samples=len(features),
        n_splits=5,
        train_ratio=0.6,  # 60% train
        val_ratio=0.2,    # 20% validation
        gap=config.prediction_horizon,
    )

    logger.info(f"✓ {len(splits)} folds creados (train/val/test)")

    all_fold_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FOLD {fold_idx}/{len(splits)}")
        logger.info(f"{'='*70}")
        logger.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

        # Split data
        train_features = features[train_idx]
        train_labels = class_labels[train_idx]
        val_features = features[val_idx]
        val_labels = class_labels[val_idx]
        test_features = features[test_idx]
        test_labels = class_labels[test_idx]

        # Crear datasets
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

        test_dataset = SequenceDataset(
            test_features,
            test_labels.reshape(-1, 1),
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=False,
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Crear modelo profundo optimizado
        model = DeepTemporalNet(
            input_dim=features.shape[1],
            sequence_length=config.sequence_length,
            **model_config,
        ).to(device)

        logger.info(f"🧠 Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")

        # Loss y optimizer
        if use_focal_loss:
            classification_criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=3).to(device)
            logger.info("✓ Usando Focal Loss para clasificación")
        else:
            # Class weights tradicional
            class_counts = np.bincount(train_labels, minlength=3)
            if (class_counts > 0).all():
                inv_weights = class_counts.sum() / class_counts
                class_weights = torch.from_numpy(
                    (inv_weights / inv_weights.mean()).astype(np.float32)
                ).to(device)
            else:
                class_weights = None

            classification_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

        criterion = MultiTaskLoss(
            class_weights=None,  # Ya está en classification_criterion
            classification_weight=1.0,
            regression_weight=0.2,
        ).to(device)  # Mover criterion a GPU
        criterion.classification_criterion = classification_criterion

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
        )

        # GradScaler con device especificado
        if use_amp:
            try:
                scaler_amp = GradScaler('cuda')
            except TypeError:
                # Fallback para versiones antiguas de PyTorch
                scaler_amp = GradScaler()
        else:
            scaler_amp = None

        # Training loop
        best_val_f1 = -float('inf')
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(device)
                batch_labels = batch_labels.squeeze(-1).to(device)

                optimizer.zero_grad()

                if use_amp:
                    # autocast con device especificado
                    try:
                        autocast_context = autocast('cuda')
                    except TypeError:
                        # Fallback para versiones antiguas
                        autocast_context = autocast()

                    with autocast_context:
                        outputs = model(batch_seq)
                        loss, _ = criterion(
                            outputs['logits'],
                            batch_labels,
                            outputs.get('regression'),
                            None,
                        )

                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    outputs = model(batch_seq)
                    loss, _ = criterion(
                        outputs['logits'],
                        batch_labels,
                        outputs.get('regression'),
                        None,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validación
            val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)

            # Learning rate step
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val F1: {val_metrics['macro_f1']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping basado en validación
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_epoch = epoch
                epochs_without_improvement = 0

                # Guardar mejor modelo
                if save_dir:
                    torch.save(model.state_dict(), save_dir / f"best_model_fold{fold_idx}.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"⏹️  Early stopping en epoch {epoch+1}")
                break

        # Cargar mejor modelo para evaluación en test
        if save_dir:
            model.load_state_dict(torch.load(save_dir / f"best_model_fold{fold_idx}.pt"))

        # Evaluación final en test set
        test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device, criterion)

        logger.info(f"\n📊 Resultados Fold {fold_idx}:")
        logger.info(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch+1})")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"  Long F1: {test_metrics['long_f1']:.4f}")
        logger.info(f"  Short F1: {test_metrics['short_f1']:.4f}")

        fold_result = {
            'fold': fold_idx,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        }
        all_fold_results.append(fold_result)

    # Promedios finales
    avg_test_acc = np.mean([r['test_metrics']['accuracy'] for r in all_fold_results])
    avg_test_f1 = np.mean([r['test_metrics']['macro_f1'] for r in all_fold_results])
    avg_long_f1 = np.mean([r['test_metrics']['long_f1'] for r in all_fold_results])
    avg_short_f1 = np.mean([r['test_metrics']['short_f1'] for r in all_fold_results])

    logger.info(f"\n{'='*70}")
    logger.info("🎯 RESULTADOS FINALES")
    logger.info(f"{'='*70}")
    logger.info(f"Avg Test Accuracy: {avg_test_acc:.4f}")
    logger.info(f"Avg Test Macro F1: {avg_test_f1:.4f}")
    logger.info(f"Avg Long F1: {avg_long_f1:.4f}")
    logger.info(f"Avg Short F1: {avg_short_f1:.4f}")

    return all_fold_results


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="5m", help="Timeframe")
@click.option("--epochs", default=200, help="Training epochs")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--sequence-length", default=48, help="Lookback window")
@click.option("--prediction-horizon", default=6, help="Forecast horizon")
@click.option("--target-return", default=0.005, help="Target return threshold")
@click.option("--max-history-days", default=1100, help="Maximum days of history")
@click.option("--device", default="cuda", help="Device (cuda/cpu)")
@click.option("--hidden-dim", default=192, help="LSTM hidden dimension")
@click.option("--lstm-layers", default=3, help="Number of LSTM layers")
@click.option("--dropout", default=0.35, help="Dropout rate")
def main(
    symbol: str,
    timeframe: str,
    epochs: int,
    batch_size: int,
    lr: float,
    sequence_length: int,
    prediction_horizon: int,
    target_return: float,
    max_history_days: int,
    device: str,
    hidden_dim: int,
    lstm_layers: int,
    dropout: float,
):
    """Entrena modelo mejorado para producción."""

    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO MEJORADO PARA PRODUCCIÓN")
    print("="*80 + "\n")

    # Check GPU
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA no disponible, usando CPU")
            device = "cpu"

    print(f"\n📊 Configuración:")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Sequence length: {sequence_length} (contexto: {sequence_length * 5} min en 5m)")
    print(f"  Prediction horizon: {prediction_horizon} ({prediction_horizon * 5} min en 5m)")
    print(f"  Target return: {target_return * 100:.2f}%")
    print(f"  Max history: {max_history_days} días")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}\n")

    # Configuración del dataset
    config = AdvancedDatasetConfig(
        symbol=symbol.replace("USDT", "/USDT") + ":USDT",
        timeframe=timeframe,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_return=target_return,
        max_history_days=max_history_days,
        use_feature_selection=True,
        n_features_to_select=50,  # De ~99 features a 50
        feature_selection_method="mutual_info",
        use_augmentation=True,
        augmentation_noise=0.01,
    )

    # Configuración del modelo mejorado
    # Dense dims scaled based on hidden_dim
    dense_dim_1 = hidden_dim * 2
    dense_dim_2 = int(hidden_dim * 1.33)
    dense_dim_3 = hidden_dim // 1.5 if hidden_dim > 96 else hidden_dim

    model_config = {
        'hidden_dim': hidden_dim,
        'lstm_layers': lstm_layers,
        'dense_dims': (dense_dim_1, dense_dim_2, int(dense_dim_3)),
        'dropout': dropout,
        'use_attention': True,
        'bidirectional': True,
        'num_classes': 3,
        'use_regression': True,
        'num_attention_heads': 8,
    }

    # Directorio de guardado
    save_dir = MODEL_DIR / symbol / timeframe
    save_dir.mkdir(parents=True, exist_ok=True)

    # Entrenar
    results = train_production_model(
        config=config,
        model_config=model_config,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-4,
        warmup_epochs=15,
        patience=30,
        gradient_clip=1.0,
        use_amp=True,
        use_focal_loss=True,
        save_dir=save_dir,
    )

    # Guardar resultados
    with open(save_dir / "production_training_results.json", "w") as f:
        json.dump({
            'config': {
                'symbol': symbol,
                'timeframe': timeframe,
                'sequence_length': sequence_length,
                'prediction_horizon': prediction_horizon,
                'target_return': target_return,
                'max_history_days': max_history_days,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
            },
            'model_config': model_config,
            'results': results,
        }, f, indent=2)

    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nModelos guardados en: {save_dir}")


if __name__ == "__main__":
    main()
