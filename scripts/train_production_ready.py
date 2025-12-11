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
import random
import os

def set_seed(seed=42):
    """Fija las semillas para reproducibilidad."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Semillas fijadas en {seed}")

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

MODEL_DIR_DEFAULT = (REPO_ROOT / "models" / "advanced").resolve()
MODEL_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)


# FocalLoss local definition removed to use the one in ml.advanced_models.temporal_model


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


def evaluate_model(model, dataloader, device_obj, criterion):
    """Evalúa el modelo y retorna métricas detalladas."""
    model.eval()
    all_preds = []
    all_labels = []
    all_returns = [] # Para PnL (Targets)
    all_reg_preds = [] # Para MSE/MAE (Predicciones)
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                batch_seq, batch_labels, batch_reg = batch
                batch_reg = batch_reg.to(device_obj)
            else:
                batch_seq, batch_labels = batch
                batch_reg = None

            batch_seq = batch_seq.to(device_obj)
            batch_labels = batch_labels.squeeze(-1).to(device_obj)

            outputs = model(batch_seq)
            loss, _ = criterion(
                outputs['logits'],
                batch_labels,
                outputs.get('regression'),
                batch_reg,
            )

            total_loss += loss.item()

            probs = torch.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            if batch_reg is not None:
                all_returns.extend(batch_reg.cpu().numpy())
                if 'regression' in outputs:
                    all_reg_preds.extend(outputs['regression'].cpu().numpy().flatten())

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

    # Métricas de Trading (Implied PnL)
    # Asumimos que si predice Long (1) compramos, Short (2) vendemos, Neutral (0) nada.
    # Usamos los retornos reales (regression targets) para calcular el resultado.
    pnl_metrics = {}
    if len(all_returns) > 0:
        all_returns = np.array(all_returns)
        all_preds = np.array(all_preds)
        
        # PnL Longs
        long_mask = (all_preds == 1)
        long_pnl = np.sum(all_returns[long_mask])
        
        # PnL Shorts (retorno inverso)
        short_mask = (all_preds == 2)
        short_pnl = np.sum(-all_returns[short_mask])
        
        total_pnl = long_pnl + short_pnl
        
        pnl_metrics = {
            'total_pnl': total_pnl,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_count': np.sum(long_mask),
            'short_count': np.sum(short_mask)
        }

    # Métricas de Regresión
    reg_metrics = {}
    if len(all_returns) > 0 and len(all_reg_preds) > 0:
        # Asegurar que tengan la misma longitud
        min_len = min(len(all_returns), len(all_reg_preds))
        y_true = np.array(all_returns[:min_len])
        y_pred = np.array(all_reg_preds[:min_len])
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        reg_metrics = {
            'mse': mse,
            'mae': mae
        }

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'neutral_f1': per_class_f1[0] if len(per_class_f1) > 0 else 0,
        'long_f1': per_class_f1[1] if len(per_class_f1) > 1 else 0,
        'short_f1': per_class_f1[2] if len(per_class_f1) > 2 else 0,
        **pnl_metrics,
        **reg_metrics
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

    # Detectar ROCm y forzar inicialización si es necesario
    is_rocm = bool(getattr(torch.version, "hip", None))
    
    if is_rocm:
        logger.info(f"🔧 ROCm detectado - Version HIP: {torch.version.hip}")
        # Desactivar AMP en ROCm por estabilidad en RX 6600
        use_amp = False
        logger.info("⚠️  AMP desactivado para AMD ROCm por estabilidad")
        
        # Workaround: ROCm a veces no detecta GPUs hasta que se intenta usarlas
        # Forzar inicialización del contexto ANTES del check de is_available()
        try:
            import os
            if 'HIP_VISIBLE_DEVICES' in os.environ or 'HSA_OVERRIDE_GFX_VERSION' in os.environ:
                logger.info(f"   HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'not set')}")
                logger.info(f"   HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")
                
                # Forzar la inicialización del contexto CUDA/ROCm
                # Esto DEBE ejecutarse antes de is_available()
                torch.cuda._lazy_init()
                logger.info("   ✅ Contexto ROCm inicializado con _lazy_init()")
                
                # Ahora verificar si se detectaron GPUs
                if torch.cuda.is_available():
                    logger.info(f"   ✅ ROCm GPUs detectadas: {torch.cuda.device_count()}")
                else:
                    logger.warning("   ⚠️  ROCm inicializado pero no se detectaron GPUs")
        except Exception as e:
            logger.warning(f"   ⚠️  Error al inicializar ROCm: {e}")

    # FORZAR USO DE GPU - NO PERMITIR CPU
    logger.info(f"📱 Device solicitado: {device}")
    logger.info(f"🔍 PyTorch CUDA disponible: {torch.cuda.is_available()}")
    
    # Verificar que GPU esté disponible
    if not torch.cuda.is_available():
        error_msg = (
            "❌ GPU NO DISPONIBLE - Entrenamiento BLOQUEADO\n"
            "   El entrenamiento en CPU está deshabilitado por configuración.\n"
            "   Razones posibles:\n"
            "   1. Variables de entorno ROCm no configuradas (HIP_VISIBLE_DEVICES, HSA_OVERRIDE_GFX_VERSION)\n"
            "   2. Drivers de GPU no cargados correctamente\n"
            "   3. PyTorch sin soporte ROCm/CUDA\n"
            "   SOLUCIÓN: Verifica las GPUs con 'rocm-smi' o 'nvidia-smi'"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # GPU disponible - continuar
    logger.info(f"✅ GPU disponible - Device count: {torch.cuda.device_count()}")
    try:
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        logger.warning(f"⚠️  Error al obtener nombre de GPU: {e}")
    
    # Crear device object - SOLO GPU, SIN FALLBACK A CPU
    try:
        device_obj = torch.device(device)
        logger.info(f"✅ Device configurado: {device_obj}")
        
        # Verificar que efectivamente podemos usar este device
        test_tensor = torch.rand(10, 10).to(device_obj)
        del test_tensor
        torch.cuda.synchronize()
        logger.info(f"✅ Test de allocación en GPU exitoso")
        
    except Exception as e:
        error_msg = f"❌ Error al configurar GPU '{device}': {e}\nUSO DE CPU BLOQUEADO"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Configurar AMP según el tipo de GPU
    if is_rocm:
        if use_amp:
            logger.warning("⚠️  ROCm detectado. Desactivando Mixed Precision (AMP)")
            logger.warning("    (Mixed precision puede causar problemas en algunas versiones de ROCm)")
            use_amp = False
        else:
            logger.info("ℹ️  ROCm detectado. Configuración optimizada para AMD GPUs")

    logger.info(f"🚀 Usando device: {device_obj} (GPU FORZADA)")
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

    # Feature selection y Scaling se mueven DENTRO del loop de folds para evitar leakage
    from ml.advanced_models.dataset import FeatureSelector
    
    # Solo inicializamos el selector si se va a usar, pero el fit se hace por fold
    feature_selector_class = FeatureSelector if config.use_feature_selection else None
    
    # Guardar features originales para referencia
    original_features = features.copy()
    original_feature_names = feature_names.copy()

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

        # --- PREPROCESAMIENTO POR FOLD (SIN LEAKAGE) ---
        
        # 1. Feature Selection
        current_train_features = features[train_idx]
        current_train_labels = class_labels[train_idx]
        
        current_val_features = features[val_idx]
        current_test_features = features[test_idx]
        
        current_feature_names = original_feature_names
        
        if feature_selector_class and len(original_feature_names) > config.n_features_to_select:
            logger.info(f"  🎯 Seleccionando features (Fold {fold_idx})...")
            selector = feature_selector_class(
                method=config.feature_selection_method,
                n_features=config.n_features_to_select,
            )
            
            # Fit solo en TRAIN
            # Usar una muestra si es muy grande para acelerar
            sample_size = min(20000, len(current_train_features))
            sample_idx_sel = np.random.choice(len(current_train_features), sample_size, replace=False)
            
            selected_names = selector.fit(
                current_train_features[sample_idx_sel], 
                current_train_labels[sample_idx_sel], 
                original_feature_names
            )
            
            # Transformar todos
            current_train_features = selector.transform(current_train_features)
            current_val_features = selector.transform(current_val_features)
            current_test_features = selector.transform(current_test_features)
            
            current_feature_names = selected_names
            
            # Guardar selector del fold
            if save_dir:
                import joblib
                joblib.dump(selector, save_dir / f"feature_selector_fold{fold_idx}.pkl")

        # 2. Scaling
        logger.info(f"  ⚙️  Escalando features (Fold {fold_idx})...")
        scaler = StandardScaler()
        
        # Fit solo en TRAIN
        current_train_features = scaler.fit_transform(current_train_features)
        
        # Transform en VAL y TEST
        current_val_features = scaler.transform(current_val_features)
        current_test_features = scaler.transform(current_test_features)
        
        if save_dir:
            import joblib
            joblib.dump(scaler, save_dir / f"scaler_fold{fold_idx}.pkl")

        # Preparar targets de regresión
        train_reg = regression_targets[train_idx] if regression_targets is not None else None
        val_reg = regression_targets[val_idx] if regression_targets is not None else None
        test_reg = regression_targets[test_idx] if regression_targets is not None else None

        # Crear datasets
        train_dataset = SequenceDataset(
            current_train_features,
            class_labels[train_idx].reshape(-1, 1),
            train_reg,
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=True,
            augmentation_noise=0.01,
        )

        val_dataset = SequenceDataset(
            current_val_features,
            class_labels[val_idx].reshape(-1, 1),
            val_reg,
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=False,
        )

        test_dataset = SequenceDataset(
            current_test_features,
            class_labels[test_idx].reshape(-1, 1),
            test_reg,
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
        # Importante: input_dim depende de las features seleccionadas en ESTE fold
        model = DeepTemporalNet(
            input_dim=current_train_features.shape[1],
            sequence_length=config.sequence_length,
            **model_config,
        ).to(device_obj)

        logger.info(f"🧠 Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")

        # Loss y optimizer
        # Calcular pesos de clase dinámicos para este fold
        train_labels = class_labels[train_idx].flatten()
        class_counts = np.bincount(train_labels, minlength=3)
        total_samples = len(train_labels)
        logger.info(f"  ⚖️  Distribución de clases: {class_counts} (Total: {total_samples})")
        
        if (class_counts > 0).all():
            # Inverse frequency weights
            weights = total_samples / (3 * class_counts)
            # Normalize to sum to 3 (or mean 1)
            weights = weights / weights.mean()
            class_weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(device_obj)
            logger.info(f"  ⚖️  Pesos calculados: {weights}")
        else:
            logger.warning("  ⚠️  Clases faltantes en train set, usando pesos uniformes")
            class_weights_tensor = None

        # Inicializar MultiTaskLoss con pesos y Focal Loss
        criterion = MultiTaskLoss(
            class_weights=class_weights_tensor if use_focal_loss else None, 
            classification_weight=1.0,
            regression_weight=0.2,
            focal_gamma=2.0
        ).to(device_obj)
        
        # Si NO usamos focal loss pero sí pesos, necesitamos configurar CrossEntropy manualmente
        # Pero MultiTaskLoss usa FocalLoss por defecto.
        # Si queremos standard CE, tendríamos que modificar MultiTaskLoss o pasar gamma=0?
        # Por ahora asumimos que use_focal_loss=True es lo deseado para "Institutional".
        if not use_focal_loss:
             # Fallback a CE standard si se solicita explícitamente no usar Focal
             criterion.classification_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device_obj)

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

            for batch in train_loader:
                if len(batch) == 3:
                    batch_seq, batch_labels, batch_reg = batch
                    batch_reg = batch_reg.to(device_obj)
                else:
                    batch_seq, batch_labels = batch
                    batch_reg = None

                batch_seq = batch_seq.to(device_obj)
                batch_labels = batch_labels.squeeze(-1).to(device_obj)

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
                            batch_reg,
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
                        batch_reg,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validación
            val_metrics, _, _ = evaluate_model(model, val_loader, device_obj, criterion)

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
        test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device_obj, criterion)

        logger.info(f"\n📊 Resultados Fold {fold_idx}:")
        logger.info(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch+1})")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
        if 'mse' in test_metrics:
            logger.info(f"  Test MSE: {test_metrics['mse']:.6f} | MAE: {test_metrics['mae']:.6f}")
        logger.info(f"  Long F1: {test_metrics['long_f1']:.4f}")
        logger.info(f"  Short F1: {test_metrics['short_f1']:.4f}")

        fold_result = {
            'fold': fold_idx,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'selected_features': current_feature_names, # Guardar features de este fold
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

    return all_fold_results, feature_names


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
@click.option("--model-dir", default=None, help="Custom model directory (for sweeps)")
@click.option("--seed", default=42, show_default=True, help="Random seed")
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
    model_dir: str,
    seed: int,
):
    """Entrena modelo mejorado para producción."""

    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO MEJORADO PARA PRODUCCIÓN (INSTITUTIONAL GRADE)")
    print("="*80 + "\n")
    
    # Fijar seed
    set_seed(seed)

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
    base_dir = Path(model_dir).resolve() if model_dir else MODEL_DIR_DEFAULT / symbol / timeframe
    base_dir.mkdir(parents=True, exist_ok=True)
    save_dir = base_dir

    # Entrenar
    results, feature_names = train_production_model(
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
            'selected_features': feature_names,
            'model_config': model_config,
            'results': results,
        }, f, indent=2)

    # Guardar meta.json para compatibilidad con el predictor
    with open(save_dir / "meta.json", "w") as f:
        json.dump({
            'symbol': symbol,
            'timeframe': timeframe,
            'sequence_length': sequence_length,
            'selected_features': feature_names,
            'model_config': model_config,
            'ensemble_size': 5
        }, f, indent=2)

    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nModelos guardados en: {save_dir}")


if __name__ == "__main__":
    main()
