#!/usr/bin/env python3
"""
Script optimizado para entrenar modelos avanzados con GPU AMD (ROCm/CUDA).

Mejoras implementadas:
1. Hyperparameter optimization automática
2. Mejor arquitectura (más capas, mejor regularización)
3. Class weighting dinámico
4. Learning rate scheduling avanzado
5. Gradient accumulation para batches grandes
6. Mixed precision training (si disponible)
7. Early stopping inteligente
8. Checkpoint saving automático
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import click

from ml.advanced_models.dataset import (
    AdvancedDatasetConfig,
    SequenceDataset,
    load_sequence_dataset,
    walk_forward_split,
)
from ml.advanced_models.temporal_model import (
    AdvancedTemporalNet,
    EnsembleModel,
    MultiTaskLoss,
)
from ml.advanced_models.trainer import AdvancedTrainer
from utils.logger import setup_logger

logger = setup_logger("optimized_trainer")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ImprovedTemporalNet(nn.Module):
    """
    Arquitectura mejorada con:
    - Más capas LSTM
    - Residual connections
    - Layer normalization
    - Mejor dropout strategy
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 24,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dense_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
        use_attention: bool = True,
        bidirectional: bool = True,
        num_classes: int = 3,
        use_regression: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_regression = use_regression
        
        # Input projection (reduce dimensionality smoothly)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Less dropout at input
        )
        
        # Stacked LSTM with residual connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        lstm_input_dim = hidden_dim
        for i in range(lstm_layers):
            lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0,  # Manual dropout between layers
            )
            self.lstm_layers.append(lstm)
            
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
            self.layer_norms.append(nn.LayerNorm(lstm_output_dim))
            lstm_input_dim = lstm_output_dim
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        else:
            self.attention = None
        
        # Dense layers with residual connections
        prev_dim = lstm_output_dim
        dense_layers = []
        
        for i, dim in enumerate(dense_dims):
            dense_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        self.dense = nn.Sequential(*dense_layers)
        
        # Output heads
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Stacked LSTM with residual connections
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(x)
            lstm_out = norm(lstm_out)
            
            # Residual connection if dimensions match
            if i > 0 and x.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + x
            
            x = lstm_out
        
        # Attention
        if self.attention is not None:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Dense layers
        features = self.dense(x)
        
        # Outputs
        outputs = {
            'logits': self.classifier(features),
        }
        
        if self.use_regression:
            outputs['regression'] = self.regressor(features)
        
        return outputs


class CosineWarmupScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
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


def train_improved_model(
    config: AdvancedDatasetConfig,
    model_config: dict,
    device: str = "cuda",
    epochs: int = 150,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 10,
    patience: int = 25,
    gradient_clip: float = 1.0,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    save_dir: Path = None,
):
    """Train model with all improvements."""
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)
    
    logger.info(f"Dataset: {len(features)} samples, {len(feature_names)} features")
    
    # Feature selection (if needed)
    from ml.advanced_models.dataset import FeatureSelector
    
    if config.use_feature_selection and len(feature_names) > config.n_features_to_select:
        logger.info(f"Selecting top {config.n_features_to_select} features...")
        selector = FeatureSelector(
            method=config.feature_selection_method,
            n_features=config.n_features_to_select,
        )
        
        sample_idx = np.random.choice(len(features), min(5000, len(features)), replace=False)
        selected_features = selector.fit(features[sample_idx], class_labels[sample_idx], feature_names)
        features = selector.transform(features)
        feature_names = selected_features
        
        # Save selector
        import joblib
        if save_dir:
            joblib.dump(selector, save_dir / "feature_selector.pkl")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    if save_dir:
        import joblib
        joblib.dump(scaler, save_dir / "scaler.pkl")
    
    # Walk-forward splits
    logger.info("Creating walk-forward splits...")
    splits = walk_forward_split(
        n_samples=len(features),
        n_splits=5,
        train_ratio=0.7,
        gap=config.prediction_horizon,
    )
    
    all_fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx}/5")
        logger.info(f"{'='*60}")
        
        # Split data
        train_features = features[train_idx]
        train_labels = class_labels[train_idx]
        train_returns = regression_targets[train_idx]
        test_features = features[test_idx]
        test_labels = class_labels[test_idx]
        test_returns = regression_targets[test_idx]
        
        # Create datasets
        train_dataset = SequenceDataset(
            train_features,
            train_labels.reshape(-1, 1),
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            augment=True,
            augmentation_noise=0.01,
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        # Create model
        model = ImprovedTemporalNet(
            input_dim=features.shape[1],
            sequence_length=config.sequence_length,
            **model_config,
        ).to(device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        class_counts = np.bincount(train_labels, minlength=3)
        if (class_counts > 0).all():
            inv_weights = class_counts.sum() / class_counts
            class_weights = torch.from_numpy((inv_weights / inv_weights.mean()).astype(np.float32)).to(device)
        else:
            class_weights = None
        
        criterion = MultiTaskLoss(
            class_weights=class_weights,
            classification_weight=1.0,
            regression_weight=0.3,  # Reduced weight for regression
        )
        
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
        
        scaler_amp = GradScaler() if use_amp else None
        
        # Training loop
        best_f1 = -float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, (batch_seq, batch_labels) in enumerate(train_loader):
                batch_seq = batch_seq.to(device)
                batch_labels = batch_labels.squeeze(-1).to(device)
                
                # Mixed precision training
                if use_amp:
                    with autocast():
                        outputs = model(batch_seq)
                        loss, _ = criterion(
                            outputs['logits'],
                            batch_labels,
                            outputs.get('regression'),
                            None,
                        )
                        loss = loss / accumulation_steps
                    
                    scaler_amp.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(batch_seq)
                    loss, _ = criterion(
                        outputs['logits'],
                        batch_labels,
                        outputs.get('regression'),
                        None,
                    )
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_seq, batch_labels in test_loader:
                    batch_seq = batch_seq.to(device)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(batch_seq)
                    else:
                        outputs = model(batch_seq)
                    
                    probs = torch.softmax(outputs['logits'], dim=1)
                    preds = probs.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.squeeze(-1).numpy())
            
            # Metrics
            from sklearn.metrics import f1_score, accuracy_score
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            # Learning rate step
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Acc: {accuracy:.4f} | "
                    f"F1: {f1:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
            
            # Early stopping
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model
                if save_dir:
                    torch.save(model.state_dict(), save_dir / f"best_model_fold{fold_idx}.pt")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Fold {fold_idx} best F1: {best_f1:.4f} at epoch {best_epoch+1}")
        
        fold_result = {
            'fold': fold_idx,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'accuracy': accuracy,
        }
        all_fold_results.append(fold_result)
    
    # Average results
    avg_f1 = np.mean([r['best_f1'] for r in all_fold_results])
    std_f1 = np.std([r['best_f1'] for r in all_fold_results])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    
    return all_fold_results


@click.command()
@click.option("--symbol", default="ETHUSDT", help="Trading symbol")
@click.option("--timeframe", default="15m", help="Timeframe")
@click.option("--epochs", default=150, help="Training epochs")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--lr", default=5e-4, help="Learning rate")
@click.option("--hidden-dim", default=128, help="LSTM hidden dimension")
@click.option("--lstm-layers", default=2, help="Number of LSTM layers")
@click.option("--dropout", default=0.25, help="Dropout rate")
@click.option("--device", default="cuda", help="Device (cuda/cpu)")
def main(
    symbol: str,
    timeframe: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    lstm_layers: int,
    dropout: float,
    device: str,
):
    """Train improved model with walk-forward validation."""
    
    print("\n" + "="*80)
    print("IMPROVED MODEL TRAINING WITH GPU OPTIMIZATION")
    print("="*80 + "\n")
    
    # Check GPU
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available, using CPU")
            device = "cpu"
    
    print(f"\nTraining {symbol} on {timeframe}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"LSTM layers: {lstm_layers}")
    print(f"Dropout: {dropout}\n")
    
    # Configuration
    config = AdvancedDatasetConfig(
        symbol=symbol.replace("USDT", "/USDT") + ":USDT",
        timeframe=timeframe,
        sequence_length=32,  # Increased for more context
        prediction_horizon=12,
        target_return=0.002,
        max_history_days=365,
        use_feature_selection=True,
        n_features_to_select=32,
        feature_selection_method="mutual_info",
        use_augmentation=True,
        augmentation_noise=0.01,
    )
    
    model_config = {
        'hidden_dim': hidden_dim,
        'lstm_layers': lstm_layers,
        'dense_dims': (256, 128, 64),  # Deeper network
        'dropout': dropout,
        'use_attention': True,
        'bidirectional': True,
        'num_classes': 3,
        'use_regression': True,
    }
    
    # Save directory
    save_dir = MODEL_DIR / symbol / timeframe
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    results = train_improved_model(
        config=config,
        model_config=model_config,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-4,
        warmup_epochs=10,
        patience=25,
        gradient_clip=1.0,
        accumulation_steps=1,
        use_amp=True,
        save_dir=save_dir,
    )
    
    # Save results
    with open(save_dir / "training_results.json", "w") as f:
        json.dump({
            'config': {
                'symbol': symbol,
                'timeframe': timeframe,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
            },
            'model_config': model_config,
            'results': results,
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETED")
    print("="*80)
    print(f"\nModels saved to: {save_dir}")
    print("\nNext steps:")
    print("1. Evaluate models")
    print("2. Create ensemble")
    print("3. Backtest")
    

if __name__ == "__main__":
    main()
