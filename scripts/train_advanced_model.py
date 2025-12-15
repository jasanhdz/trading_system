#!/usr/bin/env python3
"""
Train advanced temporal models with walk-forward validation.

This script provides a complete training pipeline with:
- Temporal sequence modeling (LSTM + Attention)
- Feature selection
- Walk-forward cross-validation
- Multi-task learning
- Ensemble models

Usage:
    python scripts/train_advanced_model.py --symbol BTCUSDT --timeframe 15m
    python scripts/train_advanced_model.py --symbol ETHUSDT --timeframe 5m --walk-forward
    python scripts/train_advanced_model.py --symbol BTCUSDT --ensemble 3
"""
import sys
from pathlib import Path
from typing import Optional

import click
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.advanced_models.dataset import (
    AdvancedDatasetConfig,
    FeatureSelector,
    SequenceDataset,
    load_sequence_dataset,
)
from ml.advanced_models.temporal_model import AdvancedTemporalNet, EnsembleModel
from ml.advanced_models.trainer import AdvancedTrainer
from utils.logger import setup_logger

logger = setup_logger("advanced_trainer")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _to_ccxt_symbol(binance_symbol: str) -> str:
    """Convert Binance symbol to CCXT format."""
    clean = binance_symbol.replace("/", "").replace(" ", "").upper()
    quote_tokens = ("USDT", "BUSD", "USDC", "BTC", "ETH")
    for quote in quote_tokens:
        if clean.endswith(quote):
            base = clean[: -len(quote)]
            return f"{base}/{quote}:USDT" if quote == "USDT" else f"{base}/{quote}"
    return clean


def _symbol_key(symbol: str) -> str:
    """Create filesystem-safe symbol key."""
    return symbol.replace("/", "").replace(":", "").replace("-", "").upper()


@click.command()
@click.option("--symbol", default="BTCUSDT", show_default=True, help="Trading symbol")
@click.option("--timeframe", default="15m", show_default=True, help="Timeframe")
@click.option("--sequence-length", default=24, show_default=True, help="Lookback window")
@click.option("--horizon", default=12, show_default=True, help="Prediction horizon")
@click.option("--target-return", default=0.002, show_default=True, help="Target return threshold")
@click.option("--epochs", default=50, show_default=True, help="Training epochs")
@click.option("--patience", default=10, show_default=True, help="Early stopping patience (epochs)")
@click.option("--batch-size", default=512, show_default=True, help="Batch size")
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate")
@click.option("--hidden-dim", default=128, show_default=True, help="LSTM hidden dimension")
@click.option("--lstm-layers", default=2, show_default=True, help="Number of LSTM layers")
@click.option("--dense-dims", default="256,128", show_default=True, help="Dense layer dimensions")
@click.option("--dropout", default=0.3, show_default=True, help="Dropout rate")
@click.option("--use-attention/--no-attention", default=True, help="Use attention mechanism")
@click.option("--bidirectional/--unidirectional", default=True, help="Bidirectional LSTM")
@click.option("--feature-selection/--no-feature-selection", default=True, help="Use feature selection")
@click.option("--n-features", default=32, show_default=True, help="Number of features to select")
@click.option("--walk-forward/--single-split", default=False, help="Use walk-forward validation")
@click.option("--n-folds", default=5, show_default=True, help="Number of walk-forward folds")
@click.option("--ensemble", default=0, show_default=True, help="Number of models in ensemble (0=single)")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device")
@click.option("--seed", default=42, show_default=True, help="Random seed")
@click.option("--history-days", type=int, default=None, help="Limit lookback history to N days (overrides defaults)")
@click.option("--loss-type", default="ce", type=click.Choice(["ce", "focal"]), help="Loss function type")
@click.option("--focal-gamma", default=2.0, show_default=True, help="Focal loss gamma")
def main(
    symbol: str,
    timeframe: str,
    sequence_length: int,
    horizon: int,
    target_return: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    lstm_layers: int,
    dense_dims: str,
    dropout: float,
    use_attention: bool,
    bidirectional: bool,
    feature_selection: bool,
    n_features: int,
    walk_forward: bool,
    n_folds: int,
    ensemble: int,
    device: str,
    seed: int,
    history_days: Optional[int],
    loss_type: str,
    focal_gamma: float,
) -> None:
    """Train advanced temporal model with modern architecture."""
    
    print("\n" + "="*80)
    print("ADVANCED TEMPORAL MODEL TRAINING")
    print("="*80 + "\n")
    
    # Parse configuration
    ccxt_symbol = _to_ccxt_symbol(symbol)
    dense_layers = tuple(int(x.strip()) for x in dense_dims.split(",") if x.strip())
    
    # Dataset configuration
    dataset_config = AdvancedDatasetConfig(
        symbol=ccxt_symbol,
        timeframe=timeframe,
        sequence_length=sequence_length,
        prediction_horizon=horizon,
        target_return=target_return,
        max_history_days=history_days if history_days is not None else (270 if timeframe == "15m" else 180),
        max_samples=100000,
        use_feature_selection=feature_selection,
        n_features_to_select=n_features,
        feature_selection_method="mutual_info",
        use_augmentation=True,
        augmentation_noise=0.01,
    )
    
    # Model configuration
    model_config = {
        'hidden_dim': hidden_dim,
        'lstm_layers': lstm_layers,
        'dense_dims': dense_layers,
        'dropout': dropout,
        'use_attention': use_attention,
        'bidirectional': bidirectional,
        'num_classes': 3,
        'use_regression': True,
    }
    
    print("Configuration:")
    print(f"  Symbol: {ccxt_symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Sequence Length: {sequence_length}")
    print(f"  Prediction Horizon: {horizon}")
    print(f"  Feature Selection: {feature_selection}")
    print(f"  Walk-Forward: {walk_forward}")
    print(f"  Patience: {patience}")
    print(f"  Ensemble Size: {ensemble if ensemble > 0 else 'Single Model'}")
    print(f"  Device: {device}\n")
    
    # Initialize trainer
    trainer = AdvancedTrainer(
        config=dataset_config,
        model_config=model_config,
        device=device,
        seed=seed,
    )
    
    # Load and prepare data
    # Load and prepare data (RAW, no selection yet)
    features, class_labels, regression_targets, feature_names = trainer.load_and_prepare_data(apply_selection=False)
    
    print(f"\nDataset Summary:")
    print(f"  Total Samples: {len(features)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Class Distribution:")
    unique, counts = np.unique(class_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        cls_name = ['Neutral', 'Long', 'Short'][cls]
        print(f"    {cls_name}: {count} ({count/len(class_labels)*100:.1f}%)")
    
    # Walk-forward validation (if enabled)
    if walk_forward:
        fold_results = trainer.walk_forward_validation(
            features,
            class_labels,
            regression_targets,
            feature_names=feature_names,
            n_splits=n_folds,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        
        # Save walk-forward results
        import json
        wf_path = MODEL_DIR / _symbol_key(symbol) / timeframe / "walk_forward_results.json"
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'config': {
                'symbol': ccxt_symbol,
                'timeframe': timeframe,
                'n_folds': n_folds,
            },
            'folds': [
                {
                    'fold': r['fold'],
                    'train_size': r['train_size'],
                    'test_size': r['test_size'],
                    'accuracy': r['metrics']['accuracy'],
                    'macro_f1': r['metrics']['macro_f1'],
                    'ap_long': r['metrics']['ap_long'],
                    'ap_short': r['metrics']['ap_short'],
                    'regression_mse': (
                        r['metrics']['regression']['mse'] if r['metrics'].get('regression') else None
                    ),
                    'regression_mae': (
                        r['metrics']['regression']['mae'] if r['metrics'].get('regression') else None
                    ),
                }
                for r in fold_results
            ],
            'average': {
                'accuracy': np.mean([r['metrics']['accuracy'] for r in fold_results]),
                'macro_f1': np.mean([r['metrics']['macro_f1'] for r in fold_results]),
                'ap_long': np.mean([r['metrics']['ap_long'] for r in fold_results]),
                'ap_short': np.mean([r['metrics']['ap_short'] for r in fold_results]),
                'regression_mse': np.mean([
                    r['metrics']['regression']['mse']
                    for r in fold_results
                    if r['metrics'].get('regression')
                ]) if all(r['metrics'].get('regression') for r in fold_results) else None,
                'regression_mae': np.mean([
                    r['metrics']['regression']['mae']
                    for r in fold_results
                    if r['metrics'].get('regression')
                ]) if all(r['metrics'].get('regression') for r in fold_results) else None,
            }
        }
        
        wf_path.write_text(json.dumps(results_to_save, indent=2))
        print(f"\nWalk-forward results saved to: {wf_path}")
        
        # Continue to train final model on all data
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL ON ALL DATA")
        print("="*80 + "\n")
    
    # Train final model(s)
    # Split into train/val/test (chronological) BEFORE scaling/selection
    n_samples = len(features)
    train_end = int(n_samples * 0.7)
    valid_end = int(n_samples * 0.85)
    
    print(f"Final Split Indices:")
    print(f"  Train: 0 - {train_end}")
    print(f"  Valid: {train_end} - {valid_end}")
    print(f"  Test: {valid_end} - {n_samples}\n")
    
    # Split raw data with lookback buffer for val/test to ensure continuity
    # We need 'sequence_length' previous samples to make the first prediction of the new set
    lookback = sequence_length
    
    X_train = features[:train_end]
    y_train_cls = class_labels[:train_end]
    y_train_reg = regression_targets[:train_end]
    
    # Val needs lookback from train
    val_start_idx = max(0, train_end - lookback)
    X_val = features[val_start_idx:valid_end]
    y_val_cls = class_labels[val_start_idx:valid_end]
    y_val_reg = regression_targets[val_start_idx:valid_end]
    
    # Test needs lookback from val
    test_start_idx = max(0, valid_end - lookback)
    X_test = features[test_start_idx:]
    y_test_cls = class_labels[test_start_idx:]
    y_test_reg = regression_targets[test_start_idx:]
    
    # Feature Selection (Fit on Train ONLY)
    if feature_selection:
        print("Performing feature selection on training set...")
        trainer.feature_selector = FeatureSelector(
            method=dataset_config.feature_selection_method,
            n_features=dataset_config.n_features_to_select,
        )
        
        # Fit on train
        sample_size = min(10000, len(X_train))
        if len(X_train) > sample_size:
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            fit_X = X_train[idx]
            fit_y = y_train_cls[idx]
        else:
            fit_X = X_train
            fit_y = y_train_cls
            
        trainer.selected_features = trainer.feature_selector.fit(fit_X, fit_y, feature_names)
        print(f"Selected {len(trainer.selected_features)} features")
        
        # Transform all
        X_train = trainer.feature_selector.transform(X_train)
        X_val = trainer.feature_selector.transform(X_val)
        X_test = trainer.feature_selector.transform(X_test)
    else:
        trainer.selected_features = feature_names

    # Scale features (Fit on Train ONLY)
    print("Scaling features (fit on train)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets (Separate datasets for strict isolation)
    train_dataset = SequenceDataset(
        X_train,
        y_train_cls,
        y_train_reg,
        sequence_length=sequence_length,
        prediction_horizon=horizon,
        augment=dataset_config.use_augmentation,
        augmentation_noise=dataset_config.augmentation_noise,
    )
    
    valid_dataset = SequenceDataset(
        X_val,
        y_val_cls,
        y_val_reg,
        sequence_length=sequence_length,
        prediction_horizon=horizon,
        augment=False,
        start_index=lookback,  # Skip buffer, start predicting at first real val sample
    )
    
    test_dataset = SequenceDataset(
        X_test,
        y_test_cls,
        y_test_reg,
        sequence_length=sequence_length,
        prediction_horizon=horizon,
        augment=False,
        start_index=lookback,  # Skip buffer, start predicting at first real test sample
    )
    
    print(f"Datasets created (Effective samples excluding buffer):")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)} (Buffer size: {lookback})")
    print(f"  Test samples: {len(test_dataset)} (Buffer size: {lookback})\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle train
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Compute class weights
    # Use y_train_cls directly since we have it separated
    class_counts = np.bincount(y_train_cls, minlength=3)
    if (class_counts > 0).all():
        inv_weights = class_counts.sum() / class_counts
        class_weights = torch.from_numpy((inv_weights / inv_weights.mean()).astype(np.float32))
    else:
        class_weights = None
    
    # Train ensemble or single model
    if ensemble > 1:
        print(f"Training ensemble of {ensemble} models...\n")
        models = []
        for i in range(ensemble):
            print(f"Training model {i+1}/{ensemble}...")
            trainer._set_seed(seed + i)  # Different seed for each model
            
            model, history = trainer.train_single_model(
                train_loader,
                valid_loader,
                class_weights=class_weights,
                epochs=epochs,
                lr=lr,
                patience=patience,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
            )
            models.append(model)
            print()
        
        # Create ensemble
        final_model = EnsembleModel(models)
        print(f"Created ensemble of {len(models)} models")
    else:
        print("Training single model...\n")
        final_model, history = trainer.train_single_model(
            train_loader,
            valid_loader,
            class_weights=class_weights,
            epochs=epochs,
            lr=lr,
            patience=patience,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )

    final_model = final_model.to(trainer.device)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    test_metrics = trainer.evaluate_model(final_model, test_loader)
    
    print("Test Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  AP Long: {test_metrics['ap_long']:.4f}")
    print(f"  AP Short: {test_metrics['ap_short']:.4f}")
    reg_metrics = test_metrics.get('regression')
    if reg_metrics:
        print(f"  Regression MSE: {reg_metrics['mse']:.6f}")
        print(f"  Regression MAE: {reg_metrics['mae']:.6f}")
    trading = test_metrics.get('trading')
    if trading:
        print(f"  Trading PnL: {trading['total_return']*100:.2f}% "
              f"(Best thr: {trading['threshold']}, trades: {trading['n_trades']})")
        print(f"  Sharpe: {trading['sharpe_ratio']:.2f} | "
              f"Win Rate: {trading['win_rate']*100:.1f}% | "
              f"Max Drawdown: {trading['max_drawdown']*100:.2f}%")
    print("\nPer-Class Metrics:")
    for cls_name, metrics in test_metrics['per_class'].items():
        print(f"  {cls_name.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    # Save model and artifacts
    symbol_dir = MODEL_DIR / _symbol_key(symbol) / timeframe
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = symbol_dir / "model.pt"
    scaler_path = symbol_dir / "scaler.pkl"
    meta_path = symbol_dir / "meta.json"
    
    # Save model
    if isinstance(final_model, EnsembleModel):
        torch.save({
            'models': [m.state_dict() for m in final_model.models],
            'weights': final_model.weights,
        }, model_path)
    else:
        torch.save(final_model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature selector (if used)
    if trainer.feature_selector:
        selector_path = symbol_dir / "feature_selector.pkl"
        joblib.dump(trainer.feature_selector, selector_path)
        print(f"Feature selector saved to: {selector_path}")
    
    # Save metadata
    import json
    meta = {
        'symbol': ccxt_symbol,
        'symbol_key': _symbol_key(symbol),
        'timeframe': timeframe,
        'sequence_length': sequence_length,
        'prediction_horizon': horizon,
        'target_return': target_return,
        'selected_features': trainer.selected_features,
        'model_config': model_config,
        'training_config': {
            'epochs': epochs,
            'patience': patience,
            'batch_size': batch_size,
            'lr': lr,
            'seed': seed,
        },
        'ensemble_size': ensemble if ensemble > 0 else 1,
        'test_metrics': test_metrics,
    }
    
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to: {meta_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1: {test_metrics['macro_f1']:.4f}")
    final_reg_metrics = test_metrics.get('regression')
    if final_reg_metrics:
        print(f"Final Test Regression MSE: {final_reg_metrics['mse']:.6f}")
        print(f"Final Test Regression MAE: {final_reg_metrics['mae']:.6f}")
    print(f"\nModel artifacts saved to: {symbol_dir}")


if __name__ == "__main__":
    main()
