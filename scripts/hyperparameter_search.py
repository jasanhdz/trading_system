#!/usr/bin/env python3
"""
Grid search optimizado para encontrar mejores hiperparámetros.

Prueba múltiples configuraciones y selecciona la mejor automáticamente.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import itertools
import torch
import click
from datetime import datetime

from ml.advanced_models.dataset import AdvancedDatasetConfig


def grid_search_configs():
    """Generate grid of hyperparameter configurations."""
    
    # Parameter grid (ordered by importance)
    param_grid = {
        'dropout': [0.20, 0.25, 0.30],  # Most important
        'lr': [3e-4, 5e-4, 7e-4],  # Second most important
        'hidden_dim': [96, 128],  # Model size
        'lstm_layers': [1, 2],  # Depth
        'sequence_length': [24, 32],  # Context window
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs


def run_training(symbol, timeframe, config, device, epochs=100):
    """Run training with specific configuration."""
    
    from train_improved_gpu import train_improved_model
    
    # Create full configuration
    data_config = AdvancedDatasetConfig(
        symbol=symbol.replace("USDT", "/USDT") + ":USDT",
        timeframe=timeframe,
        sequence_length=config['sequence_length'],
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
        'hidden_dim': config['hidden_dim'],
        'lstm_layers': config['lstm_layers'],
        'dense_dims': (256, 128, 64) if config['hidden_dim'] >= 128 else (192, 96, 48),
        'dropout': config['dropout'],
        'use_attention': True,
        'bidirectional': True,
        'num_classes': 3,
        'use_regression': True,
    }
    
    # Train
    results = train_improved_model(
        config=data_config,
        model_config=model_config,
        device=device,
        epochs=epochs,
        batch_size=256,
        lr=config['lr'],
        weight_decay=1e-4,
        warmup_epochs=10,
        patience=20,
        gradient_clip=1.0,
        accumulation_steps=1,
        use_amp=True,
        save_dir=None,  # Don't save intermediate models
    )
    
    # Calculate average F1
    avg_f1 = sum(r['best_f1'] for r in results) / len(results)
    
    return avg_f1, results


@click.command()
@click.option("--symbol", default="ETHUSDT", help="Trading symbol")
@click.option("--timeframe", default="15m", help="Timeframe")
@click.option("--epochs", default=100, help="Epochs per config")
@click.option("--max-configs", default=10, help="Max configs to try (0=all)")
@click.option("--device", default="cuda", help="Device")
def main(symbol, timeframe, epochs, max_configs, device):
    """Run grid search to find best hyperparameters."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*80 + "\n")
    
    # Check device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, using CPU")
            device = "cpu"
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Generate configurations
    all_configs = grid_search_configs()
    
    print(f"Total configurations: {len(all_configs)}")
    
    if max_configs > 0 and len(all_configs) > max_configs:
        # Sample random configs
        import random
        random.seed(42)
        configs = random.sample(all_configs, max_configs)
        print(f"Testing random sample of {max_configs} configs\n")
    else:
        configs = all_configs
        print(f"Testing all {len(configs)} configs\n")
    
    # Run grid search
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"CONFIG {i}/{len(configs)}")
        print(f"{'='*80}")
        print(f"Parameters: {config}")
        print()
        
        start_time = datetime.now()
        
        try:
            avg_f1, fold_results = run_training(
                symbol, timeframe, config, device, epochs
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                'config': config,
                'avg_f1': avg_f1,
                'fold_results': fold_results,
                'elapsed_seconds': elapsed,
                'status': 'success',
            }
            
            print(f"\n✓ Config {i} completed")
            print(f"  Average F1: {avg_f1:.4f}")
            print(f"  Time: {elapsed/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n✗ Config {i} failed: {e}")
            result = {
                'config': config,
                'avg_f1': 0.0,
                'fold_results': [],
                'elapsed_seconds': 0,
                'status': 'failed',
                'error': str(e),
            }
        
        results.append(result)
        
        # Save intermediate results
        save_path = Path(f"grid_search_{symbol}_{timeframe}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Find best configuration
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS")
    print("="*80 + "\n")
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("✗ No successful configurations")
        return
    
    # Sort by F1 score
    successful_results.sort(key=lambda x: x['avg_f1'], reverse=True)
    
    print(f"Successful configs: {len(successful_results)}/{len(results)}\n")
    print("Top 5 configurations:\n")
    
    for i, result in enumerate(successful_results[:5], 1):
        config = result['config']
        f1 = result['avg_f1']
        time_min = result['elapsed_seconds'] / 60
        
        print(f"{i}. F1: {f1:.4f} | Time: {time_min:.1f}min")
        print(f"   {config}")
        print()
    
    # Best configuration
    best = successful_results[0]
    best_config = best['config']
    best_f1 = best['avg_f1']
    
    print("="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"\nAverage F1: {best_f1:.4f}")
    print(f"\nParameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Generate training command
    print("\n" + "="*80)
    print("RETRAIN WITH BEST CONFIG")
    print("="*80)
    print("\nUse this command to retrain with best parameters:\n")
    
    cmd = f"""python scripts/train_improved_gpu.py \\
    --symbol {symbol} \\
    --timeframe {timeframe} \\
    --epochs 150 \\
    --batch-size 256 \\
    --lr {best_config['lr']} \\
    --hidden-dim {best_config['hidden_dim']} \\
    --lstm-layers {best_config['lstm_layers']} \\
    --dropout {best_config['dropout']} \\
    --device {device}"""
    
    print(cmd)
    print()
    
    # Save best config
    best_config_file = Path(f"best_config_{symbol}_{timeframe}.json")
    with open(best_config_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timeframe': timeframe,
            'best_f1': best_f1,
            'config': best_config,
            'command': cmd,
        }, f, indent=2)
    
    print(f"Best config saved to: {best_config_file}")
    print()


if __name__ == "__main__":
    main()
