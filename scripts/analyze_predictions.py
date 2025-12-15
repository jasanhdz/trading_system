import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from ml.advanced_models.dataset import AdvancedDatasetConfig, SequenceDataset, load_sequence_dataset
from ml.advanced_models.temporal_model import AdvancedTemporalNet

def analyze_predictions():
    symbol = "BTCUSDT"
    timeframe = "15m"
    model_dir = REPO_ROOT / "models" / "advanced" / symbol / timeframe
    
    print(f"Loading artifacts from {model_dir}...")
    
    # Load metadata
    import json
    meta_path = model_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
        
    # Load data
    print("Loading data...")
    config = AdvancedDatasetConfig(
        symbol="BTC/USDT:USDT",
        timeframe=timeframe,
        sequence_length=meta['sequence_length'],
        prediction_horizon=meta['prediction_horizon'],
        target_return=meta['target_return'],
        use_feature_selection=False # Load raw first
    )
    
    features, class_labels, regression_targets, feature_names = load_sequence_dataset(config)
    
    # Load feature selector and transform
    print("Applying feature selection...")
    selector = joblib.load(model_dir / "feature_selector.pkl")
    features_selected = selector.transform(features)
    
    # Load scaler and transform
    print("Scaling features...")
    scaler = joblib.load(model_dir / "scaler.pkl")
    features_scaled = scaler.transform(features_selected)
    
    # Recreate Test Split (same logic as training)
    # We need to replicate the exact split logic to get the test set
    n_samples = len(features)
    train_end = int(n_samples * 0.7)
    valid_end = int(n_samples * 0.85)
    
    lookback = meta['sequence_length']
    test_start_idx = max(0, valid_end - lookback)
    
    X_test = features_scaled[test_start_idx:]
    y_test_cls = class_labels[test_start_idx:]
    y_test_reg = regression_targets[test_start_idx:]
    
    # Create dataset
    test_dataset = SequenceDataset(
        X_test,
        y_test_cls,
        y_test_reg,
        sequence_length=meta['sequence_length'],
        prediction_horizon=meta['prediction_horizon'],
        augment=False,
        start_index=lookback # Skip buffer
    )
    
    # Load Model
    print("Loading model...")
    device = "cpu"
    model_config = meta['model_config']
    model = AdvancedTemporalNet(
        input_dim=len(meta['selected_features']),
        sequence_length=meta['sequence_length'],
        **model_config
    ).to(device)
    
    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Predict
    print("Running inference...")
    all_probs = []
    all_targets = []
    all_returns = []
    
    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            seq, cls, reg = batch
            outputs = model(seq.to(device))
            probs = torch.softmax(outputs['logits'], dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(cls.cpu().numpy())
            all_returns.append(reg.cpu().numpy())
            
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    returns = np.concatenate(all_returns)
    
    # Analysis
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # 1. Probability Distribution
    max_probs = probs.max(axis=1)
    pred_classes = probs.argmax(axis=1)
    
    print("\nConfidence Distribution:")
    bins = [0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(max_probs, bins)
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        count = counts[i]
        pct = count / len(max_probs) * 100
        print(f"  {low:.2f} - {high:.2f}: {count:5d} samples ({pct:.1f}%)")
        
    # 2. PnL by Confidence Bucket
    print("\nPerformance by Confidence Level:")
    print(f"{'Conf Range':<15} | {'Trades':<8} | {'Win Rate':<8} | {'PnL (No Cost)':<15}")
    print("-" * 60)
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (max_probs >= low) & (max_probs < high) & (pred_classes != 0) # Ignore neutral predictions for PnL
        
        if mask.sum() == 0:
            continue
            
        bucket_preds = pred_classes[mask]
        bucket_returns = returns[mask]
        
        # Calculate PnL
        pnl = np.zeros_like(bucket_returns)
        pnl[bucket_preds == 1] = bucket_returns[bucket_preds == 1] # Long
        pnl[bucket_preds == 2] = -bucket_returns[bucket_preds == 2] # Short
        
        total_pnl = pnl.sum()
        win_rate = (pnl > 0).mean()
        
        print(f"{low:.2f} - {high:.2f}    | {mask.sum():<8} | {win_rate*100:6.1f}% | {total_pnl*100:10.2f}%")

    # 3. Cumulative PnL for Best Threshold
    best_thr = meta['test_metrics']['trading']['threshold']
    print(f"\nBest Threshold Analysis ({best_thr}):")
    
    mask = (max_probs >= best_thr) & (pred_classes != 0)
    final_preds = pred_classes[mask]
    final_returns = returns[mask]
    
    pnl = np.zeros_like(final_returns)
    pnl[final_preds == 1] = final_returns[final_preds == 1]
    pnl[final_preds == 2] = -final_returns[final_preds == 2]
    
    # Apply cost
    cost = 0.0006
    pnl -= cost
    
    print(f"  Total Trades: {len(pnl)}")
    print(f"  Avg PnL per Trade: {pnl.mean()*100:.3f}%")
    print(f"  Total PnL: {pnl.sum()*100:.2f}%")
    
    # Show top 5 best and worst trades
    sorted_idx = np.argsort(pnl)
    print("\nTop 5 Worst Trades:")
    for idx in sorted_idx[:5]:
        print(f"  PnL: {pnl[idx]*100:.2f}% | Conf: {max_probs[mask][idx]:.4f} | Class: {final_preds[idx]}")
        
    print("\nTop 5 Best Trades:")
    for idx in sorted_idx[-5:]:
        print(f"  PnL: {pnl[idx]*100:.2f}% | Conf: {max_probs[mask][idx]:.4f} | Class: {final_preds[idx]}")

if __name__ == "__main__":
    analyze_predictions()
