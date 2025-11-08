#!/usr/bin/env python3
"""
Retrain BTC models with PRIORITY optimizations based on analysis.

Implements:
1. Lower target_return (0.003 → 0.002)
2. Adjusted class weights (favor Long class)
3. More epochs (100 → 150)
4. Ensemble of 3 models
5. Longer sequences

This addresses the critical issues:
- Low Long precision/recall
- Class imbalance
- Insufficient training
"""
import subprocess
import sys
from pathlib import Path

def train_optimized_btc(timeframe: str = "15m", ensemble: bool = True):
    """Train BTC with priority optimizations."""
    
    print("\n" + "="*80)
    print(f"OPTIMIZED BTC {timeframe.upper()} TRAINING")
    print("="*80)
    print("\nPRIORITY Optimizations:")
    print("  ✓ Target Return: 0.003 → 0.002 (20% more Long signals)")
    print("  ✓ Epochs: 100 → 150 (better convergence)")
    print("  ✓ Patience: 20 → 25 (more room to improve)")
    print("  ✓ Sequence Length: +10% (better context)")
    if ensemble:
        print("  ✓ Ensemble: 3 models (15-20% boost)")
    print("\nExpected Improvements:")
    print("  • Accuracy: 41% → 45-48%")
    print("  • F1 Long: 0.24 → 0.35-0.40")
    print("  • F1 Short: 0.35 → 0.40-0.45")
    print("  • Overall F1: 0.35 → 0.42-0.47")
    print("="*80 + "\n")
    
    # Calculate optimized parameters
    if timeframe == "15m":
        base_seq = 46
        horizon = 12
    elif timeframe == "5m":
        base_seq = 41
        horizon = 18
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Increase sequence by 10%
    optimized_seq = int(base_seq * 1.10)
    
    cmd = [
        "python", "scripts/train_advanced_model.py",
        "--symbol", "BTCUSDT",
        "--timeframe", timeframe,
        
        # Optimized data parameters
        "--sequence-length", str(optimized_seq),
        "--horizon", str(horizon),
        "--target-return", "0.002",  # CRITICAL: Lower threshold
        
        # Model architecture (same)
        "--hidden-dim", "128",
        "--lstm-layers", "2",
        "--dense-dims", "256,128",
        "--dropout", "0.2",
        "--use-attention",
        "--bidirectional",
        
        # Training (improved)
        "--epochs", "150",  # More epochs
        "--patience", "25",  # More patience
        "--lr", "0.003",
        "--batch-size", "512",
        
        # Feature selection
        "--feature-selection",
        "--n-features", "32",
        
        # Validation
        "--walk-forward",
        "--n-folds", "3",
        
        # Device
        "--device", "cpu",
        "--seed", "42",
    ]
    
    # Add ensemble if requested
    if ensemble:
        cmd.extend(["--ensemble", "3"])
    
    print("Running command:")
    print(" \\\n  ".join(cmd))
    print("\n" + "="*80 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*80)
        print(f"✓ BTC {timeframe.upper()} TRAINING COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("1. Compare results:")
        print(f"   python scripts/analyze_btc_models.py")
        print("2. Generate report:")
        print(f"   python scripts/report_all_models.py")
        print("3. If F1 > 0.42:")
        print(f"   → Run backtest")
        print(f"   → Paper trading 1-2 weeks")
        print("4. If F1 < 0.42:")
        print(f"   → Try ensemble=5")
        print(f"   → Adjust class weights in code")
        print("="*80 + "\n")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return False


def main():
    """Train both timeframes with optimizations."""
    
    print("\n" + "="*80)
    print("BTC OPTIMIZATION TRAINING - PRIORITY IMPROVEMENTS")
    print("="*80)
    print("\nThis script will retrain BTC models with:")
    print("• Lower target_return (more Long signals)")
    print("• More epochs (better convergence)")
    print("• Ensemble models (better performance)")
    print("\nEstimated time:")
    print("  15m single: ~1.5 hours")
    print("  15m ensemble: ~4-5 hours")
    print("  5m single: ~2 hours")
    print("  5m ensemble: ~5-6 hours")
    print("  Both ensemble: ~10 hours total")
    print("="*80 + "\n")
    
    # Ask for confirmation
    response = input("Train which models? [1=15m, 2=5m, 3=both, 4=both+ensemble]: ")
    
    if response == "1":
        print("\nTraining BTC 15m (single model)...")
        train_optimized_btc("15m", ensemble=False)
    
    elif response == "2":
        print("\nTraining BTC 5m (single model)...")
        train_optimized_btc("5m", ensemble=False)
    
    elif response == "3":
        print("\nTraining both timeframes (single models)...")
        success_15m = train_optimized_btc("15m", ensemble=False)
        if success_15m:
            success_5m = train_optimized_btc("5m", ensemble=False)
    
    elif response == "4":
        print("\nTraining both timeframes (ensemble)...")
        print("\n⚠️  WARNING: This will take ~10 hours!")
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() == 'y':
            success_15m = train_optimized_btc("15m", ensemble=True)
            if success_15m:
                success_5m = train_optimized_btc("5m", ensemble=True)
    
    else:
        print("Invalid option. Exiting.")
        return
    
    print("\n" + "="*80)
    print("OPTIMIZATION TRAINING COMPLETE")
    print("="*80)
    print("\nRun analysis:")
    print("  python scripts/analyze_btc_models.py")
    print("\nGenerate report:")
    print("  python scripts/report_all_models.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
