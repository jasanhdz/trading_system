#!/usr/bin/env python3
"""
Phantom V11: Dataset Refiner (Steroid Mode)
Filters the Twin Dataset to keep ONLY high-precision entry signals.
Pre-calculates future outcomes to handle dataset discontinuity.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Config
INPUT_PATH = ROOT_DIR / "data/dataset_phantom_twins.csv"
OUTPUT_PATH = ROOT_DIR / "data/dataset_steroid.csv"

# --- STEROID FILTERS (THRESHOLDS) ---
# Relaxed further to capture intersection (~10-20%)
SLOPE_MAX = -0.00002 
VOLUME_MIN = 0.8
WEAKNESS_MIN = 0.0
VOL_Z_MIN = -0.5
FAKEOUT_ALLOWED = 0
# Anti-Panic Filters
RSI_MIN = 25
CVD_SLOPE_LIMIT = -100000 # Relaxed from -500k to ensure we get data (100k is significant)

# Look-ahead for Reward Calculation
HORIZON = 48 # 4 Hours

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_future_outcome(df):
    """
    Pre-calculates the best Short PnL and Max Drawdown for the next 48 candles.
    This ensures that even if we filter rows, the 'truth' of that trade is preserved.
    """
    print("🔮 Pre-calculating Future Outcomes (Ground Truth)...")
    
    # We need to respect Chunks if possible, but the Twin dataset is already concatenated.
    # Assuming 'timestamp' is sorted.
    # To be safe, we iterate. Vectorized approach:
    
    # We want: 
    # 'future_min_low': Min Low in next 48
    # 'future_max_high': Max High in next 48
    # 'future_close_exit': Close at t+48
    
    # We can use Rolling windows reverted.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=HORIZON)
    
    df['future_min_low'] = df['low'].rolling(window=indexer).min()
    df['future_max_high'] = df['high'].rolling(window=indexer).max()
    df['future_close_exit'] = df['close'].shift(-HORIZON)
    
    return df

def main():
    print("離 PHANTOM V11: DATASET REFINEMENT 離")
    print("Goal: Extract 'Steroid' candles for High-Winrate training.\n")

    if not INPUT_PATH.exists():
        print(f"❌ Input not found: {INPUT_PATH}")
        return

    # 1. Load Data
    print(f"Loading Data: {INPUT_PATH.name}")
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    initial_count = len(df)
    print(f"   Initial Rows: {initial_count:,}")

    # 2. Pre-Calculate Outcomes (BEFORE FILTERING)
    # This prevents the "Time Travel" bug when rows are deleted.
    df = calculate_future_outcome(df)
    
    # Cleaning any NaNs at the end
    df.dropna(subset=['future_min_low', 'future_max_high', 'future_close_exit'], inplace=True)
    
    # 2b. CALCULATE FEATURES (DNA)
    # The dataset might only have OHLCV. We need metrics for filtering.
    from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna
    print("🧬 Calculating DNA Features for Filtering...")
    df = calculate_phantom_dna(df)
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    df.fillna(0, inplace=True)

    # 3. Apply Steroid Filters
    print("\n Applying Filters...")
    
    condition = (
        (df['slope'] < SLOPE_MAX) &           # Must be Strongly Downward
        (df['volume_ratio'] > VOLUME_MIN) &     # Must have High Volume
        (df['weakness_score'] > WEAKNESS_MIN) & # Must be Weak/Bearish
        (df['vol_z'] > VOL_Z_MIN) &           # Must be Explosive Volatility
        (df['is_fakeout'] == FAKEOUT_ALLOWED) & # Must NOT be a fakeout
        (df['rsi'] > RSI_MIN) &                 # Anti-Panic: Not in freefall bottom
        (df['cvd_slope'] < CVD_SLOPE_LIMIT)     # Smart Money must be selling
    )
    
    steroid_df = df[condition].copy()
    
    # 4. Analyze Purity
    final_count = len(steroid_df)
    
    if final_count == 0:
        print("❌ WARNING: Filters were too strict! 0 candles retained.")
        return

    retained_pct = (final_count / initial_count) * 100
    
    print(f"\n--- FILTER RESULTS ---")
    print(f"✅ Retained Rows: {final_count:,} ({retained_pct:.2f}%)")
    print(f"🗑️  Filtered Rows: {initial_count - final_count:,}")
    
    # Compare stats
    print("\n--- QUALITY METRICS (Averages) ---")
    print(f"Metric       | Original     | Steroid      | Delta")
    print("-" * 50)
    
    metrics = ['slope', 'volume_ratio', 'weakness_score', 'vol_z']
    for m in metrics:
        orig_mean = df[m].mean()
        ster_mean = steroid_df[m].mean()
        diff = ((ster_mean - orig_mean) / abs(orig_mean)) * 100
        
        print(f"{m:12s} | {orig_mean:11.4f} | {ster_mean:11.4f} | {diff:+.1f}%")

    # 5. Save Steroid Dataset
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    steroid_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n✅ REFINEMENT COMPLETE.")
    print("   Dataset contains 'ground truth' columns for safe disconnected training.")

if __name__ == "__main__":
    main()
