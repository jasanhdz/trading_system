#!/usr/bin/env python3
"""
Phantom Twin V9: Twin Hunter
Scans historical data (2022-2024) to find periods that match the "Golden Regime".
"""
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Fix path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.storage.database_manager import DatabaseManager

# Import metrics functions
try:
    from profile_regime import calculate_adx, calculate_atr, calculate_hurst, calculate_slope
except ImportError:
    # Fallback if running from different dir, though sys.path should help if strictly organized
    # We will just duplicate for robustness in this script as it's a standalone tool
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).rolling(period).mean()

    def calculate_adx(df, period=14):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = calculate_atr(df, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / tr)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period).mean()

    def calculate_hurst(series, window=100):
        series_np = series.values
        hurst_vals = np.full(len(series), np.nan)
        log_window = np.log(window)
        for i in range(window, len(series)):
            chunk = series_np[i-window:i]
            mu = np.mean(chunk)
            dc = chunk - mu
            z = np.cumsum(dc)
            r = np.max(z) - np.min(z)
            s = np.std(chunk)
            if s == 0 or r == 0:
                h = 0.5
            else:
                h = np.log(r/s) / log_window
            hurst_vals[i] = h
        return pd.Series(hurst_vals, index=series.index)

    def calculate_slope(series, period=20):
        n = period
        x = np.arange(n)
        def get_slope(y):
            try: return np.polyfit(x, y, 1)[0]
            except: return 0.0
        return series.rolling(period).apply(get_slope, raw=True) / series

# Config
DB_URL = "sqlite:////home/jasan/Develop/trading_system/data/binance_candles.db"
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"

# Search Range: 2022 to end of 2024
SEARCH_START = "2022-01-01"
SEARCH_END = "2024-12-31"

# -----------------------------------------------------------------------------
# Chunk-based Comparison Logic
# -----------------------------------------------------------------------------

def load_profile():
    profile_path = Path(__file__).parent / "regime_profile.json"
    if not profile_path.exists():
        print("❌ Data Profile not found! Run profile_regime.py first.")
        sys.exit(1)
    with open(profile_path, 'r') as f:
        return json.load(f)

def get_chunk_profile(chunk_df):
    """
    Calculates aggregated metrics for a chunk of data.
    Returns dict with means of key metrics.
    """
    # We expect pre-calculated columns: adx, hurst, slope, atr
    return {
        'adx': chunk_df['adx'].mean(),
        'hurst': chunk_df['hurst'].mean(),
        'slope': chunk_df['slope'].mean(),
        'atr': chunk_df['atr'].mean()
    }

def calculate_distance(chunk_profile, target_profile):
    """
    Calculates Euclidean distance between chunk and target.
    We normalize metrics roughly to have equal weight.
    ADX: 0-100 (Target ~35) -> Scale / 100
    Hurst: 0.5-1.0 (Target ~0.78) -> Scale * 1 (Changes are small but significant) -> Maybe * 5 for weight? 
    Slope: Tiny numbers (0.0001) -> Scale * 10000
    """
    
    # Target Values
    t_adx = target_profile['adx']['mean']
    t_hurst = target_profile['hurst']['mean']
    t_slope = target_profile['slope']['mean']
    
    # Chunk Values
    c_adx = chunk_profile['adx']
    c_hurst = chunk_profile['hurst']
    c_slope = chunk_profile['slope']
    
    # Normalized Differences
    d_adx = (c_adx - t_adx) / 100.0  # e.g. (45 - 35)/100 = 0.1
    d_hurst = (c_hurst - t_hurst) * 5.0 # e.g. (0.70 - 0.78)*5 = -0.4  (High weight on Hurst)
    d_slope = (c_slope - t_slope) * 10000.0 # e.g. (-0.0002 - -0.0001)*10000 = -1.0
    
    # Euclidean Distance
    dist = np.sqrt(d_adx**2 + d_hurst**2 + d_slope**2)
    return dist

def main():
    print("🕵️ PHANTOM TWIN V9: CHUNK MATCHING 🕵️")
    print(f"Scanning range: {SEARCH_START} to {SEARCH_END}")
    
    profile = load_profile()
    print(f"Loaded Profile (Target Hurst: {profile['hurst']['mean']:.4f})")
    
    # Load Data
    try:
        db = DatabaseManager(DB_URL)
        df = db.get_ohlcv_data(SYMBOL, TIMEFRAME, start_date=SEARCH_START, end_date=SEARCH_END)
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return
        
    if df.empty:
        print("❌ No data found.")
        return
        
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} candles. Computing metrics...")
    
    # Pre-compute metrics for entire history
    df['close'] = df['close'].ffill()
    df['atr'] = calculate_atr(df)
    df['adx'] = calculate_adx(df)
    df['slope'] = calculate_slope(df['close'])
    df['hurst'] = calculate_hurst(df['close'])
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Scanning Chunks (Sliding Window)...")
    
    # Sliding Window parameters
    # 3 Months ~ 90 Days * 288 (5m candles/day) = 25920 candles.
    CHUNK_SIZE = 25920 
    STEP_SIZE = 2880 # Step 10 days
    
    twins = []
    
    for i in range(0, len(df) - CHUNK_SIZE, STEP_SIZE):
        chunk = df.iloc[i : i + CHUNK_SIZE].copy()
        
        # Calculate Profile
        c_p = get_chunk_profile(chunk)
        
        # Score
        dist = calculate_distance(c_p, profile)
        
        # User Logic: We want SMALL distance.
        # Threshold needs tuning. Let's see results. 
        # Ideally < 0.5 based on our scaling.
        
        # Also enforce Hard Filters (Must be Downtrend)
        if c_p['slope'] > 0: # If overall positive slope, reject immediately (we want crashes)
             continue
             
        twins.append({
            'start': chunk.iloc[0]['timestamp'],
            'end': chunk.iloc[-1]['timestamp'],
            'distance': dist,
            'metrics': c_p,
            'data': chunk
        })
        
    if not twins:
        print("❌ No matching chunks found.")
        return
        
    # Sort by similarity (distance asc)
    twins.sort(key=lambda x: x['distance'])
    
    print(f"\n🔎 Evaluated {len(twins)+1} chunks. Top 5 Matches:")
    
    valid_twins = []
    
    # Filter overlapping best twins? 
    # Logic: Pick best, remove overlaps, pick next best.
    
    final_twins = []
    covered_ranges = []
    
    for t in twins:
        t_start = t['start']
        t_end = t['end']
        
        # Check overlap
        is_overlap = False
        for r_start, r_end in covered_ranges:
            # Simple overlap check
            if (t_start <= r_end) and (t_end >= r_start):
                is_overlap = True
                break
        
        if not is_overlap and t['distance'] < 1.0: # Distance Threshold
            final_twins.append(t)
            covered_ranges.append((t_start, t_end))
            print(f"✅ MATCH: {t_start.date()} -> {t_end.date()} (Dist: {t['distance']:.4f})")
            print(f"   Hurst: {t['metrics']['hurst']:.3f} | Slope: {t['metrics']['slope']:.6f}")
    
    if not final_twins:
        print("❌ No matches below distance threshold.")
        return

    # Combine Data
    print("\n📦 Building Training Dataset...")
    
    # Start with Golden Regime (2025) - Need to load it again or fetch?
    # Better to just load it fresh to be sure.
    regime_df = db.get_ohlcv_data(SYMBOL, TIMEFRAME, start_date="2025-01-01", end_date="2025-03-31")
    if 'timestamp' not in regime_df.columns: regime_df.reset_index(inplace=True)
    
    datasets = [regime_df]
    for t in final_twins:
        datasets.append(t['data'])
        
    combined_df = pd.concat(datasets).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    out_csv = Path(__file__).parent.parent.parent.parent / "data/dataset_phantom_twins.csv"
    combined_df.to_csv(out_csv, index=False)
    
    print(f"✅ SAVED: {out_csv}")
    print(f"   Total Samples: {len(combined_df)}")
    print(f"   Includes: Jan-Mar 2025 + {len(final_twins)} Historical Twins")

if __name__ == "__main__":
    main()
