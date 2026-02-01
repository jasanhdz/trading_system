#!/usr/bin/env python3
"""
Phantom Twin V9: Regime Profiler
Extracts the "DNA" of the Golden Regime (Jan-Mar 2025).
Calculates ADX, Hurst, Volatility (ATR), and Slope.
"""
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.storage.database_manager import DatabaseManager

# Config
DB_URL = "sqlite:////home/jasan/Develop/trading_system/data/binance_candles.db"
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"

# Regime Period (Jan 1 2025 - Mar 31 2025)
REGIME_START = "2025-01-01"
REGIME_END = "2025-03-31"

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(df, period)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / tr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx

def calculate_hurst(series, window=100):
    """
    Calculates Rolling Hurst Exponent using a simplified R/S analysis loop.
    """
    series_np = series.values
    hurst_vals = np.full(len(series), np.nan)
    
    # Pre-calculate log(window)
    log_window = np.log(window)
    
    for i in range(window, len(series)):
        chunk = series_np[i-window:i]
        
        # R/S Analysis
        mu = np.mean(chunk)
        dc = chunk - mu
        z = np.cumsum(dc)
        r = np.max(z) - np.min(z)
        s = np.std(chunk)
        
        if s == 0 or r == 0:
            h = 0.5
        else:
            rs = r / s
            h = np.log(rs) / log_window
            
        hurst_vals[i] = h
        
    return pd.Series(hurst_vals, index=series.index)

def calculate_slope(series, period=20):
    """
    Calculates Linear Regression Slope over a rolling window.
    Normalized by price to be comparable.
    """
    # Rolling slope using polyfit on normalized data
    # Improving speed: We only need the slope.
    # Slope = (N * Sum(xy) - Sum(x)Sum(y)) / (N * Sum(x^2) - (Sum(x))^2)
    # x is always 0..N-1
    
    n = period
    x = np.arange(n)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)
    denom = n * sum_x2 - sum_x**2
    
    # We need rolling sum of y and xy
    # This is still a bit complex to vectorize perfectly with pandas rolling without custom object
    # So we'll stick to the apply method which is safer for correctness, even if slower.
    # 5m data for a few months isn't too huge.
    
    def get_slope(y):
        try:
            return np.polyfit(x, y, 1)[0]
        except:
            return 0.0

    slope = series.rolling(period).apply(get_slope, raw=True)
    return slope / series # Normalize

def main():
    print("🔎 PHANTOM TWIN V9: REGIME PROFILING 🔎")
    print(f"Target: {SYMBOL} | Regime: {REGIME_START} to {REGIME_END}")
    
    try:
        db = DatabaseManager(DB_URL)
        # Load enough data to initialize indicators before 2025
        # 100 candles for Hurst is minimum, so start Dec 1st 2024
        df = db.get_ohlcv_data(SYMBOL, TIMEFRAME, start_date='2024-12-01')
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return
    
    if df.empty:
        print("❌ No data found!")
        return

    if 'timestamp' not in df.columns:
        df = df.reset_index()

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} candles.")

    # --- Calculate Metrics ---
    print("Calculating Metrics (this may take a minute)...")
    
    # Fill NaN to avoid compounding errors, though rolling handles it partly
    df['close'] = df['close'].ffill()
    
    df['atr'] = calculate_atr(df)
    df['adx'] = calculate_adx(df)
    print("- ADX & ATR done.")
    
    df['slope'] = calculate_slope(df['close'])
    print("- Slope done.")
    
    df['hurst'] = calculate_hurst(df['close'])
    print("- Hurst done.")
    
    # --- Filter for Golden Regime ---
    mask = (df['timestamp'] >= REGIME_START) & (df['timestamp'] <= REGIME_END)
    regime_df = df[mask].copy()
    
    if regime_df.empty:
        print("❌ No data in regime period!")
        return

    # --- Profile Analysis ---
    print("\n📊 REGIME FINGERPRINT (Jan-Mar 2025) 📊")
    
    metrics = ['adx', 'hurst', 'atr', 'slope']
    
    stats = {}
    for m in metrics:
        clean_series = regime_df[m].dropna()
        if clean_series.empty:
            print(f"⚠️ Warning: {m} is all NaN")
            continue
            
        stats[m] = {
            'mean': clean_series.mean(),
            'std': clean_series.std(),
            'min': clean_series.min(),
            'max': clean_series.max(),
            'q25': clean_series.quantile(0.25),
            'q75': clean_series.quantile(0.75)
        }
        print(f"\nMetric: {m.upper()}")
        print(f"  Mean: {stats[m]['mean']:.6f}")
        print(f"  Std:  {stats[m]['std']:.6f}")
        print(f"  Min/Max: [{stats[m]['min']:.6f}, {stats[m]['max']:.6f}]")
        print(f"  IQR:   [{stats[m]['q25']:.6f}, {stats[m]['q75']:.6f}]")

    # Save Profile to JSON
    profile_path = Path(__file__).parent / "regime_profile.json"
    
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        return o

    with open(profile_path, 'w') as f:
        json.dump(stats, f, default=convert, indent=4)
        
    print(f"\n✅ Profile saved to {profile_path}")

if __name__ == "__main__":
    main()
