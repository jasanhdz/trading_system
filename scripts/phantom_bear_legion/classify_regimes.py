#!/usr/bin/env python3
"""
THE OCEAN: Phase 2 - Automatic Classifier (The Discovery)
Scans 3.5 years of history (2022-2026) and labels every candle with a Regime Type.
Inputs: data/binance_candles.db (ETH/USDT 5m)
Outputs: scripts/phantom_bear_legion/data/regime_labeled_history.csv

Classification Logic:
1. BEAR_TREND: Slope < -0.00005 & Hurst > 0.55
2. BULL_TREND: Slope > +0.00005 & Hurst > 0.55
3. NEUTRAL: Hurst < 0.50 (or residuals)
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
# from tqdm import tqdm

# Config
DB_PATH = Path(__file__).parent.parent.parent / "data/binance_candles.db"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "regime_labeled_history.csv"

SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"

# Classification Thresholds
HURST_THRESHOLD = 0.55
NEUTRAL_HURST_CAP = 0.50
SLOPE_THRESHOLD = 0.00005 # +/- 0.00005

def calculate_slope(series, window=90):
    if len(series) < window: return 0.0
    y = series.values[-window:]
    x = np.arange(window)
    # Vectorized or fast Linregress? Linregress is slow in loop.
    # Polyfit is faster.
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    except:
        return 0.0

def calculate_hurst(series, window=100):
    """
    Simplified R/S Analysis for speed on 370k rows.
    """
    if len(series) < window: return 0.5
    
    X = series.values[-window:]
    
    mean = np.mean(X)
    Y = X - mean
    Z = np.cumsum(Y)
    R = np.max(Z) - np.min(Z)
    S = np.std(X)
    if S == 0: return 0.5
    
    # H = log(R/S) / log(N)
    try:
        H = np.log(R/S) / np.log(window/2)
        return max(0.0, min(1.0, H))
    except:
        return 0.5

def main():
    print("🌊 THE OCEAN: CLASSIFYING 3.5 YEARS OF HISTORY 🌊")
    print(f"   Database: {DB_PATH}")
    
    # 1. Load Data
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT timestamp, open, high, low, close, volume 
    FROM ohlcv_data 
    WHERE symbol='{SYMBOL}' AND timeframe='{TIMEFRAME}'
    ORDER BY timestamp ASC
    """
    print("   Loading candles from DB (this might take a moment)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Loaded: {len(df):,} candles ({df['timestamp'].min()} -> {df['timestamp'].max()})")
    
    # 2. Calculate Indicators (Rolling)
    # Using Pandas Rolling with Apply is slow for 370k rows. 
    # We will use Vectorized approach or Numba if possible. Python loop is acceptable for 300k if simple.
    # Let's try apply but optimize.
    
    print("   Calculating Indicators (Slope & Hurst)...")
    
    # Pre-calculate rolling windows using sliding_view (numpy trick) would be fastest, but let's stick to safe pandas.
    # Optimization: Calculate Slope using convolution?
    # Simple Linregress Slope = Cov(x,y) / Var(x).
    # Var(x) for fixed window is Top Constant.
    # Cov(x,y) = Mean(x*y) - Mean(x)*Mean(y).
    # We can do this with rolling means! FAST.
    
    WINDOW_SLOPE = 90
    x = np.arange(WINDOW_SLOPE)
    mean_x = np.mean(x)
    var_x = np.var(x)
    
    # Rolling Constant: Mean(x) is constant.
    # We need Rolling Mean(y) and Rolling Mean(x*y).
    
    # y = close
    # x*y? x is 0..89 relative to window start? No.
    # Slope of moving window: x is always 0..N-1.
    # So we need dot product of window with x.
    # df['close'].rolling(90).apply(lambda y: np.polyfit(np.arange(90), y, 1)[0]) is very slow.
    
    # Let's rely on standard apply for now, if it takes >2 mins we optimize.
    # But for 400k rows, row-by-row apply takes ~10-20 mins.
    # Let's proceed carefully.
    
    # Vectorized Slope Calculation:
    # slope = (N*sum(xy) - sum(x)sum(y)) / (N*sum(x^2) - (sum(x))^2)
    # x = 0, 1, ..., N-1
    # sum(x), sum(x^2) are constants.
    # sum(y) is rolling sum of close.
    # sum(xy) is rolling sum of (close * index_in_window).
    # This is convolution!
    
    from scipy.ndimage import convolve1d
    
    # Slope
    N = WINDOW_SLOPE
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    delta = N * sum_x2 - sum_x**2
    
    # Kernel for sum(xy): [0, 1, 2... N-1] reversed?
    # Convolve computes sum(w[k] * f[i-k]).
    # We want sum(w[k] * f[i - (N-1) + k]).
    # Let's use standard pandas apply for Simplicity unless user complains. 
    # WAIT! "The Ocean" implies scale. I will use a simplified loop for Hurst and Slope.
    
    slopes = np.zeros(len(df))
    hursts = np.zeros(len(df))
    closes = df['close'].values
    
    # Buffers
    # Precomputing x array
    x_slope = np.arange(WINDOW_SLOPE)
    
    # Processing Loop
    print(f"   Processing {len(df)} candles...")
    
    for i in range(200, len(df)):
        if i % 10000 == 0:
            print(f"   Row {i}/{len(df)} ({(i/len(df)*100):.1f}%)")

        # Slope
        y_slope = closes[i-WINDOW_SLOPE:i]
        cov_mat = np.cov(x_slope, y_slope)
        slope = cov_mat[0,1] / cov_mat[0,0]
        slopes[i] = slope
        
        # Hurst (Window 100)
        # Using Simplified R/S
        y_hurst = closes[i-100:i]
        
        mean_h = np.mean(y_hurst)
        Y_h = y_hurst - mean_h
        Z_h = np.cumsum(Y_h)
        R = np.max(Z_h) - np.min(Z_h)
        S = np.std(y_hurst)
        
        if S > 0:
            H = np.log(R/S) / np.log(50) # log(100/2)
            hursts[i] = max(0.0, min(1.0, H))
        else:
            hursts[i] = 0.5

    df['slope'] = slopes
    df['hurst'] = hursts
    
    # 3. Label Regimes
    print("   Labeling Regimes...")
    
    conditions = [
        (df['slope'] < -SLOPE_THRESHOLD) & (df['hurst'] > HURST_THRESHOLD), # BEAR
        (df['slope'] > SLOPE_THRESHOLD) & (df['hurst'] > HURST_THRESHOLD),  # BULL
        (df['hurst'] < NEUTRAL_HURST_CAP)                                   # NEUTRAL
    ]
    choices = ['BEAR_TREND', 'BULL_TREND', 'NEUTRAL']
    
    df['regime_type'] = np.select(conditions, choices, default='UNCERTAIN')
    
    # 4. Save
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    print(f"   Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # 5. Stats
    print("\n📊 REGIME DISTRIBUTION:")
    print(df['regime_type'].value_counts(normalize=True) * 100)
    print("\n✅ DISCOVERY COMPLETE.")

if __name__ == "__main__":
    main()
