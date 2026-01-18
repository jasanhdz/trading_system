#!/usr/bin/env python3
"""
Project Wraith: Distribution Top Detector
Target: Identify High-Probability Short Setups in Futures.
Physics:
1. Buyer Exhaustion (Volume Divergence)
2. Momentum Deceleration (Negative Acceleration)
3. Resistance Compression (EMA 200)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger("wraith_detector")

# Config
DB_URL = "sqlite:///data/binance_candles.db"
db_manager = DatabaseManager(DB_URL)

def calculate_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for Project Wraith:
    - EMA 200 (The Ceiling)
    - Velocity & Acceleration (Momentum Physics)
    - Volume Divergence
    - Volatility Compression
    """
    df = df.copy()
    
    # 1. The Ceiling (EMA 200)
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['dist_to_ema'] = (df['close'] - df['ema_200']) / df['ema_200']
    
    # 2. Momentum Physics (Velocity & Acceleration)
    # Velocity = Price Change (1st Derivative)
    df['velocity'] = df['close'].diff()
    # Acceleration = Velocity Change (2nd Derivative)
    df['acceleration'] = df['velocity'].diff()
    
    # Smoothed for noise reduction
    df['velocity_sm'] = df['velocity'].rolling(5).mean()
    df['acceleration_sm'] = df['acceleration'].rolling(5).mean()
    
    # 3. Volume Physics
    # Bullish Volume vs Bearish Volume
    df['delta'] = df['close'].diff()
    df['bull_vol'] = np.where(df['delta'] > 0, df['volume'], 0)
    df['bear_vol'] = np.where(df['delta'] < 0, df['volume'], 0)
    
    # Smoothed Volume
    df['vol_sm'] = df['volume'].rolling(10).mean()
    
    # 4. Volatility (Compression)
    # Normalized ATR or StdDev
    df['std_20'] = df['close'].rolling(20).std()
    df['volatility_z'] = (df['std_20'] - df['std_20'].rolling(200).mean()) / df['std_20'].rolling(200).std()
    
    # 5. Bollinger Bands (Upper Interaction)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_dist'] = (df['bb_upper'] - df['close']) / df['close'] # Distance to upper band
    
    return df

def detect_wraith_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Wraith Logic to detect setups.
    """
    candidates = []
    
    # Iterate (skipping first 200 for EMA)
    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- LOGIC 1: Resistance Proximity ---
        # Price is below EMA 200 but close (within 0.5%) OR slightly above (fakeout)
        # Ideally, we want to catch the rejection AT the EMA.
        # Let's define "Near Ceiling" as -1.0% to +0.5% around EMA 200
        near_ceiling = -0.01 < row['dist_to_ema'] < 0.005
        
        if not near_ceiling:
            continue
            
        # --- LOGIC 2: Momentum Deceleration (The Stall) ---
        # Price is still high/rising (Velocity > 0 or small), but Acceleration is Negative
        # Or Price made a Higher High but Velocity made a Lower High (Divergence)
        
        # Simple Physics: Deceleration
        is_decelerating = row['acceleration_sm'] < 0
        
        # --- LOGIC 3: Buyer Exhaustion (Volume) ---
        # Price is near local high, but Volume is low/dropping
        vol_drying = row['volume'] < row['vol_sm']
        
        # --- LOGIC 4: Compression (Optional but powerful) ---
        # Volatility is low (Z-Score < -1.0) implies a squeeze before the move
        is_compressed = row['volatility_z'] < -0.5
        
        # --- TRIGGER ---
        # We need a "Weakness" signal. e.g., a Red Candle after touching the zone.
        is_red = row['close'] < row['open']
        
        if near_ceiling and is_decelerating and (vol_drying or is_compressed) and is_red:
            candidates.append(i)
            
    # Create result DF
    results = df.iloc[candidates].copy()
    return results

def plot_candidates(df: pd.DataFrame, candidates: pd.DataFrame, symbol: str):
    """
    Visualizes the setups.
    """
    if candidates.empty:
        print("No candidates to plot.")
        return
        
    plt.figure(figsize=(14, 8))
    
    # Plot Price & EMA
    plt.plot(df.index, df['close'], label='Price', color='black', alpha=0.6)
    plt.plot(df.index, df['ema_200'], label='EMA 200', color='blue', linewidth=1.5)
    
    # Plot Candidates
    plt.scatter(candidates.index, candidates['close'], color='red', label='Wraith Short', marker='v', s=100, zorder=5)
    
    plt.title(f"Project Wraith: Distribution Tops - {symbol}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = f"reports/wraith_detection_{symbol.replace('/', '_')}.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    
    logger.info(f"Scanning for Wraith Setups on {args.symbol} {args.timeframe}...")
    
    # 1. Load Data
    df = db_manager.get_ohlcv_data(args.symbol, args.timeframe, limit=args.limit)
    if df.empty:
        logger.error("No data found.")
        return
        
    # Ensure timestamp is a column
    if 'timestamp' not in df.columns:
        df = df.reset_index()
        
    # 2. Engineer Features
    df = calculate_physics_features(df)
    
    # 3. Detect
    candidates = detect_wraith_setups(df)
    
    logger.info(f"Found {len(candidates)} Wraith Candidates.")
    
    if not candidates.empty:
        print("\nTop 5 Candidates:")
        print(candidates[['timestamp', 'close', 'ema_200', 'dist_to_ema', 'acceleration_sm', 'volatility_z']].tail())
        
        # 4. Plot
        # We need numeric index for plotting if using matplotlib simple plot, 
        # but df index is usually RangeIndex or Datetime. 
        # db_manager returns RangeIndex usually.
        plot_candidates(df, candidates, args.symbol)
        
    else:
        print("No setups found matching criteria.")

if __name__ == "__main__":
    main()
