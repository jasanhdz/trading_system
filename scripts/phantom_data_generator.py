#!/usr/bin/env python3
"""
Project Phantom: ETH High-Density Data Generator
Target: Extract 1000+ training examples from historical ETH drops.
Method: Find every 3%+ drop in 6 hours and map the pre-drop signature.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager

# Config
DB_URL = "sqlite:///data/binance_candles.db"
TIMEFRAME = "5m"

# Drop Detection Params
MIN_DROP_PCT = 0.03  # 3% minimum drop to qualify
DROP_HORIZON = 72    # 6 hours (72 * 5min)
LOOKBACK = 6         # 30 minutes before entry (6 * 5min)

def load_dual_data(limit=100000):
    """Load BTC and ETH data for maximum coverage."""
    print("📊 Loading historical data...")
    
    db_manager = DatabaseManager(DB_URL)
    
    # Load BTC
    btc = db_manager.get_ohlcv_data("BTC/USDT", TIMEFRAME, limit=limit)
    if 'timestamp' not in btc.columns:
        btc = btc.reset_index()
    btc = btc.add_suffix('_btc')
    btc = btc.rename(columns={'timestamp_btc': 'timestamp'})
    
    # Load ETH
    eth = db_manager.get_ohlcv_data("ETH/USDT", TIMEFRAME, limit=limit)
    if 'timestamp' not in eth.columns:
        eth = eth.reset_index()
    eth = eth.add_suffix('_eth')
    eth = eth.rename(columns={'timestamp_eth': 'timestamp'})
    
    # Merge
    merged = pd.merge(btc, eth, on='timestamp', how='inner')
    
    print(f"✅ Loaded {len(merged)} candles")
    return merged

def calculate_cvd_proxy(df):
    """
    CVD Proxy: Cumulative Volume Delta
    Approximation using candle body direction * volume.
    Positive = aggressive buying, Negative = aggressive selling.
    """
    print("🔬 Calculating CVD Proxy...")
    
    # Direction: 1 if green, -1 if red
    df['direction'] = np.where(df['close_eth'] > df['open_eth'], 1, -1)
    
    # Volume Delta per candle
    df['volume_delta'] = df['direction'] * df['volume_eth']
    
    # CVD (cumulative over rolling window)
    df['cvd_5'] = df['volume_delta'].rolling(5).sum()
    df['cvd_20'] = df['volume_delta'].rolling(20).sum()
    
    # CVD slope (momentum of accumulation/distribution)
    df['cvd_slope'] = df['cvd_20'].diff(5)
    
    # Normalized CVD
    df['cvd_z'] = (df['cvd_20'] - df['cvd_20'].rolling(50).mean()) / (df['cvd_20'].rolling(50).std() + 1e-8)
    
    return df

def calculate_phantom_features(df):
    """Calculate all Phantom-specific features."""
    print("🦅 Calculating Phantom features...")
    
    # ETH/BTC Ratio
    df['eth_btc_ratio'] = df['close_eth'] / df['close_btc']
    df['eth_btc_ema'] = df['eth_btc_ratio'].ewm(span=20).mean()
    df['weakness_score'] = (df['eth_btc_ema'] - df['eth_btc_ratio']) / (df['eth_btc_ema'] + 1e-8) * 100
    
    # Volatility
    df['returns'] = df['close_eth'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_z'] = (df['volatility'] - df['volatility'].expanding().mean()) / (df['volatility'].expanding().std() + 1e-8)
    
    # Price Features
    df['ema_20'] = df['close_eth'].ewm(span=20).mean()
    df['ema_200'] = df['close_eth'].ewm(span=200).mean()
    df['dist_ema20'] = (df['close_eth'] - df['ema_20']) / df['close_eth']
    df['dist_ema200'] = (df['close_eth'] - df['ema_200']) / df['close_eth']
    
    # Momentum
    df['velocity'] = df['close_eth'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['velocity_sm'] = df['velocity'].ewm(span=5).mean()
    df['acceleration_sm'] = df['acceleration'].ewm(span=5).mean()
    
    # Fakeout Detection
    df['body'] = abs(df['open_eth'] - df['close_eth'])
    df['upper_wick'] = df['high_eth'] - df[['open_eth', 'close_eth']].max(axis=1)
    df['lower_wick'] = df[['open_eth', 'close_eth']].min(axis=1) - df['low_eth']
    df['is_fakeout'] = df['upper_wick'] > (df['body'] * 1.5)
    
    # Volume Analysis
    df['vol_sma'] = df['volume_eth'].rolling(20).mean()
    df['vol_ratio'] = df['volume_eth'] / (df['vol_sma'] + 1e-8)
    
    # Staleness (consecutive doji candles)
    df['is_doji'] = df['body'] < (df['high_eth'] - df['low_eth']) * 0.1
    df['staleness'] = df['is_doji'].rolling(10).sum()
    
    # CVD features
    df = calculate_cvd_proxy(df)
    
    return df

def find_significant_drops(df):
    """
    Find all candles that precede a 3%+ drop in the next 6 hours.
    These are "golden" short entry points.
    """
    print("🔍 Scanning for significant drops...")
    
    drop_candidates = []
    
    for i in range(LOOKBACK, len(df) - DROP_HORIZON):
        entry_price = df.iloc[i]['close_eth']
        
        # Look at future 6 hours
        future = df.iloc[i+1:i+DROP_HORIZON+1]
        
        # Find max drop
        min_price = future['low_eth'].min()
        max_drop = (entry_price - min_price) / entry_price
        
        if max_drop >= MIN_DROP_PCT:
            # This is a valid drop candidate
            candidate = {
                'idx': i,
                'timestamp': df.iloc[i]['timestamp'],
                'entry_price': entry_price,
                'min_price': min_price,
                'drop_pct': max_drop * 100,
                # Pre-drop features
                'cvd_z': df.iloc[i]['cvd_z'],
                'cvd_slope': df.iloc[i]['cvd_slope'],
                'weakness_score': df.iloc[i]['weakness_score'],
                'volatility_z': df.iloc[i]['volatility_z'],
                'is_fakeout': df.iloc[i]['is_fakeout'],
                'vol_ratio': df.iloc[i]['vol_ratio'],
                'staleness': df.iloc[i]['staleness'],
                'velocity_sm': df.iloc[i]['velocity_sm'],
                'acceleration_sm': df.iloc[i]['acceleration_sm'],
                'dist_ema20': df.iloc[i]['dist_ema20'],
                'dist_ema200': df.iloc[i]['dist_ema200'],
            }
            drop_candidates.append(candidate)
    
    print(f"✅ Found {len(drop_candidates)} drop candidates (>={MIN_DROP_PCT*100}%)")
    return pd.DataFrame(drop_candidates)

def analyze_drop_patterns(candidates):
    """Analyze what the pre-drop signature looks like."""
    print("\n📊 PHANTOM DROP PATTERN ANALYSIS:")
    print("=" * 60)
    
    if candidates.empty:
        print("❌ No candidates to analyze")
        return
    
    print(f"\n📈 Candidates: {len(candidates)}")
    print(f"📉 Avg Drop: {candidates['drop_pct'].mean():.2f}%")
    print(f"📉 Max Drop: {candidates['drop_pct'].max():.2f}%")
    
    # CVD Analysis
    print("\n🔬 CVD (Volume Delta) Before Drops:")
    print(f"   Avg CVD Z-Score: {candidates['cvd_z'].mean():.2f}")
    print(f"   Avg CVD Slope: {candidates['cvd_slope'].mean():.4f}")
    
    # Weakness Analysis
    print("\n📉 ETH/BTC Weakness Before Drops:")
    print(f"   Avg Weakness Score: {candidates['weakness_score'].mean():.2f}")
    weak_pct = (candidates['weakness_score'] > 0).sum() / len(candidates) * 100
    print(f"   % Weaker than BTC: {weak_pct:.1f}%")
    
    # Fakeout Analysis
    fakeout_pct = candidates['is_fakeout'].sum() / len(candidates) * 100
    print(f"\n🎭 Fakeout (Long Upper Wick) Rate: {fakeout_pct:.1f}%")
    
    # Volume Analysis
    high_vol_pct = (candidates['vol_ratio'] > 1.5).sum() / len(candidates) * 100
    print(f"📊 High Volume Rate (>1.5x): {high_vol_pct:.1f}%")
    
    # Staleness Analysis
    stale_pct = (candidates['staleness'] > 3).sum() / len(candidates) * 100
    print(f"⏸️ Staleness (>3 Doji): {stale_pct:.1f}%")
    
    return candidates

def generate_phantom_training_data(df, candidates):
    """
    Generate training data for Phantom AI.
    Format: 12-feature state vector + reward (drop size).
    """
    print("\n🧠 Generating Phantom Training Data...")
    
    training_data = []
    
    for _, row in candidates.iterrows():
        idx = row['idx']
        
        # 12-feature state vector
        state = [
            # CVD features (unique to Phantom)
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] if not pd.isna(row['cvd_slope']) else 0,
            
            # ETH/BTC weakness
            row['weakness_score'] if not pd.isna(row['weakness_score']) else 0,
            
            # Volatility
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            
            # Fakeout
            float(row['is_fakeout']),
            
            # Volume
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            
            # Staleness
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            
            # Momentum
            row['velocity_sm'] / row['entry_price'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['entry_price'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            
            # Price position
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            
            # Reserved for BTC context
            0  # Placeholder for future BTC momentum feature
        ]
        
        # Reward (asymmetric: PnL squared)
        drop_pct = row['drop_pct'] / 100
        reward = (drop_pct ** 2) * 1000  # Amplify big drops
        
        training_data.append({
            'timestamp': row['timestamp'],
            'state': state,
            'action': 1,  # All these are SHORT candidates
            'reward': reward,
            'drop_pct': row['drop_pct']
        })
    
    # Also generate "PASS" examples (non-drop periods)
    print("🔄 Generating PASS examples (non-drop periods)...")
    pass_count = 0
    
    for i in range(LOOKBACK, len(df) - DROP_HORIZON, 50):  # Sample every 50th candle
        # Check if this is NOT a drop candidate
        entry_price = df.iloc[i]['close_eth']
        future = df.iloc[i+1:i+DROP_HORIZON+1]
        min_price = future['low_eth'].min()
        max_drop = (entry_price - min_price) / entry_price
        
        if max_drop < 0.015:  # Less than 1.5% drop = should PASS
            state = [
                df.iloc[i]['cvd_z'] if not pd.isna(df.iloc[i]['cvd_z']) else 0,
                df.iloc[i]['cvd_slope'] if not pd.isna(df.iloc[i]['cvd_slope']) else 0,
                df.iloc[i]['weakness_score'] if not pd.isna(df.iloc[i]['weakness_score']) else 0,
                df.iloc[i]['volatility_z'] if not pd.isna(df.iloc[i]['volatility_z']) else 0,
                float(df.iloc[i]['is_fakeout']),
                df.iloc[i]['vol_ratio'] - 1.0 if not pd.isna(df.iloc[i]['vol_ratio']) else 0,
                df.iloc[i]['staleness'] / 10 if not pd.isna(df.iloc[i]['staleness']) else 0,
                df.iloc[i]['velocity_sm'] / df.iloc[i]['close_eth'] * 1000 if not pd.isna(df.iloc[i]['velocity_sm']) else 0,
                df.iloc[i]['acceleration_sm'] / df.iloc[i]['close_eth'] * 1000 if not pd.isna(df.iloc[i]['acceleration_sm']) else 0,
                df.iloc[i]['dist_ema20'] * 100 if not pd.isna(df.iloc[i]['dist_ema20']) else 0,
                df.iloc[i]['dist_ema200'] * 100 if not pd.isna(df.iloc[i]['dist_ema200']) else 0,
                0
            ]
            
            training_data.append({
                'timestamp': df.iloc[i]['timestamp'],
                'state': state,
                'action': 0,  # PASS
                'reward': 0.1,  # Small positive for correct pass
                'drop_pct': max_drop * 100
            })
            pass_count += 1
    
    print(f"✅ Generated {len(training_data)} training examples")
    print(f"   SHORT: {len(candidates)}")
    print(f"   PASS: {pass_count}")
    
    return pd.DataFrame(training_data)

def main():
    print("🦅 PROJECT PHANTOM: ETH DATA GENERATOR 🦅")
    print("=" * 60)
    print("📋 Target: Generate 1000+ high-quality training examples")
    print("📋 Method: Find all 3%+ drops in 6h and map pre-drop signature")
    print("=" * 60)
    
    # Load maximum data
    df = load_dual_data(limit=100000)
    
    # Calculate features
    df = calculate_phantom_features(df)
    
    # Find drops
    candidates = find_significant_drops(df)
    
    # Analyze patterns
    candidates = analyze_drop_patterns(candidates)
    
    # Generate training data
    training_data = generate_phantom_training_data(df, candidates)
    
    # Save
    df.to_csv("data/phantom_features.csv", index=False)
    candidates.to_csv("data/phantom_candidates.csv", index=False)
    training_data.to_pickle("data/phantom_training.pkl")
    
    print(f"\n📁 Data saved:")
    print(f"   - data/phantom_features.csv ({len(df)} rows)")
    print(f"   - data/phantom_candidates.csv ({len(candidates)} rows)")
    print(f"   - data/phantom_training.pkl ({len(training_data)} examples)")
    
    return df, candidates, training_data

if __name__ == "__main__":
    main()
