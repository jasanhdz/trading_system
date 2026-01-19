#!/usr/bin/env python3
"""
Export Phantom V8 Backtest Data
Target: Export the exact candles used in the backtest to a JSON file for TS verification.
"""
import pandas as pd
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Config
FEATURES_PATH = "data/phantom_features.csv"
OUTPUT_PATH = "data/phantom_backtest_candles.json"
HORIZON = 288

def main():
    print("🦅 EXPORTING PHANTOM V8 DATA 🦅")
    
    # Load features
    if not Path(FEATURES_PATH).exists():
        print("❌ Features not found. Run phantom_data_generator.py first.")
        return
    
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter range used in backtest (200 to len - HORIZON)
    # We export ALL data so the TS bot can simulate the same range
    # But we need to make sure column names match what TS expects (or what we'll parse)
    
    # The TS bot needs: timestamp, open, high, low, close, volume
    # Our features file has: open_eth, high_eth, low_eth, close_eth, volume_eth
    
    export_data = []
    
    for idx, row in df.iterrows():
        export_data.append({
            "timestamp": row['timestamp'].timestamp() * 1000, # MS for TS
            "open": row['open_eth'],
            "high": row['high_eth'],
            "low": row['low_eth'],
            "close": row['close_eth'],
            "volume": row['volume_eth'],
            # Include BTC data if needed for context, but ML service fetches it live usually.
            # For backtest injection, we might need to inject BTC too if ML service uses it.
            # ML service uses fetch_candles("BTCUSDT").
            # We should probably export BTC candles too if we want full isolation.
            # But for now let's assume we just inject the ETH candles and maybe ML service can still fetch BTC?
            # No, if we want deterministic backtest, we should inject both.
            # But let's start with ETH injection.
        })
        
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(export_data, f)
        
    print(f"✅ Exported {len(export_data)} candles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
