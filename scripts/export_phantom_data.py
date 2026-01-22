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
    
    export_data_eth = []
    export_data_btc = []
    
    for idx, row in df.iterrows():
        # ETH Candle
        export_data_eth.append({
            "timestamp": row['timestamp'].timestamp() * 1000,
            "open": row['open_eth'],
            "high": row['high_eth'],
            "low": row['low_eth'],
            "close": row['close_eth'],
            "volume": row['volume_eth']
        })
        
        # BTC Candle
        export_data_btc.append({
            "timestamp": row['timestamp'].timestamp() * 1000,
            "open": row['open_btc'],
            "high": row['high_btc'],
            "low": row['low_btc'],
            "close": row['close_btc'],
            "volume": row['volume_btc']
        })
        
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(export_data_eth, f)
        
    btc_output_path = OUTPUT_PATH.replace("candles.json", "btc_candles.json")
    with open(btc_output_path, 'w') as f:
        json.dump(export_data_btc, f)
        
    print(f"✅ Exported {len(export_data_eth)} ETH candles to {OUTPUT_PATH}")
    print(f"✅ Exported {len(export_data_btc)} BTC candles to {btc_output_path}")

if __name__ == "__main__":
    main()
