import pandas as pd
import json
import os

# Paths
FEATURES_PATH = "data/phantom_features.csv"
ETH_JSON_PATH = "data/phantom_backtest_candles.json"
BTC_JSON_PATH = "data/phantom_backtest_btc_candles.json"

def convert_data():
    print(f"Reading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # ETH Candles
    print("Extracting ETH candles...")
    eth_candles = []
    for _, row in df.iterrows():
        eth_candles.append({
            "timestamp": int(row['timestamp'].timestamp() * 1000), # ms
            "open": float(row['open_eth']),
            "high": float(row['high_eth']),
            "low": float(row['low_eth']),
            "close": float(row['close_eth']),
            "volume": float(row['volume_eth'])
        })
    
    with open(ETH_JSON_PATH, 'w') as f:
        json.dump(eth_candles, f)
    print(f"Saved {len(eth_candles)} ETH candles to {ETH_JSON_PATH}")

    # BTC Candles
    print("Extracting BTC candles...")
    btc_candles = []
    for _, row in df.iterrows():
        btc_candles.append({
            "timestamp": int(row['timestamp'].timestamp() * 1000), # ms
            "open": float(row['open_btc']),
            "high": float(row['high_btc']),
            "low": float(row['low_btc']),
            "close": float(row['close_btc']),
            "volume": float(row['volume_btc'])
        })
        
    with open(BTC_JSON_PATH, 'w') as f:
        json.dump(btc_candles, f)
    print(f"Saved {len(btc_candles)} BTC candles to {BTC_JSON_PATH}")

if __name__ == "__main__":
    convert_data()
