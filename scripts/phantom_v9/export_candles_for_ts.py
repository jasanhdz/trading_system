#!/usr/bin/env python3
"""
Export Candles for TS Backtest
Exports the exact same 50k candles used in Python backtest to JSON.
"""
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
OUTPUT_PATH = "data/phantom_v9_ts_data.json"

def main():
    print(f"👻 Exporting Candles for TS... 👻")
    
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, '5m', limit=50000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    # Convert timestamps to int (milliseconds)
    df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6
    
    # Convert to list of dicts
    records = df.to_dict(orient='records')
    
    # Save to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(records, f)
        
    print(f"✅ Exported {len(records)} candles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
