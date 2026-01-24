#!/usr/bin/env python3
"""
Export Phantom V9 Data for TypeScript Backtest
Generates a JSON file with Candles + Model Signals.
"""
import pandas as pd
import numpy as np
import torch
import sys
import json
from pathlib import Path

# Fix path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna, detect_eth_setups
from scripts.phantom_v9.train_phantom_dqn import PhantomNet

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
MODEL_PATH = "models/phantom_v9/phantom_v9_best.pth" # MkII Model
OUTPUT_PATH = "data/phantom_v9_ts_data.json"

def main():
    print(f"📦 Exporting Phantom V9 Data for TS...")
    
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, '5m', limit=50000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    # 1. Calculate Features
    df = calculate_phantom_dna(df)
    
    # 2. Load Model
    device = torch.device("cpu")
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 3. Generate Signals
    print("Generating Signals...")
    signals = []
    
    # We need to iterate all rows to generate signals for the TS bot
    # The TS bot runs on every candle.
    # However, to save space/time, we can just export the ones that meet the "Detection" criteria?
    # No, the TS bot needs the candles for simulation.
    # But we can pre-calculate the signal for every candle.
    
    # Optimization: Vectorize or batch?
    # For 50k rows, loop is fine.
    
    export_data = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Default Signal
        signal = {
            "action": "PASS",
            "confidence": 0.0,
            "short_prob": 0.0
        }
        
        # Only run model if it passes detection (Optimization)
        # Re-implement detection logic inline or check if it's in candidates?
        # Let's just run detection logic here to be safe.
        
        # Detection Logic (from detect_phantom_tops.py)
        # 1. Resistance
        near_resistance = abs(row['dist_ema_20']) < 0.005 
        # 2. Staleness
        is_tired = row['staleness'] > 15
        # 3. Volatility
        is_volatile = abs(row['vol_z']) > 0.2
        # 4. Rejection
        is_rejection = (row['close'] < row['open']) or (row['is_fakeout'] == 1)
        
        if near_resistance and is_tired and is_volatile and is_rejection:
            # Run Model
            state = np.array([
                row['velocity'] / row['close'] * 10000,
                row['acceleration'] / row['close'] * 10000,
                row['cvd_slope'] / 1e6,
                row['bear_trap'],
                row['vol_z'],
                row['volume_ratio'],
                row['dist_ema_20'] * 100,
                row['dist_ema_200'] * 100,
                row['staleness'] / 50.0,
                row['weakness_score'],
                row['is_fakeout'],
                row['reserved']
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_t)
                action_idx = torch.argmax(q_values).item()
                probs = torch.softmax(q_values, dim=1)[0]
                confidence = probs[1].item() # Short Prob
                
            if action_idx == 1:
                signal["action"] = "SHORT"
                signal["confidence"] = float(confidence)
                signal["short_prob"] = float(confidence)
        
        # Append to export
        export_data.append({
            "timestamp": int(row['timestamp'].timestamp() * 1000), # TS expects ms
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row['volume']),
            "signal": signal
        })
        
        if i % 5000 == 0:
            print(f"Processed {i}/{len(df)}")
            
    # Save to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(export_data, f)
        
    print(f"✅ Exported {len(export_data)} candles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
