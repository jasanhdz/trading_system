#!/usr/bin/env python3
"""
Debug Sequential Logic
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Config
FEATURES_PATH = "data/phantom_features.csv"
MODEL_PATH = "models/phantom_eth/phantom_net_best.pth"
CONFIDENCE_THRESHOLD = 0.60 # Match V8
HORIZON = 288

# Exit Params
SL_PCT = 0.015
TP_PCT = 0.06
TRAILING_DEV = 0.015
BE_TRIGGER_ROE = 0.10

class PhantomNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PhantomNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

def simulate_trade(entry_price, future_candles, leverage):
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.003)
    peak_price = entry_price
    is_breakeven = False
    
    for i, row in future_candles.iterrows():
        # Update Peak
        if row['low_eth'] < peak_price:
            peak_price = row['low_eth']
        
        # Check BE Trigger
        current_roe = (entry_price - row['low_eth']) / entry_price * leverage
        if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
            sl_price = be_price
            is_breakeven = True
        
        # Check Trailing
        if is_breakeven:
            trailing_sl = peak_price * (1 + TRAILING_DEV)
            if row['high_eth'] >= trailing_sl:
                return trailing_sl, "TRAILING", is_breakeven, i + 1
        
        # Check SL
        if row['high_eth'] >= sl_price:
            return sl_price, "STOP_LOSS", is_breakeven, i + 1
        
        # Check TP
        if row['low_eth'] <= tp_price:
            return tp_price, "TAKE_PROFIT", is_breakeven, i + 1
    
    return future_candles.iloc[-1]['close_eth'], "TIME_LIMIT", is_breakeven, len(future_candles)

def main():
    print("DEBUG SEQUENTIAL LOGIC")
    
    # Load Data
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhantomNet(12, 64, 2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    next_free_idx = 0
    trades_taken = 0
    signals_skipped = 0
    
    print(f"\nScanning {len(df)} candles...")
    
    for idx in range(200, len(df) - HORIZON):
        row = df.iloc[idx]
        
        # 1. Check Signal
        state = np.array([
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] / 10000 if not pd.isna(row['cvd_slope']) else 0,
            row['weakness_score'] if not pd.isna(row['weakness_score']) else 0,
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            float(row['is_fakeout']) if not pd.isna(row['is_fakeout']) else 0,
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            row['velocity_sm'] / row['close_eth'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['close_eth'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            0
        ], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
            
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            # SIGNAL DETECTED
            
            # Check if blocked
            if idx < next_free_idx:
                signals_skipped += 1
                # print(f"⚠️ SKIPPED Signal at {row['timestamp']} (Blocked until {df.iloc[next_free_idx]['timestamp']})")
            else:
                # TAKE TRADE
                entry_price = row['close_eth']
                future = df.iloc[idx+1 : idx+HORIZON+1]
                exit_price, reason, hit_be, duration = simulate_trade(entry_price, future, 5)
                
                trades_taken += 1
                next_free_idx = idx + duration
                
                print(f"✅ TRADE #{trades_taken} at {row['timestamp']}")
                print(f"   Duration: {duration} candles ({duration*5/60:.1f} hours)")
                print(f"   Reason: {reason}")
                print(f"   Blocked until: {df.iloc[next_free_idx]['timestamp']}\n")

    print(f"\nSUMMARY:")
    print(f"Trades Taken: {trades_taken}")
    print(f"Signals Skipped: {signals_skipped}")

if __name__ == "__main__":
    main()
