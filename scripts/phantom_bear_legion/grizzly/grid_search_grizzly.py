#!/usr/bin/env python3
"""
GRIZZLY GRID SEARCH 🦖
Optimizing SL/TP for the Crash Specialist.
Focused on High Volatility Bear Candles.
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import itertools
from pathlib import Path

# Fix path to ROOT (parent of grizzly is phantom_bear_legion, parent of that is scripts, parent is trading_system)
# Script is in: scripts/phantom_bear_legion/grizzly/
# ROOT is:      trading_system/
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# --- CONFIG ---
DATA_PATH = ROOT_DIR / "scripts/phantom_bear_legion/data/regime_labeled_history.csv"
MODEL_GRIZZLY_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/grizzly_v1.pth"

# PARAMETER GRID
# Grizzly Likes Wide Stops and Deep Targets
SL_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0] # %
TP_GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0] # %
LEVERAGE = 20.0
FIXED_MARGIN = 20.0
VOL_THRESHOLD = 0.005 # 0.5% (The Crash Gate)
HORIZON = 48 # 4 Hours

# --- MODEL ---
class BearLegionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BearLegionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        # Dropout not needed for inference

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def main():
    print(f"🦖 GRIZZLY GRID SEARCH (VECTORIZED)")
    
    # 1. Load Data
    if not DATA_PATH.exists(): return
    df = pd.read_csv(DATA_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    needed = ['velocity', 'weakness_score', 'vol_z']
    if not all(c in df.columns for c in needed):
         df = calculate_phantom_dna(df)
    df.fillna(0, inplace=True)
    df['ema_macro'] = df['close'].ewm(span=200*24*12).mean()
    
    # 2. Extract Features & Arrays
    feature_cols = ['velocity', 'acceleration', 'cvd_slope', 'bear_trap', 'vol_z', 'volume_ratio', 'dist_ema_20', 'dist_ema_200', 'staleness', 'weakness_score', 'is_fakeout']
    close_arr = df['close'].values.astype(np.float32)
    feats = df[feature_cols].values.astype(np.float32)
    
    # Norm
    eps = 1e-9
    feats[:, 0] = feats[:, 0] / (close_arr + eps) * 10000 
    feats[:, 1] = feats[:, 1] / (close_arr + eps) * 10000 
    feats[:, 2] = feats[:, 2] / 1e6 
    feats[:, 6] = feats[:, 6] * 100 
    feats[:, 8] = feats[:, 8] / 50.0 
    feats[:, 9] = feats[:, 9] / 0.05 
    feats = np.hstack([feats, np.zeros((len(feats), 1), dtype=np.float32)]) # Pad to 12
    
    # 3. Model Inference (Get Potential Triggers)
    device = torch.device("cpu")
    model = BearLegionNet(12, 2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_GRIZZLY_PATH, map_location=device))
        model.eval()
    except:
        print("❌ Model load failed. Check structure.")
        return

    t_feats = torch.FloatTensor(feats).to(device)
    with torch.no_grad():
        logits = model(t_feats)
        actions = torch.argmax(logits, dim=1).numpy()
        
    # 4. Filter Context (Where Grizzly Is Allowed To Hunt)
    # Rule 1: Bear Trend
    mask_regime = (df['regime_type'] == 'BEAR_TREND').values
    # Rule 2: Macro Shield (Price < EMA 200)
    mask_macro = (close_arr < df['ema_macro'].values)
    # Rule 3: High Volatility (The Grizzly Domain)
    volCheck = ((df['high'] - df['low']) / df['open']).values
    mask_vol = (volCheck >= VOL_THRESHOLD)
    
    # Trigger Mask
    # Note: We do NOT apply Judge here to find RAW Potential first.
    mask_trigger = mask_regime & mask_macro & mask_vol & (actions == 1)
    
    entry_indices = np.where(mask_trigger)[0]
    print(f"   Potential Triggers found (Raw Grizzly): {len(entry_indices)}")
    
    if len(entry_indices) == 0:
        print("   ❌ No triggers found. Grizzly is sleeping. Try lowering Volatility Threshold or retraining.")
        return

    # 5. Grid Search Loop
    print(f"   🧪 Testing {len(SL_GRID) * len(TP_GRID)} Configurations...")
    
    results = []
    
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    for sl_pct_raw, tp_pct_raw in itertools.product(SL_GRID, TP_GRID):
        sl_pct = sl_pct_raw / 100.0
        tp_pct = tp_pct_raw / 100.0
        
        balance = 100.0
        trades_count = 0
        wins = 0
        last_exit = -1
        
        for idx in entry_indices:
            if idx <= last_exit: continue
            if idx >= len(df) - HORIZON: continue
            
            entry = close_arr[idx]
            sl_price = entry * (1 + sl_pct)
            tp_price = entry * (1 - tp_pct)
            
            future_high = high_arr[idx+1 : idx+1+HORIZON]
            future_low = low_arr[idx+1 : idx+1+HORIZON]
            future_close = close_arr[idx+1 : idx+1+HORIZON]
            
            hit_sl = future_high >= sl_price
            hit_tp = future_low <= tp_price
            
            sl_idx = np.argmax(hit_sl) if np.any(hit_sl) else 9999
            tp_idx = np.argmax(hit_tp) if np.any(hit_tp) else 9999
            
            reason = "TIME"
            pnl = 0.0
            steps = HORIZON
            
            if sl_idx < tp_idx and sl_idx < 9999:
                pnl = (entry - sl_price)/entry
                steps = sl_idx + 1
                reason = "SL"
            elif tp_idx < sl_idx and tp_idx < 9999:
                pnl = (entry - tp_price)/entry
                steps = tp_idx + 1
                reason = "TP"
            else:
                pnl = (entry - future_close[-1])/entry
                
            balance += (pnl * LEVERAGE * FIXED_MARGIN)
            trades_count += 1
            if pnl > 0: wins += 1
            last_exit = idx + steps
            
        net_return = ((balance - 100)/100)*100
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        
        results.append({
            'sl': sl_pct_raw,
            'tp': tp_pct_raw,
            'return': net_return,
            'trades': trades_count,
            'wr': win_rate
        })
        
        # print(f"   SL: {sl_pct_raw}% | TP: {tp_pct_raw}% | Ret: {net_return:6.2f}% | Trades: {trades_count}")
        
    # Sort and Report
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print("\n🏆 TOP 5 CONFIGURATIONS (GRIZZLY):")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"   {i+1}. SL: {r['sl']}% | TP: {r['tp']}% | Return: {r['return']:8.2f}% | Trades: {r['trades']} | WR: {r['wr']:.1f}%")

if __name__ == "__main__":
    main()
