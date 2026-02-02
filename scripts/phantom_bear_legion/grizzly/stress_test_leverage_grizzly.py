#!/usr/bin/env python3
"""
GRIZZLY LEVERAGE STRESS TEST ☢️
Testing the Awakening Config (SL 2.5% / TP 7.0%) with NUCLEAR Leverage.
Levels: 30x, 40x, 50x, 60x, 80x.
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# --- CONFIG ---
DATA_PATH = ROOT_DIR / "scripts/phantom_bear_legion/data/regime_labeled_history.csv"
MODEL_GRIZZLY_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/grizzly_v1.pth"

# OPTIMIZED CONFIG
SL_PCT = 0.025 # 2.5%
TP_PCT = 0.070 # 7.0%
HORIZON = 48
VOL_THRESHOLD = 0.005
FIXED_MARGIN = 20.0

LEVERAGE_LEVELS = [30.0, 40.0, 50.0, 60.0, 80.0]

# --- MODEL ---
class BearLegionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BearLegionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def main():
    print(f"☢️ GRIZZLY LEVERAGE TEST: NUCLEAR MODE")
    print(f"   Config: SL {SL_PCT*100}% | TP {TP_PCT*100}%")
    
    # 1. Load Data
    if not DATA_PATH.exists(): return
    df = pd.read_csv(DATA_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    needed = ['velocity', 'weakness_score', 'vol_z']
    if not all(c in df.columns for c in needed):
         df = calculate_phantom_dna(df)
    df.fillna(0, inplace=True)
    df['ema_macro'] = df['close'].ewm(span=200*24*12).mean()
    
    # 2. Features
    feature_cols = ['velocity', 'acceleration', 'cvd_slope', 'bear_trap', 'vol_z', 'volume_ratio', 'dist_ema_20', 'dist_ema_200', 'staleness', 'weakness_score', 'is_fakeout']
    close_arr = df['close'].values.astype(np.float32)
    feats = df[feature_cols].values.astype(np.float32)
    
    eps = 1e-9
    feats[:, 0] = feats[:, 0] / (close_arr + eps) * 10000 
    feats[:, 1] = feats[:, 1] / (close_arr + eps) * 10000 
    feats[:, 2] = feats[:, 2] / 1e6 
    feats[:, 6] = feats[:, 6] * 100 
    feats[:, 8] = feats[:, 8] / 50.0 
    feats[:, 9] = feats[:, 9] / 0.05 
    feats = np.hstack([feats, np.zeros((len(feats), 1), dtype=np.float32)])
    
    # 3. Model
    device = torch.device("cpu")
    model = BearLegionNet(12, 2).to(device)
    model.load_state_dict(torch.load(MODEL_GRIZZLY_PATH, map_location=device))
    model.eval()
    
    t_feats = torch.FloatTensor(feats).to(device)
    with torch.no_grad():
        actions = torch.argmax(model(t_feats), dim=1).numpy()
        
    # 4. Triggers
    mask_regime = (df['regime_type'] == 'BEAR_TREND').values
    mask_macro = (close_arr < df['ema_macro'].values)
    volCheck = ((df['high'] - df['low']) / df['open']).values
    mask_vol = (volCheck >= VOL_THRESHOLD)
    
    entry_indices = np.where(mask_regime & mask_macro & mask_vol & (actions == 1))[0]
    print(f"   Triggers found: {len(entry_indices)}")
    
    # 5. Simulation
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    for lev in LEVERAGE_LEVELS:
        balance = 100.0
        last_exit = -1
        liq_count = 0
        
        # Calculate Risk of Ruin per trade
        # Liquidation Price (Long/Short logic distinct, here Short)
        # Short Liq Price ~= Entry * (1 + (1/Leverage))
        # e.g. 40x -> 1/40 = 2.5% move up.
        # Since SL is 2.5%, at 40x, SL HIT = LIQUIDATION (100% loss).
        
        liq_threshold_pct = 1.0 / lev
        
        for idx in entry_indices:
            if idx <= last_exit: continue
            if idx >= len(df) - HORIZON: continue
            
            entry = close_arr[idx]
            sl_price = entry * (1 + SL_PCT)
            tp_price = entry * (1 - TP_PCT)
            
            future_high = high_arr[idx+1:idx+1+HORIZON]
            future_low = low_arr[idx+1:idx+1+HORIZON]
            future_close = close_arr[idx+1:idx+1+HORIZON]
            
            # Check Liquidation FIRST
            # Did price go up by 1/Leverage %?
            liq_price = entry * (1 + liq_threshold_pct)
            
            hit_liq = future_high >= liq_price
            hit_sl = future_high >= sl_price
            hit_tp = future_low <= tp_price
            
            liq_idx = np.argmax(hit_liq) if np.any(hit_liq) else 9999
            sl_idx = np.argmax(hit_sl) if np.any(hit_sl) else 9999
            tp_idx = np.argmax(hit_tp) if np.any(hit_tp) else 9999
            
            raw_pnl = 0.0
            steps = HORIZON
            
            # Outcome Logic
            if liq_idx < tp_idx and liq_idx < 9999:
                # LIQUIDATED
                raw_pnl = -1.0 / lev # Loss is 100% of margin (normalized logic)
                # Actually, pnl pct = (entry - exit) / entry. 
                # If exit = liq_price, pnl = -liq_threshold_pct.
                # lev * pnl = lev * (-1/lev) = -1.0 (100% Loss).
                realized_pnl_pct = -1.0 
                steps = liq_idx + 1
                liq_count += 1
            elif sl_idx < tp_idx and sl_idx < 9999:
                # SL HIT
                pnl = (entry - sl_price)/entry # Negative
                realized_pnl_pct = pnl * lev
                steps = sl_idx + 1
            elif tp_idx < sl_idx and tp_idx < 9999:
                # TP HIT
                pnl = (entry - tp_price)/entry # Positive
                realized_pnl_pct = pnl * lev
                steps = tp_idx + 1
            else:
                # TIME
                pnl = (entry - future_close[-1])/entry
                realized_pnl_pct = pnl * lev
                
            # Apply to Balance
            # FIXED MARGIN: We bet $20. 
            # If realized_pnl_pct is -1.0 (Liq), we lose $20.
            # If realized_pnl_pct is -1.5 (Gap risk), we lose $30 (eating into balance).
            profit = FIXED_MARGIN * realized_pnl_pct
            balance += profit
            last_exit = idx + steps
            
        ret = ((balance - 100)/100)*100
        print(f"   Leverage {lev}x | Return: {ret:8.2f}% | Final: ${balance:8.2f} | Liquidations: {liq_count}")
        
        if lev * SL_PCT >= 1.0:
            print(f"      ⚠️ WARNING: SL ({SL_PCT*100}%) >= Liquidation threshold. SL acts as Liq.")

if __name__ == "__main__":
    main()
