#!/usr/bin/env python3
"""
Phantom V11 Sniper: The Steroid Validator
Tests the V11 model on the FILTERED Steroid Dataset.
Uses 20% Risk Per Trade (Aggressive Compounding).
"""
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_twin_v9.smart_money_markdown.train_specialist import PhantomNet, SL_PCT, TP_PCT, DRAWDOWN_PENALTY

# Config
DATASET_PATH = ROOT_DIR / "data/dataset_steroid.csv"
MODEL_PATH = ROOT_DIR / "models/phantom_v11_twin/phantom_v11_final.pth"
INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.20 # 20% of Balance at Risk (Aggressive!)

def main():
    print(f"🎯 PHANTOM V11: STEROID VALIDATION (20% RISK) 🎯")
    
    # 1. Load Data
    if not DATASET_PATH.exists():
        print("❌ Dataset not found.")
        return
    
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Extract Validation Set (Last 20% - SAME AS TRAINING)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()
    val_df.reset_index(drop=True, inplace=True)
    
    print(f"📊 Dataset Info: {len(val_df)} Steroid Candles (Validation)")
    
    # 3. Features (Should be in CSV, but recalculate to be safe)
    from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna
    # Check if cols exist
    if 'velocity' not in val_df.columns:
         print("🧬 Calculating DNA...")
         val_df = calculate_phantom_dna(val_df)
         val_df.fillna(0, inplace=True)
    
    # 4. Load Model
    device = torch.device("cpu") 
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Model V11 (Steroid) Loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 5. Run Simulation
    balance = INITIAL_BALANCE
    trades = []
    
    print("🚀 Running Backtest on PURE DATA...")
    
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        
        # Inference
        state = np.array([
            row.get('velocity', 0) / row['close'] * 10000,
            row.get('acceleration', 0) / row['close'] * 10000,
            row.get('cvd_slope', 0) / 1e6,
            row.get('bear_trap', 0),
            row.get('vol_z', 0),
            row.get('volume_ratio', 1),
            row.get('dist_ema_20', 0) * 100,
            row.get('dist_ema_200', 0) * 100,
            row.get('staleness', 0) / 50.0,
            row.get('weakness_score', 0),
            row.get('is_fakeout', 0),
            row.get('reserved', 0)
        ], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q = model(state_t)
            action = torch.argmax(q).item()
            conf = torch.softmax(q, dim=1)[0][1].item()
        
        # Threshold: 0.5 (Standard)
        if action == 1 and conf > 0.50:
            # --- TIME FILTERS (Dead Hours & Red Tuesdays) ---
            # 1. Block Tuesdays (Day 1)
            if row['timestamp'].dayofweek == 1:
                continue 
            
            # 2. Block Dead Hours (21:00 - 06:00 UTC)
            # Assuming timestamp is UTC.
            h = row['timestamp'].hour
            if h >= 21 or h < 6:
                continue

            entry_price = row['close']
            
            # --- AGGRESSIVE RISK MANAGEMENT ---
            # Risk 20% of Balance.
            risk_amount = balance * RISK_PER_TRADE
            
            # Position Size = Risk / SL%
            position_size_usd = risk_amount / SL_PCT
            
            # --- OUTCOME (From Ground Truth) ---
            # We assume the trade happens here.
            # Using pre-calculated future columns.
            if 'future_max_high' not in row:
                print("❌ Missing ground truth columns (future_max_high). Run refine_dataset.py correctly.")
                break
                
            max_high = row['future_max_high']
            min_low = row['future_min_low']
            close_exit = row['future_close_exit']
            
            sl_price = entry_price * (1 + SL_PCT)
            tp_price = entry_price * (1 - TP_PCT)
            
            exit_price = None
            reason = None
            
            if max_high >= sl_price:
                exit_price = sl_price
                reason = "SL"
            elif min_low <= tp_price:
                exit_price = tp_price
                reason = "TP"
            else:
                exit_price = close_exit
                reason = "TIME"
                
            # PnL Calc
            raw_pnl_pct = (entry_price - exit_price) / entry_price
            
            # Realized PnL = Position Size * PnL
            # Fee: 0.12% of Position
            fees = position_size_usd * 0.0012
            pnl_usd = (position_size_usd * raw_pnl_pct) - fees
            
            balance += pnl_usd
            
            trade_res = {
                'entry_time': row['timestamp'],
                'reason': reason,
                'pnl_usd': pnl_usd,
                'pnl_pct_account': (pnl_usd / (balance - pnl_usd)) * 100,
                'balance': balance,
                'conf': conf
            }
            trades.append(trade_res)
            
            if balance <= 10:
                print("💀 LIQUIDATION.")
                break

    # 6. Report
    print("-" * 40)
    print(f"🏁 STEROID RESULTS")
    print("-" * 40)
    print(f"Final Balance: ${balance:.2f} (Start: ${INITIAL_BALANCE})")
    print(f"Total Trades: {len(trades)}")
    
    if trades:
        rdf = pd.DataFrame(trades)
        wins = rdf[rdf['pnl_usd'] > 0]
        win_rate = len(wins) / len(trades) * 100
        
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Avg PnL (Account %): {rdf['pnl_pct_account'].mean():.2f}%")
        print(f"Max Drawdown Trade: {rdf['pnl_pct_account'].min():.2f}%")
        print(f"Max Win Trade: {rdf['pnl_pct_account'].max():.2f}%")
        
        out_path = ROOT_DIR / "reports/phantom_v11_steroid_validation.csv"
        rdf.to_csv(out_path, index=False)
        print(f"\n📄 Details saved to {out_path}")

if __name__ == "__main__":
    main()
