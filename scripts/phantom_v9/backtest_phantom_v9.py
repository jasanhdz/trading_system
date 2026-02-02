#!/usr/bin/env python3
"""
Phantom V9: The Reborn Backtest (Crash Forensics Edition)
Wraith Architecture + Phantom DNA for ETH.
"""
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna, detect_eth_setups
from scripts.phantom_v9.train_phantom_dqn import PhantomNet

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
MODEL_PATH = "models/phantom_v9/phantom_v9_best.pth"
INITIAL_BALANCE = 20.0
LEVERAGE = 20.0 
CONFIDENCE_THRESHOLD = 0.55
FORBIDDEN_HOURS = [0, 1, 22, 23] 
FORBIDDEN_DAYS = ['Tuesday'] 

# 🧟 ZOMBIE FILTER (Bollinger)
# Block 0.18 - 0.37
BB_LOWER = 0.18
BB_UPPER = 0.37

def is_forbidden_time(ts):
    return ts.strftime('%A') in FORBIDDEN_DAYS or ts.hour in FORBIDDEN_HOURS

def calculate_bb_pct(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    diff = upper - lower
    diff = diff.replace(0, np.nan)
    pct = (series - lower) / diff
    return pct

def main():
    print(f"👻 PHANTOM V9: CRASH FORENSICS (100% RISK + DATA HARVEST) 🕵️‍♂️")
    
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, '5m', limit=400000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    df = calculate_phantom_dna(df)
    candidates = detect_eth_setups(df)
    
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-12-31")
    candidates = candidates[(candidates['timestamp'] >= start_date) & (candidates['timestamp'] <= end_date)]
    
    # --- FORENSICS PREP ---
    print("🧬 CALCULATING ADVANCED INDICATORS FOR FORENSICS...")
    
    # 1. Bollinger
    df['bb_pct'] = calculate_bb_pct(df['close'], 20, 2)
    # 2. RSI (Vanilla)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # 3. ATR %
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    # 4. EMA Distance
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close'] * 100
    
    # Merge Features to Candidates
    forensics_cols = ['bb_pct', 'rsi_14', 'atr_pct', 'dist_ema_50']
    df_unique = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')[forensics_cols]
    candidates = candidates.merge(df_unique, left_on='timestamp', right_index=True)
    
    # --- APPLY ZOMBIE FILTER ---
    candidates['zombie_approved'] = (candidates['bb_pct'] < BB_LOWER) | (candidates['bb_pct'] > BB_UPPER)
    
    # --- APPLY BULL TRAP FILTER (Crash Forensics) ---
    # Killers were buying Highs (BB > 0.60) above EMA (>0).
    # Filter: Reject if BB > 0.55 AND DistEMA > 0.05
    # (Buying a breakout that fails)
    candidates['bull_trap_detected'] = (candidates['bb_pct'] > 0.55) & (candidates['dist_ema_50'] > 0.05)
    
    # Final Approval: Zombie Approved AND NOT Bull Trap
    candidates['final_approved'] = (candidates['zombie_approved'] == True) & (candidates['bull_trap_detected'] == False)
    
    initial = len(candidates)
    candidates = candidates[candidates['final_approved'] == True]
    filtered = len(candidates)
    print(f"   Filters: {initial} -> {filtered} Candidates (Zombie + Bull Trap)")
    
    device = torch.device("cpu")
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        print("❌ Model Not Found")
        return
        
    model.eval()
    
    balance = INITIAL_BALANCE
    max_balance = INITIAL_BALANCE
    trades = []
    balance_locked_until = None
    
    for _, row in candidates.iterrows():
        
        if is_forbidden_time(row['timestamp']): continue
        
        if balance_locked_until is not None:
            if row['timestamp'] < balance_locked_until:
                continue
            else:
                balance_locked_until = None
        
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
            0.0
        ], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
            
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            entry = row['close']
            
            start_idx = df['timestamp'].searchsorted(row['timestamp'])
            if start_idx + 49 >= len(df): continue
            future = df.iloc[start_idx+1 : start_idx+49]
            
            SL_PCT = 0.035
            BREAKEVEN_ACTIVATION = 999.0
            
            exit_price = future.iloc[-1]['close']
            exit_reason = "TIME"
            peak_roe = -999.0
            be_active = False
            trade_active = True
            candles_held = 0
            
            for _, f_row in future.iterrows():
                candles_held += 1
                current_sl_price = entry if be_active else entry * (1 + SL_PCT)
                
                if f_row['high'] > current_sl_price:
                    exit_price = current_sl_price
                    exit_reason = "BREAKEVEN" if be_active else "SL"
                    trade_active = False
                    break
                
                current_roe = (entry - f_row['low']) / entry * LEVERAGE
                if current_roe > peak_roe: peak_roe = current_roe
                if not be_active and peak_roe >= BREAKEVEN_ACTIVATION: be_active = True
            
            # --- AGGRESSIVE RISK (100% COMPOUNDING) ---
            margin = balance 
            
            pnl_pct_price = (entry - exit_price) / entry
            net_pnl = margin * pnl_pct_price * LEVERAGE
            trade_roe = pnl_pct_price * LEVERAGE
            
            trades.append({
                'time': row['timestamp'],
                'pnl': net_pnl,
                'balance_before': balance,
                'balance_after': balance + net_pnl,
                'reason': exit_reason,
                'roe': trade_roe,
                # Forensics
                'rsi_14': row['rsi_14'],
                'bb_pct': row['bb_pct'],
                'atr_pct': row['atr_pct'],
                'dist_ema_50': row['dist_ema_50']
            })
            
            balance += net_pnl
            if balance > max_balance: max_balance = balance
            if balance <= 0: break
            
            trade_duration = pd.Timedelta(minutes=candles_held * 5)
            balance_locked_until = row['timestamp'] + trade_duration

    print(f"\nFinal Balance: ${balance:,.2f}")
    print(f"Max Balance:   ${max_balance:,.2f}")
    print(f"Total Trades:  {len(trades)}")
    
    if len(trades) > 0:
        df_res = pd.DataFrame(trades)
        print(df_res.tail())
    if len(trades) > 0:
        df_res = pd.DataFrame(trades)
        print(df_res.tail())
        df_res.to_csv("scripts/phantom_v9/v9_bulltrap_results.csv", index=False)
        print("📄 Forensics saved to scripts/phantom_v9/v9_bulltrap_results.csv")

if __name__ == "__main__":
    main()
