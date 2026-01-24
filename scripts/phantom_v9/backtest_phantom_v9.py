#!/usr/bin/env python3
"""
Phantom V9: The Reborn Backtest
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
LEVERAGE = 20.0 # Breakeven Test (20x)
CONFIDENCE_THRESHOLD = 0.55 # V9 Threshold
FORBIDDEN_HOURS = [0, 1, 22, 23] # ETH duerme menos, prohibimos solo horas muertas
FORBIDDEN_DAYS = ['Tuesday'] # Mantenemos la regla del martes

def is_forbidden_time(ts):
    return ts.strftime('%A') in FORBIDDEN_DAYS or ts.hour in FORBIDDEN_HOURS

def main():
    print(f"👻 PHANTOM V9: THE REBORN (20x + BREAKEVEN TRIGGER) 👻")
    
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, '5m', limit=50000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    df = calculate_phantom_dna(df)
    candidates = detect_eth_setups(df)
    
    device = torch.device("cpu")
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    balance = INITIAL_BALANCE
    trades = []
    
    # Capital Lock Protocol
    balance_locked_until = None
    
    print(f"Testing on {len(candidates)} candidates...")
    
    for i in range(len(candidates)):
        cand_idx = candidates.index[i]
        row = candidates.iloc[i]
        
        if is_forbidden_time(row['timestamp']): continue
        
        # Check if capital is locked
        if balance_locked_until is not None:
            if row['timestamp'] < balance_locked_until:
                continue
            else:
                balance_locked_until = None
        
        # Crear Estado (Debe ser IDÉNTICO al del training)
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
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
            
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            entry = row['close']
            loc = df.index.get_loc(cand_idx)
            future = df.iloc[loc+1 : loc+49] # 4 horas (48 velas)
            
            if future.empty: continue
            
            # --- BREAKEVEN TRIGGER (20x + BE at 15%) ---
            SL_PCT = 0.035 # 3.5% Hard Stop
            BREAKEVEN_ACTIVATION = 999.0 # Disabled
            
            exit_price = future.iloc[-1]['close'] # Default exit at end
            exit_reason = "TIME"
            peak_roe = -999.0
            be_active = False
            
            trade_active = True
            candles_held = 0
            
            for _, f_row in future.iterrows():
                candles_held += 1
                
                # 1. Check Hard SL (Initial)
                # If BE is active, SL is Entry. If not, SL is Entry * 1.025
                current_sl_price = entry if be_active else entry * (1 + SL_PCT)
                
                if f_row['high'] > current_sl_price:
                    exit_price = current_sl_price
                    exit_reason = "BREAKEVEN" if be_active else "SL"
                    trade_active = False
                    break
                
                # 2. Update Peak ROE (Short: Lower price is better)
                current_roe = (entry - f_row['low']) / entry * LEVERAGE
                if current_roe > peak_roe:
                    peak_roe = current_roe
                
                # 3. Activate Breakeven
                if not be_active and peak_roe >= BREAKEVEN_ACTIVATION:
                    be_active = True
            
            # Calculate Final ROE
            final_roe = (entry - exit_price) / entry * LEVERAGE
            pnl_pct = (entry - exit_price) / entry
            net_pnl = balance * pnl_pct * LEVERAGE
            
            # Lock capital for ACTUAL duration
            trade_duration = pd.Timedelta(minutes=candles_held * 5)
            balance_locked_until = row['timestamp'] + trade_duration
            
            trades.append({
                'time': row['timestamp'],
                'pnl': net_pnl,
                'balance': balance + net_pnl,
                'reason': exit_reason,
                'peak_roe': peak_roe,
                'final_roe': final_roe,
                'duration': candles_held
            })
            
            balance += net_pnl
            if balance <= 0: break

    print(f"\nFinal Balance: ${balance:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    if len(trades) > 0:
        df_res = pd.DataFrame(trades)
        print(df_res.tail())
        df_res.to_csv("reports/phantom_v9_results.csv", index=False)
    
    if len(trades) > 0:
        df_res = pd.DataFrame(trades)
        
        # --- STATS ANALYSIS ---
        avg_peak = df_res['peak_roe'].mean() * 100
        avg_final = df_res['final_roe'].mean() * 100
        avg_giveback = avg_peak - avg_final
        
        print("\n📊 TRADE STATISTICS (4H Horizon)")
        print(f"Average Peak ROE:  {avg_peak:.2f}%")
        print(f"Average Final ROE: {avg_final:.2f}%")
        print(f"Average Giveback:  {avg_giveback:.2f}%")
        print("-" * 30)
        print(f"Max Peak ROE:      {df_res['peak_roe'].max() * 100:.2f}%")
        print(f"Median Peak ROE:   {df_res['peak_roe'].median() * 100:.2f}%")
        
        df_res.to_csv("reports/phantom_v9_stats.csv", index=False)

if __name__ == "__main__":
    main()
