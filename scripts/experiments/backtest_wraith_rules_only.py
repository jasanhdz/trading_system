#!/usr/bin/env python3
"""
Experiment: Wraith V6 Ablation Study (Rules Only - No AI)
Target: Validate if the DQN adds value over simple physics rules.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.detect_distribution_tops import calculate_physics_features, detect_wraith_setups
# No WraithNet import needed

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# V6 Sentinel Params (Optimized)
INITIAL_BALANCE = 20.0
FEE_RATE = 0.0005
BASE_LEVERAGE = 5
SL_PCT = 0.015
TP_PCT = 0.06
TRAILING_DEV = 0.010
BE_TRIGGER_ROE = 0.05
# CONFIDENCE_THRESHOLD = 0.85 # Not used
HORIZON = 288
RVOL_THRESHOLD = 2.0

# Equity Protection
HOUSE_MONEY_MULTIPLIER = 2.0
HOUSE_MONEY_REDUCTION = 0.5
CIRCUIT_BREAKER_DD = 0.15
CIRCUIT_BREAKER_CANDLES = 288

# TIME SENTINEL CONFIG - KEPT ACTIVE FOR FAIR COMPARISON
FORBIDDEN_HOURS = [1, 4, 5, 10, 13, 18, 19, 23]
FORBIDDEN_DAYS = ['Tuesday']

def is_forbidden_time(timestamp):
    """Time Sentinel: Block trades during high-loss periods."""
    hour = timestamp.hour
    day = timestamp.strftime('%A')
    return day in FORBIDDEN_DAYS or hour in FORBIDDEN_HOURS

def check_bos(df, idx):
    if idx < 20: return False
    current = df.iloc[idx]
    previous = df.iloc[idx-1]
    vol_avg = df['volume'].iloc[idx-20:idx].mean()
    return current['close'] < previous['low'] and current['volume'] > vol_avg * RVOL_THRESHOLD

def simulate_trade(entry_price, future_candles, leverage):
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.002)
    peak_price = entry_price
    is_breakeven = False
    
    for i, row in future_candles.iterrows():
        if row['low'] < peak_price:
            peak_price = row['low']
            
        max_roe = (entry_price - row['low']) / entry_price * leverage
        
        if max_roe >= BE_TRIGGER_ROE and not is_breakeven:
            sl_price = be_price
            is_breakeven = True
            
        if is_breakeven:
            trailing_sl = peak_price * (1 + TRAILING_DEV)
            if row['high'] >= trailing_sl:
                return trailing_sl, "TRAILING", row['timestamp'], is_breakeven
                
        if row['high'] >= sl_price:
            return sl_price, "STOP_LOSS", row['timestamp'], is_breakeven
            
        if row['low'] <= tp_price:
            return tp_price, "TAKE_PROFIT", row['timestamp'], is_breakeven
            
    return future_candles.iloc[-1]['close'], "TIME_LIMIT", future_candles.iloc[-1]['timestamp'], is_breakeven

def run_experiment():
    print(f"🧪 Experiment: Wraith V6 - RULES ONLY (No AI)")
    
    db_manager = DatabaseManager(DB_URL)
    df = db_manager.get_ohlcv_data(SYMBOL, TIMEFRAME, limit=50000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    df = calculate_physics_features(df)
    candidates = detect_wraith_setups(df)
    
    # No Model Loading
    
    # State
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    circuit_breaker_until = None
    blocked_by_sentinel = 0
    
    for i in range(len(candidates)):
        cand_idx = candidates.index[i]
        row = candidates.iloc[i]
        
        # Circuit Breaker Check
        if circuit_breaker_until is not None:
            if cand_idx < circuit_breaker_until:
                continue
            else:
                circuit_breaker_until = None
        
        if not check_bos(df, cand_idx):
            continue
        
        # TIME SENTINEL CHECK
        if is_forbidden_time(row['timestamp']):
            blocked_by_sentinel += 1
            continue
            
        # NO AI CHECK - TAKE ALL TRADES
        # if action == 1 and confidence > CONFIDENCE_THRESHOLD:
        if True:
            entry_price = row['close']
            
            # House Money Rule
            leverage = BASE_LEVERAGE
            if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            position_size = balance * leverage
            quantity = position_size / entry_price
            
            loc = df.index.get_loc(cand_idx)
            future = df.iloc[loc+1 : loc+HORIZON+1]
            if len(future) < HORIZON: continue
            
            exit_price, reason, exit_time, hit_be = simulate_trade(entry_price, future, leverage)
            
            raw_pnl = (entry_price - exit_price) * quantity
            fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
            net_pnl = raw_pnl - fees
            
            balance += net_pnl
            if balance < 0: balance = 0
            
            if balance > peak_balance:
                peak_balance = balance
                
            # Circuit Breaker
            current_dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            if current_dd >= CIRCUIT_BREAKER_DD:
                circuit_breaker_until = cand_idx + CIRCUIT_BREAKER_CANDLES
            
            trades.append({
                'entry_time': row['timestamp'],
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'reason': reason,
                'leverage': leverage,
                'pnl': net_pnl,
                'balance': balance,
                'peak_balance': peak_balance
            })
            
            if balance <= 0: break
    
    print(f"\n🚫 Blocked by Time Sentinel: {blocked_by_sentinel}")
    
    if not trades:
        print("No trades.")
        return
        
    df_trades = pd.DataFrame(trades)
    final = df_trades.iloc[-1]['balance']
    ret = ((final - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    peak = df_trades['peak_balance'].max()
    mdd = ((df_trades['balance'].cummax() - df_trades['balance']) / df_trades['balance'].cummax()).max() * 100
    
    # Metrics
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades)
    
    print("\n" + "="*60)
    print("🧪 ABLATION RESULTS (RULES ONLY - NO AI)")
    print("="*60)
    print(f"💰 Final Balance:   ${final:.2f}")
    print(f"📈 Total Return:    {ret:.2f}%")
    print(f"📉 Max Drawdown:    {mdd:.2f}%")
    print(f"🔫 Trades:          {len(df_trades)}")
    print(f"✅ Win Rate:        {win_rate*100:.2f}%")
    print(f"⚖️ Profit Factor:   {profit_factor:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
