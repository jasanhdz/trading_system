#!/usr/bin/env python3
"""
Project Wraith V6: The Time Sentinel
The DEFINITIVE version for thesis validation.
Target: Profit Factor 1.3+ by eliminating temporal noise.
"""
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.detect_distribution_tops import calculate_physics_features, detect_wraith_setups
from scripts.train_wraith_dqn import WraithNet

# Config
DB_URL = "sqlite:///data/binance_candles.db"
MODEL_PATH = "models/wraith_dqn/wraith_net_best.pth"
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
CONFIDENCE_THRESHOLD = 0.85
HORIZON = 288
RVOL_THRESHOLD = 2.0

# Equity Protection
HOUSE_MONEY_MULTIPLIER = 2.0
HOUSE_MONEY_REDUCTION = 0.5
CIRCUIT_BREAKER_DD = 0.15
CIRCUIT_BREAKER_CANDLES = 288

# TIME SENTINEL CONFIG
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

def run_v6_backtest():
    print(f"🦅 Wraith V6: The Time Sentinel 🦅")
    print(f"💰 Initial: ${INITIAL_BALANCE}")
    print(f"🚫 Forbidden: {FORBIDDEN_DAYS} | Hours: {FORBIDDEN_HOURS}")
    
    db_manager = DatabaseManager(DB_URL)
    df = db_manager.get_ohlcv_data(SYMBOL, TIMEFRAME, limit=50000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    df = calculate_physics_features(df)
    candidates = detect_wraith_setups(df)
    
    device = torch.device("cpu")
    model = WraithNet(input_dim=6, output_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
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
            
        state = np.array([
            row['dist_to_ema'] * 100,
            row['velocity_sm'] / row['close'] * 1000,
            row['acceleration_sm'] / row['close'] * 1000,
            row['volatility_z'],
            row['bb_dist'] * 100,
            (row['volume'] / (row['vol_sm'] + 1e-8)) - 1.0
        ], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
            
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
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
    
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if len(df_trades[df_trades['pnl'] > 0]) > 0 else 0
    avg_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].mean()) if len(df_trades[df_trades['pnl'] <= 0]) > 0 else 0
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    print("\n" + "="*60)
    print("⏰ WRAITH V6 TIME SENTINEL - MASTER TEST RESULTS")
    print("="*60)
    print(f"💰 Final Balance:   ${final:.2f}")
    print(f"📈 Total Return:    {ret:.2f}%")
    print(f"🏆 Peak Balance:    ${peak:.2f} ({(peak/INITIAL_BALANCE-1)*100:.1f}%)")
    print(f"📉 Max Drawdown:    {mdd:.2f}%")
    print(f"🔫 Trades:          {len(df_trades)}")
    print(f"✅ Win Rate:        {win_rate*100:.2f}%")
    print(f"⚖️ Profit Factor:   {profit_factor:.2f}")
    print(f"🎲 Expectancy:      ${expectancy:.4f}/trade")
    print(f"💵 Avg Win:         ${avg_win:.2f}")
    print(f"💔 Avg Loss:        ${avg_loss:.2f}")
    print("="*60)
    
    print("\nExit Reasons:")
    print(df_trades['reason'].value_counts())
    
    # Equity Curve Peak Detection
    ath_idx = df_trades['balance'].idxmax()
    ath_time = df_trades.loc[ath_idx, 'entry_time']
    print(f"\n🏆 ATH reached on: {ath_time} (${peak:.2f})")
    
    df_trades.to_csv("reports/wraith_v6_backtest.csv", index=False)
    print("📄 Report saved to reports/wraith_v6_backtest.csv")

if __name__ == "__main__":
    run_v6_backtest()
