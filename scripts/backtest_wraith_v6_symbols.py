#!/usr/bin/env python3
"""
Project Wraith V6: Symbol-Specific Backtest
Target: Test ETH and SOL with their dedicated DQN models.
"""
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.detect_distribution_tops import calculate_physics_features, detect_wraith_setups
from scripts.train_wraith_symbol import WraithNet  # Use symbol-specific architecture

# Config
DB_URL = "sqlite:///data/binance_candles.db"
TIMEFRAME = "5m"

# V6 Params
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

# Time Sentinel
FORBIDDEN_HOURS = [1, 4, 5, 10, 13, 18, 19, 23]
FORBIDDEN_DAYS = ['Tuesday']

def get_model_path(symbol):
    """Get model path for symbol-specific model."""
    symbol_clean = symbol.replace('/', '_').replace(':', '_').lower()
    return f"models/wraith_{symbol_clean}/wraith_net_best.pth"

def is_forbidden_time(timestamp):
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
                return trailing_sl, "TRAILING", is_breakeven
                
        if row['high'] >= sl_price:
            return sl_price, "STOP_LOSS", is_breakeven
            
        if row['low'] <= tp_price:
            return tp_price, "TAKE_PROFIT", is_breakeven
            
    return future_candles.iloc[-1]['close'], "TIME_LIMIT", is_breakeven

def run_backtest_for_symbol(symbol, device):
    print(f"\n{'='*60}")
    print(f"🦅 V6 BACKTEST WITH DEDICATED MODEL: {symbol}")
    print(f"{'='*60}")
    
    # Load Symbol-Specific Model
    model_path = get_model_path(symbol)
    print(f"📁 Model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = WraithNet(input_dim=6, output_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Data
    db_manager = DatabaseManager(DB_URL)
    df = db_manager.get_ohlcv_data(symbol, TIMEFRAME, limit=50000)
    
    if df.empty or len(df) < 1000:
        print(f"❌ Insufficient data for {symbol}")
        return None
    
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    
    print(f"📊 Data: {len(df)} candles")
    
    df = calculate_physics_features(df)
    candidates = detect_wraith_setups(df)
    print(f"🔬 Physics Candidates: {len(candidates)}")
    
    # State
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    circuit_breaker_until = None
    blocked_by_sentinel = 0
    
    for i in range(len(candidates)):
        cand_idx = candidates.index[i]
        row = candidates.iloc[i]
        
        # Circuit Breaker
        if circuit_breaker_until is not None:
            if cand_idx < circuit_breaker_until:
                continue
            else:
                circuit_breaker_until = None
        
        # Time Sentinel
        if is_forbidden_time(row['timestamp']):
            blocked_by_sentinel += 1
            continue
        
        # BOS Check
        if not check_bos(df, cand_idx):
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
            
            # House Money
            leverage = BASE_LEVERAGE
            if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            position_size = balance * leverage
            quantity = position_size / entry_price
            
            loc = df.index.get_loc(cand_idx)
            future = df.iloc[loc+1 : loc+HORIZON+1]
            if len(future) < HORIZON: continue
            
            exit_price, reason, hit_be = simulate_trade(entry_price, future, leverage)
            
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
                'symbol': symbol,
                'entry_time': row['timestamp'],
                'reason': reason,
                'pnl': net_pnl,
                'balance': balance
            })
            
            if balance <= 0: break
    
    if not trades:
        print(f"❌ No trades for {symbol}")
        return {'symbol': symbol, 'trades': 0, 'final_balance': INITIAL_BALANCE, 'return': 0, 'win_rate': 0, 'profit_factor': 0}
        
    df_trades = pd.DataFrame(trades)
    final = df_trades.iloc[-1]['balance']
    ret = ((final - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100
    mdd = ((df_trades['balance'].cummax() - df_trades['balance']) / df_trades['balance'].cummax()).max() * 100
    
    print(f"\n💰 Final: ${final:.2f} ({ret:+.2f}%)")
    print(f"🏆 Peak: ${peak_balance:.2f}")
    print(f"📉 Max DD: {mdd:.2f}%")
    print(f"🔫 Trades: {len(df_trades)}")
    print(f"✅ Win Rate: {win_rate:.2f}%")
    print(f"⚖️ Profit Factor: {profit_factor:.2f}")
    print(f"🚫 Sentinel Blocked: {blocked_by_sentinel}")
    
    # Save trades
    symbol_clean = symbol.replace('/', '_').lower()
    df_trades.to_csv(f"reports/wraith_v6_{symbol_clean}_backtest.csv", index=False)
    
    return {
        'symbol': symbol,
        'trades': len(df_trades),
        'final_balance': final,
        'return': ret,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_dd': mdd,
        'peak': peak_balance
    }

def main():
    print("🦅 WRAITH V6 SYMBOL-SPECIFIC BACKTEST 🦅")
    
    device = torch.device("cpu")
    
    symbols = ['ETH/USDT', 'SOL/USDT']
    results = []
    
    for symbol in symbols:
        result = run_backtest_for_symbol(symbol, device)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 V6 SYMBOL-SPECIFIC MODEL COMPARISON")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    print("\n🔍 VIABILITY:")
    for r in results:
        if r['profit_factor'] >= 1.0:
            print(f"  ✅ {r['symbol']}: PF {r['profit_factor']:.2f} - VIABLE")
        else:
            print(f"  ❌ {r['symbol']}: PF {r['profit_factor']:.2f} - NOT VIABLE")
    
    df_results.to_csv("reports/wraith_v6_symbol_specific.csv", index=False)
    print("\n📄 Report: reports/wraith_v6_symbol_specific.csv")

if __name__ == "__main__":
    main()
