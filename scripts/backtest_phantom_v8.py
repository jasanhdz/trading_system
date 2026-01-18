#!/usr/bin/env python3
"""
Project Phantom V8: ETH Specialist Backtest
Target: Validate Phantom model trained on 11,519 drop examples.
"""
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_phantom_dqn import PhantomNet

# Config
MODEL_PATH = "models/phantom_eth/phantom_net_best.pth"
FEATURES_PATH = "data/phantom_features.csv"

# V8 Params (Optimized for ETH institutional moves)
INITIAL_BALANCE = 20.0
FEE_RATE = 0.0005
BASE_LEVERAGE = 5

# Exit Params (Deep Breath + Patience)
SL_PCT = 0.015       # 1.5% SL
TP_PCT = 0.06        # 6% TP
TRAILING_DEV = 0.015 # 1.5% trailing
BE_TRIGGER_ROE = 0.10  # 10% ROE before BE

CONFIDENCE_THRESHOLD = 0.55  # Slightly relaxed for more trades
HORIZON = 288

# Equity Protection
HOUSE_MONEY_MULTIPLIER = 2.0
HOUSE_MONEY_REDUCTION = 0.5
CIRCUIT_BREAKER_DD = 0.15
CIRCUIT_BREAKER_CANDLES = 288

# Time Sentinel
FORBIDDEN_HOURS = [1, 4, 5, 10, 13, 18, 19, 23]
FORBIDDEN_DAYS = ['Tuesday']

def is_forbidden_time(timestamp):
    hour = timestamp.hour
    day = timestamp.strftime('%A')
    return day in FORBIDDEN_DAYS or hour in FORBIDDEN_HOURS

def check_phantom_trigger(df, idx):
    """
    Phantom Trigger: CVD-based entry.
    Requires negative CVD slope + weakness.
    """
    if idx < 50:
        return False
    
    row = df.iloc[idx]
    
    # 1. CVD Slope negative (distribution)
    if pd.isna(row['cvd_slope']) or row['cvd_slope'] > 0:
        return False
    
    # 2. CVD Z-Score below average (selling pressure)
    if pd.isna(row['cvd_z']) or row['cvd_z'] > 0.5:
        return False
    
    # 3. Bearish candle
    if row['close_eth'] >= row['open_eth']:
        return False
    
    # 4. ETH weaker than BTC
    if pd.isna(row['weakness_score']) or row['weakness_score'] < 0:
        return False
    
    return True

def simulate_trade(entry_price, future_candles, leverage):
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.003)
    peak_price = entry_price
    is_breakeven = False
    
    for i, row in future_candles.iterrows():
        if row['low_eth'] < peak_price:
            peak_price = row['low_eth']
        
        current_roe = (entry_price - row['low_eth']) / entry_price * leverage
        
        if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
            sl_price = be_price
            is_breakeven = True
        
        if is_breakeven:
            trailing_sl = peak_price * (1 + TRAILING_DEV)
            if row['high_eth'] >= trailing_sl:
                return trailing_sl, "TRAILING", is_breakeven
        
        if row['high_eth'] >= sl_price:
            return sl_price, "STOP_LOSS", is_breakeven
        
        if row['low_eth'] <= tp_price:
            return tp_price, "TAKE_PROFIT", is_breakeven
    
    return future_candles.iloc[-1]['close_eth'], "TIME_LIMIT", is_breakeven

def main():
    print("🦅 PROJECT PHANTOM V8: ETH BACKTEST 🦅")
    print("=" * 60)
    print("📋 Key Features:")
    print("   - CVD Proxy trigger")
    print("   - 12-feature PhantomNet")
    print("   - Deep Breath exit (10% BE ROE)")
    print("   - Trained on 11,519 drop examples")
    print("=" * 60)
    
    # Load features
    if not Path(FEATURES_PATH).exists():
        print("❌ Features not found. Run phantom_data_generator.py first.")
        return
    
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n📊 Data: {len(df)} candles")
    
    # Load model
    if not Path(MODEL_PATH).exists():
        print("❌ Model not found. Run train_phantom_dqn.py first.")
        return
    
    device = torch.device("cpu")
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("✅ Phantom model loaded")
    
    # State
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    circuit_breaker_until = None
    blocked_by_sentinel = 0
    phantom_triggers = 0
    
    for idx in range(200, len(df) - HORIZON):
        row = df.iloc[idx]
        
        # Circuit Breaker
        if circuit_breaker_until is not None:
            if idx < circuit_breaker_until:
                continue
            else:
                circuit_breaker_until = None
        
        # Time Sentinel
        if is_forbidden_time(row['timestamp']):
            blocked_by_sentinel += 1
            continue
        
        # Phantom Trigger
        if not check_phantom_trigger(df, idx):
            continue
        
        phantom_triggers += 1
        
        # Build state vector (12 features)
        state = np.array([
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] / 10000 if not pd.isna(row['cvd_slope']) else 0,  # Normalized
            row['weakness_score'] if not pd.isna(row['weakness_score']) else 0,
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            float(row['is_fakeout']) if not pd.isna(row['is_fakeout']) else 0,
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            row['velocity_sm'] / row['close_eth'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['close_eth'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            0  # Reserved
        ], dtype=np.float32)
        
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
        
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            entry_price = row['close_eth']
            
            # House Money
            leverage = BASE_LEVERAGE
            if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            position_size = balance * leverage
            quantity = position_size / entry_price
            
            future = df.iloc[idx+1 : idx+HORIZON+1]
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
                circuit_breaker_until = idx + CIRCUIT_BREAKER_CANDLES
            
            trades.append({
                'entry_time': row['timestamp'],
                'reason': reason,
                'pnl': net_pnl,
                'balance': balance,
                'hit_be': hit_be,
                'confidence': confidence
            })
            
            if balance <= 0: break
    
    print(f"\n📊 PHANTOM V8 RESULTS:")
    print(f"  Phantom Triggers: {phantom_triggers}")
    print(f"  Sentinel Blocked: {blocked_by_sentinel}")
    
    if not trades:
        print("  ❌ No trades executed")
        return
    
    df_trades = pd.DataFrame(trades)
    final = df_trades.iloc[-1]['balance']
    ret = ((final - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100
    mdd = ((df_trades['balance'].cummax() - df_trades['balance']) / df_trades['balance'].cummax()).max() * 100
    
    # Exit Analysis
    reasons = df_trades['reason'].value_counts()
    be_hits = df_trades['hit_be'].sum()
    
    print(f"\n💰 Final: ${final:.2f} ({ret:+.2f}%)")
    print(f"🏆 Peak: ${peak_balance:.2f}")
    print(f"📉 Max DD: {mdd:.2f}%")
    print(f"🔫 Trades: {len(df_trades)}")
    print(f"✅ Win Rate: {win_rate:.2f}%")
    print(f"⚖️ Profit Factor: {profit_factor:.2f}")
    print(f"\n📊 Exit Analysis:")
    for reason, count in reasons.items():
        print(f"   {reason}: {count}")
    print(f"   BE Activations: {be_hits}")
    
    # Evolution Comparison
    print("\n📈 ETH MODEL EVOLUTION:")
    print("  Spectre V6:  WR 65%, PF 0.70, $12.59")
    print("  Spectre V7:  WR 44%, PF 0.67, $15.25")
    print(f"  Phantom V8:  WR {win_rate:.1f}%, PF {profit_factor:.2f}, ${final:.2f}")
    
    # Verdict
    print("\n🔬 PHANTOM V8 VERDICT:")
    if profit_factor >= 1.5:
        print("  🏆 PHANTOM SUCCESS - ETH CONQUERED!")
    elif profit_factor >= 1.0:
        print("  ✅ PHANTOM VIABLE - Positive edge achieved!")
    else:
        if profit_factor > 0.70:
            print("  🟡 PHANTOM IMPROVING - Continue optimization")
        else:
            print("  ❌ PHANTOM NEEDS WORK - Different approach needed")
    
    df_trades.to_csv("reports/phantom_v8_backtest.csv", index=False)
    print("\n📄 Report: reports/phantom_v8_backtest.csv")

if __name__ == "__main__":
    main()
