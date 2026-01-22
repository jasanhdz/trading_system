#!/usr/bin/env python3
"""
Project Phantom V8: Exit-on-New-Signal Strategy
Cierra posición actual cuando detecta nueva señal.
Esto simula un bot realista que maximiza oportunidades sin overlapping.
"""
import pandas as pd
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_phantom_dqn import PhantomNet

# Config
MODEL_PATH = "models/phantom_eth/phantom_net_best.pth"
FEATURES_PATH = "data/phantom_features.csv"

# V8 Params
INITIAL_BALANCE = 20.0
FEE_RATE = 0.0005
BASE_LEVERAGE = 5

# Exit Params
SL_PCT = 0.015
TP_PCT = 0.06
TRAILING_DEV = 0.015
BE_TRIGGER_ROE = 0.10

CONFIDENCE_THRESHOLD = 0.55
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
    if idx < 50:
        return False
    
    row = df.iloc[idx]
    
    if pd.isna(row['cvd_slope']) or row['cvd_slope'] > 0:
        return False
    
    if pd.isna(row['cvd_z']) or row['cvd_z'] > 0.5:
        return False
    
    if row['close_eth'] >= row['open_eth']:
        return False
    
    if pd.isna(row['weakness_score']) or row['weakness_score'] < 0:
        return False
    
    return True

def simulate_trade(entry_price, entry_idx, df, leverage, open_trades):
    """
    Simula el trade hasta que se cierre naturalmente O hasta que se detecte nueva señal.
    """
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.003)
    peak_price = entry_price
    is_breakeven = False
    
    # Tracking
    open_trades['active'] = True
    open_trades['entry_idx'] = entry_idx
    open_trades['entry_price'] = entry_price
    open_trades['sl_price'] = sl_price
    open_trades['tp_price'] = tp_price
    open_trades['be_price'] = be_price
    open_trades['peak_price'] = peak_price
    open_trades['is_breakeven'] = is_breakeven
    open_trades['leverage'] = leverage

def check_exit(current_row, current_idx, open_trades):
    """
    Chequea si el trade actual debe cerrarse en esta vela.
    Retorna: (should_exit, exit_price, exit_reason)
    """
    if not open_trades['active']:
        return False, None, None
    
    entry_price = open_trades['entry_price']
    sl_price = open_trades['sl_price']
    tp_price = open_trades['tp_price']
    peak_price = open_trades['peak_price']
    is_breakeven = open_trades['is_breakeven']
    leverage = open_trades['leverage']
    
    # Update peak
    if current_row['low_eth'] < peak_price:
        peak_price = current_row['low_eth']
        open_trades['peak_price'] = peak_price
    
    # Check BE activation
    current_roe = (entry_price - current_row['low_eth']) / entry_price * leverage
    if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
        sl_price = open_trades['be_price']
        open_trades['sl_price'] = sl_price
        open_trades['is_breakeven'] = True
        is_breakeven = True
    
    # Check Trailing (solo si BE activado)
    if is_breakeven:
        trailing_sl = peak_price * (1 + TRAILING_DEV)
        if current_row['high_eth'] >= trailing_sl:
            return True, trailing_sl, "TRAILING"
    
    # Check SL
    if current_row['high_eth'] >= sl_price:
        return True, sl_price, "STOP_LOSS"
    
    # Check TP
    if current_row['low_eth'] <= tp_price:
        return True, tp_price, "TAKE_PROFIT"
    
    # Check TIME_LIMIT (288 candles = HORIZON)
    duration = current_idx - open_trades['entry_idx']
    if duration >= HORIZON:
        return True, current_row['close_eth'], "TIME_LIMIT"
    
    return False, None, None

def close_trade(exit_price, exit_reason, open_trades):
    """
    Cierra el trade actual.
    """
    open_trades['active'] = False
    open_trades['exit_price'] = exit_price
    open_trades['exit_reason'] = exit_reason
    return {
        'entry_price': open_trades['entry_price'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'is_breakeven': open_trades['is_breakeven']
    }

def main():
    print("🚀 PROJECT PHANTOM V8: EXIT-ON-NEW-SIGNAL")
    print("=" * 60)
    print(" Strategy:")
    print("   - 1 posición a la vez (realista)")
    print("   - Nueva señal → Cierra actual + Abre nueva")
    print("   - 100% capital en cada trade")
    print("=" * 60)
    
    # Load features
    if not Path(FEATURES_PATH).exists():
        print("❌ Features not found.")
        return
    
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n📈 Data: {len(df)} candles")
    
    # Load model
    if not Path(MODEL_PATH).exists():
        print("❌ Model not found.")
        return
    
    device = torch.device("cpu")
    model = PhantomNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Phantom model loaded\n")
    
    # State
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    circuit_breaker_until = None
    blocked_by_sentinel = 0
    phantom_triggers = 0
    
    # Open trade tracking
    open_trades = {'active': False}
    signal_overrides = 0
    
    for idx in range(200, len(df) - HORIZON):
        row = df.iloc[idx]
        
        # 1. Check if current position needs to be closed
        if open_trades['active']:
            should_exit, exit_price, exit_reason = check_exit(row, idx, open_trades)
            if should_exit:
                # Close trade
                trade_info = close_trade(exit_price, exit_reason, open_trades)
                
                # Calculate PnL
                entry_price = trade_info['entry_price']
                quantity = (balance * open_trades['leverage']) / entry_price
                raw_pnl = (entry_price - exit_price) * quantity
                fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
                net_pnl = raw_pnl - fees
                
                balance += net_pnl
                if balance < 0: balance = 0
                if balance > peak_balance:
                    peak_balance = balance
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'net_pnl': net_pnl,
                    'balance': balance
                })
                
                # Circuit Breaker check
                drawdown_pct = (peak_balance - balance) / peak_balance
                if drawdown_pct >= CIRCUIT_BREAKER_DD:
                    circuit_breaker_until = idx + CIRCUIT_BREAKER_CANDLES
        
        # 2. Circuit Breaker
        if circuit_breaker_until is not None:
            if idx < circuit_breaker_until:
                continue
            else:
                circuit_breaker_until = None
        
        # 3. Time Sentinel
        if is_forbidden_time(row['timestamp']):
            blocked_by_sentinel += 1
            continue
        
        # 4. Phantom Trigger
        if not check_phantom_trigger(df, idx):
            continue
        
        phantom_triggers += 1
        
        # 5. Build state vector
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
        
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1)[0][1].item()
        
        # 6. Exit-on-New-Signal Logic
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            # Si hay trade abierto, cerrarlo primero
            if open_trades['active']:
                exit_price = row['close_eth']
                exit_reason = "NEW_SIGNAL"
                trade_info = close_trade(exit_price, exit_reason, open_trades)
                
                # Calculate PnL
                entry_price = trade_info['entry_price']
                quantity = (balance * open_trades['leverage']) / entry_price
                raw_pnl = (entry_price - exit_price) * quantity
                fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
                net_pnl = raw_pnl - fees
                
                balance += net_pnl
                if balance < 0: balance = 0
                if balance > peak_balance:
                    peak_balance = balance
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'net_pnl': net_pnl,
                    'balance': balance
                })
                
                signal_overrides += 1
            
            # Abrir nuevo trade
            entry_price = row['close_eth']
            
            # House Money
            leverage = BASE_LEVERAGE
            if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            simulate_trade(entry_price, idx, df, leverage, open_trades)
    
    # Close any remaining open trade
    if open_trades['active']:
        last_row = df.iloc[-1]
        exit_price = last_row['close_eth']
        exit_reason = "END_OF_DATA"
        trade_info = close_trade(exit_price, exit_reason, open_trades)
        
        entry_price = trade_info['entry_price']
        quantity = (balance * open_trades['leverage']) / entry_price
        raw_pnl = (entry_price - exit_price) * quantity
        fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
        net_pnl = raw_pnl - fees
        
        balance += net_pnl
        trades.append({
            'timestamp': last_row['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'net_pnl': net_pnl,
            'balance': balance
        })
    
    # Results
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    print(f"\n{'='*60}")
    print("📊 RESULTADOS: EXIT-ON-NEW-SIGNAL")
    print(f"{'='*60}")
    print(f"\n💰 Balance Final: ${balance:,.2f}")
    print(f"📈 ROI: {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:,.2f}%")
    print(f"🎯 Phantom Triggers: {phantom_triggers}")
    print(f"⛔ Sentinel Blocked: {blocked_by_sentinel}")
    print(f"🔄 Signal Overrides: {signal_overrides}")
    print(f"\n📊 Trades: {len(trades)}")
    print(f"✅ Win Rate: {len(wins) / len(trades) * 100:.2f}%")
    
    if len(losses) > 0:
        profit_factor = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
        print(f"⚖️ Profit Factor: {profit_factor:.2f}")
    
    # Exit breakdown
    print(f"\n📊 Exit Breakdown:")
    for reason in df_trades['exit_reason'].unique():
        count = len(df_trades[df_trades['exit_reason'] == reason])
        print(f"   {reason}: {count}")
    
    # Save report
    output_path = "reports/phantom_v8_exit_on_signal.csv"
    df_trades.to_csv(output_path, index=False)
    print(f"\n💾 Report: {output_path}")

if __name__ == "__main__":
    main()
