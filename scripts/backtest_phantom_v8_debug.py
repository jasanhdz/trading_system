#!/usr/bin/env python3
"""
Project Phantom V8: ETH Service Client Backtest (DEBUG VERSION)
Target: Verify ML Service Logic by decoupling backtest execution.
"""
import pandas as pd
import numpy as np
import sys
import os
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Config
FEATURES_PATH = "data/phantom_features.csv"
ML_SERVICE_URL = "http://127.0.0.1:8002/ml-v2/backtest_predict"

# V8 Params (Optimized for ETH institutional moves)
INITIAL_BALANCE = 20.0
FEE_RATE = 0.0005
BASE_LEVERAGE = 5

# Exit Params (Deep Breath + Patience)
SL_PCT = 0.015       # 1.5% SL
TP_PCT = 0.06        # 6% TP
TRAILING_DEV = 0.015 # 1.5% trailing
BE_TRIGGER_ROE = 0.10  # 10% ROE before BE

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

def get_service_prediction(row):
    """
    Call the ML Service for a prediction using precalculated features.
    """
    try:
        # Construct payload with precalculated features
        payload = {
            "symbol": "ETHUSDT",
            "custom_candles": [], # Required by schema but ignored if precalculated_features is present
            "precalculated_features": {
                "cvd_slope": float(row['cvd_slope']) if not pd.isna(row['cvd_slope']) else 0.0,
                "cvd_z": float(row['cvd_z']) if not pd.isna(row['cvd_z']) else 0.0,
                "weakness_score": float(row['weakness_score']) if not pd.isna(row['weakness_score']) else 0.0,
                "close": float(row['close_eth']),
                "open": float(row['open_eth'])
            }
        }
        
        response = requests.post(ML_SERVICE_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("action", "PASS"), data.get("confidence", 0.0)
        else:
            print(f"Service Error {response.status_code}: {response.text}")
            return "PASS", 0.0
            
    except Exception as e:
        print(f"Service Exception: {e}")
        return "PASS", 0.0

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
            print(f"[PY DEBUG SL] Time: {row['timestamp']} | High: {row['high_eth']} | SL: {sl_price} | Entry: {entry_price}")
            return sl_price, "STOP_LOSS", is_breakeven
        
        if row['low_eth'] <= tp_price:
            return tp_price, "TAKE_PROFIT", is_breakeven
    
    return future_candles.iloc[-1]['close_eth'], "TIME_LIMIT", is_breakeven

def main():
    print("礪 PROJECT PHANTOM V8: SERVICE CLIENT BACKTEST (DEBUG) 礪")
    print("=" * 60)
    
    # Load features
    if not Path(FEATURES_PATH).exists():
        print("❌ Features not found.")
        return
    
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n Data: {len(df)} candles")
    
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
        
        # Call Service
        action, confidence = get_service_prediction(row)
        
        if action == "SHORT":
            phantom_triggers += 1
            
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
                'entry_price': entry_price,
                'exit_price': exit_price,
                'reason': reason,
                'pnl': net_pnl,
                'balance': balance,
                'hit_be': hit_be,
                'confidence': confidence
            })
            
            if balance <= 0: break
    
    print(f"\n SERVICE CLIENT RESULTS:")
    print(f"  Phantom Triggers: {phantom_triggers}")
    print(f"  Sentinel Blocked: {blocked_by_sentinel}")
    
    if not trades:
        print("  ❌ No trades executed")
        return
    
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('reports/python_trades.csv', index=False)
    print(f" Report: reports/python_trades.csv")
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
    
    print(f"\n Final: ${final:.2f} ({ret:+.2f}%)")
    print(f" Peak: ${peak_balance:.2f}")
    print(f" Max DD: {mdd:.2f}%")
    print(f" Trades: {len(df_trades)}")
    print(f"✅ Win Rate: {win_rate:.2f}%")
    print(f"⚖️ Profit Factor: {profit_factor:.2f}")
    print(f"\n Exit Analysis:")
    for reason, count in reasons.items():
        print(f"   {reason}: {count}")
    print(f"   BE Activations: {be_hits}")
    
    df_trades.to_csv("reports/phantom_v8_service_client_results.csv", index=False)
    print("\n Report: reports/phantom_v8_service_client_results.csv")

if __name__ == "__main__":
    main()
