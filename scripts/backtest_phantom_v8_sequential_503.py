#!/usr/bin/env python3
"""
Project Phantom V8: Sequential Execution of 503 Original Signals
Ejecuta las 503 señales originales de manera secuencial con interés compuesto.
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

def simulate_trade(entry_price, future_candles, leverage):
    """Simula un trade completo hasta su cierre natural."""
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

def extract_original_signals(df, model, device):
    """
    FASE 1: Extrae las 503 señales originales (con overlapping).
    Retorna lista de índices donde ocurrieron las señales.
    """
    print("\n🔍 FASE 1: Extrayendo las 503 señales originales...")
    
    signals = []
    circuit_breaker_until = None
    
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
            continue
        
        # Phantom Trigger
        if not check_phantom_trigger(df, idx):
            continue
        
        # Build state vector
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
        
        if action == 1 and confidence > CONFIDENCE_THRESHOLD:
            signals.append({
                'idx': idx,
                'timestamp': row['timestamp'],
                'confidence': confidence
            })
    
    print(f"✅ Encontradas {len(signals)} señales originales")
    return signals

def execute_signals_sequentially(df, signals, model, device):
    """
    FASE 2: Ejecuta las señales de manera secuencial con interés compuesto.
    """
    print("\n💰 FASE 2: Ejecutando señales secuencialmente con interés compuesto...")
    
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    next_available_idx = 200  # Índice donde el bot puede abrir siguiente trade
    
    for signal_num, signal in enumerate(signals, 1):
        idx = signal['idx']
        
        # Solo ejecutar si estamos disponibles
        if idx < next_available_idx:
            continue
        
        row = df.iloc[idx]
        entry_price = row['close_eth']
        
        # House Money
        leverage = BASE_LEVERAGE
        if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
            leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
        
        position_size = balance * leverage
        quantity = position_size / entry_price
        
        future = df.iloc[idx+1 : idx+HORIZON+1]
        if len(future) < HORIZON:
            continue
        
        # Simulate Trade
        exit_price, reason, hit_be = simulate_trade(entry_price, future, leverage)
        
        raw_pnl = (entry_price - exit_price) * quantity
        fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
        net_pnl = raw_pnl - fees
        
        balance += net_pnl
        if balance < 0:
            balance = 0
        if balance > peak_balance:
            peak_balance = balance
        
        trades.append({
            'signal_num': signal_num,
            'timestamp': row['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': reason,
            'net_pnl': net_pnl,
            'balance': balance,
            'confidence': signal['confidence']
        })
        
        # Actualizar next_available_idx basado en duración del trade
        # Encontrar el índice de cierre
        for i, future_row in future.iterrows():
            exit_conditions_met = False
            
            # Misma lógica de simulate_trade para encontrar cuándo se cerró
            current_sl = entry_price * (1 + SL_PCT)
            current_tp = tp_price = entry_price * (1 - TP_PCT)
            
            if future_row['high_eth'] >= current_sl or \
               future_row['low_eth'] <= current_tp:
                next_available_idx = i + 1
                exit_conditions_met = True
                break
        
        if not exit_conditions_met:
            # TIME_LIMIT
            next_available_idx = idx + HORIZON + 1
        
        if signal_num % 50 == 0:
            print(f"  Ejecutado {signal_num}/{len(signals)} trades | Balance: ${balance:,.2f}")
    
    return trades, balance

def main():
    print("🎯 PROJECT PHANTOM V8: SEQUENTIAL 503")
    print("=" * 60)
    print(" Estrategia:")
    print("   - Identificar las 503 señales originales")
    print("   - Ejecutarlas secuencialmente (1 a la vez)")
    print("   - Usar 100% capital acumulado en cada trade")
    print("   - Interés compuesto realista")
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
    print("✅ Phantom model loaded")
    
    # FASE 1: Extraer señales originales
    signals = extract_original_signals(df, model, device)
    
    # FASE 2: Ejecutar secuencialmente
    trades, final_balance = execute_signals_sequentially(df, signals, model, device)
    
    # Results
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    print(f"\n{'='*60}")
    print("📊 RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"\n💰 Balance Final: ${final_balance:,.2f}")
    print(f"📈 ROI: {(final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:,.2f}%")
    print(f"\n📊 Trades Ejecutados: {len(trades)} / {len(signals)}")
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
    output_path = "reports/phantom_v8_sequential_503.csv"
    df_trades.to_csv(output_path, index=False)
    print(f"\n💾 Report: {output_path}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("📊 COMPARACIÓN CON ORIGINAL")
    print(f"{'='*60}")
    print(f"Original (Overlapping):  503 trades → $25.6M")
    print(f"Sequential 503:          {len(trades)} trades → ${final_balance:,.2f}")

if __name__ == "__main__":
    main()
