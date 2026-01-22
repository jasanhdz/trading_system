#!/usr/bin/env python3
"""
Project Phantom V8: Sequential Execution of Original 503 Trades
Ejecuta las 503 señales originales exactas, cerrando prematuramente si es necesario.
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

def extract_original_503_signals(df, model, device):
    """
    FASE 1: Replica el backtest EXACTO original para identificar las 503 señales.
    Ejecuta con overlapping para obtener las mismas señales que generaron $79M.
    """
    print("\n🔍 FASE 1: Identificando las 503 señales originales...")
    print("(Replicando backtest con overlapping)")
    
    balance_temp = INITIAL_BALANCE  # Balance temporal para circuit breaker
    peak_balance_temp = INITIAL_BALANCE
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
            # Guardar esta señal
            signals.append({
                'idx': idx,
                'timestamp': row['timestamp'],
                'price': row['close_eth'],
                'confidence': confidence
            })
            
            # Simular trade para actualizar circuit breaker (como el original)
            entry_price = row['close_eth']
            leverage_temp = BASE_LEVERAGE
            if balance_temp >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage_temp = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            position_size = balance_temp * leverage_temp
            quantity = position_size / entry_price
            
            future = df.iloc[idx+1 : idx+HORIZON+1]
            if len(future) >= HORIZON:
                # Simular cierre (simplificado para circuit breaker)
                exit_price, reason, _ = simulate_trade_simple(entry_price, future, leverage_temp)
                
                raw_pnl = (entry_price - exit_price) * quantity
                fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
                net_pnl = raw_pnl - fees
                
                balance_temp += net_pnl
                if balance_temp < 0: balance_temp = 0
                if balance_temp > peak_balance_temp:
                    peak_balance_temp = balance_temp
                
                # Circuit Breaker check
                drawdown_pct = (peak_balance_temp - balance_temp) / peak_balance_temp
                if drawdown_pct >= CIRCUIT_BREAKER_DD:
                    circuit_breaker_until = idx + CIRCUIT_BREAKER_CANDLES
    
    print(f"✅ Identificadas {len(signals)} señales (esperábamos 503)")
    return signals

def simulate_trade_simple(entry_price, future_candles, leverage):
    """Versión simplificada de simulate_trade para extracción de señales."""
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

def execute_503_sequentially_with_override(df, signals):
    """
    FASE 2: Ejecuta las 503 señales secuencialmente.
    Si llega nueva señal antes de cerrar la actual → cierra la actual y abre la nueva.
    """
    print("\n💰 FASE 2: Ejecutando 503 señales secuencialmente con override...")
    
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    
    current_trade = None  # Trade actualmente abierto
    overrides = 0
    
    for signal_num, signal in enumerate(signals, 1):
        signal_idx = signal['idx']
        signal_row = df.iloc[signal_idx]
        
        # Si hay trade abierto, verificar si debemos cerrarlo
        if current_trade is not None:
            # Verificar desde el índice de apertura hasta ahora
            entry_idx = current_trade['entry_idx']
            entry_price = current_trade['entry_price']
            leverage = current_trade['leverage']
            
            # Buscar si el trade se habría cerrado naturalmente antes de esta nueva señal
            closed_naturally = False
            exit_price = None
            exit_reason = None
            
            sl_price = entry_price * (1 + SL_PCT)
            tp_price = entry_price * (1 - TP_PCT)
            be_price = entry_price * (1 - 0.003)
            peak_price = entry_price
            is_breakeven = False
            
            for check_idx in range(entry_idx + 1, signal_idx):
                check_row = df.iloc[check_idx]
                
                if check_row['low_eth'] < peak_price:
                    peak_price = check_row['low_eth']
                
                current_roe = (entry_price - check_row['low_eth']) / entry_price * leverage
                
                if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
                    sl_price = be_price
                    is_breakeven = True
                
                if is_breakeven:
                    trailing_sl = peak_price * (1 + TRAILING_DEV)
                    if check_row['high_eth'] >= trailing_sl:
                        exit_price = trailing_sl
                        exit_reason = "TRAILING"
                        closed_naturally = True
                        break
                
                if check_row['high_eth'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "STOP_LOSS"
                    closed_naturally = True
                    break
                
                if check_row['low_eth'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"
                    closed_naturally = True
                    break
                
                # Check TIME_LIMIT
                if check_idx - entry_idx >= HORIZON:
                    exit_price = check_row['close_eth']
                    exit_reason = "TIME_LIMIT"
                    closed_naturally = True
                    break
            
            if not closed_naturally:
                # Cerrar al precio de mercado de la nueva señal (OVERRIDE)
                exit_price = signal_row['close_eth']
                exit_reason = "OVERRIDE_NEW_SIGNAL"
                overrides += 1
            
            # Calcular PnL del trade cerrado
            quantity = (balance * current_trade['leverage']) / current_trade['entry_price']
            raw_pnl = (current_trade['entry_price'] - exit_price) * quantity
            fees = (current_trade['entry_price'] * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
            net_pnl = raw_pnl - fees
            
            balance += net_pnl
            if balance < 0:
                balance = 0
            if balance > peak_balance:
                peak_balance = balance
            
            trades.append({
                'signal_num': current_trade['signal_num'],
                'entry_timestamp': current_trade['entry_timestamp'],
                'exit_timestamp': signal_row['timestamp'] if not closed_naturally else df.iloc[check_idx]['timestamp'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'net_pnl': net_pnl,
                'balance': balance
            })
            
            current_trade = None
        
        # Abrir nuevo trade con la señal actual
        entry_price = signal['price']
        
        # House Money
        leverage = BASE_LEVERAGE
        if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
            leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
        
        current_trade = {
            'signal_num': signal_num,
            'entry_idx': signal_idx,
            'entry_timestamp': signal['timestamp'],
            'entry_price': entry_price,
            'leverage': leverage
        }
        
        if signal_num % 50 == 0:
            print(f"  Procesado {signal_num}/{len(signals)} señales | Balance: ${balance:,.2f}")
    
    # Cerrar último trade si quedó abierto
    if current_trade is not None:
        # Simular hasta el final
        entry_idx = current_trade['entry_idx']
        entry_price = current_trade['entry_price']
        leverage = current_trade['leverage']
        
        future = df.iloc[entry_idx+1 : entry_idx+HORIZON+1]
        if len(future) > 0:
            exit_price, exit_reason, _ = simulate_trade_simple(entry_price, future, leverage)
        else:
            exit_price = df.iloc[-1]['close_eth']
            exit_reason = "END_OF_DATA"
        
        quantity = (balance * leverage) / entry_price
        raw_pnl = (entry_price - exit_price) * quantity
        fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
        net_pnl = raw_pnl - fees
        
        balance += net_pnl
        if balance < 0: balance = 0
        
        trades.append({
            'signal_num': current_trade['signal_num'],
            'entry_timestamp': current_trade['entry_timestamp'],
            'exit_timestamp': df.iloc[min(entry_idx + len(future), len(df)-1)]['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'net_pnl': net_pnl,
            'balance': balance
        })
    
    print(f"\n✅ Overrides: {overrides} trades cerrados prematuramente")
    return trades, balance, overrides

def main():
    print("🎯 PROJECT PHANTOM V8: ORIGINAL 503 SEQUENTIAL")
    print("=" * 60)
    print(" Estrategia:")
    print("   - Fase 1: Identificar las 503 señales exactas del backtest original")
    print("   - Fase 2: Ejecutarlas secuencialmente con 100% capital")
    print("   - Si nueva señal → Cerrar actual + Abrir nueva")
    print("=" * 60)
    
    # Load features
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n📈 Data: {len(df)} candles")
    
    # Load model
    device = torch.device("cpu")
    model = PhantomNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Phantom model loaded")
    
    # FASE 1: Extraer las 503 señales originales
    signals_503 = extract_original_503_signals(df, model, device)
    
    # FASE 2: Ejecutar secuencialmente con override
    trades, final_balance, overrides = execute_503_sequentially_with_override(df, signals_503)
    
    # Results
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    print(f"\n{'='*60}")
    print("📊 RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"\n💰 Balance Final: ${final_balance:,.2f}")
    print(f"📈 ROI: {(final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:,.2f}%")
    print(f"\n📊 Señales Originales: {len(signals_503)}")
    print(f"📊 Trades Ejecutados: {len(trades)}")
    print(f"🔄 Overrides (cerrados prematuramente): {overrides}")
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
    output_path = "reports/phantom_v8_original_503_sequential.csv"
    df_trades.to_csv(output_path, index=False)
    print(f"\n💾 Report: {output_path}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("📊 COMPARACIÓN")
    print(f"{'='*60}")
    print(f"Original (Overlapping):     {len(signals_503)} signals → $25.6M")
    print(f"Sequential con Override:    {len(trades)} trades → ${final_balance:,.2f}")

if __name__ == "__main__":
    main()
