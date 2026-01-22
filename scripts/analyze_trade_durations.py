#!/usr/bin/env python3
"""
Analizar la duración de los 503 trades del backtest original de Phantom V8.
Este script modifica temporalmente simulate_trade para capturar la duración de cada trade.
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

def simulate_trade(entry_price, future_candles, leverage, entry_idx):
    """
    Versión modificada que captura la duración exacta del trade.
    """
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.003)
    peak_price = entry_price
    is_breakeven = False
    
    trade_duration = 0  # En candles (5 minutos cada una)
    exit_candle_idx = None
    
    for relative_i, (abs_idx, row) in enumerate(future_candles.iterrows()):
        if row['low_eth'] < peak_price:
            peak_price = row['low_eth']
        
        current_roe = (entry_price - row['low_eth']) / entry_price * leverage
        
        if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
            sl_price = be_price
            is_breakeven = True
        
        if is_breakeven:
            trailing_sl = peak_price * (1 + TRAILING_DEV)
            if row['high_eth'] >= trailing_sl:
                trade_duration = relative_i + 1
                exit_candle_idx = abs_idx
                return trailing_sl, "TRAILING", is_breakeven, trade_duration, exit_candle_idx
        
        if row['high_eth'] >= sl_price:
            trade_duration = relative_i + 1
            exit_candle_idx = abs_idx
            return sl_price, "STOP_LOSS", is_breakeven, trade_duration, exit_candle_idx
        
        if row['low_eth'] <= tp_price:
            trade_duration = relative_i + 1
            exit_candle_idx = abs_idx
            return tp_price, "TAKE_PROFIT", is_breakeven, trade_duration, exit_candle_idx
    
    trade_duration = len(future_candles)
    exit_candle_idx = future_candles.index[-1]
    return future_candles.iloc[-1]['close_eth'], "TIME_LIMIT", is_breakeven, trade_duration, exit_candle_idx

def main():
    print("📊 ANÁLISIS DE DURACIÓN: 503 TRADES ORIGINALES")
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
    
    # Force CPU to avoid ROCm issues
    device = torch.device("cpu")
    model = PhantomNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Handle both possible checkpoint structures
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Phantom model loaded (CPU mode)\n")
    
    # State
    balance = INITIAL_BALANCE
    trades = []
    peak_balance = INITIAL_BALANCE
    circuit_breaker_until = None
    blocked_by_sentinel = 0
    phantom_triggers = 0
    
    # ANÁLISIS DE DURACIONES
    durations = []
    
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
            entry_time = row['timestamp']
            
            # House Money
            leverage = BASE_LEVERAGE
            if balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
                leverage = BASE_LEVERAGE * HOUSE_MONEY_REDUCTION
            
            position_size = balance * leverage
            quantity = position_size / entry_price
            
            future = df.iloc[idx+1 : idx+HORIZON+1]
            if len(future) < HORIZON: continue
            
            # Simulate Trade (CON CAPTURA DE DURACIÓN)
            exit_price, reason, hit_be, duration, exit_idx = simulate_trade(entry_price, future, leverage, idx)
            
            exit_time = df.loc[exit_idx, 'timestamp']
            duration_hours = duration * 5 / 60  # 5 min candles
            duration_days = duration_hours / 24
            
            raw_pnl = (entry_price - exit_price) * quantity
            fees = (entry_price * quantity * FEE_RATE) + (exit_price * quantity * FEE_RATE)
            net_pnl = raw_pnl - fees
            
            balance += net_pnl
            if balance < 0: balance = 0
            if balance > peak_balance:
                peak_balance = balance
            
            # Guardar duración
            durations.append({
                'trade_num': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration_candles': duration,
                'duration_hours': duration_hours,
                'duration_days': duration_days,
                'exit_reason': reason,
                'pnl': net_pnl,
                'balance': balance
            })
            
            trades.append({
                'timestamp': row['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': reason,
                'net_pnl': net_pnl,
                'confidence': confidence,
                'balance': balance
            })
            
            # Circuit Breaker
            drawdown_pct = (peak_balance - balance) / peak_balance
            if drawdown_pct >= CIRCUIT_BREAKER_DD:
                circuit_breaker_until = idx + CIRCUIT_BREAKER_CANDLES
    
    # Crear DataFrame de duraciones
    df_durations = pd.DataFrame(durations)
    
    print(f"\n📊 RESULTADOS:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Final Balance: ${balance:.2f}")
    print(f"  ROI: {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:.2f}%\n")
    
    print("📈 ESTADÍSTICAS DE DURACIÓN:")
    print(f"  Media: {df_durations['duration_hours'].mean():.1f} horas ({df_durations['duration_days'].mean():.2f} días)")
    print(f"  Mediana: {df_durations['duration_hours'].median():.1f} horas ({df_durations['duration_days'].median():.2f} días)")
    print(f"  Mínimo: {df_durations['duration_hours'].min():.1f} horas ({df_durations['duration_days'].min():.2f} días)")
    print(f"  Máximo: {df_durations['duration_hours'].max():.1f} horas ({df_durations['duration_days'].max():.2f} días)")
    print(f"  Desv. Est: {df_durations['duration_hours'].std():.1f} horas\n")
    
    # Distribución por exit reason
    print("📊 DISTRIBUCIÓN POR EXIT REASON:")
    for reason in df_durations['exit_reason'].unique():
        subset = df_durations[df_durations['exit_reason'] == reason]
        print(f"  {reason}:")
        print(f"    Count: {len(subset)}")
        print(f"    Avg Duration: {subset['duration_hours'].mean():.1f} horas ({subset['duration_days'].mean():.2f} días)")
    
    # Guardar reporte detallado
    output_path = "reports/trade_durations_503.csv"
    df_durations.to_csv(output_path, index=False)
    print(f"\n💾 Reporte guardado: {output_path}")
    
    # Top 10 trades más largos
    print("\n🐢 TOP 10 TRADES MÁS LARGOS:")
    top_10_longest = df_durations.nlargest(10, 'duration_hours')
    for _, trade in top_10_longest.iterrows():
        print(f"  Trade #{int(trade['trade_num'])}: {trade['duration_days']:.1f} días ({trade['exit_reason']}) | Entry: {trade['entry_time']} | Exit: {trade['exit_time']}")
    
    # Top 10 trades más cortos
    print("\n🐇 TOP 10 TRADES MÁS CORTOS:")
    top_10_shortest = df_durations.nsmallest(10, 'duration_hours')
    for _, trade in top_10_shortest.iterrows():
        print(f"  Trade #{int(trade['trade_num'])}: {trade['duration_hours']:.1f} horas ({trade['exit_reason']}) | Entry: {trade['entry_time']} | Exit: {trade['exit_time']}")

if __name__ == "__main__":
    main()
