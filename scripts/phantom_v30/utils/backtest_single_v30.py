#!/usr/bin/env python3
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
import warnings

warnings.filterwarnings('ignore')

print("🚀 Iniciando Motor de Backtest Single (V30 Entrada+Salida)")

# === CONFIGURACIÓN ===
ENTRY_MODEL_PATH = "models/phantom_v30_champion.zip"
EXIT_MODEL_PATH = "models/phantom_exit_champion_v2.zip"
INITIAL_BALANCE = 20.0
LEVERAGE = 10.0
FEE = 0.0004
ENTRY_CONFIDENCE = 0.55
EXIT_CONFIDENCE = 0.60
TRAILING_ACTIVATION = 0.30  # 30% ROE para activar trailing
TRAILING_CALLBACK = 0.015   # 1.5% callback

# 1. CARGAR DATOS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.storage.database_manager import DatabaseManager
from config.settings import settings
db = DatabaseManager(settings.DATABASE_URL)
df = db.get_ohlcv_data(settings.SYMBOL, "5m")
if df is None or df.empty:
    raise ValueError("No se encontraron datos en la base de datos")

df.reset_index(inplace=True)
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Obtener últimos 20 días + 200 velas (para EMAs)
cutoff = df['timestamp'].max() - pd.Timedelta(days=21)
df_full = df[df['timestamp'] >= (cutoff - pd.Timedelta(days=1))].copy()

print("📊 Calculando features de mercado...")
# 2. FEATURE ENGINEERING (Vectorizado para todo el dataset)
# => BASE V30
df_full['log_ret'] = np.log(df_full['close'] / df_full['close'].shift(1)).fillna(0)
df_full['high_norm'] = np.log(df_full['high'] / df_full['close']).fillna(0)
df_full['low_norm'] = np.log(df_full['low'] / df_full['close']).fillna(0)
vol_ma = df_full['volume'].rolling(window=24).mean()
df_full['vol_norm'] = (df_full['volume'] / (vol_ma + 1e-8)).fillna(0).clip(0, 10)

delta = df_full['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-8)
rsi = 100.0 - (100.0 / (1.0 + rs))
df_full['rsi_norm'] = ((rsi - 50.0) / 50.0).fillna(0)

ema_9 = df_full['close'].ewm(span=9, adjust=False).mean()
ema_21 = df_full['close'].ewm(span=21, adjust=False).mean()
ema_200 = df_full['close'].ewm(span=200, adjust=False).mean()
df_full['ema_9_norm'] = np.log(df_full['close'] / ema_9).fillna(0)
df_full['ema_21_norm'] = np.log(df_full['close'] / ema_21).fillna(0)
df_full['ema_200_norm'] = np.log(df_full['close'] / ema_200).fillna(0)

# = CVD =
buy_volume = df_full['buy_volume'].values.astype(np.float32)
buy_volume = np.nan_to_num(buy_volume, nan=df_full['volume'].values / 2.0)
cvd_raw = (2 * buy_volume) - df_full['volume'].values
cvd_diff = np.zeros_like(cvd_raw)
cvd_diff[1:] = np.diff(cvd_raw)
cvd_diff[0] = cvd_raw[0]

cvd_ewm_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
cvd_ewm_std = pd.Series(cvd_diff).ewm(span=20).std().values
cvd_z = (cvd_diff - cvd_ewm_mean) / (cvd_ewm_std + 1e-8)
cvd_z = np.clip(cvd_z, -5, 5)
cvd_z[~np.isfinite(cvd_z)] = 0.0
df_full['cvd_z'] = cvd_z

cvd_roc = np.zeros_like(cvd_z)
cvd_roc[1:] = cvd_z[1:] - cvd_z[:-1]
cvd_roc = np.clip(cvd_roc, -2, 2)
cvd_roc[~np.isfinite(cvd_roc)] = 0.0
df_full['cvd_roc'] = cvd_roc

df_full['candle_progress'] = 1.0 # Para backtest consideramos candle cerrada

# => BASE EXIT (ATR & Vol Ratio)
high = df_full['high'].values
low = df_full['low'].values
close = df_full['close'].values
prev_close = np.roll(close, 1)
prev_close[0] = close[0]

tr1 = high - low
tr2 = np.abs(high - prev_close)
tr3 = np.abs(low - prev_close)
tr = np.maximum(tr1, np.maximum(tr2, tr3))
atr = np.zeros_like(tr)
atr[0] = tr[0]
for i in range(1, len(tr)):
    atr[i] = (atr[i-1] * 13 + tr[i]) / 14
df_full['atr'] = atr
df_full['vol_ratio'] = np.clip(df_full['volume'] / (df_full['volume'].rolling(20).mean() + 1e-10), 0, 10)

# CORTAR LOS 20 DIAS EXACTOS (quitando el warmup de 200 velas)
df_play = df_full[df_full['timestamp'] >= cutoff].copy().reset_index(drop=True)
market_features = ['log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc', 'candle_progress']
X = df_play[market_features].values.astype(np.float32)

print(f"✅ Datos listos: {len(df_play)} velas de 5 min (Últimos 20 días)")

# 3. CARGAR MODELOS CPU
import torch
print("\n🧠 Cargando IA Única (V31 Champion)...")
entry_model = PPO.load(ENTRY_MODEL_PATH, device="cpu")

# 4. SIMULACION STEP-BY-STEP
balance = INITIAL_BALANCE
in_trade = False
side = 0
entry_price = 0.0

mfe_roe = 0.0
mae_roe = 0.0
peak_price = 0.0
trade_step = 0
entry_timestamp = None

trades = []

print("\n🚀 INICIANDO SIMULACION...")

# Necesitamos window_size=64 para el V30
WINDOW = 64
for i in range(WINDOW, len(df_play) - 1):
    current_close = df_play['close'].iloc[i]
    current_high = df_play['high'].iloc[i]
    current_low = df_play['low'].iloc[i]
    timestamp = df_play['timestamp'].iloc[i]
    
    # ---------------- EXIT LOGIC ----------------
    if in_trade:
        trade_step += 1
        
        # Calcular ROE (Return On Equity) y actualizar peaks
        if side == 1:
            current_roe = ((current_close - entry_price) / entry_price) * LEVERAGE
            best_roe = ((current_high - entry_price) / entry_price) * LEVERAGE
            worst_roe = ((current_low - entry_price) / entry_price) * LEVERAGE
            new_peak = max(peak_price, current_high)
            peak_price = new_peak
        else:
            current_roe = ((entry_price - current_close) / entry_price) * LEVERAGE
            best_roe = ((entry_price - current_low) / entry_price) * LEVERAGE
            worst_roe = ((entry_price - current_high) / entry_price) * LEVERAGE
            new_peak = min(peak_price, current_low)
            peak_price = new_peak
            
        mfe_roe = max(mfe_roe, best_roe)
        mae_roe = min(mae_roe, worst_roe)
        
        close_reason = None
        realized_pnl = 0.0
        
        # A) Hard Stop Loss Mecánico (-50% ROE)
        if worst_roe <= -0.50:
            close_reason = "SL_HIT"
            realized_pnl = -(balance * 0.50) - (balance * LEVERAGE * FEE * 2) 
            
        # B) Trailing Stop Mecánico (Guardian)
        elif mfe_roe >= TRAILING_ACTIVATION:
            callback_dist = mfe_roe * TRAILING_CALLBACK
            trail_level = mfe_roe - callback_dist
            
            # Chequear si tocamos el nivel de trailing DURANTE el progreso de la vela
            if worst_roe <= trail_level:
                close_reason = "TRAILING_SAFETY_NET"
                ret_pct = trail_level
                realized_pnl = balance * ret_pct - (balance * LEVERAGE * FEE * 2)
                
        # C) AI EXIT V31 (Lógica original de único modelo)
        if close_reason is None:
            window_market_exit = X[i-WINDOW+1 : i+1]
            account_exit = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_exit = {
                'market': window_market_exit,
                'account': account_exit
            }
            
            action_exit, _ = entry_model.predict(obs_exit, deterministic=True)
            a_exit = int(action_exit)
            
            try:
                obs_t = entry_model.policy.obs_to_tensor(obs_exit)[0]
                dist = entry_model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.detach().cpu().numpy().flatten()
            except:
                probs = [0, 0, 0, 0]
                
            if probs[3] > EXIT_CONFIDENCE:
                close_reason = "AI_PANIC_CLOSE_V31"
                realized_pnl = (balance * current_roe) - (balance * LEVERAGE * FEE * 2)

        if close_reason:
            balance += realized_pnl
            trades.append({
                'entry_time': entry_timestamp,
                'exit_time': timestamp,
                'side': "LONG" if side == 1 else "SHORT",
                'entry_price': entry_price,
                'exit_price': current_close,
                'roe': current_roe * 100,
                'mfe': mfe_roe * 100,
                'pnl_usd': realized_pnl,
                'balance': balance,
                'reason': close_reason
            })
            in_trade = False
            
            if balance <= 1.0:
                print(f"💀 LIQUIDADO en {timestamp} | Balance: {balance}")
                break
                
    # ---------------- ENTRY LOGIC ----------------
    if not in_trade:
        window_market = X[i-WINDOW+1 : i+1]
        account = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        obs_entry = {
            'market': window_market,
            'account': account
        }
        
        action_entry, _ = entry_model.predict(obs_entry, deterministic=True)
        a = int(action_entry)
        
        try:
            obs_t = entry_model.policy.obs_to_tensor(obs_entry)[0]
            dist = entry_model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy().flatten()
        except:
            probs = [0, 0, 0, 0]
            
        if a == 1 and probs[1] > ENTRY_CONFIDENCE: # LONG
            side = 1
            entry_price = current_close
            in_trade = True
            mfe_roe = 0.0
            mae_roe = 0.0
            peak_price = entry_price
            trade_step = 0
            entry_timestamp = timestamp
        elif a == 2 and probs[2] > ENTRY_CONFIDENCE: # SHORT
            side = -1
            entry_price = current_close
            in_trade = True
            mfe_roe = 0.0
            mae_roe = 0.0
            peak_price = entry_price
            trade_step = 0
            entry_timestamp = timestamp

# === REPORT ===
print(f"\n{'='*50}")
print(f"🏆 BACKTEST SINGLE COMPLETADO (Últimos 20 días)")
print(f"{'='*50}")
print(f"💰 Balance Inicial: ${INITIAL_BALANCE:.2f}")
print(f"💰 Balance Final:   ${balance:.2f} ({(balance-INITIAL_BALANCE)/INITIAL_BALANCE*100:.1f}%)")

if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades['pnl_usd'] > 0).mean() * 100
    longs = len(df_trades[df_trades['side'] == 'LONG'])
    shorts = len(df_trades[df_trades['side'] == 'SHORT'])
    avg_pnl = df_trades['pnl_usd'].mean()
    net_pnl = df_trades['pnl_usd'].sum()
    
    print(f"📊 Trades Totales:  {len(trades)}")
    print(f"🎯 Win Rate:        {win_rate:.1f}%")
    print(f"📈 Longs: {longs} | 📉 Shorts: {shorts}")
    print(f"💵 Avg PnL/Trade:   ${avg_pnl:.2f}")
    print(f"💵 Net PnL Total:   ${net_pnl:.2f}")
    
    print("\n🛡️ Razones de Salida:")
    print(df_trades['reason'].value_counts().to_string())
    
    print("\n🧾 Últimos 10 Trades:")
    print(df_trades[['exit_time', 'side', 'roe', 'pnl_usd', 'reason']].tail(10).to_string())
else:
    print("⚠️ No se ejecutó ningún trade con thresholds > 55%.")
