#!/usr/bin/env python3
"""
Phantom V30 Inference Server
Self-contained FastAPI server for ML predictions.
Loads the V30 champion model and serves predictions to the TypeScript bot.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import torch
import sys
import os
from pathlib import Path
from stable_baselines3 import PPO

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.phantom_v30.train_v30 import TransformerExtractor  # Required for model loading
from scripts.phantom_v30.env import PhantomEnv

app = FastAPI()

# === CONFIG ===
MODEL_PATH = "models/phantom_v30_champion.zip"
EXIT_MODEL_PATH = "models/phantom_exit_champion_v2.zip"
WINDOW_SIZE = 64

# Global model reference
model = None
model_mtime = 0  # Track file modification time for hot-reload

exit_model = None
exit_model_mtime = 0

def load_model():
    """Load the champion model into memory. Auto-detects file changes."""
    global model, model_mtime
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ No champion model at {MODEL_PATH}")
        return False
    
    current_mtime = os.path.getmtime(MODEL_PATH)
    
    # Skip if model already loaded and file hasn't changed
    if model is not None and current_mtime == model_mtime:
        return True
    
    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        model_mtime = current_mtime
        print(f"🔄 Model {'reloaded' if model_mtime > 0 else 'loaded'}: {MODEL_PATH} (mtime={int(current_mtime)})")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def load_exit_model():
    """Load the exit champion model into memory. Auto-detects file changes."""
    global exit_model, exit_model_mtime
    if not os.path.exists(EXIT_MODEL_PATH):
        print(f"⚠️ No exit champion model at {EXIT_MODEL_PATH}")
        return False
    
    current_mtime = os.path.getmtime(EXIT_MODEL_PATH)
    
    # Skip if model already loaded and file hasn't changed
    if exit_model is not None and current_mtime == exit_model_mtime:
        return True
    
    try:
        exit_model = PPO.load(EXIT_MODEL_PATH, device="cpu")
        exit_model_mtime = current_mtime
        print(f"🔄 Exit Model {'reloaded' if exit_model_mtime > 0 else 'loaded'}: {EXIT_MODEL_PATH} (mtime={int(current_mtime)})")
        return True
    except Exception as e:
        print(f"❌ Failed to load exit model: {e}")
        return False

def get_signal(df: pd.DataFrame):
    """
    Run inference on market data.
    Returns: (action, confidence, leverage)
    - action: 0=IDLE, 1=LONG, 2=SHORT, 3=CLOSE
    - confidence: 0.0 to 1.0
    - leverage: suggested leverage (fixed at 1 for now)
    """
    global model
    if model is None:
        return 0, 0.5, 1, 0.0, 0.0, [0.5, 0.25, 0.25, 0.0]  # Default: IDLE
    
    # Feature Engineering (must match env.py)
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['high_norm'] = np.log(df['high'] / df['close']).fillna(0)
    df['low_norm'] = np.log(df['low'] / df['close']).fillna(0)
    vol_ma = df['volume'].rolling(window=24).mean()
    df['vol_norm'] = (df['volume'] / (vol_ma + 1e-8)).fillna(0).clip(0, 10)
    
    # 4. RSI(14) Normalized [-1, 1]
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['rsi_norm'] = ((rsi - 50.0) / 50.0).fillna(0)
    
    # 5. EMA Distances
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()
    
    df['ema_9_norm'] = np.log(df['close'] / ema_9).fillna(0)
    df['ema_21_norm'] = np.log(df['close'] / ema_21).fillna(0)
    df['ema_200_norm'] = np.log(df['close'] / ema_200).fillna(0)

    # 6. CVD — EWMA Differential (matches tensor_loader.py V33.51)
    buy_volume = df['buy_volume'].values.astype(np.float32)
    buy_volume = np.nan_to_num(buy_volume, nan=df['volume'].values / 2.0)
    cvd_raw = (2 * buy_volume) - df['volume'].values
    
    # Differential CVD (stationary)
    cvd_diff = np.zeros_like(cvd_raw)
    cvd_diff[1:] = np.diff(cvd_raw)
    cvd_diff[0] = cvd_raw[0]
    
    # EWMA Z-Score
    cvd_ewm_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
    cvd_ewm_std = pd.Series(cvd_diff).ewm(span=20).std().values
    cvd_z = (cvd_diff - cvd_ewm_mean) / (cvd_ewm_std + 1e-8)
    cvd_z = np.clip(cvd_z, -5, 5)
    cvd_z[~np.isfinite(cvd_z)] = 0.0
    df['cvd_z'] = cvd_z
    
    # CVD ROC based on Z-Score
    cvd_roc = np.zeros_like(cvd_z)
    cvd_roc[1:] = cvd_z[1:] - cvd_z[:-1]
    cvd_roc = np.clip(cvd_roc, -2, 2)
    cvd_roc[~np.isfinite(cvd_roc)] = 0.0
    df['cvd_roc'] = cvd_roc

    # 7. Candle Progress — use wall-clock time for the last candle
    # Historical candle timestamps are exact 5-min multiples (always 0).
    # Only the LAST candle has meaningful progress based on current time.
    import time
    now_ms = int(time.time() * 1000)
    candle_progress_arr = np.zeros(len(df), dtype=np.float32)
    candle_progress_arr[-1] = (now_ms % 300000) / 300000.0  # Only last candle matters
    df['candle_progress'] = candle_progress_arr

    # --- Advanced Expert Features ---
    # 8. MTF EMA Slopes (1H = 12 periods, 4H = 48 periods)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_48 = df['close'].ewm(span=48, adjust=False).mean()
    df['ema_1h_slope'] = ((ema_12.diff() / (ema_12.shift(1) + 1e-8)).clip(-10/1000, 10/1000) * 1000).fillna(0).astype(np.float32)
    df['ema_4h_slope'] = ((ema_48.diff() / (ema_48.shift(1) + 1e-8)).clip(-10/1000, 10/1000) * 1000).fillna(0).astype(np.float32)
    
    df['ema_1h_accel'] = (df['ema_1h_slope'].diff() * 2000).fillna(0).clip(-30, 30).astype(np.float32)
    df['ema_4h_accel'] = (df['ema_4h_slope'].diff() * 2000).fillna(0).clip(-30, 30).astype(np.float32)
    df['cvd_accel'] = (df['cvd_roc'].diff() * 2).fillna(0).clip(-4, 4).astype(np.float32)
    
    # 11. ADX(14) Normalized
    def _calc_adx_live(h, l, c, period=14):
        h_s, l_s, c_s = pd.Series(h), pd.Series(l), pd.Series(c)
        plus_dm = h_s.diff().clip(lower=0)
        minus_dm = (-l_s.diff()).clip(lower=0)
        tr1 = h_s - l_s
        tr2 = (h_s - c_s.shift(1)).abs()
        tr3 = (l_s - c_s.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        return adx.fillna(0).values
    
    adx_raw = _calc_adx_live(df['high'].values, df['low'].values, df['close'].values)
    df['adx_norm'] = np.clip((adx_raw / 50.0) - 1.0, -1.0, 1.0).astype(np.float32)
    
    # 12. Trend Efficiency
    df['trend_efficiency'] = np.clip(
        np.abs(df['close'] - df['open']) / np.maximum(df['high'] - df['low'], 1e-10),
        0, 1
    ).astype(np.float32)
    
    # 13. Volatility Regime (ATR14 / close)
    tr1_vr = df['high'] - df['low']
    tr2_vr = (df['high'] - df['close'].shift(1)).abs()
    tr3_vr = (df['low'] - df['close'].shift(1)).abs()
    tr_vr = pd.concat([tr1_vr, tr2_vr, tr3_vr], axis=1).max(axis=1)
    atr14_vr = tr_vr.ewm(alpha=1/14, min_periods=14).mean().fillna(tr_vr)
    df['vol_regime'] = np.clip((atr14_vr / df['close'] - 0.01) / 0.02, -1, 1).astype(np.float32)
    # 9. Volume Z-Score (Climax Detection)
    vol_s = df['volume']
    vol_ma20 = vol_s.rolling(window=20).mean()
    vol_std20 = vol_s.rolling(window=20).std()
    df['vol_z'] = ((vol_s - vol_ma20) / (vol_std20 + 1e-8)).fillna(0).clip(-5, 10).astype(np.float32)
    
    # 10. CVD Divergence Flag
    price_roc3 = df['close'].diff(3).fillna(0).values
    cvd_roc3 = pd.Series(cvd_z).diff(3).fillna(0).values
    cvd_div = np.zeros_like(cvd_z)
    cvd_div[(price_roc3 < -df['close'].values*0.001) & (cvd_roc3 > 0.5)] = 1.0  # Bullish Div
    cvd_div[(price_roc3 > df['close'].values*0.001) & (cvd_roc3 < -0.5)] = -1.0 # Bearish Div
    df['cvd_div'] = cvd_div.astype(np.float32)

    # Auto-detect model's expected feature count (backward compat)
    try:
        model_n_features = model.observation_space['market'].shape[-1]
        # DEBUG: print(f"--- Inference: Detected Model Features = {model_n_features} ---")
    except Exception:
        model_n_features = 10
    
    if model_n_features >= 21:
        market_features = [
            'log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 
            'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc', 'candle_progress',
            'ema_1h_slope', 'ema_4h_slope', 'vol_z', 'cvd_div',
            'ema_1h_accel', 'ema_4h_accel', 'cvd_accel',
            'adx_norm', 'trend_efficiency', 'vol_regime'
        ]
    elif model_n_features >= 18:
        # fallback para modelos antiguos
        market_features = [
            'log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 
            'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc', 'candle_progress',
            'ema_1h_slope', 'ema_4h_slope', 'vol_z', 'cvd_div',
            'ema_1h_accel', 'ema_4h_accel', 'cvd_accel'
        ]
    elif model_n_features >= 15:
        market_features = [
            'log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 
            'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc', 'candle_progress',
            'ema_1h_slope', 'ema_4h_slope', 'vol_z', 'cvd_div'
        ]
    elif model_n_features >= 11:
        market_features = ['log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc', 'candle_progress']
    else:
        market_features = ['log_ret', 'high_norm', 'low_norm', 'vol_norm', 'rsi_norm', 'ema_9_norm', 'ema_21_norm', 'ema_200_norm', 'cvd_z', 'cvd_roc']
    
    # Build observation window
    if len(df) < WINDOW_SIZE:
        return 0, 0.5, 1, 0.0, 0.0, [0.5, 0.25, 0.25, 0.0]  # Not enough data
    
    window = df[market_features].iloc[-WINDOW_SIZE:].values.astype(np.float32)
    
    # Account state (flat position for inference)
    # Must match matrix_env.py exactly when flat:
    # [balance_norm=1.0, leverage_used=0.0, pnl_pct=0.0, in_trade=0.0]
    account = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    obs = {
        'market': window,
        'account': account
    }
    
    # Predict
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)
    
    # Get REAL action probabilities from the policy's softmax
    try:
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()
        # probs[0]=IDLE, probs[1]=LONG, probs[2]=SHORT, probs[3]=CLOSE
        confidence = float(probs[action])
        all_probs = [float(p) for p in probs]
    except:
        confidence = 0.5
        all_probs = [0.5, 0.25, 0.25, 0.0]  # fallback
    
    # Return CVD values for the API response
    cvd_z_last = float(df['cvd_z'].iloc[-1]) if 'cvd_z' in df.columns else 0.0
    cvd_roc_last = float(df['cvd_roc'].iloc[-1]) if 'cvd_roc' in df.columns else 0.0
    return action, confidence, 0, cvd_z_last, cvd_roc_last, all_probs


class PredictRequest(BaseModel):
    symbol: str

class ExitRequest(BaseModel):
    symbol: str
    entry_price: float
    current_pnl: float
    mfe: float
    mae: float
    duration_minutes: float
    leverage: float

@app.on_event("startup")
async def startup_event():
    load_model()
    load_exit_model()


@app.post("/ml-v2/predict")
async def predict(req: PredictRequest):
    # Hot-reload: check if champion.zip has been updated
    load_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Fetch data directly via Binance fapi to get full 11 columns (Taker Buy Volume is at index 9)
    try:
        import ccxt
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
        # Note: req.symbol is already native "ETHUSDT" coming from TS
        raw_klines = exchange.fapiPublicGetKlines({'symbol': req.symbol, 'interval': '5m', 'limit': 1000})
        
        ohlcv = []
        for k in raw_klines:
            # Timestamp, Open, High, Low, Close, Volume, Taker Buy Base Asset Volume
            ohlcv.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])])
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'buy_volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")
    
    # Run Inference
    action, conf, lev, cvd_z_live, cvd_roc_live, all_probs = get_signal(df)
    
    # Use REAL softmax probabilities from the model
    # all_probs: [IDLE, LONG, SHORT, CLOSE]
    idle_prob = all_probs[0] if len(all_probs) > 0 else 0.5
    long_prob = all_probs[1] if len(all_probs) > 1 else 0.25
    short_prob = all_probs[2] if len(all_probs) > 2 else 0.25
    close_prob = all_probs[3] if len(all_probs) > 3 else 0.0
    neutral_prob = idle_prob  # Separated out the close probability!
        
    cvd_z_val = cvd_z_live
    cvd_roc_val = cvd_roc_live
        
    return {
        "symbol": req.symbol,
        "long_prob": float(long_prob),
        "short_prob": float(short_prob),
        "close_prob": float(close_prob),
        "neutral_prob": float(neutral_prob),
        "consensus_level": int(lev),
        "smart_leverage": int(lev),
        "meta_verdict": "PHANTOM_V33",
        "features": {
            "cvd_z": cvd_z_val,
            "cvd_slope": cvd_roc_val,
            "weakness": 0.5
        }
    }

@app.post("/ml-v2/exit_signal")
async def get_exit_signal(req: ExitRequest):
    load_exit_model()
    
    if exit_model is None:
        return {"action": "HOLD", "confidence": 0.0, "reason": "model_not_loaded"}
        
    try:
        import ccxt
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
        
        # Fetch 100 limit to ensure EWMA for CVD has enough warmup
        raw_klines = exchange.fapiPublicGetKlines({'symbol': req.symbol, 'interval': '5m', 'limit': 1000})
        
        ohlcv = []
        for k in raw_klines:
            # Timestamp, Open, High, Low, Close, Volume, Taker Buy Base Asset Volume
            ohlcv.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])])
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'buy_volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
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
        
        current_atr = atr[-1]
        
        # Build features
        # obs: [current_pnl, mfe, mae, time_decay, atr_norm]
        time_decay = (req.duration_minutes / 5.0) / 24.0  # 24 velas x 5min = 2h (synced with MatrixExitEnv)
        atr_norm = (current_atr / req.entry_price) * req.leverage
        
        # Calculate CVD Z-Score & ROC
        buy_volume = df['buy_volume'].values.astype(np.float32)
        buy_volume = np.nan_to_num(buy_volume, nan=df['volume'].values / 2.0)
        cvd_raw = (2 * buy_volume) - df['volume'].values
        
        cvd_diff = np.zeros_like(cvd_raw)
        cvd_diff[1:] = np.diff(cvd_raw)
        cvd_diff[0] = cvd_raw[0]
        
        cvd_ewm_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
        cvd_ewm_std = pd.Series(cvd_diff).ewm(span=20).std().values
        cvd_z_arr = (cvd_diff - cvd_ewm_mean) / (cvd_ewm_std + 1e-8)
        cvd_z_arr = np.clip(cvd_z_arr, -5, 5)
        cvd_z_arr[~np.isfinite(cvd_z_arr)] = 0.0
        
        cvd_roc_arr = np.zeros_like(cvd_z_arr)
        cvd_roc_arr[1:] = cvd_z_arr[1:] - cvd_z_arr[:-1]
        cvd_roc_arr = np.clip(cvd_roc_arr, -2, 2)
        cvd_roc_arr[~np.isfinite(cvd_roc_arr)] = 0.0
        
        cvd_z_val = float(cvd_z_arr[-1])
        cvd_roc_val = float(cvd_roc_arr[-1])
        
        # 5. Drawdown from peak
        drawdown_from_peak = (req.mfe - req.current_pnl) / max(req.mfe, 1e-10) if req.mfe > 0 else 0.0
        drawdown_from_peak = np.clip(drawdown_from_peak, 0.0, 5.0)
        
        # Reconstruct ROE history for velocity & acceleration using close prices
        close_prices = df['close'].values
        price_diff = close_prices[-1] - req.entry_price
        # Infer side (1.0 for LONG, -1.0 for SHORT) using current PnL direction
        if abs(price_diff) > 1e-8:
            side = 1.0 if (req.current_pnl * price_diff) > 0 else -1.0
        else:
            side = 1.0
            
        roe_history = []
        for i in range(-3, 0):
            price = close_prices[i]
            raw_pct = ((price - req.entry_price) / req.entry_price) * side
            roe_history.append(raw_pct * req.leverage)
            
        roe_velocity = np.clip(roe_history[-1] - roe_history[-2], -2.0, 2.0)
        vel_now = roe_history[-1] - roe_history[-2]
        vel_prev = roe_history[-2] - roe_history[-3]
        roe_acceleration = np.clip(vel_now - vel_prev, -2.0, 2.0)
        
        # Distance to TP (Assumes 1.50 = 150% ROE default TP)
        TAKE_PROFIT_ROE = 1.50
        distance_to_tp = max(0.0, (TAKE_PROFIT_ROE - req.current_pnl) / TAKE_PROFIT_ROE)
        distance_to_tp = np.clip(distance_to_tp, 0.0, 5.0)
        
        # Volume ratio
        vol = float(df['volume'].values[-1])
        vol_ma = float(df['volume'].rolling(window=20).mean().values[-1])
        volume_ratio = np.clip(vol / max(vol_ma, 1e-10), 0.0, 10.0)

        obs = np.array([
            req.current_pnl,      # [0]
            req.mfe,              # [1]
            req.mae,              # [2]
            time_decay,           # [3]
            atr_norm,             # [4]
            drawdown_from_peak,   # [5]
            roe_velocity,         # [6]
            roe_acceleration,     # [7]  
            cvd_z_val,            # [8]
            cvd_roc_val,          # [9]
            distance_to_tp,       # [10]
            volume_ratio,         # [11]
        ], dtype=np.float32)
        
        action, _states = exit_model.predict(obs, deterministic=True)
        action_val = int(action)
        
        # Calculate confidence
        try:
            obs_tensor = exit_model.policy.obs_to_tensor(obs)[0]
            dist = exit_model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy().flatten()
            confidence = float(probs[action_val])
        except:
            confidence = 0.5
        
        str_action = "CLOSE" if action_val == 1 else "HOLD"
        
        return {
            "action": str_action,
            "confidence": confidence,
            "features_used": {
                "current_pnl": float(req.current_pnl),
                "mfe": float(req.mfe),
                "mae": float(req.mae),
                "time_decay": float(time_decay),
                "atr_norm": float(atr_norm),
                "drawdown_from_peak": float(drawdown_from_peak),
                "roe_velocity": float(roe_velocity),
                "roe_acceleration": float(roe_acceleration),
                "cvd_z": float(cvd_z_val),
                "cvd_roc": float(cvd_roc_val),
                "distance_to_tp": float(distance_to_tp),
                "volume_ratio": float(volume_ratio)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exit signal prediction failed: {e}")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
