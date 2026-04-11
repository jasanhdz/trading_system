#!/usr/bin/env python3
"""
Phantom V30 Matrix — Tensor Data Loader
Loads OHLCV data directly into GPU VRAM as PyTorch tensors.
No pandas at runtime — pure tensor operations.
"""
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.storage.database_manager import DatabaseManager
from config.settings import settings


def load_tensor_data(device: str = "cuda:0", days: int | None = None, split: str = "train") -> dict:
    """
    Load OHLCV from DB, compute features, return GPU tensors.
    
    Returns dict with:
        'features': (N, 10) tensor [log_ret, high_norm, low_norm, vol_norm, rsi_norm, ema9, ema21, ema200, cvd_z, cvd_roc]
        'close':    (N,) tensor of close prices
        'n_candles': int
    """
    print(f"📦 Loading market data to {device}...")
    
    db = DatabaseManager(settings.DATABASE_URL)
    limit = int(days * 288) if days else None
    
    # Sort by timestamp DESC to get latest candles, then sort back to ASC
    df = db.get_ohlcv_data(settings.SYMBOL, "5m", limit=limit)
    if not df.empty and limit:
        # get_ohlcv_data with limit in sqlalchemy returns first N records (oldest) if order is ASC
        # We need to manually slice the last `limit` records if it returned entire ASC list
        df = df.tail(limit)
    
    if df.empty:
        raise ValueError("No data found in database.")
    
    # --- Feature Engineering (numpy, then move to GPU once) ---
    close = df['close'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    volume = df['volume'].values.astype(np.float32)
    
    # 1. Log Returns
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    
    # 2. High/Low shadows
    high_norm = np.log(high / close)
    low_norm = np.log(low / close)
    
    # 3. Volume normalization (rolling 24-period MA)
    vol_ma = pd.Series(volume).rolling(24).mean().fillna(volume[0]).values
    vol_norm = volume / (vol_ma + 1e-8)
    vol_norm = np.clip(vol_norm, 0, 10)
    
    # 4. RSI(14) Normalized [-1, 1]
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_norm = ((rsi - 50.0) / 50.0).fillna(0).values.astype(np.float32)
    
    # 5. EMA Distances
    ema_9 = close_s.ewm(span=9, adjust=False).mean()
    ema_21 = close_s.ewm(span=21, adjust=False).mean()
    ema_200 = close_s.ewm(span=200, adjust=False).mean()
    
    ema_9_norm = np.log(close_s / ema_9).fillna(0).values.astype(np.float32)
    ema_21_norm = np.log(close_s / ema_21).fillna(0).values.astype(np.float32)
    ema_200_norm = np.log(close_s / ema_200).fillna(0).values.astype(np.float32)
    
    # Clean NaN/Inf
    for arr in [log_ret, high_norm, low_norm, vol_norm, rsi_norm, ema_9_norm, ema_21_norm, ema_200_norm]:
        arr[~np.isfinite(arr)] = 0.0
        
    # 6. CVD (Cumulative Volume Delta) & ROC
    buy_volume = df['buy_volume'].values.astype(np.float32)
    buy_volume = np.nan_to_num(buy_volume, nan=volume / 2.0)
    cvd_raw = (2 * buy_volume) - volume
    
    # 1. Calcular CVD diferencial (no acumulativo) para estacionariedad
    cvd_raw_arr = np.asarray(cvd_raw)
    cvd_diff = np.zeros_like(cvd_raw_arr)
    cvd_diff[1:] = np.diff(cvd_raw_arr)
    cvd_diff[0] = cvd_raw_arr[0]

    # 2. Usar EWMA (exponencial) en lugar de media móvil simple para reactividad
    cvd_ewm_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
    cvd_ewm_std = pd.Series(cvd_diff).ewm(span=20).std().values

    # 3. Z-score del diferencial, no del acumulado
    cvd_z = (cvd_diff - cvd_ewm_mean) / (cvd_ewm_std + 1e-8)
    cvd_z = np.clip(cvd_z, -5, 5).astype(np.float32)
    cvd_z[~np.isfinite(cvd_z)] = 0.0
    
    # 4. CVD Rate of Change (Momentum) based on Z-Score
    cvd_roc = np.zeros_like(cvd_z)
    cvd_roc[1:] = cvd_z[1:] - cvd_z[:-1]
    cvd_roc = np.clip(cvd_roc, -2, 2).astype(np.float32)
    cvd_roc[~np.isfinite(cvd_roc)] = 0.0
    
    # 7. Candle Progress (Timestamp modulo)
    timestamps = pd.to_numeric(df.index).values  # Usually in ns or ms
    if timestamps[0] > 1e16:
        timestamps = timestamps / 1e6
    candle_progress = (timestamps % 300000) / 300000.0  # 5 min = 300k ms
    candle_progress = candle_progress.astype(np.float32)
    
    # --- Advanced Expert Features ---
    # 8. MTF EMA Slopes (1H = 12 periods, 4H = 48 periods)
    ema_12 = close_s.ewm(span=12, adjust=False).mean()
    ema_48 = close_s.ewm(span=48, adjust=False).mean()
    ema_1h_slope = (ema_12.diff() / (ema_12.shift(1) + 1e-8)).fillna(0).values.astype(np.float32)
    ema_4h_slope = (ema_48.diff() / (ema_48.shift(1) + 1e-8)).fillna(0).values.astype(np.float32)
    ema_1h_slope = np.clip(ema_1h_slope * 1000, -10, 10)  # scale up for nn
    ema_4h_slope = np.clip(ema_4h_slope * 1000, -10, 10)
    
    # 9. Volume Z-Score (Climax Detection)
    vol_s = pd.Series(volume)
    vol_ma20 = vol_s.rolling(window=20).mean()
    vol_std20 = vol_s.rolling(window=20).std()
    vol_z_arr = ((vol_s - vol_ma20) / (vol_std20 + 1e-8)).fillna(0).clip(-5, 10).values.astype(np.float32)
    
    # 10. CVD Divergence Flag
    price_roc3 = close_s.diff(3).fillna(0).values
    cvd_roc3 = pd.Series(cvd_z).diff(3).fillna(0).values
    cvd_div = np.zeros_like(cvd_z)
    cvd_div[(price_roc3 < -close*0.001) & (cvd_roc3 > 0.5)] = 1.0  # Bullish Div (Price down > 0.1%, but CVD strongly up)
    cvd_div[(price_roc3 > close*0.001) & (cvd_roc3 < -0.5)] = -1.0 # Bearish Div
    cvd_div = cvd_div.astype(np.float32)

    # Stack features: (N, 15)
    features = np.stack([
        log_ret, high_norm, low_norm, vol_norm, rsi_norm, 
        ema_9_norm, ema_21_norm, ema_200_norm, 
        cvd_z, cvd_roc, candle_progress,
        ema_1h_slope, ema_4h_slope, vol_z_arr, cvd_div
    ], axis=1)    
    # Skip first 24 rows (NaN from rolling window)
    features = features[24:]
    close = close[24:]
    
    # --- Walk-forward Split ---
    total_candles = len(features)
    
    # 14 days of 5m candles = 14 * 24 * 12 = 4032 candles
    val_candles = 14 * 288
    
    if total_candles > val_candles + 1000:
        train_end = total_candles - val_candles
    else:
        # Fallback if we have very little data
        train_end = int(total_candles * 0.8)
        
    gap = 200  # Match EMA200 memory — first val candle has zero train residue
    
    if split == "train":
        features = features[:train_end]
        close = close[:train_end]
        split_name = "Training (All except last 14d)"
    elif split == "val":
        features = features[train_end + gap:]
        close = close[train_end + gap:]
        split_name = "Validation (Last 14d)"
    elif split == "all":
        split_name = "Full Dataset"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Just use original features
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    close_tensor = torch.tensor(close, dtype=torch.float32, device=device)
    
    n_candles = features_tensor.shape[0]
    mem_mb = (features_tensor.nelement() + close_tensor.nelement()) * 4 / 1e6
    
    print(f"✅ Loaded {n_candles:,} candles for {split_name}")
    print(f"   VRAM usage: {mem_mb:.1f} MB on {device}")
    
    return {
        'features': features_tensor,
        'close': close_tensor,
        'n_candles': n_candles,
    }


if __name__ == "__main__":
    data = load_tensor_data("cuda:0")
    print(f"\nFeatures shape: {data['features'].shape}")
    print(f"Close shape: {data['close'].shape}")
    print(f"Sample features[100]: {data['features'][100].cpu().numpy()}")
    print(f"Sample close[100]: {data['close'][100].cpu().item():.2f}")
