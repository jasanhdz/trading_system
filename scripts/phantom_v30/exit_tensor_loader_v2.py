#!/usr/bin/env python3
"""
Exit Tensor Loader V2 — Enhanced for Momentum Features
=======================================================
Loads OHLCV + ATR + CVD + Volume data for the Exit Agent V2.
Adds high/low prices and volume moving average for intra-candle
bracket checks and volume ratio calculations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.storage.database_manager import DatabaseManager
from config.settings import settings


def load_exit_tensors_v2(device: str = "cpu", split: str = "train", 
                         symbol: str = "ETH/USDT", timeframe: str = "5m") -> dict:
    """
    Load OHLCV data for Exit Agent V2 — includes high/low and volume MA.
    SYNCHRONIZED with Champion data split (4yr train / 14d val / 200 gap).
    """
    print(f"📦 Loading Exit V2 Data [{split}] to {device}...")
    
    db = DatabaseManager(settings.DATABASE_URL)
    df = db.get_ohlcv_data(symbol, timeframe, limit=None)
    
    if df.empty:
        raise ValueError(f"No database records found for {symbol} {timeframe}")
    
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    close = df['close'].values.astype(np.float32)
    volume = df['volume'].values.astype(np.float32)
    
    # ATR (14-period smoothed)
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
    
    # CVD features
    buy_volume = df['buy_volume'].values.astype(np.float32)
    buy_volume = np.nan_to_num(buy_volume, nan=volume / 2.0)
    cvd_raw = (2 * buy_volume) - volume
    
    cvd_diff = np.zeros_like(cvd_raw)
    cvd_diff[1:] = np.diff(cvd_raw)
    cvd_diff[0] = cvd_raw[0]
    
    cvd_ewm_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
    cvd_ewm_std = pd.Series(cvd_diff).ewm(span=20).std().values
    cvd_z = (cvd_diff - cvd_ewm_mean) / (cvd_ewm_std + 1e-8)
    cvd_z = np.clip(cvd_z, -5, 5).astype(np.float32)
    cvd_z[~np.isfinite(cvd_z)] = 0.0

    cvd_roc = np.zeros_like(cvd_z)
    cvd_roc[1:] = cvd_z[1:] - cvd_z[:-1]
    cvd_roc = np.clip(cvd_roc, -2, 2).astype(np.float32)
    cvd_roc[~np.isfinite(cvd_roc)] = 0.0
    
    # Volume MA (24-period = 2 hours)
    vol_ma = pd.Series(volume).rolling(24, min_periods=1).mean().values.astype(np.float32)
    
    # Trim warmup (align with Champion's 24)
    warmup = 24
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    atr = atr[warmup:]
    cvd_z = cvd_z[warmup:]
    cvd_roc = cvd_roc[warmup:]
    volume = volume[warmup:]
    vol_ma = vol_ma[warmup:]
    
    # Walk-forward Split — IDENTICAL logic to tensor_loader.py
    total_candles = len(close)
    val_candles = 14 * 288  # 14 days
    
    if total_candles > val_candles + 1000:
        train_end = total_candles - val_candles
    else:
        train_end = int(total_candles * 0.8)
    
    gap = 200
    
    if split == "train":
        sl = slice(0, train_end)
        split_name = "Training"
    elif split == "val":
        sl = slice(train_end + gap, None)
        split_name = "Validation"
    elif split == "all":
        sl = slice(None)
        split_name = "Full Dataset"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    close = close[sl]
    high = high[sl]
    low = low[sl]
    atr = atr[sl]
    cvd_z = cvd_z[sl]
    cvd_roc = cvd_roc[sl]
    volume = volume[sl]
    vol_ma = vol_ma[sl]
    
    n_candles = len(close)
    
    result = {
        'close': torch.tensor(close, dtype=torch.float32, device=device),
        'high': torch.tensor(high, dtype=torch.float32, device=device),
        'low': torch.tensor(low, dtype=torch.float32, device=device),
        'atr': torch.tensor(atr, dtype=torch.float32, device=device),
        'cvd_z': torch.tensor(cvd_z, dtype=torch.float32, device=device),
        'cvd_roc': torch.tensor(cvd_roc, dtype=torch.float32, device=device),
        'volume': torch.tensor(volume, dtype=torch.float32, device=device),
        'volume_ma': torch.tensor(vol_ma, dtype=torch.float32, device=device),
        'n_candles': n_candles,
    }
    
    print(f"✅ Loaded {n_candles:,} candles for Exit V2 [{split_name}]")
    print(f"   Arrays: close, high, low, atr, cvd_z, cvd_roc, volume, volume_ma")
    
    return result


if __name__ == "__main__":
    tensors = load_exit_tensors_v2(split="train")
    print(f"Close shape: {tensors['close'].shape}")
    print(f"High range:  {tensors['high'].min().item():.2f} to {tensors['high'].max().item():.2f}")
    print(f"ATR range:   {tensors['atr'].min().item():.4f} to {tensors['atr'].max().item():.4f}")
    print(f"CVD_Z range: {tensors['cvd_z'].min().item():.2f} to {tensors['cvd_z'].max().item():.2f}")
    print(f"Vol ratio sample: {(tensors['volume'][100] / tensors['volume_ma'][100]).item():.2f}")
