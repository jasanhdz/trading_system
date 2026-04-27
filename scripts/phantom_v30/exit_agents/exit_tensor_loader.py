import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Import project settings
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.storage.database_manager import DatabaseManager
from config.settings import settings


def load_exit_tensors(device: str = "cpu", split: str = "train", symbol: str = "ETH/USDT", timeframe: str = "5m") -> dict:
    """
    Load OHLCV data for the Exit Agent — SYNCHRONIZED with Champion training.
    Uses the exact same DB, full history, and train/val split as tensor_loader.py.
    
    Args:
        split: "train" (all except last 14d) or "val" (last 14d only)
    """
    print(f"📦 Loading Exit Data [{split}] to {device}...")
    
    # 1. Load ALL data from the same DB as the Champion
    db = DatabaseManager(settings.DATABASE_URL)
    df = db.get_ohlcv_data(symbol, timeframe, limit=None)
    
    if df.empty:
        raise ValueError(f"No database records found for {symbol} {timeframe}")
    
    # 2. Calculate ATR (Average True Range) for Volatility
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    close = df['close'].values.astype(np.float32)
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed 14-period ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # 3. CVD features — same calc as tensor_loader.py
    volume = df['volume'].values.astype(np.float32)
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
        
    # 4. Trim warmup (ATR needs 14, but align with Champion's 24)
    warmup = 24
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    atr = atr[warmup:]
    cvd_z = cvd_z[warmup:]
    cvd_roc = cvd_roc[warmup:]
    
    # 5. Walk-forward Split — IDENTICAL logic to tensor_loader.py
    total_candles = len(close)
    val_candles = 14 * 288  # 14 days
    
    if total_candles > val_candles + 1000:
        train_end = total_candles - val_candles
    else:
        train_end = int(total_candles * 0.8)
    
    gap = 200  # Match EMA200 memory gap from Champion
    
    if split == "train":
        sl = slice(0, train_end)
        split_name = "Training (All except last 14d)"
    elif split == "val":
        sl = slice(train_end + gap, None)
        split_name = "Validation (Last 14d)"
    elif split == "all":
        sl = slice(None)
        split_name = "Full Dataset"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    close = close[sl]
    atr = atr[sl]
    cvd_z = cvd_z[sl]
    cvd_roc = cvd_roc[sl]
    
    n_candles = len(close)
    
    # 6. Port to PyTorch Tensors
    t_close = torch.tensor(close, dtype=torch.float32, device=device)
    t_atr = torch.tensor(atr, dtype=torch.float32, device=device)
    t_cvd_z = torch.tensor(cvd_z, dtype=torch.float32, device=device)
    t_cvd_roc = torch.tensor(cvd_roc, dtype=torch.float32, device=device)
    
    print(f"✅ Loaded {n_candles:,} candles for Exit Agent [{split_name}]")
    
    return {
        'close': t_close,
        'atr': t_atr,
        'cvd_z': t_cvd_z,
        'cvd_roc': t_cvd_roc,
        'n_candles': n_candles
    }

if __name__ == "__main__":
    tensors = load_exit_tensors(split="train")
    print("Close shape:", tensors['close'].shape)
    print("ATR range:", tensors['atr'].min().item(), "to", tensors['atr'].max().item())
    print("CVD_Z range:", tensors['cvd_z'].min().item(), "to", tensors['cvd_z'].max().item())
