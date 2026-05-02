from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_ret",
    "high_norm",
    "low_norm",
    "vol_norm",
    "rsi_norm",
    "ema_9_norm",
    "ema_21_norm",
    "ema_200_norm",
    "cvd_z",
    "cvd_roc",
    "candle_progress",
    "ema_1h_slope",
    "ema_4h_slope",
    "vol_z",
    "cvd_div",
    "ema_1h_accel",
    "ema_4h_accel",
    "cvd_accel",
    "adx_norm",
    "trend_efficiency",
    "vol_regime",
]


def _clean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    h_s, l_s, c_s = pd.Series(high), pd.Series(low), pd.Series(close)
    plus_dm = h_s.diff().clip(lower=0)
    minus_dm = (-l_s.diff()).clip(lower=0)
    tr = pd.concat(
        [h_s - l_s, (h_s - c_s.shift(1)).abs(), (l_s - c_s.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1 / period, min_periods=period).mean().fillna(0).values.astype(np.float32)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing_ohlcv_columns: {sorted(missing)}")

    out = df.copy()
    close = out["close"].astype(float).values.astype(np.float32)
    high = out["high"].astype(float).values.astype(np.float32)
    low = out["low"].astype(float).values.astype(np.float32)
    open_ = out["open"].astype(float).values.astype(np.float32)
    volume = out["volume"].astype(float).values.astype(np.float32)
    close_s = pd.Series(close)

    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
    out["log_ret"] = _clean(log_ret)
    out["high_norm"] = _clean(np.log(high / np.maximum(close, 1e-10)))
    out["low_norm"] = _clean(np.log(low / np.maximum(close, 1e-10)))

    vol_ma = pd.Series(volume).rolling(24).mean().fillna(volume[0]).values
    out["vol_norm"] = _clean(np.clip(volume / (vol_ma + 1e-8), 0, 10))

    delta = close_s.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-8)))
    out["rsi_norm"] = _clean(((rsi - 50.0) / 50.0).fillna(0).values)

    ema_9 = close_s.ewm(span=9, adjust=False).mean()
    ema_21 = close_s.ewm(span=21, adjust=False).mean()
    ema_200 = close_s.ewm(span=200, adjust=False).mean()
    out["ema_9_norm"] = _clean(np.log(close_s / ema_9).fillna(0).values)
    out["ema_21_norm"] = _clean(np.log(close_s / ema_21).fillna(0).values)
    out["ema_200_norm"] = _clean(np.log(close_s / ema_200).fillna(0).values)

    buy_volume = out.get("buy_volume", pd.Series(volume / 2.0)).astype(float).values.astype(np.float32)
    buy_volume = np.nan_to_num(buy_volume, nan=volume / 2.0)
    cvd_raw = (2 * buy_volume) - volume
    cvd_diff = np.zeros_like(cvd_raw)
    cvd_diff[1:] = np.diff(cvd_raw)
    cvd_diff[0] = cvd_raw[0]
    cvd_mean = pd.Series(cvd_diff).ewm(span=20).mean().values
    cvd_std = pd.Series(cvd_diff).ewm(span=20).std().values
    cvd_z = np.clip((cvd_diff - cvd_mean) / (cvd_std + 1e-8), -5, 5).astype(np.float32)
    cvd_z[~np.isfinite(cvd_z)] = 0.0
    out["cvd_z"] = cvd_z
    cvd_roc = np.zeros_like(cvd_z)
    cvd_roc[1:] = cvd_z[1:] - cvd_z[:-1]
    out["cvd_roc"] = _clean(np.clip(cvd_roc, -2, 2))

    timestamps = pd.to_numeric(out.index).values
    if len(timestamps) and timestamps[0] > 1e16:
        timestamps = timestamps / 1e6
    out["candle_progress"] = _clean((timestamps % 300000) / 300000.0)

    ema_12 = close_s.ewm(span=12, adjust=False).mean()
    ema_48 = close_s.ewm(span=48, adjust=False).mean()
    ema_1h_slope = np.clip((ema_12.diff() / (ema_12.shift(1) + 1e-8)).fillna(0).values * 1000, -10, 10)
    ema_4h_slope = np.clip((ema_48.diff() / (ema_48.shift(1) + 1e-8)).fillna(0).values * 1000, -10, 10)
    out["ema_1h_slope"] = _clean(ema_1h_slope)
    out["ema_4h_slope"] = _clean(ema_4h_slope)

    vol_s = pd.Series(volume)
    out["vol_z"] = _clean(((vol_s - vol_s.rolling(20).mean()) / (vol_s.rolling(20).std() + 1e-8)).fillna(0).clip(-5, 10).values)

    price_roc3 = close_s.diff(3).fillna(0).values
    cvd_roc3 = pd.Series(cvd_z).diff(3).fillna(0).values
    cvd_div = np.zeros_like(cvd_z)
    cvd_div[(price_roc3 < -close * 0.001) & (cvd_roc3 > 0.5)] = 1.0
    cvd_div[(price_roc3 > close * 0.001) & (cvd_roc3 < -0.5)] = -1.0
    out["cvd_div"] = _clean(cvd_div)

    ema_1h_accel = np.zeros_like(ema_1h_slope)
    ema_1h_accel[1:] = ema_1h_slope[1:] - ema_1h_slope[:-1]
    ema_4h_accel = np.zeros_like(ema_4h_slope)
    ema_4h_accel[1:] = ema_4h_slope[1:] - ema_4h_slope[:-1]
    cvd_accel = np.zeros_like(cvd_roc)
    cvd_accel[1:] = cvd_roc[1:] - cvd_roc[:-1]
    out["ema_1h_accel"] = _clean(np.clip(ema_1h_accel * 2000, -30, 30))
    out["ema_4h_accel"] = _clean(np.clip(ema_4h_accel * 2000, -30, 30))
    out["cvd_accel"] = _clean(np.clip(cvd_accel * 2, -4, 4))

    adx_raw = _calc_adx(high, low, close)
    out["adx_norm"] = _clean(np.clip((adx_raw / 50.0) - 1.0, -1.0, 1.0))
    out["trend_efficiency"] = _clean(np.clip(np.abs(close - open_) / np.maximum(high - low, 1e-10), 0.0, 1.0))

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).ewm(alpha=1 / 14, min_periods=14).mean().fillna(pd.Series(tr)).values
    out["vol_regime"] = _clean(np.clip((atr14 / np.maximum(close, 1e-10) - 0.01) / 0.02, -1.0, 1.0))

    return out.iloc[24:].copy()


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    frame = build_feature_frame(df)
    return frame[FEATURE_COLUMNS].values.astype(np.float32)
