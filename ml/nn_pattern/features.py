"""Feature engineering helpers for the neural pattern model."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from analysis.features.technical_indicators import TechnicalIndicators
from ml.nn_pattern.regime_features import calculate_regime_features, get_regime_feature_names

PRICE_COLS = ["open", "high", "low", "close", "volume"]

MOMENTUM_COLS = [
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "roc_10",
    "roc_20",
    "momentum_10",
    "cci_14",
    "cci_20",
]

TREND_COLS = [
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_10",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_histogram",
    "adx",
    "plus_di",
    "minus_di",
    "sar",
    "aroon_up",
    "aroon_down",
    "aroon_oscillator",
]

VOLATILITY_COLS = [
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",
    "kc_upper",
    "kc_lower",
    "kc_width",
    "atr_14",
    "atr_20",
    "natr_14",
    "natr_20",
    "hist_vol_10",
    "hist_vol_20",
    "hist_vol_50",
    "vol_ratio",
]

VOLUME_COLS = [
    "volume_sma_10",
    "volume_sma_20",
    "volume_sma_50",
    "volume_ratio_10",
    "volume_ratio_20",
    "obv",
    "obv_sma",
    "ad_line",
    "cmf",
    "vpt",
    "mfi",
]

CUSTOM_FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "roll_vol_10",
    "roll_vol_30",
    "roll_vol_60",
    "volume_zscore_20",
    "atr_pct",
    "price_location",
    "volume_flow",
]

# Features de régimen de mercado (35 features adicionales)
REGIME_FEATURES = get_regime_feature_names()

ALL_FEATURES = (
    MOMENTUM_COLS
    + TREND_COLS
    + VOLATILITY_COLS
    + VOLUME_COLS
    + CUSTOM_FEATURES
    + REGIME_FEATURES
)


def _strip_base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove OHLCV columns when joining indicator frames."""
    return frame.drop(columns=[c for c in PRICE_COLS if c in frame.columns])


def _safe_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return frame restricted to desired columns, filling with NaNs if missing."""
    return frame.reindex(columns=columns)


def _build_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional handcrafted statistics not covered by TechnicalIndicators."""
    feat = pd.DataFrame(index=df.index)
    feat["return_1"] = df["close"].pct_change()
    feat["return_3"] = df["close"].pct_change(periods=3)
    feat["return_6"] = df["close"].pct_change(periods=6)
    feat["return_12"] = df["close"].pct_change(periods=12)

    feat["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    feat["log_return_3"] = np.log(df["close"] / df["close"].shift(3))
    feat["log_return_6"] = np.log(df["close"] / df["close"].shift(6))

    feat["roll_vol_10"] = df["close"].pct_change().rolling(10).std()
    feat["roll_vol_30"] = df["close"].pct_change().rolling(30).std()
    feat["roll_vol_60"] = df["close"].pct_change().rolling(60).std()

    feat["volume_zscore_20"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / (df["volume"].rolling(20).std() + 1e-9)
    )
    
    # New Features
    # 1. ATR Percentage (Normalized Volatility)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    feat["atr_pct"] = atr_14 / df["close"]

    # 2. Simulated Order Flow (Buying/Selling Pressure)
    # Approximation: Volume * (Close location within High-Low range)
    range_len = (df["high"] - df["low"]).replace(0, 1e-9)
    feat["price_location"] = (df["close"] - df["low"]) / range_len
    feat["volume_flow"] = df["volume"] * (2 * feat["price_location"] - 1) # -1 to 1 scale

    return feat


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate a feature dataframe aligned with the input price index.

    Returns:
        feature_frame: DataFrame containing engineered features.
        feature_columns: Ordered list of feature column names.
    """
    if not set(PRICE_COLS).issubset(df.columns):
        missing = sorted(set(PRICE_COLS) - set(df.columns))
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df.sort_index().copy()
    ti = TechnicalIndicators(df)

    # Calcular features de régimen de mercado
    regime_frame = calculate_regime_features(df)

    frames = [
        _safe_columns(_strip_base_columns(ti.momentum_indicators()), MOMENTUM_COLS),
        _safe_columns(_strip_base_columns(ti.trend_indicators()), TREND_COLS),
        _safe_columns(_strip_base_columns(ti.volatility_indicators()), VOLATILITY_COLS),
        _safe_columns(_strip_base_columns(ti.volume_indicators()), VOLUME_COLS),
        _safe_columns(_build_custom_features(df), CUSTOM_FEATURES),
        _safe_columns(regime_frame, REGIME_FEATURES),
    ]

    feature_frame = pd.concat(frames, axis=1)

    # Clean up infinities, forward-fill indicator warm-up gaps, and keep consistent column order
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.ffill()
    feature_frame = feature_frame.dropna()

    return feature_frame, ALL_FEATURES
