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


def _normalize_price_features(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza features basadas en precio absoluto para ML.
    
    Convierte indicadores de precio/volumen absolutos a valores relativos.
    Esto previene que el scaler aprenda parámetros gigantes y mejora la generalización.
    """
    normalized = features_df.copy()
    close = df.loc[features_df.index, 'close']
    volume = df.loc[features_df.index, 'volume']
    
    # 1. Moving Averages: Convertir a % distancia del precio
    ma_features = ['sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20']
    for ma in ma_features:
        if ma in normalized.columns:
            # (close - MA) / close * 100 = % arriba/abajo de la MA
            normalized[ma] = ((close - normalized[ma]) / close * 100).replace([np.inf, -np.inf], np.nan)
    
    # 2. Bollinger Bands: Usar posición relativa y normalizar
    if 'bb_upper' in normalized.columns and 'bb_lower' in normalized.columns:
        bb_range = (normalized['bb_upper'] - normalized['bb_lower']).replace(0, np.nan)
        # Posición del precio dentro de las bandas (-1 = en lower, 0 = en middle, 1 = en upper)
        normalized['bb_position'] = ((close - normalized['bb_middle']) / (bb_range / 2)).replace([np.inf, -np.inf], np.nan)
        # Width normalizado (% del precio)
        normalized['bb_width'] = (bb_range / close * 100).replace([np.inf, -np.inf], np.nan)
        # Eliminar valores absolutos
        normalized = normalized.drop(columns=['bb_upper', 'bb_middle', 'bb_lower'], errors='ignore')
    
    # 3. Keltner Channels: Mismo tratamiento
    if 'kc_upper' in normalized.columns and 'kc_lower' in normalized.columns:
        kc_range = (normalized['kc_upper'] - normalized['kc_lower']).replace(0, np.nan)
        kc_middle = (normalized['kc_upper'] + normalized['kc_lower']) / 2
        normalized['kc_position'] = ((close - kc_middle) / (kc_range / 2)).replace([np.inf, -np.inf], np.nan)
        normalized['kc_width'] = (kc_range / close * 100).replace([np.inf, -np.inf], np.nan)
        normalized = normalized.drop(columns=['kc_upper', 'kc_lower'], errors='ignore')
    
    # 4. Parabolic SAR: Distancia relativa
    if 'sar' in normalized.columns:
        normalized['sar'] = ((close - normalized['sar']) / close * 100).replace([np.inf, -np.inf], np.nan)
    
    # 5. Momentum: Convertir a % en vez de diferencia absoluta
    if 'momentum_10' in normalized.columns:
        # Momentum es close - close.shift(10), convertir a %
        normalized['momentum_10'] = (normalized['momentum_10'] / close * 100).replace([np.inf, -np.inf], np.nan)
    
    # 6. Volume Features: Normalizar por volumen promedio reciente
    volume_ma_20 = volume.rolling(20, min_periods=1).mean()
    volume_sma_features = ['volume_sma_10', 'volume_sma_20', 'volume_sma_50']
    for vol_feat in volume_sma_features:
        if vol_feat in normalized.columns:
            # Ratio vs volumen promedio (ej. 1.5 = 50% más que el promedio)
            normalized[vol_feat] = (normalized[vol_feat] / volume_ma_20).replace([np.inf, -np.inf], np.nan)
    
    # 7. Indicadores acumulativos (OBV, A/D, VPT): Usar z-score en ventana móvil
    # Estos indicadores crecen sin límite, mejor usar su posición relativa reciente
    window = 50
    
    # OBV (On Balance Volume)
    if 'obv' in normalized.columns:
        obv = normalized['obv']
        obv_mean = obv.rolling(window, min_periods=10).mean()
        obv_std = obv.rolling(window, min_periods=10).std()
        normalized['obv_zscore'] = ((obv - obv_mean) / (obv_std + 1e-9)).replace([np.inf, -np.inf], np.nan)
        normalized = normalized.drop(columns=['obv'], errors='ignore')
    
    if 'obv_sma' in normalized.columns:
        normalized = normalized.drop(columns=['obv_sma'], errors='ignore')
    
    # A/D Line (Accumulation/Distribution)
    if 'ad_line' in normalized.columns:
        ad = normalized['ad_line']
        ad_mean = ad.rolling(window, min_periods=10).mean()
        ad_std = ad.rolling(window, min_periods=10).std()
        normalized['ad_zscore'] = ((ad - ad_mean) / (ad_std + 1e-9)).replace([np.inf, -np.inf], np.nan)
        normalized = normalized.drop(columns=['ad_line'], errors='ignore')
    
    # VPT (Volume Price Trend)
    if 'vpt' in normalized.columns:
        vpt = normalized['vpt']
        vpt_mean = vpt.rolling(window, min_periods=10).mean()
        vpt_std = vpt.rolling(window, min_periods=10).std()
        normalized['vpt_zscore'] = ((vpt - vpt_mean) / (vpt_std + 1e-9)).replace([np.inf, -np.inf], np.nan)
        normalized = normalized.drop(columns=['vpt'], errors='ignore')
    
    # 8. Volume Flow: Ya está parcialmente normalizado pero verificar
    if 'volume_flow' in normalized.columns:
        # Normalizar por volumen promedio
        normalized['volume_flow'] = (normalized['volume_flow'] / volume_ma_20).replace([np.inf, -np.inf], np.nan)
    
    return normalized

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

    # Normalizar features basadas en precio ANTES de limpiar
    feature_frame = _normalize_price_features(df, feature_frame)

    # Clean up infinities, forward-fill indicator warm-up gaps, and keep consistent column order
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.ffill()
    feature_frame = feature_frame.dropna()
    
    # Actualizar lista de features (algunas columnas fueron reemplazadas)
    final_features = feature_frame.columns.tolist()

    return feature_frame, final_features
