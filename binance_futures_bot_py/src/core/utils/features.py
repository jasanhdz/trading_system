"""Feature extraction utilities for ML models."""

import numpy as np
from typing import Dict, List, Optional

from ..indicators import adx, atr, ema
from ..types import Candle
from .candles import get_closes, get_highs, get_lows, volume_avg


def compute_features(
    candles: List[Candle],
    lookback: int = 50
) -> Dict[str, float]:
    """
    Compute features for ML model or strategy.
    
    Args:
        candles: List of candles
        lookback: Period for calculations
        
    Returns:
        Dictionary of features
    """
    if len(candles) < lookback:
        raise ValueError(f"Need at least {lookback} candles")
    
    closes = get_closes(candles)
    highs = get_highs(candles)
    lows = get_lows(candles)
    
    # Price features
    close = closes[-1]
    sma_20 = np.mean(closes[-20:])
    sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
    
    # EMAs
    ema_9 = ema(closes, 9)[-1]
    ema_21 = ema(closes, 21)[-1]
    
    # ATR
    atr_14 = atr(highs, lows, closes, 14)[-1]
    atr_pct = (atr_14 / close) * 100 if close > 0 else 0
    
    # ADX
    adx_result = adx(highs, lows, closes, 14)
    adx_val = adx_result.get('adx', 0)
    plus_di = adx_result.get('plus_di', 0)
    minus_di = adx_result.get('minus_di', 0)
    
    # Volatility
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns[-20:]) * np.sqrt(252) * 100  # Annualized
    
    # Volume
    avg_volume = volume_avg(candles, 20)
    volume_ratio = candles[-1].volume / avg_volume if avg_volume > 0 else 1
    
    # Price position
    high_20 = max(highs[-20:])
    low_20 = min(lows[-20:])
    price_position = (close - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
    
    # Momentum
    roc_10 = ((close - closes[-11]) / closes[-11]) * 100 if len(closes) > 10 else 0
    
    # RSI
    rsi = calculate_rsi(closes, 14)
    
    features = {
        'close': close,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'ema_9': ema_9,
        'ema_21': ema_21,
        'atr_14': atr_14,
        'atr_pct': atr_pct,
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'price_position': price_position,
        'roc_10': roc_10,
        'rsi_14': rsi,
        
        # Derived features
        'trend_strength': abs(ema_9 - ema_21) / atr_14 if atr_14 > 0 else 0,
        'is_trending': adx_val > 25,
        'trend_direction': 1 if plus_di > minus_di else -1,
    }
    
    return features


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    RSI clásico de Wilder (RMA con semilla SMA y suavizado en toda la serie).
    Devuelve el ÚLTIMO valor (float). Para series muy cortas, retorna 50.0.
    """
    p = np.asarray(prices, dtype=float)
    if p.size < period + 1:
        return 50.0

    deltas = np.diff(p)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Semilla (SMA inicial)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Suavizado RMA (Wilder) en toda la serie
    for i in range(period, gains.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def normalize_features(
    features: Dict[str, float],
    normalization_params: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """
    Normalize features for ML model input.
    
    Args:
        features: Raw features
        normalization_params: Min/max values for normalization
        
    Returns:
        Normalized features
    """
    if normalization_params is None:
        # Default normalization ranges
        normalization_params = {
            'rsi_14': {'min': 0, 'max': 100},
            'adx': {'min': 0, 'max': 100},
            'plus_di': {'min': 0, 'max': 100},
            'minus_di': {'min': 0, 'max': 100},
            'price_position': {'min': 0, 'max': 1},
            'volume_ratio': {'min': 0, 'max': 5},
            'atr_pct': {'min': 0, 'max': 10},
            'volatility': {'min': 0, 'max': 100},
        }
    
    normalized = features.copy()
    
    for key, params in normalization_params.items():
        if key in normalized:
            val = normalized[key]
            min_val = params['min']
            max_val = params['max']
            
            # Min-max normalization to [0, 1]
            if max_val > min_val:
                normalized[key] = (val - min_val) / (max_val - min_val)
                normalized[key] = max(0, min(1, normalized[key]))  # Clip to [0, 1]
    
    return normalized


def feature_vector(
    features: Dict[str, float],
    selected_features: List[str]
) -> np.ndarray:
    """
    Convert features dict to numpy array.
    
    Args:
        features: Feature dictionary
        selected_features: List of feature names to include
        
    Returns:
        Feature vector as numpy array
    """
    vector = []
    for feature in selected_features:
        if feature not in features:
            raise ValueError(f"Missing feature: {feature}")
        vector.append(features[feature])
    
    return np.array(vector)
