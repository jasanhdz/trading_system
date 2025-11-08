"""Average True Range (ATR) indicator."""

import numpy as np
from typing import List, Union


def atr(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        period: ATR period (default: 14)
        
    Returns:
        Array of ATR values
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    
    if len(high) < 2:
        return np.array([np.nan] * len(high))
    
    # Calculate True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR using Wilder's smoothing
    atr_values = np.full(len(tr), np.nan)
    
    if len(tr) >= period:
        # First ATR is SMA of first 'period' TR values
        atr_values[period-1] = np.mean(tr[:period])
        
        # Subsequent ATR values use Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    return atr_values


def atr_percent(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14
) -> np.ndarray:
    """
    Calculate ATR as percentage of close price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
        
    Returns:
        Array of ATR percentage values
    """
    close = np.asarray(close, dtype=float)
    atr_values = atr(high, low, close, period)
    
    # Calculate as percentage of close
    atr_pct = np.where(close != 0, (atr_values / close) * 100, np.nan)
    
    return atr_pct


def atr_bands(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
    multiplier: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate ATR-based bands around close price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
        multiplier: Band distance multiplier (default: 2.0)
        
    Returns:
        Tuple of (upper_band, lower_band)
    """
    close = np.asarray(close, dtype=float)
    atr_values = atr(high, low, close, period)
    
    upper_band = close + (atr_values * multiplier)
    lower_band = close - (atr_values * multiplier)
    
    return upper_band, lower_band


def atr_stop(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
    multiplier: float = 2.0,
    side: str = "LONG"
) -> float:
    """
    Calculate ATR-based stop loss level.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
        multiplier: Stop distance multiplier (default: 2.0)
        side: Position side ("LONG" or "SHORT")
        
    Returns:
        Stop loss price level
    """
    close = np.asarray(close, dtype=float)
    atr_values = atr(high, low, close, period)
    
    if np.isnan(atr_values[-1]):
        return np.nan
    
    current_atr = atr_values[-1]
    current_close = close[-1]
    
    if side.upper() == "LONG":
        return current_close - (current_atr * multiplier)
    else:  # SHORT
        return current_close + (current_atr * multiplier)
