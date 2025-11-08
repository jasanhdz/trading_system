"""Exponential Moving Average (EMA) indicator."""

import numpy as np
from typing import List, Union


def ema(
    values: Union[List[float], np.ndarray], 
    period: int
) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    
    Args:
        values: Price series
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    if len(values) < period:
        return np.array([np.nan] * len(values))
    
    values = np.asarray(values, dtype=float)
    
    # Use pandas-like EMA calculation
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with SMA for first period
    ema_values = np.empty_like(values)
    ema_values[:period-1] = np.nan
    ema_values[period-1] = np.mean(values[:period])
    
    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i-1]
    
    return ema_values


def double_ema(
    values: Union[List[float], np.ndarray], 
    fast_period: int,
    slow_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate two EMAs at once.
    
    Args:
        values: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        Tuple of (fast_ema, slow_ema)
    """
    fast_ema = ema(values, fast_period)
    slow_ema = ema(values, slow_period)
    return fast_ema, slow_ema


def ema_cross(
    values: Union[List[float], np.ndarray],
    fast_period: int,
    slow_period: int
) -> tuple[bool, bool]:
    """
    Check for EMA crossovers.
    
    Args:
        values: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        Tuple of (bullish_cross, bearish_cross)
    """
    fast_ema, slow_ema = double_ema(values, fast_period, slow_period)
    
    if len(fast_ema) < 2:
        return False, False
    
    # Current and previous values
    fast_curr, fast_prev = fast_ema[-1], fast_ema[-2]
    slow_curr, slow_prev = slow_ema[-1], slow_ema[-2]
    
    # Check for NaN values
    if any(np.isnan([fast_curr, fast_prev, slow_curr, slow_prev])):
        return False, False
    
    # Bullish cross: fast crosses above slow
    bullish = fast_prev <= slow_prev and fast_curr > slow_curr
    
    # Bearish cross: fast crosses below slow
    bearish = fast_prev >= slow_prev and fast_curr < slow_curr
    
    return bullish, bearish
