"""Candle utility functions."""

import numpy as np
from typing import List, Optional, Tuple

from ..types import Candle


def last(candles: List[Candle]) -> Candle:
    """Get the last candle."""
    if not candles:
        raise ValueError("Empty candles list")
    return candles[-1]


def get_closes(candles: List[Candle]) -> np.ndarray:
    """Extract close prices from candles."""
    return np.array([c.close for c in candles])


def get_highs(candles: List[Candle]) -> np.ndarray:
    """Extract high prices from candles."""
    return np.array([c.high for c in candles])


def get_lows(candles: List[Candle]) -> np.ndarray:
    """Extract low prices from candles."""
    return np.array([c.low for c in candles])


def get_opens(candles: List[Candle]) -> np.ndarray:
    """Extract open prices from candles."""
    return np.array([c.open for c in candles])


def get_volumes(candles: List[Candle]) -> np.ndarray:
    """Extract volumes from candles."""
    return np.array([c.volume for c in candles])


def volume_avg(candles: List[Candle], period: int = 20) -> float:
    """Calculate average volume over period."""
    if len(candles) < period:
        period = len(candles)
    
    volumes = get_volumes(candles)[-period:]
    return float(np.mean(volumes))


def count_streak(values: List[bool]) -> int:
    """
    Count consecutive True values from the end.
    
    Args:
        values: List of boolean values
        
    Returns:
        Count of consecutive True values from end
    """
    if not values:
        return 0
    
    count = 0
    for i in range(len(values) - 1, -1, -1):
        if values[i]:
            count += 1
        else:
            break
    
    return count


def is_green(candle: Candle) -> bool:
    """Check if candle is green (close > open)."""
    return candle.close > candle.open


def is_red(candle: Candle) -> bool:
    """Check if candle is red (close < open)."""
    return candle.close < candle.open


def candle_body(candle: Candle) -> float:
    """Calculate candle body size."""
    return abs(candle.close - candle.open)


def candle_range(candle: Candle) -> float:
    """Calculate full candle range (high - low)."""
    return candle.high - candle.low


def upper_wick(candle: Candle) -> float:
    """Calculate upper wick size."""
    if is_green(candle):
        return candle.high - candle.close
    else:
        return candle.high - candle.open


def lower_wick(candle: Candle) -> float:
    """Calculate lower wick size."""
    if is_green(candle):
        return candle.open - candle.low
    else:
        return candle.close - candle.low


def body_percentage(candle: Candle) -> float:
    """Calculate body as percentage of full range."""
    range_val = candle_range(candle)
    if range_val == 0:
        return 0.0
    return (candle_body(candle) / range_val) * 100


def is_doji(candle: Candle, threshold: float = 0.1) -> bool:
    """
    Check if candle is a doji.
    
    Args:
        candle: Candle to check
        threshold: Max body percentage for doji (default: 0.1%)
        
    Returns:
        True if doji pattern
    """
    return body_percentage(candle) < threshold


def is_hammer(candle: Candle, body_ratio: float = 0.3) -> bool:
    """
    Check if candle is a hammer pattern.
    
    Args:
        candle: Candle to check
        body_ratio: Max body/range ratio (default: 0.3)
        
    Returns:
        True if hammer pattern
    """
    body = candle_body(candle)
    range_val = candle_range(candle)
    lower = lower_wick(candle)
    upper = upper_wick(candle)
    
    if range_val == 0:
        return False
    
    # Hammer: small body at top, long lower wick
    return (
        body / range_val < body_ratio and
        lower > body * 2 and
        upper < body
    )


def highest_high(candles: List[Candle], period: Optional[int] = None) -> float:
    """Get highest high over period."""
    if period:
        candles = candles[-period:]
    
    if not candles:
        raise ValueError("Empty candles list")
    
    return max(c.high for c in candles)


def lowest_low(candles: List[Candle], period: Optional[int] = None) -> float:
    """Get lowest low over period."""
    if period:
        candles = candles[-period:]
    
    if not candles:
        raise ValueError("Empty candles list")
    
    return min(c.low for c in candles)


def price_range(candles: List[Candle], period: Optional[int] = None) -> Tuple[float, float]:
    """Get price range (low, high) over period."""
    return lowest_low(candles, period), highest_high(candles, period)
