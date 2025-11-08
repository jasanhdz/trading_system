"""Trend analysis utilities."""

import numpy as np
from typing import List, Optional, Tuple, Dict

from ..indicators import ema
from ..types import Candle


def identify_trend(
    candles: List[Candle],
    short_period: int = 20,
    long_period: int = 50
) -> str:
    """
    Identify current trend based on EMAs.
    
    Args:
        candles: List of candles
        short_period: Short EMA period
        long_period: Long EMA period
        
    Returns:
        "BULLISH", "BEARISH", or "NEUTRAL"
    """
    if len(candles) < long_period:
        return "NEUTRAL"
    
    closes = np.array([c.close for c in candles])
    
    short_ema = ema(closes, short_period)[-1]
    long_ema = ema(closes, long_period)[-1]
    
    if np.isnan(short_ema) or np.isnan(long_ema):
        return "NEUTRAL"
    
    # Check EMA relationship
    if short_ema > long_ema * 1.001:  # 0.1% buffer
        return "BULLISH"
    elif short_ema < long_ema * 0.999:  # 0.1% buffer
        return "BEARISH"
    else:
        return "NEUTRAL"


def trend_strength(
    candles: List[Candle],
    period: int = 14
) -> float:
    """
    Calculate trend strength (0-100).
    
    Uses ADX or similar calculation.
    
    Args:
        candles: List of candles
        period: Period for calculation
        
    Returns:
        Trend strength percentage
    """
    from ..indicators import adx
    
    if len(candles) < period + 10:
        return 0.0
    
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    
    adx_result = adx(highs, lows, closes, period)
    return float(adx_result.get('adx', 0))


def support_resistance_levels(
    candles: List[Candle],
    lookback: int = 100,
    min_touches: int = 2
) -> Tuple[List[float], List[float]]:
    """
    Find support and resistance levels.
    
    Args:
        candles: List of candles
        lookback: Number of candles to analyze
        min_touches: Minimum touches to confirm level
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if len(candles) < lookback:
        lookback = len(candles)
    
    recent_candles = candles[-lookback:]
    
    # Get all highs and lows
    highs = [c.high for c in recent_candles]
    lows = [c.low for c in recent_candles]
    
    # Find local extremes
    support_levels = []
    resistance_levels = []
    
    # Simple approach: cluster prices
    all_prices = highs + lows
    price_clusters = cluster_prices(all_prices, tolerance=0.001)  # 0.1% tolerance
    
    current_price = candles[-1].close
    
    for level, count in price_clusters:
        if count >= min_touches:
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    # Sort levels
    support_levels.sort(reverse=True)  # Highest to lowest
    resistance_levels.sort()  # Lowest to highest
    
    return support_levels[:3], resistance_levels[:3]  # Return top 3 of each


def cluster_prices(
    prices: List[float],
    tolerance: float = 0.001
) -> List[Tuple[float, int]]:
    """
    Cluster similar prices together.
    
    Args:
        prices: List of prices
        tolerance: Relative tolerance for clustering
        
    Returns:
        List of (level, count) tuples
    """
    if not prices:
        return []
    
    sorted_prices = sorted(prices)
    clusters = []
    current_cluster = [sorted_prices[0]]
    
    for price in sorted_prices[1:]:
        if price <= current_cluster[-1] * (1 + tolerance):
            current_cluster.append(price)
        else:
            # Start new cluster
            level = np.mean(current_cluster)
            clusters.append((level, len(current_cluster)))
            current_cluster = [price]
    
    # Add final cluster
    if current_cluster:
        level = np.mean(current_cluster)
        clusters.append((level, len(current_cluster)))
    
    # Sort by count (descending)
    clusters.sort(key=lambda x: x[1], reverse=True)
    
    return clusters


def is_breakout(
    candles: List[Candle],
    level: float,
    direction: str = "UP",
    confirmation_candles: int = 2
) -> bool:
    """
    Check if price has broken out of a level.
    
    Args:
        candles: List of candles
        level: Price level to check
        direction: "UP" or "DOWN"
        confirmation_candles: Number of candles for confirmation
        
    Returns:
        True if breakout confirmed
    """
    if len(candles) < confirmation_candles:
        return False
    
    recent = candles[-confirmation_candles:]
    
    if direction.upper() == "UP":
        # All recent closes should be above level
        return all(c.close > level for c in recent)
    else:  # DOWN
        # All recent closes should be below level
        return all(c.close < level for c in recent)


def pivot_points(candle: Candle) -> Dict[str, float]:
    """
    Calculate pivot points from a candle.
    
    Args:
        candle: Single candle (typically daily)
        
    Returns:
        Dictionary with pivot levels
    """
    pivot = (candle.high + candle.low + candle.close) / 3
    
    r1 = 2 * pivot - candle.low
    r2 = pivot + (candle.high - candle.low)
    r3 = r1 + (candle.high - candle.low)
    
    s1 = 2 * pivot - candle.high
    s2 = pivot - (candle.high - candle.low)
    s3 = s1 - (candle.high - candle.low)
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }


def linear_regression_slope(
    values: np.ndarray,
    period: Optional[int] = None
) -> float:
    """
    Calculate linear regression slope.
    
    Args:
        values: Price values
        period: Period for calculation
        
    Returns:
        Slope value (positive = uptrend, negative = downtrend)
    """
    if period:
        values = values[-period:]
    
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    
    # Calculate linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
    
    # Normalize by average value
    avg_val = np.mean(values)
    if avg_val != 0:
        slope = slope / avg_val
    
    return float(slope)
