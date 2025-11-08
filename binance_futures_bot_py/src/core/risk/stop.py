"""Stop loss calculations and management."""
from typing import List, Optional
from ..indicators import atr
from ..types import Candle, Side

def atr_stop_loss(
    candles: List[Candle],
    side: Side,
    atr_multiplier: float = 2.0,
    period: int = 14
) -> float:
    """
    Calculate ATR-based stop loss.
    
    Args:
        candles: Historical candles
        side: Position side
        atr_multiplier: ATR multiplier for stop distance
        period: ATR period
        
    Returns:
        Stop loss price
    """
    if len(candles) < period:
        raise ValueError(f"Need at least {period} candles for ATR")
    
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    
    atr_values = atr(highs, lows, closes, period)
    current_atr = atr_values[-1]
    current_price = candles[-1].close
    
    if side == Side.LONG:
        return current_price - (current_atr * atr_multiplier)
    else:  # SHORT
        return current_price + (current_atr * atr_multiplier)


def percentage_stop_loss(
    entry_price: float,
    side: Side,
    stop_percent: float
) -> float:
    """
    Calculate percentage-based stop loss.
    
    Args:
        entry_price: Entry price
        side: Position side
        stop_percent: Stop loss percentage
        
    Returns:
        Stop loss price
    """
    if side == Side.LONG:
        return entry_price * (1 - stop_percent / 100)
    else:  # SHORT
        return entry_price * (1 + stop_percent / 100)


def swing_stop_loss(
    candles: List[Candle],
    side: Side,
    lookback: int = 20,
    buffer_percent: float = 0.1
) -> float:
    """
    Calculate stop loss based on recent swing high/low.
    
    Args:
        candles: Historical candles
        side: Position side
        lookback: Number of candles to look back
        buffer_percent: Buffer percentage beyond swing
        
    Returns:
        Stop loss price
    """
    if len(candles) < lookback:
        lookback = len(candles)
    
    recent = candles[-lookback:]
    
    if side == Side.LONG:
        # Find recent swing low
        swing_low = min(c.low for c in recent)
        return swing_low * (1 - buffer_percent / 100)
    else:  # SHORT
        # Find recent swing high
        swing_high = max(c.high for c in recent)
        return swing_high * (1 + buffer_percent / 100)


def trailing_stop_update(
    current_price: float,
    side: Side,
    current_stop: float,
    trail_percent: float,
    entry_price: Optional[float] = None
) -> float:
    """
    Update trailing stop loss.
    
    Args:
        current_price: Current market price
        side: Position side
        current_stop: Current stop loss price
        trail_percent: Trailing percentage
        entry_price: Original entry price (for break-even check)
        
    Returns:
        Updated stop loss price
    """
    if side == Side.LONG:
        new_stop = current_price * (1 - trail_percent / 100)
        
        # Only update if new stop is higher
        if new_stop > current_stop:
            # Optional: Ensure stop is at least break-even
            if entry_price and new_stop < entry_price:
                return entry_price
            return new_stop
    else:  # SHORT
        new_stop = current_price * (1 + trail_percent / 100)
        
        # Only update if new stop is lower
        if new_stop < current_stop:
            # Optional: Ensure stop is at least break-even
            if entry_price and new_stop > entry_price:
                return entry_price
            return new_stop
    
    return current_stop


def chandelier_exit(
    candles: List[Candle],
    side: Side,
    period: int = 22,
    atr_multiplier: float = 3.0
) -> float:
    """
    Calculate Chandelier Exit stop loss.
    
    Args:
        candles: Historical candles
        side: Position side
        period: Lookback period for highest/lowest
        atr_multiplier: ATR multiplier
        
    Returns:
        Stop loss price
    """
    if len(candles) < max(period, 14):  # Need data for ATR too
        raise ValueError("Insufficient candles for Chandelier Exit")
    
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    
    # Calculate ATR
    atr_values = atr(highs, lows, closes, 14)
    current_atr = atr_values[-1]
    
    # Get recent highs/lows
    recent_candles = candles[-period:]
    
    if side == Side.LONG:
        # Highest high - ATR * multiplier
        highest = max(c.high for c in recent_candles)
        return highest - (current_atr * atr_multiplier)
    else:  # SHORT
        # Lowest low + ATR * multiplier
        lowest = min(c.low for c in recent_candles)
        return lowest + (current_atr * atr_multiplier)


def parabolic_sar_stop(
    candles: List[Candle],
    side: Side,
    initial_af: float = 0.02,
    max_af: float = 0.2,
    af_increment: float = 0.02
) -> float:
    """
    Calculate Parabolic SAR stop loss.
    
    Simplified implementation for stop loss purposes.
    
    Args:
        candles: Historical candles
        side: Position side
        initial_af: Initial acceleration factor
        max_af: Maximum acceleration factor
        af_increment: AF increment
        
    Returns:
        Stop loss price
    """
    if len(candles) < 5:
        raise ValueError("Need at least 5 candles for PSAR")
    
    # Simplified PSAR calculation
    if side == Side.LONG:
        # Start with lowest low
        sar = min(c.low for c in candles[-5:])
        ep = max(c.high for c in candles[-5:])  # Extreme point
    else:  # SHORT
        # Start with highest high
        sar = max(c.high for c in candles[-5:])
        ep = min(c.low for c in candles[-5:])  # Extreme point
    
    af = initial_af
    
    # Update SAR for recent candles
    for candle in candles[-4:]:
        # Update SAR
        sar = sar + af * (ep - sar)
        
        # Update extreme point and AF
        if side == Side.LONG:
            if candle.high > ep:
                ep = candle.high
                af = min(af + af_increment, max_af)
        else:  # SHORT
            if candle.low < ep:
                ep = candle.low
                af = min(af + af_increment, max_af)
    
    return sar


def apply_price_filter(
    stop_price: float,
    tick_size: float,
    price_precision: int
) -> float:
    """
    Apply exchange price filters to stop price.
    
    Args:
        stop_price: Raw stop price
        tick_size: Minimum price increment
        price_precision: Price decimal precision
        
    Returns:
        Filtered stop price
    """
    if tick_size > 0:
        # Round to nearest tick
        stop_price = round(stop_price / tick_size) * tick_size
    
    # Apply precision
    stop_price = round(stop_price, price_precision)
    
    return stop_price

def round_to_tick(price: float, tick_size: float, price_precision: int) -> float:
    """
    Redondea un precio al múltiplo de tick_size y precisión deseada.
    """
    if tick_size <= 0:
        return round(price, price_precision)
    steps = round(price / tick_size)
    return round(steps * tick_size, price_precision)


def compute_stop_from_liq_ticks(
    side: Side,
    liq_price: float,
    current_price: float,
    entry_price: float,
    tick_size: float,
    price_precision: int,
    ticks_above_liq: int,
) -> float:
    """
    Stop a una distancia (en ticks) desde el precio de liquidación,
    asegurando que quede del lado correcto respecto a entry/price.
    - LONG: stop = liq + ticks * tick_size  (y siempre por debajo de entry/price)
    - SHORT: stop = liq - ticks * tick_size (y siempre por encima de entry/price)
    """
    offset = ticks_above_liq * tick_size
    if side == Side.LONG:
        raw = liq_price + offset
        limit = min(entry_price, current_price) - tick_size
        stop = min(raw, limit)
    else:
        raw = liq_price - offset
        limit = max(entry_price, current_price) + tick_size
        stop = max(raw, limit)

    return round_to_tick(stop, tick_size, price_precision)


def compute_stop_from_max_loss(
    *,
    side: Side,
    entry_price: float,
    tick_size: float,
    price_precision: int,
    ticks_from_entry: int,
) -> float:
    """SL a 'ticks_from_entry * tick_size' desde el precio de ENTRADA."""
    delta = tick_size * float(ticks_from_entry)
    stop_raw = entry_price - delta if side == Side.LONG else entry_price + delta
    return round_to_tick(stop_raw, tick_size, price_precision)
