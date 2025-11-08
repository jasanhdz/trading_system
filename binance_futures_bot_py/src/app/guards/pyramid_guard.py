"""Pyramid guard - handles position pyramiding."""
import math
from typing import Any
from ...core.types import BotMode, Side
from ...core.indicators.atr import atr
from ...core.risk.sizing import floor_to_step
from ...core.utils.candles import get_highs, get_lows, get_closes
from ...infra.config import CONFIG

async def pyramid_guard(
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
) -> None:
    """Handle position pyramiding logic."""
    st = await state.get()
    if not st or st.mode == BotMode.IDLE:
        return
    
    if not st.last_side or not st.last_entry_price:
        return
    
    # Check if pyramiding is enabled
    if CONFIG.PYRAMID_MAX_UNITS <= 0:
        return
    
    # Check current pyramid units
    if (st.pyramid_units or 0) >= CONFIG.PYRAMID_MAX_UNITS:
        return
    
    # Get current position
    pos = await exchange.read_active_position(symbol, st.last_side)
    if not pos:
        return
    
    # Get current price
    price = await exchange.get_mark_price(symbol)
    
    # Get candles for ATR
    candles = await exchange.get_candles(symbol, CONFIG.ENTRY_TIMEFRAME, 200)
    atr_arr = atr(get_highs(candles), get_lows(candles), get_closes(candles), CONFIG.ATR_LEN)
    atr_value = float(atr_arr[-1]) if not math.isnan(atr_arr[-1]) else 0.0
    
    if not atr_value or atr_value <= 0:
        return
    
    # Calculate distance from last pyramid price
    last_pyramid = st.last_pyramid_price or st.last_entry_price
    distance = abs(price - last_pyramid)
    
    # Check if price moved enough (in ATR terms)
    if distance < CONFIG.PYRAMID_STEP_ATR * atr_value:
        return
    
    # Check if move is in favorable direction
    if st.last_side == Side.LONG:
        if price <= last_pyramid:
            return  # Price must be higher for long pyramid
    else:
        if price >= last_pyramid:
            return  # Price must be lower for short pyramid
    
    # Calculate pyramid size
    entry_qty = st.last_entry_qty or pos.qty_abs
    pyramid_qty = entry_qty * CONFIG.PYRAMID_UNIT_PCT_OF_ENTRY
    
    # Get filters for rounding
    symbol_cfg = CONFIG.get_symbol_config(symbol)
    filters = await exchange.get_symbol_filters(symbol, symbol_cfg.leverage)
    pyramid_qty = floor_to_step(pyramid_qty, filters.step_size, filters.qty_precision)
    
    if pyramid_qty <= 0:
        return
    
    # Check minimum notional
    notional = pyramid_qty * price
    if notional < filters.min_notional:
        logger.debug("pyramid_min_notional_not_met", {
            "qty": pyramid_qty,
            "notional": notional,
            "minNotional": filters.min_notional,
        })
        return
    
    logger.info("pyramid_adding", {
        "side": st.last_side.value,
        "units": st.pyramid_units,
        "qty": pyramid_qty,
        "price": price,
        "lastPyramid": last_pyramid,
        "distance": distance,
        "atr": atr_value,
    })
    
    # Add to position
    result = await exchange.market_open(symbol, st.last_side, pyramid_qty)
    
    # Update state
    st.pyramid_units = (st.pyramid_units or 0) + 1
    st.last_pyramid_price = result.avg_price or price
    await state.set(st)
    
    logger.info("pyramid_added", {
        "side": st.last_side.value,
        "units": st.pyramid_units,
        "avgPrice": result.avg_price,
    })
    
    # May need to adjust stops/TP after pyramiding
    # This could be implemented based on strategy needs
