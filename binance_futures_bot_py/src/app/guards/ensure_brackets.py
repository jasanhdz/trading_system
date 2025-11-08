"""Ensure brackets guard - maintains stop loss and take profit orders."""
import time
from typing import Any

from ...core.types import BotMode, Side
from ...core.risk.stop import compute_stop_from_liq_ticks, round_to_tick
from ...infra.config import CONFIG

async def brackets_guard(
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
) -> None:
    """Ensure stop loss and take profit orders are in place."""
    st = await state.get()
    if not st or st.mode == BotMode.IDLE or not st.last_side or not st.last_entry_price:
        return

    pos = await exchange.read_active_position(symbol, st.last_side)
    if not pos:
        return

    symbol_cfg = CONFIG.get_symbol_config(symbol)
    leverage = getattr(pos, "leverage", None) or symbol_cfg.leverage
    filters = await exchange.get_symbol_filters(symbol, leverage)
    price = await exchange.get_mark_price(symbol)

    # ========== STOP basado en ENTRADA (ticks fijos), NO liquidación ==========
    stop_order = await exchange.open_stop_for_side(symbol, st.last_side)
    if not stop_order:
        liq_price = await exchange.read_liquidation_price(symbol, st.last_side)
        if liq_price is None:
            liq_price = price
        ticks = CONFIG.SL_TICKS_ABOVE_LIQ_MAP.get(symbol, CONFIG.SL_TICKS_ABOVE_LIQ_DEFAULT)
        liq_buffer = getattr(CONFIG, "STOP_LIQ_BUFFER_RATIO", 0.08)

        stop = compute_stop_from_liq_ticks(
            side=st.last_side,
            liq_price=liq_price,
            current_price=price,
            entry_price=st.last_entry_price,
            tick_size=filters.tick_size,
            price_precision=filters.price_precision,
            ticks_above_liq=ticks,
        )

        await exchange.place_stop_close(symbol, st.last_side, stop)
        logger.info(
            "ensure_stop_created",
            {
                "symbol": symbol,
                "side": st.last_side.value,
                "stop": stop,
                "liq": liq_price,
                "ticks": ticks,
            },
        )
        st.lastTrailStop = stop
        await state.set(st)

    # TP
    tp_order = await exchange.open_tp_for_side(symbol, st.last_side)
    if not tp_order:
        r = CONFIG.TP_ROE
        fee = CONFIG.FEE_BUFFER_PCT
        lev = getattr(pos, "leverage", None) or st.last_leverage or symbol_cfg.leverage
        if st.last_side == Side.LONG:
            tp_raw = st.last_entry_price * (1 + r / lev + fee)
        else:
            tp_raw = st.last_entry_price * (1 - r / lev - fee)
        tp = round_to_tick(tp_raw, filters.tick_size, filters.price_precision)
        await exchange.place_tp_close(symbol, st.last_side, tp)
        logger.info("ensure_tp_created", {"symbol": symbol, "side": st.last_side.value, "tp": tp})

    stop_now = stop_order or (await exchange.open_stop_for_side(symbol, st.last_side))
    tp_now = tp_order or (await exchange.open_tp_for_side(symbol, st.last_side))
    if stop_now and tp_now and not st.brackets_armed_at:
        st.brackets_armed_at = int(time.time() * 1000)
        await state.set(st)
        logger.debug("brackets_armed", {"symbol": symbol, "side": st.last_side.value})
