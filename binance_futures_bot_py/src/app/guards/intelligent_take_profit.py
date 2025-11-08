"""Intelligent take-profit guard (parity with TS implementation)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...core.types import BotMode, Side
from ...core.indicators.ema import ema
from ...core.indicators.adx import adx as adx_calc
from ...core.utils.candles import get_closes, get_highs, get_lows
from ...core.utils.features import calculate_rsi
from ...infra.config import CONFIG


async def _cancel_brackets(exchange: Any, symbol: str, side: Side, logger: Any) -> None:
    try:
        if hasattr(exchange, "cancel_close_orders_for_side"):
            await exchange.cancel_close_orders_for_side(symbol, side)
        else:
            # Fallback: best-effort wipe of any reduce-only stops/tps
            if hasattr(exchange, "open_orders"):
                try:
                    orders = await exchange.open_orders(symbol)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - logging only
                    logger.debug("tp_dynamic_cancel_fetch_fail", {"symbol": symbol, "err": str(exc)})
                else:
                    for order in orders:
                        otype = str(order.get("type", ""))
                        if otype in {"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
                            try:
                                await exchange.cancel_order_by_id(symbol, str(order.get("orderId")))
                            except Exception as exc:  # pragma: no cover - logging only
                                logger.debug(
                                    "tp_dynamic_cancel_individual_fail",
                                    {"symbol": symbol, "orderId": order.get("orderId"), "err": str(exc)},
                                )
    except Exception as exc:  # pragma: no cover - logging only
        logger.debug("tp_dynamic_cancel_fail", {"symbol": symbol, "err": str(exc)})


def _compute_roe(side: Side, entry: float, mark: float, qty: float, leverage: float) -> float:
    if not entry or not qty or not leverage:
        return 0.0
    direction = 1 if side == Side.LONG else -1
    pnl = (mark - entry) * qty * direction
    notional = mark * qty
    margin = notional / max(1.0, leverage)
    return pnl / margin if margin else 0.0


async def intelligent_take_profit(symbol: str, exchange: Any, state_store: Any, logger: Any) -> None:
    state = await state_store.get()
    if not state or state.mode == BotMode.IDLE or not state.last_side or not state.last_entry_price:
        return

    position = await exchange.read_active_position(symbol, state.last_side)
    if not position or not position.qty_abs:
        return

    mark = await exchange.get_mark_price(symbol)
    leverage = (
        getattr(position, "leverage", None)
        or state.last_leverage
        or CONFIG.get_symbol_config(symbol).leverage
        or CONFIG.LEVERAGE
    )

    roe = _compute_roe(state.last_side, state.last_entry_price, mark, position.qty_abs, max(leverage, 1.0))

    peak = state.peak_roe or roe
    new_peak = max(peak, roe)
    updated_peak = False
    if new_peak != peak:
        state.peak_roe = new_peak
        updated_peak = True

    min_roe = getattr(CONFIG, "INT_TP_MIN_ROE", 0.20)

    now_ms = int(time.time() * 1000)
    cooldown_ms = getattr(CONFIG, "INT_TP_COOLDOWN_MS", 15_000)
    if state.last_intelli_tp_at and now_ms - state.last_intelli_tp_at < cooldown_ms:
        if updated_peak:
            await state_store.set(state)
        return

    lookback = max(40, getattr(CONFIG, "INT_TP_LOOKBACK", 40))
    timeframe = CONFIG.ENTRY_TIMEFRAME
    candles = await exchange.get_candles(symbol, timeframe, max(lookback * 2, 160))
    if len(candles) < lookback:
        if updated_peak:
            await state_store.set(state)
        return

    closes = get_closes(candles)
    highs = get_highs(candles)
    lows = get_lows(candles)

    # EMAs (match TS periods)
    ema_fast = ema(closes, 13)
    ema_slow = ema(closes, 34)
    fast = float(ema_fast[-1]) if ema_fast.size else float("nan")
    slow = float(ema_slow[-1]) if ema_slow.size else float("nan")

    adx_vals = adx_calc(highs, lows, closes, 14)
    adx_val = adx_vals.get("adx", 0.0)

    rsi = float(calculate_rsi(np.array(closes, dtype=float), 14))
    last_close = closes[-1]

    adx_min = getattr(CONFIG, "INT_TP_TREND_ADX", 18.0)
    trail_drop = max(0.0, min(1.0, getattr(CONFIG, "INT_TP_TRAIL_DROP", 0.35)))
    drop = new_peak - roe if new_peak > 0 else 0.0
    drop = max(0.0, drop / max(new_peak, 1e-9)) if new_peak > 0 else 0.0

    trend_strong_long = (
        fast > slow
        and adx_val >= adx_min
        and rsi >= 45.0
        and last_close >= slow
    )
    trend_strong_short = (
        fast < slow
        and adx_val >= adx_min
        and rsi <= 55.0
        and last_close <= slow
    )

    should_ride = (
        trend_strong_long if state.last_side == Side.LONG else trend_strong_short
    ) and drop < trail_drop

    if should_ride:
        state.last_intelli_tp_at = now_ms
        state.intelli_tp_state = "ride"
        state.peak_roe = new_peak
        await state_store.set(state)
        logger.debug(
            "tp_dynamic_hold",
            {
                "symbol": symbol,
                "roe": roe,
                "peak": new_peak,
                "drop": drop,
                "trendStrong": should_ride,
                "rsi": rsi,
                "adx": adx_val,
            },
        )
        return

    if roe < min_roe:
        state.last_intelli_tp_at = now_ms
        state.intelli_tp_state = "watch"
        state.peak_roe = new_peak
        await state_store.set(state)
        logger.debug(
            "tp_dynamic_watch",
            {
                "symbol": symbol,
                "roe": roe,
                "peak": new_peak,
                "drop": drop,
                "rsi": rsi,
                "adx": adx_val,
            },
        )
        return

    # Close only when ROE >= minimum threshold
    await exchange.close_side_market_safe(symbol, state.last_side, position.qty_abs, position.side_mode)
    await _cancel_brackets(exchange, symbol, state.last_side, logger)

    state.mode = BotMode.IDLE
    state.last_exit_reason = "tp_dynamic"
    state.last_exit_at = now_ms
    state.last_intelli_tp_at = now_ms
    state.intelli_tp_state = "exit"
    state.peak_roe = 0.0
    await state_store.set(state)

    logger.info(
        "tp_dynamic_close",
        {
            "symbol": symbol,
            "roe": roe,
            "peak": new_peak,
            "drop": drop,
            "rsi": rsi,
            "adx": adx_val,
        },
    )
