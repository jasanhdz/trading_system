# src/app/guards/profit_guard.py
"""Profit guard - profit protection with breakeven & ROE trailing."""

from typing import Any
from ...core.types import BotMode, Side
from ...core.risk.stop import round_to_tick
from ...infra.config import CONFIG


# ==== Helpers de configuración con fallback (no rompe si faltan en CONFIG) ====

def _get(name: str, default):
    return getattr(CONFIG, name, default)

# Trailing por ROE: bandas y porcentaje de “ganancia bloqueada” según pico
PG_ENABLED            = _get("PROFIT_TRAIL_ENABLED", True)
PG_ARM_ROE            = _get("PROFIT_TRAIL_ARM_ROE", 0.12)   # arma trailing desde 12% ROE
PG_PEAK_BAND_LOW      = _get("PROFIT_TRAIL_PEAK_BAND_LOW", 0.30)  # 30%
PG_PEAK_BAND_MID      = _get("PROFIT_TRAIL_PEAK_BAND_MID", 0.50)  # 50%
PG_KEEP_LOW           = _get("PROFIT_TRAIL_KEEP_LOW", 0.70)  # <30% pico, bloquea 70%
PG_KEEP_MID           = _get("PROFIT_TRAIL_KEEP_MID", 0.75)  # 30–50% pico, bloquea 75%
PG_KEEP_HIGH          = _get("PROFIT_TRAIL_KEEP_HIGH", 0.80) # >50% pico, bloquea 80%
PG_MIN_LOCK           = _get("PROFIT_TRAIL_MIN_LOCK", 0.10)  # nunca salgas con menos de 10% ROE
PG_USE_STOP           = _get("PROFIT_TRAIL_USE_STOP", False) # False=cierre a mercado; True=ajusta stop
PG_HYSTERESIS         = _get("PROFIT_TRAIL_HYSTERESIS", 0.002)  # 0.2% ROE de colchón anti-ruido

# Breakeven lock (igual que tu lógica actual)
BE_AT_ROE             = _get("PROFIT_LOCK_BE_AT_ROE", 0.02)  # ejemplo: 2% ROE

# (Opcional) compatibilidad con tu esquema anterior de "giveback" absoluto/relativo
GB_ARM_ROE            = _get("PROFIT_GIVEBACK_ARM_ROE", None)
GB_DROP_REL           = _get("PROFIT_GIVEBACK_DROP_REL", None)
GB_DROP_MIN           = _get("PROFIT_GIVEBACK_DROP_MIN", None)


def _trail_keep_ratio(peak_roe: float) -> float:
    """Devuelve el % del pico que queremos conservar (bloquear)."""
    if peak_roe < PG_PEAK_BAND_LOW:
        return PG_KEEP_LOW
    if peak_roe < PG_PEAK_BAND_MID:
        return PG_KEEP_MID
    return PG_KEEP_HIGH


def _roe_to_price(entry: float, roe: float, leverage: float, side: Side) -> float:
    """
    Convierte un ROE objetivo a precio umbral.
    LONG:  roe = ((P - entry)/entry) * lev  =>  P = entry * (1 + roe/lev)
    SHORT: roe = ((entry - P)/entry) * lev  =>  P = entry * (1 - roe/lev)
    """
    if leverage <= 0:
        leverage = 1.0
    if side == Side.LONG:
        return entry * (1.0 + roe / leverage)
    else:
        return entry * (1.0 - roe / leverage)


async def enforce_profit_guard(
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
) -> None:
    """Enforce profit protection rules (breakeven + ROE trailing)."""
    st = await state.get()
    if not st or st.mode == BotMode.IDLE:
        return

    if not st.last_side or not st.last_entry_price:
        return

    # Posición y precio actual
    pos = await exchange.read_active_position(symbol, st.last_side)
    if not pos:
        return

    price = await exchange.get_mark_price(symbol)
    symbol_cfg = CONFIG.get_symbol_config(symbol)
    lev = float(symbol_cfg.leverage)

    # ROE actual
    if st.last_side == Side.LONG:
        roe = (price - st.last_entry_price) / st.last_entry_price * lev
    else:
        roe = (st.last_entry_price - price) / st.last_entry_price * lev

    # Actualiza pico
    if roe > (st.peak_roe or 0):
        st.peak_roe = roe
        await state.set(st)
        logger.debug("peak_roe_updated", {"roe": roe})

    peak = float(st.peak_roe or 0.0)

    # ---- Breakeven lock (como ya tenías) ----
    if roe >= BE_AT_ROE:
        filters = await exchange.get_symbol_filters(symbol, symbol_cfg.leverage)
        current_stop = await exchange.open_stop_for_side(symbol, st.last_side)

        be_price = st.last_entry_price
        if st.last_side == Side.LONG:
            if current_stop and current_stop["stopPrice"] < be_price:
                new_stop = round_to_tick(
                    be_price + filters.tick_size, filters.tick_size, filters.price_precision
                )
                await exchange.cancel_order_by_id(symbol, current_stop["orderId"])
                await exchange.place_stop_close(symbol, st.last_side, new_stop)
                logger.info("stop_to_breakeven", {
                    "side": st.last_side.value,
                    "oldStop": current_stop["stopPrice"],
                    "newStop": new_stop,
                })
        else:
            if current_stop and current_stop["stopPrice"] > be_price:
                new_stop = round_to_tick(
                    be_price - filters.tick_size, filters.tick_size, filters.price_precision
                )
                await exchange.cancel_order_by_id(symbol, current_stop["orderId"])
                await exchange.place_stop_close(symbol, st.last_side, new_stop)
                logger.info("stop_to_breakeven", {
                    "side": st.last_side.value,
                    "oldStop": current_stop["stopPrice"],
                    "newStop": new_stop,
                })

    # ---- Trailing por ROE anclado al pico ----
    if PG_ENABLED and peak >= PG_ARM_ROE:
        keep = _trail_keep_ratio(peak)                 # p.ej. 0.70 con pico < 30%
        locked_roe = max(PG_MIN_LOCK, peak * keep)     # nunca menos de PG_MIN_LOCK
        trigger_roe = locked_roe - PG_HYSTERESIS       # pequeño colchón anti-ruido

        # Log de estado (útil para depurar)
        logger.debug("profit_guard_status", {
            "roe": roe,
            "peak": peak,
            "keep": keep,
            "lockedRoe": locked_roe,
            "triggerRoe": trigger_roe,
        })

        if roe <= trigger_roe:
            # Opción A: cierre inmediato a mercado (default)
            if not PG_USE_STOP:
                logger.warn("profit_trail_exit", {
                    "peakRoe": peak,
                    "exitRoe": roe,
                    "lockedRoe": locked_roe,
                })
                await exchange.close_side_market_safe(symbol, st.last_side, pos.qty_abs, pos.side_mode)
                if hasattr(exchange, "cancel_close_orders_for_side"):
                    try:
                        await exchange.cancel_close_orders_for_side(symbol, st.last_side)
                    except Exception as exc:  # pragma: no cover - logging only
                        logger.debug("profit_guard_cancel_fail", {"symbol": symbol, "err": str(exc)})
                st.mode = BotMode.IDLE
                st.last_exit_reason = "profit_trail"
                await state.set(st)
                logger.info("exited_profit_trail", {
                    "side": st.last_side.value,
                    "peakRoe": peak,
                    "exitRoe": roe,
                })
                return

            # Opción B: subir stop a nivel del locked_roe (reduce-only en el exchange)
            filters = await exchange.get_symbol_filters(symbol, symbol_cfg.leverage)
            target_price = _roe_to_price(st.last_entry_price, locked_roe, lev, st.last_side)
            target_price = round_to_tick(target_price, filters.tick_size, filters.price_precision)
            current_stop = await exchange.open_stop_for_side(symbol, st.last_side)

            # Solo movemos el stop si mejora la protección (nunca la empeora)
            if st.last_side == Side.LONG:
                must_move = (not current_stop) or (current_stop["stopPrice"] < target_price)
            else:
                must_move = (not current_stop) or (current_stop["stopPrice"] > target_price)

            if must_move:
                if current_stop:
                    await exchange.cancel_order_by_id(symbol, current_stop["orderId"])
                await exchange.place_stop_close(symbol, st.last_side, target_price)
                logger.info("profit_trail_stop_upserted", {
                    "side": st.last_side.value,
                    "targetStop": target_price,
                    "lockedRoe": locked_roe,
                })

        # Si usas también el viejo esquema de giveback, lo ignoramos cuando el trail está activo
        return

    # ---- (Opcional) Esquema de giveback anterior (si quieres mantenerlo como fallback) ----
    if GB_ARM_ROE is not None and GB_DROP_REL is not None and GB_DROP_MIN is not None:
        if peak >= GB_ARM_ROE:
            drop = peak - roe
            drop_rel = drop / peak if peak > 0 else 0.0
            if (drop_rel >= GB_DROP_REL) and (drop >= GB_DROP_MIN):
                logger.warn("profit_giveback_exit", {
                    "peakRoe": peak,
                    "currentRoe": roe,
                    "drop": drop,
                    "dropRel": drop_rel,
                })
                await exchange.close_side_market_safe(symbol, st.last_side, pos.qty_abs, pos.side_mode)
                if hasattr(exchange, "cancel_close_orders_for_side"):
                    try:
                        await exchange.cancel_close_orders_for_side(symbol, st.last_side)
                    except Exception as exc:  # pragma: no cover - logging only
                        logger.debug("profit_guard_cancel_fail", {"symbol": symbol, "err": str(exc)})
                st.mode = BotMode.IDLE
                st.last_exit_reason = "profit_giveback"
                await state.set(st)
                logger.info("exited_profit_giveback", {
                    "side": st.last_side.value,
                    "peakRoe": peak,
                    "exitRoe": roe,
                })
