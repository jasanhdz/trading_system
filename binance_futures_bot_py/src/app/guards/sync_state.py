# src/app/guards/sync_state.py
from typing import Any, Optional
from ...core.types import BotMode, Side

def _get_entry_price_from_pos(pos: Any) -> Optional[float]:
    """
    Intenta obtener el precio de entrada desde distintas formas de posición
    (objeto con atributos, dict tipo CCXT, etc).
    """
    cand = []
    for k in ("entry_price", "avg_price", "entry", "avgPrice", "entryPrice"):
        v = getattr(pos, k, None) if hasattr(pos, k) else (pos.get(k) if isinstance(pos, dict) else None)
        if v is not None:
            try:
                cand.append(float(v))
            except Exception:
                pass
    return cand[0] if cand else None

def _side_of_pos(pos: Any) -> Optional[Side]:
    """
    Obtiene el Side a partir de la posición (propiedad explícita o por qty).
    """
    s = getattr(pos, "side", None) or (pos.get("side") if isinstance(pos, dict) else None)
    if isinstance(s, Side):
        return s
    if isinstance(s, str):
        s = s.upper()
        if s in ("LONG", "BUY"):
            return Side.LONG
        if s in ("SHORT", "SELL"):
            return Side.SHORT
    # Fallback por cantidad
    qty = getattr(pos, "qty_abs", None)
    if qty is None and isinstance(pos, dict):
        qty = pos.get("positionAmt") or pos.get("qty") or pos.get("qty_abs")
    try:
        if qty is not None and float(qty) > 0:
            # Si no hay side explícito, asumimos LONG si mark < entry o al menos LONG por convención
            # (en modo hedge real tu exchange debería darnos el side explícitamente)
            # Aquí no inferimos por PnL, solo decimos: “hay una posición”. Si tienes oneway, con una sola basta.
            pass
    except Exception:
        pass
    return None  # Si el exchange ya expone read_active_position(side) no necesitamos inferir

async def sync_state_guard(
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
) -> None:
    """
    Sincroniza el estado con posiciones abiertas (aunque se abran manualmente).
    - Si encuentra LONG o SHORT, adjunta el bot a esa posición.
    - Si no encuentra ninguna, deja el estado en IDLE.
    """
    st = await state.get()

    # Intentamos leer ambas (soporta hedge/onway)
    long_pos = await exchange.read_active_position(symbol, Side.LONG)
    short_pos = await exchange.read_active_position(symbol, Side.SHORT)

    pos = long_pos or short_pos
    if pos:
        side = Side.LONG if long_pos else Side.SHORT
        entry = _get_entry_price_from_pos(pos)
        qty_abs = getattr(pos, "qty_abs", None)
        lev = getattr(pos, "leverage", None)

        # Si el estado está vacío o no coincide, lo rearmamos
        needs_attach = (
            not st
            or st.mode == BotMode.IDLE
            or st.last_side != side
            or not st.last_entry_price
        )

        if needs_attach and entry:
            if not st:
                # crea contenedor mínimo
                from types import SimpleNamespace
                st = SimpleNamespace()

            st.mode = BotMode.LONG_RIDE if side == Side.LONG else BotMode.SHORT_RIDE
            st.last_side = side
            st.last_entry_price = float(entry)
            # No armamos brackets aquí; dejamos que brackets_guard haga su trabajo
            if getattr(st, "brackets_armed_at", None) is None:
                st.brackets_armed_at = None

            await state.set(st)
            logger.info("sync_attach_to_open_position", {
                "side": side.value,
                "entry": float(entry),
                "lev": lev if lev is not None else "-",
                "qtyAbs": float(qty_abs) if qty_abs is not None else "-",
            })
        return

    # Si no hay posiciones vivas y el bot “creía” estar adentro, reseteamos
    if st and st.mode != BotMode.IDLE:
        st.mode = BotMode.IDLE
        st.last_side = None
        st.last_entry_price = None
        st.peak_roe = None
        st.brackets_armed_at = None
        await state.set(st)
        logger.info("sync_no_position")
