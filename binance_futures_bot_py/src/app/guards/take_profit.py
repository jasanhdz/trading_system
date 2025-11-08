"""Take profit guard - monitors and handles take profit conditions."""

import time
from typing import Any
from ...core.types import BotMode
from ...infra.config import CONFIG


async def check_take_profit(
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
) -> None:
    """Check if position was closed by take profit."""
    st = await state.get()
    if not st or st.mode == BotMode.IDLE:
        return
    
    if not st.last_side:
        return
    
    # Check if position still exists
    pos = await exchange.read_active_position(symbol, st.last_side)
    
    if not pos:
        # Position closed, likely by TP or SL
        logger.info("position_closed_detected", {
            "side": st.last_side.value,
            "lastEntry": st.last_entry_price,
        })
        
        # Check if we should re-enter (if TP was hit and reenter is enabled)
        if CONFIG.REENTER_ON_TP and st.last_tp_at:
            cooldown = CONFIG.REENTER_COOLDOWN_MS
            if int(time.time() * 1000) - st.last_tp_at < cooldown:
                logger.debug("reenter_cooldown_active", {
                    "remaining": cooldown - (int(time.time() * 1000) - st.last_tp_at),
                })
            else:
                # Could trigger re-entry logic here
                logger.debug("reenter_cooldown_expired")
        
        # Update state to IDLE
        st.mode = BotMode.IDLE
        st.last_exit_reason = "tp_or_sl"
        st.last_exit_at = int(time.time() * 1000)
        await state.set(st)
        
        logger.info("state_to_idle", {"reason": "position_closed"})
    else:
        # Position still open, check if TP was partially filled
        tp_order = await exchange.open_tp_for_side(symbol, st.last_side)
        if not tp_order and st.brackets_armed_at:
            # TP might have been hit
            st.last_tp_at = int(time.time() * 1000)
            await state.set(st)
            logger.info("tp_possibly_hit", {"side": st.last_side.value})
