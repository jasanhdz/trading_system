"""Strategy runner module."""

import os
import time
from typing import Any, Optional, TYPE_CHECKING

from ..core.utils.candles import get_highs, get_lows, get_closes
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from ..core.ports.state_store import StateStore
from ..core.types import BotMode, Side
from ..core.risk.sizing import size_by_budget, floor_to_step, ceil_to_step
from ..core.risk.stop import compute_stop_from_liq_ticks, round_to_tick
from ..core.indicators.atr import atr
from ..strategies.base import Strategy
if TYPE_CHECKING:
    from ..infra.config import Config, SymbolSettings


class StrategyRunner:
    """Executes trading strategies."""

    def __init__(
        self,
        exchange: Exchange,
        logger: Logger,
        state: StateStore,
        strategy: Strategy,
        config: "Config",
        symbol_config: "SymbolSettings",
    ):
        """Initialize strategy runner."""
        self.exchange = exchange
        self.logger = logger
        self.state = state
        self.strategy = strategy
        self.config = config
        self.symbol_config = symbol_config

    async def tick(self, symbol: str) -> None:
        """Execute one strategy tick."""
        st_before = await self.state.get()
        self.logger.debug(
            "state_snapshot",
            {
                "mode": st_before.mode.value if st_before else "IDLE",
                "lastSide": st_before.last_side.value if st_before and st_before.last_side else None,
            },
        )

        mark_price_cached: Optional[float] = None
        active_position = None

        # Get signal from strategy
        sig = await self.strategy.evaluate(
            symbol=symbol,
            exchange=self.exchange,
            config=self.config,
            state=st_before,
            now=int(time.time() * 1000),
            logger=self.logger,
        )

        self.logger.info("signal", {"symbol": symbol, **sig})

        has_active_position = (
            st_before
            and st_before.mode != BotMode.IDLE
            and st_before.last_side is not None
        )
        if has_active_position:
            try:
                active_position = await self.exchange.read_active_position(symbol, st_before.last_side)  # type: ignore[arg-type]
            except Exception as exc:
                self.logger.warn("position_read_fail", {"symbol": symbol, "err": str(exc)})

        if has_active_position and st_before and st_before.last_side:
            qty_abs = 0.0
            entry_price = None
            leverage_used = None
            if active_position:
                qty_abs = active_position.qty_abs or 0.0
                entry_price = active_position.entry_price or entry_price
                leverage_used = active_position.leverage or leverage_used
            if st_before.last_entry_qty:
                qty_abs = qty_abs or st_before.last_entry_qty
            if st_before.last_entry_price:
                entry_price = entry_price or st_before.last_entry_price
            if st_before.last_leverage:
                leverage_used = leverage_used or st_before.last_leverage
            leverage_used = leverage_used or self.symbol_config.leverage

            if qty_abs and entry_price:
                if mark_price_cached is None:
                    mark_price_cached = await self.exchange.get_mark_price(symbol)
                mark = mark_price_cached
                pnl_usd = None
                if mark is not None:
                    direction = 1 if st_before.last_side == Side.LONG else -1
                    pnl_usd = (mark - entry_price) * qty_abs * direction
                margin = None
                if mark is not None:
                    notional = mark * qty_abs
                    margin = notional / max(1, leverage_used)
                roi_pct = None
                if pnl_usd is not None and margin:
                    roi_pct = (pnl_usd / max(1e-9, margin)) * 100
                open_ms = (
                    int(time.time() * 1000) - st_before.last_entry_at
                    if st_before.last_entry_at
                    else None
                )
                self.logger.info(
                    "position_snapshot",
                    {
                        "symbol": symbol,
                        "side": st_before.last_side.value,
                        "entry": entry_price,
                        "mark": mark,
                        "leverage": leverage_used,
                        "qtyAbs": qty_abs,
                        "roiPct": roi_pct,
                        "pnlUsd": pnl_usd,
                        "openMs": open_ms,
                    },
                )
                if sig["action"] in ("ENTER_LONG", "ENTER_SHORT"):
                    self.logger.debug(
                        "entry_blocked_existing_position",
                        {"symbol": symbol, "requested": sig["action"], "reason": "existing_position"},
                    )
                    return

        # Handle EXIT signal
        if sig["action"] == "EXIT":
            if st_before and st_before.last_side:
                pos = await self.exchange.read_active_position(symbol, st_before.last_side)
                if pos:
                    self.logger.info(
                        "exit_request",
                        {
                            "side": st_before.last_side.value,
                            "qtyAbs": pos.qty_abs,
                        },
                    )
                    await self.exchange.close_side_market_safe(
                        symbol, st_before.last_side, pos.qty_abs, pos.side_mode
                    )
                    if hasattr(self.exchange, "cancel_close_orders_for_side"):
                        try:
                            await self.exchange.cancel_close_orders_for_side(symbol, st_before.last_side)
                        except Exception as exc:
                            self.logger.debug(
                                "cancel_brackets_fail",
                                {"symbol": symbol, "side": st_before.last_side.value, "err": str(exc)},
                            )
                    st_before.mode = BotMode.IDLE
                    st_before.last_exit_reason = sig.get("reason", "exit_by_strategy")
                    await self.state.set(st_before)
                    self.logger.info("exit_done", {"reason": sig.get("reason")})
            return

        # Handle IDLE signal
        if sig["action"] == "IDLE":
            self.logger.debug("idle_noop")
            return

        # --- Handle ENTRY signals ---
        side = Side.LONG if sig["action"] == "ENTER_LONG" else Side.SHORT
        leverage = self.symbol_config.leverage

        await self.exchange.set_leverage(symbol, leverage)
        if mark_price_cached is None:
            price = await self.exchange.get_mark_price(symbol)
        else:
            price = mark_price_cached
        filters = await self.exchange.get_symbol_filters(symbol, leverage)

        self.logger.debug("filters", vars(filters))

        usdt = await self.exchange.get_usdt_balance()

        # Daily drawdown kill-switch
        dd_max = getattr(self.config, "DAILY_DD_MAX_PCT", 0.0)
        if dd_max > 0:
            sod_key = f"BAL_SOD_{symbol}"
            sod = float(os.environ.get(sod_key, str(usdt)))
            os.environ[sod_key] = str(sod)
            dd = (sod - usdt) / max(1e-9, sod)
            if dd >= dd_max:
                self.logger.warn(
                    "daily_kill_switch",
                    {
                        "dd": dd,
                        "sod": sod,
                        "bal": usdt,
                    },
                )
                return

        # Base sizing by budget
        sized = size_by_budget(
            usdt_balance=usdt,
            reserve=self.config.MIN_WALLET_RESERVE_USDT,
            capital_pct=self.symbol_config.capital_usage_pct,
            price=price,
            leverage=leverage,
            fee_pct=self.config.FEE_BUFFER_PCT,
            filters=filters,
        )

        if sized.get("qty", 0) == 0:
            self.logger.warn("sizing_rejected", sized)
            return

        qty = sized["qty"]
        capital_pct_used = sized.get("capital_pct_used", self.symbol_config.capital_usage_pct)
        self.logger.info(
            "sizing_plan",
            {
                "symbol": symbol,
                "side": side.value,
                "capitalPctRequested": self.symbol_config.capital_usage_pct,
                "capitalPctUsed": capital_pct_used,
                "capitalRequested": sized.get("capital_requested"),
                "capitalUsed": sized.get("capital"),
                "qty": qty,
                "price": price,
                "notional": qty * price,
                "usdtBalance": usdt,
            },
        )

        # Risk overlay: limit qty by provisional stop (ATR-based)
        max_risk_pct = getattr(self.config, "MAX_RISK_PCT", 0.0)
        if max_risk_pct > 0:
            candles = await self.exchange.get_candles(symbol, self.config.ENTRY_TIMEFRAME, 200)
            a_arr = atr(
                get_highs(candles),
                get_lows(candles),
                get_closes(candles),
                getattr(self.config, "ATR_LEN", 14),
            )
            a = float(a_arr[-1]) if a_arr.size > 0 else 0.0

            if a is not None and a > 0:
                base_mult = getattr(
                    self.config,
                    "TRAIL_ATR_MULT_BASE",
                    getattr(self.config, "TRAIL_ATR_MULT", 2.5),
                )

                if side == Side.LONG:
                    plan_stop = price - base_mult * a
                    plan_stop = min(plan_stop, price - filters.tick_size)
                else:
                    plan_stop = price + base_mult * a
                    plan_stop = max(plan_stop, price + filters.tick_size)

                plan_stop = round_to_tick(plan_stop, filters.tick_size, filters.price_precision)

                stop_dist = (
                    max(0, price - plan_stop)
                    if side == Side.LONG
                    else max(0, plan_stop - price)
                )

                if stop_dist > 0:
                    risk_usdt = usdt * max_risk_pct
                    qty_by_risk = floor_to_step(
                        risk_usdt / stop_dist,
                        filters.step_size,
                        filters.qty_precision,
                    )

                    if qty_by_risk <= 0:
                        self.logger.warn(
                            "sizing_rejected_by_risk",
                            {
                                "stopDist": stop_dist,
                                "riskUSDT": risk_usdt,
                            },
                        )
                        return

                    if qty > qty_by_risk:
                        self.logger.info(
                            "sizing_capped_by_risk",
                            {
                                "from": qty,
                                "to": qty_by_risk,
                                "stopDist": stop_dist,
                                "riskPct": max_risk_pct,
                            },
                        )
                        qty = qty_by_risk

        # Re-validate minimum notional after risk cap
        min_qty_by_notional = ceil_to_step(
            filters.min_notional / price,
            filters.step_size,
            filters.qty_precision,
        )

        if qty < min_qty_by_notional:
            self.logger.warn(
                "min_notional_not_met_after_risk",
                {
                    "qty": qty,
                    "minQtyByNotional": min_qty_by_notional,
                    "price": price,
                    "notional": round(qty * price, filters.price_precision),
                    "minNotional": filters.min_notional,
                },
            )
            return

        self.logger.info(
            "sizing_ok",
            {
                "side": side.value,
                "qty": qty,
                "price": price,
                "usdt": usdt,
            },
        )

        # Open market position
        t_open = int(time.time() * 1000)
        result = await self.exchange.market_open(symbol, side, qty)
        avg_price = result.avg_price or price

        self.logger.info(
            "market_opened",
            {
                "side": side.value,
                "qty": qty,
                "price": price,
                "avgPrice": avg_price,
                "ms": int(time.time() * 1000) - t_open,
            },
        )

        # Set initial stop loss
        ticks = self.config.SL_TICKS_ABOVE_LIQ_MAP.get(symbol, self.config.SL_TICKS_ABOVE_LIQ_DEFAULT)
        liq = await self.exchange.read_liquidation_price(symbol, side) or price

        stop = compute_stop_from_liq_ticks(
            side=side,
            liq_price=liq,
            current_price=price,
            entry_price=avg_price,
            tick_size=filters.tick_size,
            price_precision=filters.price_precision,
            ticks_above_liq=ticks,
        )

        await self.exchange.place_stop_close(symbol, side, stop)
        self.logger.info(
            "stop_upserted",
            {
                "side": side.value,
                "stop": stop,
                "liq": liq,
                "ticks": ticks,
            },
        )

        # Set take profit
        r = self.config.TP_ROE
        fee = self.config.FEE_BUFFER_PCT

        if side == Side.LONG:
            tp_raw = avg_price * (1 + r / leverage + fee)
        else:
            tp_raw = avg_price * (1 - r / leverage - fee)

        tp = round_to_tick(tp_raw, filters.tick_size, filters.price_precision)

        await self.exchange.place_tp_close(symbol, side, tp)
        self.logger.info(
            "tp_upserted",
            {
                "side": side.value,
                "tp": tp,
                "roe": r,
            },
        )

        # Update state
        if not st_before:
            from ..core.types import BotState

            st_before = BotState(mode=BotMode.IDLE)

        st_before.mode = BotMode.LONG_RIDE if side == Side.LONG else BotMode.SHORT_RIDE
        st_before.last_side = side
        st_before.last_entry_price = avg_price
        st_before.last_leverage = leverage
        st_before.last_entry_at = int(time.time() * 1000)
        st_before.peak_roe = 0.0
        st_before.brackets_armed_at = int(time.time() * 1000)
        st_before.last_entry_qty = qty
        st_before.pyramid_units = 0
        st_before.last_pyramid_price = avg_price
        st_before.last_trail_stop = None

        await self.state.set(st_before)

        self.logger.info(
            "state_entered",
            {
                "mode": st_before.mode.value,
            },
        )
