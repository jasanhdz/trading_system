# -*- coding: utf-8 -*-
# src/strategies/trend_following.py
from __future__ import annotations
import math
from typing import Optional, Any, List
import numpy as np

from .base import Strategy
from ..core.types import Signal, BotState, Side
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger

from ..core.indicators.ema import ema
from ..core.indicators.adx import adx as adx_calc
from ..core.indicators.atr import atr
from ..core.utils.candles import last
from ..core.utils.bars import last_closed_and_series
from ..core.utils.features import compute_features

def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)

def _ema(values: List[float], n: int) -> List[float]:
    out = ema(values, n)
    return out if isinstance(out, list) else list(out)

class TrendFollowing(Strategy):
    def __init__(self):
        super().__init__(name="trend_following", timeframe="5m")

    async def evaluate(
        self,
        symbol: str,
        exchange: Exchange,
        config: Any,
        state: Optional[BotState],
        now: int,
        logger: Logger,
    ) -> Signal:
        tf = _cfg(config, "ENTRY_TIMEFRAME", "5m")

        raw = await exchange.get_candles(symbol, tf, 300)
        if len(raw) < 120:
            return {"action": "IDLE", "reason": "few_candles"}

        L, cs = last_closed_and_series(raw)

        closes = [c.close for c in cs]
        highs = [c.high for c in cs]
        lows = [c.low for c in cs]
        opens = [c.open for c in cs]

        ema7_arr  = _ema(closes, 7)
        ema25_arr = _ema(closes, 25)
        ema99_arr = _ema(closes, 99)

        ema7  = ema7_arr[-1]
        ema25 = ema25_arr[-1]
        ema99 = ema99_arr[-1]

        adx_res = adx_calc(highs, lows, closes, 14)
        adx_now  = float(adx_res.get("adx", 0.0))
        di_plus  = float(adx_res.get("plus_di", 0.0))
        di_minus = float(adx_res.get("minus_di", 0.0))

        atr_len = int(_cfg(config, "ATR_LEN", 14))
        atr_now = float(atr(highs, lows, closes, atr_len)[-1]) if len(closes) > atr_len else 0.0

        ema7_prev  = ema7_arr[-2] if len(ema7_arr)  >= 2 else ema7
        ema25_prev = ema25_arr[-2] if len(ema25_arr) >= 2 else ema25
        ema7_up  = ema7 > ema7_prev
        ema25_up = ema25 > ema25_prev

        adx_min = float(_cfg(config, "TF_ADX_MIN", 22.0))
        NO_TRADE_BAND = float(_cfg(config, "NO_TRADE_BAND_AROUND_EMA_SLOW", 0.003))

        # # banda no-trade cerca de EMA99
        # if NO_TRADE_BAND > 0 and ema99 > 0:
        #     if abs(L.close - ema99) / ema99 <= NO_TRADE_BAND:
        #         return {"action": "IDLE", "reason": "tf_near_ema99"}

        # fuerza de tendencia + direccionalidad DI
        di_ok_long = (di_plus > di_minus) if (di_plus or di_minus) else True
        trend_up = (ema25 > ema99) and (adx_now >= adx_min) and di_ok_long

        # definición de pullback auténtico
        in_pb_zone = (L.close <= max(ema7, ema25)) and (L.close >= min(ema7, ema25))
        touched_ema7_prev = (len(cs) >= 2) and (cs[-2].low <= ema7_arr[-2] or cs[-2].close <= ema7_arr[-2])

        # confirmación (rebote + momentum suave)
        close_above_ema7 = (L.close > ema7)
        break_prev_high = (len(highs) >= 2 and L.close > highs[-2])
        resume_confirm = touched_ema7_prev and close_above_ema7 and ema7_up and (break_prev_high or ema25_up)

        # anti-knife
        body = abs(L.close - (cs[-1].open if len(cs)>=1 else L.close))
        big_red = (atr_now > 0) and (L.close < (cs[-1].open if len(cs)>=1 else L.close)) and (body > 1.2 * atr_now)

        # cooldown
        tf_min_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        tf_min = tf_min_map.get(tf, 5)
        bar_ms = tf_min * 60 * 1000

        cooldown_bars = int(_cfg(config, "TF_COOLDOWN_BARS", 6))
        def bars_since(ms_then: Optional[int]) -> int:
            if not ms_then:
                return 10 ** 9
            return max(0, (now - int(ms_then)) // bar_ms)

        long_cool_ok = True
        if state:
            long_cool_ok = bars_since(getattr(state, "tf_last_long_at", None)) >= cooldown_bars

        allow_pyramiding = bool(_cfg(config, "ALLOW_PYRAMIDING", False))
        if not allow_pyramiding and state and state.last_side == Side.LONG and state.mode and "RIDE" in state.mode.value:
            return {"action": "IDLE", "reason": "pyramiding_disabled"}

        # RSI (para logging / filtro ligero)
        feats = compute_features(cs)
        rsi = float(feats.get("rsi_14", 50.0))

        long_ok = (
            _cfg(config, "ALLOW_LONGS", True)
            and trend_up
            and in_pb_zone
            and resume_confirm
            and not big_red
            and long_cool_ok
        )

        allow_shorts = bool(_cfg(config, "ALLOW_SHORTS", False))
        short_ok = False
        if allow_shorts:
            di_ok_short = (di_minus > di_plus) if (di_plus or di_minus) else True
            trend_down = (ema25 < ema99) and (adx_now >= adx_min) and di_ok_short
            in_pb_zone_s = (L.close >= min(ema7, ema25)) and (L.close <= max(ema7, ema25))
            close_below_ema7 = (L.close < ema7)
            break_prev_low = (len(lows) >= 2 and L.close < lows[-2])
            ema7_down = ema7 < ema7_prev
            ema25_down = ema25 < ema25_prev
            touched_ema7_prev_s = (len(cs) >= 2) and (cs[-2].high >= ema7_arr[-2] or cs[-2].close >= ema7_arr[-2])
            resume_confirm_s = touched_ema7_prev_s and close_below_ema7 and ema7_down and (break_prev_low or ema25_down)
            big_green = (atr_now > 0) and (L.close > (cs[-1].open if len(cs)>=1 else L.close)) and (body > 1.2 * atr_now)

            short_cool_ok = True
            if state:
                short_cool_ok = bars_since(getattr(state, "tf_last_short_at", None)) >= cooldown_bars

            if not allow_pyramiding and state and state.last_side == Side.SHORT and state.mode and "RIDE" in state.mode.value:
                pass
            else:
                short_ok = (
                    allow_shorts
                    and trend_down
                    and in_pb_zone_s
                    and resume_confirm_s
                    and not big_green
                    and short_cool_ok
                )

        logger.info("tf_filters", {
            "trend": int(trend_up if not allow_shorts else (trend_up or (ema25 < ema99 and adx_now >= adx_min))),
            "pb": int(in_pb_zone),
            "rsi": float(rsi),
            "adx": round(adx_now, 1),
        })


        if long_ok:
            if state:
                setattr(state, "tf_last_long_at", int(now))
            return {"action": "ENTER_LONG", "reason": f"TF_long pb ema7~ema25 bounce=True adx={adx_now:.1f} rsi={rsi:.1f}"}

        if short_ok:
            if state:
                setattr(state, "tf_last_short_at", int(now))
            return {"action": "ENTER_SHORT", "reason": f"TF_short pb ema7~ema25 bounce=True adx={adx_now:.1f} rsi={rsi:.1f}"}

        return {"action": "IDLE", "reason": "tf_no_entry"}
