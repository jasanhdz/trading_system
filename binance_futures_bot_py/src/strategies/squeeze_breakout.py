# -*- coding: utf-8 -*-
# src/strategies/squeeze_breakout.py
from __future__ import annotations
import math
from typing import Any, Optional, Dict, List
import numpy as np

from .base import Strategy
from ..core.types import Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from ..core.utils.candles import last, volume_avg
from ..core.utils.bars import last_closed_and_series
from ..core.indicators.ema import ema
from ..core.indicators.adx import adx as adx_calc

def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)

def _stdev(arr: List[float], n: int = 20) -> float:
    k = min(len(arr), n)
    if k <= 1:
        return float("nan")
    s = arr[-k:]
    m = float(np.mean(s))
    v = sum((x - m) ** 2 for x in s) / (k - 1)
    return math.sqrt(max(0.0, v))

def _atr(high: List[float], low: List[float], close: List[float], n: int = 14) -> float:
    if len(close) < n + 1:
        return float("nan")
    trs = []
    for i in range(1, len(close)):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        trs.append(max(a, b, c))
    arr = trs[-n:]
    v = arr[0]
    k = 1.0 / n
    for i in range(1, len(arr)):
        v = v * (1 - k) + arr[i] * k
    return float(v)

class SqueezeBreakout(Strategy):
    def __init__(self):
        super().__init__(name="squeeze_breakout", timeframe="5m")

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
        highs  = [c.high  for c in cs]
        lows   = [c.low   for c in cs]
        vols   = [c.volume for c in cs]

        ema25 = ema(closes, 25)[-1]
        ema99 = ema(closes, 99)[-1]
        ema_up   = ema25 > ema99
        ema_down = ema25 < ema99

        ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
        sd20 = _stdev(closes, 20)
        upper = ma20 + 2 * sd20
        lower = ma20 - 2 * sd20
        bbw = (upper - lower) / ma20 if (ma20 > 0 and not math.isnan(ma20)) else float("nan")

        # Historial de BW para percentil y expansión
        bbw_hist = []
        for i in range(20, len(closes) + 1):
            m = float(np.mean(closes[i - 20:i]))
            s = _stdev(closes[:i], 20)
            u, l = m + 2 * s, m - 2 * s
            bbw_hist.append((u - l) / m if m > 0 and not math.isnan(m) else float("nan"))
        bbw_hist = [x for x in bbw_hist if math.isfinite(x)]
        if len(bbw_hist) < 60:
            return {"action": "IDLE", "reason": "bbw_history_short"}

        bbw_pctl = np.percentile(bbw_hist[-120:], _cfg(config, "SB_BBW_PCTL", 25))
        bbw_prev = float(bbw_hist[-2]) if len(bbw_hist) >= 2 else float("nan")
        expanding = (math.isfinite(bbw_prev) and math.isfinite(bbw) and bbw > bbw_prev)

        adx_now = float(adx_calc(highs, lows, closes, 14).get("adx", 0.0))
        atr_now = _atr(highs, lows, closes, 14)

        dc_len = int(_cfg(config, "SB_DC_LEN", 20))
        dc_high = max(highs[-dc_len:])
        dc_low  = min(lows[-dc_len:])

        vavg = volume_avg(cs, max(20, _cfg(config, "VOL_AVG_LEN", 20)))
        vol_mult = (L.volume / vavg) if vavg > 0 else 1.0

        ADX_MIN_EXPAND   = float(_cfg(config, "SB_ADX_MIN", 18.0))
        VOL_MIN          = float(_cfg(config, "SB_VOL_MIN", 1.2))
        VOL_MAX          = float(_cfg(config, "SB_VOL_MAX", 3.5))
        EPS              = float(_cfg(config, "SB_EPS", 0.001))
        MAX_ATR_CHASE    = float(_cfg(config, "SB_MAX_ATR_CHASE", 1.2))
        REQUIRE_SQUEEZE  = bool(_cfg(config, "SB_REQUIRE_SQUEEZE", True))
        NO_TRADE_BAND    = float(_cfg(config, "NO_TRADE_BAND_AROUND_EMA_SLOW", 0.003))

        # Banda no-trade en EMA99
        if NO_TRADE_BAND > 0 and ema99 > 0:
            if abs(L.close - ema99) / ema99 <= NO_TRADE_BAND:
                return {"action": "IDLE", "reason": "sb_near_ema99"}

        in_squeeze = (not math.isnan(bbw)) and (bbw <= bbw_pctl) and (adx_now <= ADX_MIN_EXPAND)

        dist_from_ema = abs(L.close - ema25) / (atr_now if atr_now > 0 else 1.0)
        long_break = (L.close >= max(upper, dc_high * (1 + EPS)))
        long_ok = (
            ema_up and long_break and expanding and
            (vol_mult >= VOL_MIN) and (vol_mult <= VOL_MAX) and
            (dist_from_ema <= MAX_ATR_CHASE) and
            (in_squeeze if REQUIRE_SQUEEZE else True)
        )
        if long_ok:
            return {
                "action": "ENTER_LONG",
                "reason": f"SB_long bbw={bbw:.3f}<=pctl({bbw_pctl:.3f})↑ vol={vol_mult:.2f} adx={adx_now:.1f} distATR={dist_from_ema:.2f}",
            }

        short_break = (L.close <= min(lower, dc_low * (1 - EPS)))
        short_ok = (
            ema_down and short_break and expanding and
            (vol_mult >= VOL_MIN) and (vol_mult <= VOL_MAX) and
            (dist_from_ema <= MAX_ATR_CHASE) and
            (in_squeeze if REQUIRE_SQUEEZE else True)
        )
        if short_ok:
            return {
                "action": "ENTER_SHORT",
                "reason": f"SB_short bbw={bbw:.3f}<=pctl({bbw_pctl:.3f})↑ vol={vol_mult:.2f} adx={adx_now:.1f} distATR={dist_from_ema:.2f}",
            }

        return {"action": "IDLE", "reason": f"SB_idle bbw={bbw:.3f} pctl={bbw_pctl:.3f} exp={bool(expanding)} adx={adx_now:.1f} vol={vol_mult:.2f}"}
