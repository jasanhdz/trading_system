# -*- coding: utf-8 -*-
# src/strategies/mean_reversion.py
import math
from typing import List, Optional, Any
import numpy as np

from .base import Strategy
from ..core.types import Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from ..core.utils.candles import last, volume_avg
from ..core.utils.bars import last_closed_and_series
from ..core.indicators.ema import ema
from ..core.indicators.adx import adx as adx_calc
from ..core.utils.features import compute_features

def stdev(arr: List[float], n: int = 20) -> float:
    k = min(len(arr), n)
    if k <= 1:
        return float('nan')
    s = arr[-k:]
    m = sum(s) / k
    v = sum((x - m) ** 2 for x in s) / (k - 1)
    return math.sqrt(max(0, v))

async def confirm_bear_on_1h(exchange: Exchange, symbol: str, adx_min: float) -> bool:
    candles_1h = await exchange.get_candles(symbol, "1h", 200)
    if len(candles_1h) < 30:
        return False
    closes = [c.close for c in candles_1h]
    highs = [c.high for c in candles_1h]
    lows  = [c.low  for c in candles_1h]
    ema25_arr = ema(closes, 25)
    ema99_arr = ema(closes, 99)
    ema25_1h = ema25_arr[-1]
    ema99_1h = ema99_arr[-1]
    a = adx_calc(highs, lows, closes, 14).get("adx", 0.0)
    return (ema25_1h < ema99_1h) and (a >= adx_min)

class MeanReversion(Strategy):
    def __init__(self):
        super().__init__(name="mean_reversion", timeframe="5m")

    async def evaluate(
        self,
        symbol: str,
        exchange: Exchange,
        config: Any,
        state: Optional[BotState],
        now: int,
        logger: Logger,
    ) -> Signal:
        raw = await exchange.get_candles(symbol, config.ENTRY_TIMEFRAME, 300)
        if len(raw) < 60:
            return {"action": "IDLE", "reason": "few_candles"}

        L, cs = last_closed_and_series(raw)

        closes = [c.close for c in cs]
        highs  = [c.high  for c in cs]
        lows   = [c.low   for c in cs]

        adx_now = float(adx_calc(highs, lows, closes, 14).get("adx", 0.0))

        if len(closes) >= 20:
            ma20 = float(np.mean(closes[-20:]))
        else:
            ma20 = float(np.mean(closes))

        sd20 = stdev(closes, 20)
        upper = ma20 + 2 * sd20
        lower = ma20 - 2 * sd20
        bandwidth = (upper - lower) / ma20 if (ma20 > 0 and not math.isnan(ma20)) else float('nan')
        bw_str = f"{bandwidth:.3f}" if not math.isnan(bandwidth) else "NaN"

        is_range = (
            adx_now <= config.MR_ADX_MAX
            and not math.isnan(bandwidth)
            and bandwidth <= config.MR_BB_WIDTH_MAX
        )
        if not is_range:
            return {"action": "IDLE", "reason": f"no_range adx={adx_now:.1f} bw={bw_str}"}

        vavg = volume_avg(cs, max(20, config.VOL_AVG_LEN))
        spike = (vavg > 0) and (L.volume >= config.MR_SPIKE_VOL_FACTOR * vavg)
        if spike:
            mult = (L.volume / vavg) if vavg > 0 else 0.0
            return {"action": "IDLE", "reason": f"spike_vol={mult:.2f}x"}

        feats = compute_features(cs)
        rsi = float(feats.get("rsi_14", 50.0))

        eps = config.MR_TOUCH_EPS
        near_lower = (not math.isnan(lower)) and (L.close <= lower * (1 + eps))
        near_upper = (not math.isnan(upper)) and (L.close >= upper * (1 - eps))

        from ..core.utils.candles import count_streak
        red_flags = [c.close < c.open for c in cs]
        green_flags = [c.close > c.open for c in cs]
        red_streak = count_streak(red_flags)
        green_streak = count_streak(green_flags)
        need_streak = max(1, config.MR_MIN_STREAK)

        ema25_5m = ema(closes, 25)[-1]
        ema99_5m = ema(closes, 99)[-1]
        bull_ma = ema25_5m > ema99_5m
        bear_ma = ema25_5m < ema99_5m

        # z-score para asegurar desvío real
        z = (L.close - ma20) / (sd20 if sd20 > 0 else 1e-12)

        long_ok = (
            config.ALLOW_LONGS
            and near_lower
            and z <= -1.3
            and rsi <= config.MR_RSI_LOW
            and red_streak >= need_streak
            and not bear_ma
        )
        if long_ok:
            return {"action": "ENTER_LONG",
                    "reason": f"MR_long rsi={rsi:.1f} z={z:.2f} streak={red_streak} bw={bw_str} adx={adx_now:.1f}"}

        extra_short_ok = True
        if config.MR_STRICT_SHORTS:
            if config.MR_SHORT_CONFIRM_1H:
                extra_short_ok = await confirm_bear_on_1h(exchange, symbol, config.MR_SHORT_1H_ADX_MIN)
            else:
                extra_short_ok = not bull_ma

        short_ok = (
            config.ALLOW_SHORTS
            and near_upper
            and z >= 1.3
            and rsi >= config.MR_RSI_HIGH
            and green_streak >= need_streak
            and extra_short_ok
        )
        if short_ok:
            return {"action": "ENTER_SHORT",
                    "reason": (f"MR_short rsi={rsi:.1f} z={z:.2f} streak={green_streak} bw={bw_str} "
                               f"adx={adx_now:.1f} 1hOK={extra_short_ok if config.MR_SHORT_CONFIRM_1H else True}")}

        return {"action": "IDLE",
                "reason": (f"mr_filters nearL={near_lower} nearU={near_upper} rsi={rsi:.1f} z={z:.2f} "
                           f"red={red_streak} green={green_streak} bw={bw_str} adx={adx_now:.1f}") }
