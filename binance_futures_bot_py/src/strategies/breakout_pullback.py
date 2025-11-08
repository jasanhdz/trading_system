# -*- coding: utf-8 -*-
# src/strategies/breakout_pullback.py
from __future__ import annotations
import math
from typing import Any, Optional
import numpy as np

from .base import Strategy
from ..core.types import Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from ..core.utils.candles import last, volume_avg
from ..core.utils.bars import last_closed_and_series
from ..core.indicators.ema import ema
from ..core.indicators.atr import atr
from ..core.indicators.adx import adx as adx_calc
from ..core.utils.features import compute_features


def _pct(a: float, b: float) -> float:
    if b == 0 or math.isnan(a) or math.isnan(b):
        return float("nan")
    return (a / b) - 1.0


def _safe_num(x):
    try:
        if x is True: return 1
        if x is False: return 0
        if x is None or (isinstance(x, float) and math.isnan(x)): return None
        return float(x)
    except Exception:
        return None


class BreakoutPullback(Strategy):
    """
    Híbrida:
      1) Breakout de rango con volumen + ADX/DI.
      2) Pullback a EMA25 tras breakout.
      3) Continuation cerca de EMA25 con ADX alto.
    """

    def __init__(self):
        super().__init__(name="breakout_pullback", timeframe="5m")

    async def evaluate(
        self,
        symbol: str,
        exchange: Exchange,
        config: Any,
        state: Optional[BotState],
        now: int,
        logger: Logger,
    ) -> Signal:
        tf = getattr(config, "ENTRY_TIMEFRAME", "5m")
        raw = await exchange.get_candles(symbol, tf, 300)
        if len(raw) < 120:
            return {"action": "IDLE", "reason": "few_candles"}

        # Usar siempre vela cerrada
        L, cs = last_closed_and_series(raw)

        closes = [c.close for c in cs]
        highs  = [c.high  for c in cs]
        lows   = [c.low   for c in cs]

        # --- Config ---
        BRK_LOOKBACK       = getattr(config, "BRK_LOOKBACK", 24)
        BRK_RANGE_MAX_W    = getattr(config, "BRK_RANGE_MAX_W", 0.025)
        BRK_BREAK_VOL      = getattr(config, "BRK_BREAK_VOL", 1.25)
        BRK_PULLBACK_VOL   = getattr(config, "BRK_PULLBACK_VOL", 1.15)
        BRK_EPS            = getattr(config, "BRK_EPS", 0.0005)
        BRK_ADX_MIN        = getattr(config, "BRK_ADX_MIN", 18.0)
        BRK_RSI_LONG_MAX   = getattr(config, "BRK_RSI_LONG_MAX", 75.0)
        BRK_RSI_SHORT_MIN  = getattr(config, "BRK_RSI_SHORT_MIN", 25.0)
        BRK_PULL_NEAR_E25  = getattr(config, "BRK_PULL_NEAR_E25", 0.006)
        BRK_REQ_ALIGN_EMA  = getattr(config, "BRK_REQ_ALIGN_EMA", True)

        CONT_ADX_MIN       = getattr(config, "BRK_CONT_ADX_MIN", 28.0)
        CONT_LOOKBACK      = getattr(config, "BRK_CONT_LOOK", 4)
        CONT_MAX_E25_EXT   = getattr(config, "BRK_CONT_MAX_E25_EXT", 0.015)

        NO_TRADE_BAND      = getattr(config, "NO_TRADE_BAND_AROUND_EMA_SLOW", 0.003)

        # --- Indicadores ---
        ema7  = ema(closes, 7)[-1]
        ema25 = ema(closes, 25)[-1]
        ema99 = ema(closes, 99)[-1]

        # ATR para umbral dinámico
        atr14 = float(atr(highs, lows, closes, 14)[-1]) if len(closes) > 14 else 0.0
        atrp  = (atr14 / L.close) if (L.close > 0 and atr14 > 0) else 0.0
        EPS_R = max(BRK_EPS, 0.25 * atrp)  # no romper por “milímetros”

        # ADX + DI
        adx_res = adx_calc(highs, lows, closes, 14)
        adx_now  = float(adx_res.get("adx", 0.0))
        di_plus  = float(adx_res.get("plus_di", 0.0))
        di_minus = float(adx_res.get("minus_di", 0.0))

        feats = compute_features(cs)
        rsi = float(feats.get("rsi_14", 50.0))

        vavg20 = volume_avg(cs, max(20, getattr(config, "VOL_AVG_LEN", 20)))
        vol_ok_break = (L.volume >= BRK_BREAK_VOL * vavg20) if vavg20 > 0 else False
        vol_ok_pull  = (L.volume <= BRK_PULLBACK_VOL * vavg20) if vavg20 > 0 else True

        # --- Caja de consolidación (actual y previa) ---
        hh_curr = max(highs[-BRK_LOOKBACK:])
        ll_curr = min(lows[-BRK_LOOKBACK:])
        mid = float(np.mean(closes[-BRK_LOOKBACK:]))
        box_w = abs(hh_curr - ll_curr) / mid if mid > 0 else float("inf")

        # *previa* excluyendo la última barra
        if len(highs) >= BRK_LOOKBACK + 1:
            hh_prev = max(highs[-BRK_LOOKBACK-1:-1])
            ll_prev = min(lows[-BRK_LOOKBACK-1:-1])
        else:
            hh_prev, ll_prev = hh_curr, ll_curr

        # --- Alineación EMAs + DI
        bull_align = (ema7 > ema25 > ema99) and (di_plus > di_minus)
        bear_align = (ema7 < ema25 < ema99) and (di_minus > di_plus)

        # --- Banda no-trade alrededor de EMA99
        if NO_TRADE_BAND > 0 and ema99 > 0:
            if abs(L.close - ema99) / ema99 <= NO_TRADE_BAND:
                return {"action": "IDLE", "reason": "bp_near_ema99"}

        # --- Breakout puro ---
        long_breakout  = (
            (L.close >= hh_curr * (1.0 + EPS_R)) and
            (adx_now >= BRK_ADX_MIN) and vol_ok_break and
            (rsi <= BRK_RSI_LONG_MAX) and
            (not BRK_REQ_ALIGN_EMA or bull_align)
        )
        short_breakout = (
            (L.close <= ll_curr * (1.0 - EPS_R)) and
            (adx_now >= BRK_ADX_MIN) and vol_ok_break and
            (rsi >= BRK_RSI_SHORT_MIN) and
            (not BRK_REQ_ALIGN_EMA or bear_align)
        )

        # --- Pullback tras ruptura previa (real)
        prev = cs[-2] if len(cs) >= 2 else cs[-1]
        prev_break_long  = prev.close >= hh_prev * (1.0 + EPS_R)
        prev_break_short = prev.close <= ll_prev * (1.0 - EPS_R)
        near_e25 = abs(_pct(L.close, ema25)) <= BRK_PULL_NEAR_E25

        long_pullback = (
            prev_break_long and bull_align and (adx_now >= BRK_ADX_MIN) and
            near_e25 and vol_ok_pull and rsi >= 45.0
        )
        short_pullback = (
            prev_break_short and bear_align and (adx_now >= BRK_ADX_MIN) and
            near_e25 and vol_ok_pull and rsi <= 55.0
        )

        # --- Continuation ---
        dist_e25 = abs(L.close - ema25) / ema25 if ema25 > 0 else 9.9
        made_new_low  = L.close < min(lows[-CONT_LOOKBACK:])
        made_new_high = L.close > max(highs[-CONT_LOOKBACK:])
        continuation_short = bear_align and (adx_now >= CONT_ADX_MIN) and made_new_low  and (dist_e25 <= CONT_MAX_E25_EXT)
        continuation_long  = bull_align and (adx_now >= CONT_ADX_MIN) and made_new_high and (dist_e25 <= CONT_MAX_E25_EXT)

        # --- Filtro de caja razonable ---
        in_box_regime = box_w <= BRK_RANGE_MAX_W

        # --- Señales (prioridad) ---
        if in_box_regime and long_breakout:
            return {"action": "ENTER_LONG",
                    "reason": f"BRK_long >HH epsR={EPS_R:.4f} volx={_safe_num(L.volume / vavg20)} adx={adx_now:.1f} rsi={rsi:.1f} bw={box_w:.3f}"}

        if in_box_regime and short_breakout:
            return {"action": "ENTER_SHORT",
                    "reason": f"BRK_short <LL epsR={EPS_R:.4f} volx={_safe_num(L.volume / vavg20)} adx={adx_now:.1f} rsi={rsi:.1f} bw={box_w:.3f}"}

        if continuation_short:
            return {"action": "ENTER_SHORT",
                    "reason": f"BRK_cont_short newLow{CONT_LOOKBACK} adx={adx_now:.1f} distE25={dist_e25:.3f}"}

        if continuation_long:
            return {"action": "ENTER_LONG",
                    "reason": f"BRK_cont_long newHigh{CONT_LOOKBACK} adx={adx_now:.1f} distE25={dist_e25:.3f}"}

        if long_pullback:
            return {"action": "ENTER_LONG",
                    "reason": f"PULL_long nearE25 volOK adx={adx_now:.1f} rsi={rsi:.1f} bw={box_w:.3f}"}

        if short_pullback:
            return {"action": "ENTER_SHORT",
                    "reason": f"PULL_short nearE25 volOK adx={adx_now:.1f} rsi={rsi:.1f} bw={box_w:.3f}"}

        volx = (L.volume / vavg20) if vavg20 > 0 else None
        return {"action": "IDLE",
                "reason": f"bp_watch bw={box_w:.3f} adx={adx_now:.1f} bull={bool(bull_align)} bear={bool(bear_align)} rsi={rsi:.1f} volx={_safe_num(volx)} nearE25={_safe_num(abs(_pct(L.close, ema25)))}"}
