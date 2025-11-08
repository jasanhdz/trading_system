# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Any, Optional, List
import numpy as np

from .base import Strategy
from ..core.types import Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from ..core.utils.candles import last, volume_avg
from ..core.indicators.ema import ema
from ..core.indicators.adx import adx as adx_calc
from ..core.utils.features import compute_features


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


def _bbw(closes: List[float], n: int = 20) -> float:
    """Bollinger BandWidth = (upper-lower)/MA."""
    if not closes:
        return float("nan")
    k = min(len(closes), n)
    sample = closes[-k:]
    m = float(np.mean(sample))
    if m <= 0:
        return float("nan")
    sd = _stdev(closes, n)
    up = m + 2 * sd
    lo = m - 2 * sd
    return (up - lo) / m


class VolatilityCompressionRebound(Strategy):
    """
    VCR: entra a favor de un *rebote* corto cuando hay compresión de volatilidad
    (BBW bajo) + mercado sin dirección (ADX bajo) + ligera sobreventa y giro
    sobre EMA corta con algo de volumen.

    Entra LONG/SHORT simétricamente, TP corto por diseño (lo maneja el runner/TP global).
    """

    def __init__(self):
        super().__init__(name="volatility_compression_rebound", timeframe="5m")
        # slots internos (no persistentes entre procesos)
        self._last_signal_ts: Optional[int] = None

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
        cs = await exchange.get_candles(symbol, tf, 300)
        if len(cs) < 120:
            return {"action": "IDLE", "reason": "few_candles"}

        L = last(cs)
        closes = [c.close for c in cs]
        highs  = [c.high for c in cs]
        lows   = [c.low for c in cs]
        vols   = [c.volume for c in cs]

        # --- features base
        ema7  = ema(closes, 7)[-1]
        ema25 = ema(closes, 25)[-1]
        ema99 = ema(closes, 99)[-1]

        adx_now = float(adx_calc(highs, lows, closes, 14).get("adx", 0.0))
        bbw_now = _bbw(closes, 20)

        feats = compute_features(cs)
        rsi14 = float(feats.get("rsi_14", 50.0))

        vavg20 = volume_avg(cs, max(20, _cfg(config, "VOL_AVG_LEN", 20)))
        vol_rel = (L.volume / vavg20) if vavg20 > 0 else 1.0

        # --- thresholds (overridables por CONFIG)
        BBW_PCTL     = float(_cfg(config, "VCR_BBW_PCTL", 30))    # compresión: ancho en pctil bajo
        ADX_MAX      = float(_cfg(config, "VCR_ADX_MAX", 20.0))   # mercado no direccional
        VOL_MIN_X    = float(_cfg(config, "VCR_MIN_VOL_X", 1.05)) # confirmación de giro
        COOLDOWN_BAR = int(_cfg(config, "VCR_COOLDOWN_BARS", 6))  # 6 velas por defecto
        RSI_LONG_MAX = float(_cfg(config, "VCR_RSI_LONG_MAX", 45.0))
        RSI_SHORT_MIN= float(_cfg(config, "VCR_RSI_SHORT_MIN", 55.0))
        TOUCH_EPS    = float(_cfg(config, "VCR_TOUCH_EPS", 0.0015))  # 0.15% de tolerancia

        # calculamos el percentil dinámico de BBW en la ventana reciente
        # (tomamos el 30p por defecto para “compresión” relativa)
        def _bbw_percentile(vals: List[float], n: int = 20, window: int = 120, pctl: float = BBW_PCTL) -> float:
            series = []
            for i in range(n, len(vals) + 1):
                series.append(_bbw(vals[:i], n))
            series = [x for x in series if math.isfinite(x)]
            if not series:
                return float("nan")
            tail = series[-window:] if len(series) >= window else series
            return float(np.percentile(tail, pctl))

        bbw_pctl = _bbw_percentile(closes)

        # --- reglas de compresión
        in_compression = (math.isfinite(bbw_now) and math.isfinite(bbw_pctl) and bbw_now <= bbw_pctl)
        not_trending   = (adx_now <= ADX_MAX)

        # --- "giro" técnico básico
        bull_flip = (L.close >= ema7 * (1 + TOUCH_EPS)) and (ema7 >= ema25 or L.close >= ema25)
        bear_flip = (L.close <= ema7 * (1 - TOUCH_EPS)) and (ema7 <= ema25 or L.close <= ema25)

        # --- cooldown por barras (sobre el state para que lo respete todo el ciclo)
        # guardamos/consultamos st.vcr_cooldown_until_bar como índice "lógico" = cantidad de ticks vistos
        # más simple: comparamos por tiempo ms contra un estimado (5m barras)
        st = state
        bar_ms = 5 * 60 * 1000 if tf.endswith("5m") else 60 * 1000  # aproximado
        if st and getattr(st, "vcr_cooldown_until", None):
            if now < st.vcr_cooldown_until:
                return {"action": "IDLE", "reason": "vcr_cooldown"}

        # --- LONG setup (rebote tras compresión bajista)
        long_ok = (
            in_compression and not_trending and
            rsi14 <= RSI_LONG_MAX and
            bull_flip and
            vol_rel >= VOL_MIN_X
        )

        if long_ok and _cfg(config, "ALLOW_LONGS", True):
            # activar cooldown
            if st:
                st.vcr_cooldown_until = now + COOLDOWN_BAR * bar_ms
            reason = f"VCR_long bbw={bbw_now:.3f}≤p{BBW_PCTL}:{bbw_pctl:.3f} adx={adx_now:.1f} rsi={rsi14:.1f} volx={vol_rel:.2f}"
            return {"action": "ENTER_LONG", "reason": reason}

        # --- SHORT simétrico (rebote bajista tras compresión alcista)
        short_ok = (
            in_compression and not_trending and
            rsi14 >= RSI_SHORT_MIN and
            bear_flip and
            vol_rel >= VOL_MIN_X
        )

        if short_ok and _cfg(config, "ALLOW_SHORTS", True):
            if st:
                st.vcr_cooldown_until = now + COOLDOWN_BAR * bar_ms
            reason = f"VCR_short bbw={bbw_now:.3f}≤p{BBW_PCTL}:{bbw_pctl:.3f} adx={adx_now:.1f} rsi={rsi14:.1f} volx={vol_rel:.2f}"
            return {"action": "ENTER_SHORT", "reason": reason}

        # fallback: log de diagnóstico
        return {
            "action": "IDLE",
            "reason": (
                f"vcr_filters comp={in_compression} adx={adx_now:.1f} "
                f"rsi={rsi14:.1f} bull={bull_flip} bear={bear_flip} volx={vol_rel:.2f}"
            ),
        }
