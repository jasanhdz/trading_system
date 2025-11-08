# src/strategies/regime_switcher.py
"""RegimeSwitcher MTF (1h-15m-5m-3m)

Este módulo reemplaza/moderniza tu `RegimeSwitcher` para aplicar las
estrategias que pediste (Momentum MTF, Breakout+Pullback y Mean Reversion
con Bandas de Bollinger) usando tu arquitectura actual.

Puntos clave:
- **Marco superior (1h)** define el sesgo con EMAs (EMA_FAST/EMA_SLOW) y ADX.
- **Confirmación (15m)** valida estructura/volumen.
- **Timing (5m y micro 3m)** dispara la entrada con pullback-control de
  extensión y filtros de volumen.
- **Salidas**: invalidación de estructura, objetivos naturales (media/banda),
  y dejá que tus guards manejen BE + trailing por ROE.

**Nota Binance**: no existe timeframe "2m"; usamos **"3m"** como micro.

No requiere cambios en `StrategyRunner` ni en `Config`. Este archivo usa
`getattr(CONFIG, name, default)` para nuevas knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from ..core.types import Side, Signal, Candle, BotMode
from ..core.indicators import ema
from ..core.indicators.adx import adx as adx_wilder
from ..core.utils.candles import (
    get_closes, get_highs, get_lows, get_volumes, count_streak, is_green, is_red
)
from ..core.utils.features import calculate_rsi
from .base import Strategy


# =====================
# Utilidades locales
# =====================

def _bbands(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if closes.size < period:
        m = np.full_like(closes, np.nan)
        return m, m, m
    ma = np.convolve(closes, np.ones(period)/period, mode="valid")
    # Alinear con la serie original
    pad = closes.size - ma.size
    ma = np.concatenate([np.full(pad, np.nan), ma])
    # Desv std rolling
    std = np.full_like(closes, np.nan)
    for i in range(period-1, closes.size):
        std[i] = closes[i-period+1:i+1].std(ddof=0)
    upper = ma + mult * std
    lower = ma - mult * std
    return lower, ma, upper


def _bb_width(lower: float, mid: float, upper: float) -> float:
    if mid and mid != 0 and np.isfinite(mid) and np.isfinite(upper) and np.isfinite(lower):
        return (upper - lower) / abs(mid)
    return 0.0


def _v_ratio(candles: List[Candle], lookback: int) -> float:
    if not candles:
        return 1.0
    vols = get_volumes(candles)
    k = min(len(vols), lookback)
    avg = float(vols[-k:].mean()) if k > 0 else 0.0
    last = float(vols[-1])
    return (last / avg) if avg > 0 else 1.0


def _ext_from(value: float, ref: float) -> float:
    return abs(value - ref) / ref if ref else 0.0


def _timeframe_micro(config) -> str:
    # Binance no tiene 2m; por defecto usamos 3m. Permitimos override via ENV.
    return getattr(config, "ENTRY_MICRO_TF", "3m")


@dataclass
class _Context:
    # Pre-cálculos para evitar repetir trabajo entre sub-estrategias
    h1: List[Candle]
    m15: List[Candle]
    m5: List[Candle]
    m3: List[Candle]
    ema_fast_h1: float
    ema_mid_h1: float
    ema_slow_h1: float
    adx_h1: float
    bb_lower_h1: float
    bb_mid_h1: float
    bb_upper_h1: float
    v_ratio_5m: float


async def _load_context(symbol: str, exchange, config, logger) -> Optional[_Context]:
    try:
        # Carga de velas
        h1 = await exchange.get_candles(symbol, "1h", 300)
        m15 = await exchange.get_candles(symbol, "15m", 300)
        m5 = await exchange.get_candles(symbol, "5m", 300)
        m3 = await exchange.get_candles(symbol, _timeframe_micro(config), 300)
        if len(h1) < 60 or len(m15) < 60 or len(m5) < 60 or len(m3) < 60:
            return None

        closes_h1 = get_closes(h1)
        closes_m15 = get_closes(m15)
        closes_m5 = get_closes(m5)

        # EMAs marco superior (usa knobs existentes)
        efast = getattr(config, "EMA_FAST", 7)
        emid = getattr(config, "EMA_MID", 25)
        eslow = getattr(config, "EMA_SLOW", 99)
        ema_fast_h1 = float(ema(closes_h1, efast)[-1])
        ema_mid_h1 = float(ema(closes_h1, emid)[-1])
        ema_slow_h1 = float(ema(closes_h1, eslow)[-1])

        # ADX marco superior
        adx_len = getattr(config, "ADX_LEN", 14)
        adx_res = adx_wilder(get_highs(h1), get_lows(h1), get_closes(h1), adx_len)
        adx_h1 = float(adx_res.get("adx", 0.0))

        # BB (para detectar rango)
        bb_period = getattr(config, "BOLL_PERIOD", 20)
        bb_mult = getattr(config, "BOLL_STD", 2.0)
        lo_h1, mid_h1, up_h1 = _bbands(closes_h1, bb_period, bb_mult)
        bb_lower_h1 = float(lo_h1[-1]) if lo_h1.size else np.nan
        bb_mid_h1 = float(mid_h1[-1]) if mid_h1.size else np.nan
        bb_upper_h1 = float(up_h1[-1]) if up_h1.size else np.nan

        # Volumen intradía 5m
        v_ratio_5m = _v_ratio(m5, getattr(config, "VOL_AVG_LEN", 20))

        return _Context(
            h1=h1, m15=m15, m5=m5, m3=m3,
            ema_fast_h1=ema_fast_h1,
            ema_mid_h1=ema_mid_h1,
            ema_slow_h1=ema_slow_h1,
            adx_h1=adx_h1,
            bb_lower_h1=bb_lower_h1,
            bb_mid_h1=bb_mid_h1,
            bb_upper_h1=bb_upper_h1,
            v_ratio_5m=v_ratio_5m,
        )
    except Exception as e:
        logger.warn("ctx_load_fail", {"err": str(e)})
        return None


# =====================
# Sub-estrategias
# =====================

class _MomentumMTF:
    """Tendencia con sesgo 1h y timing 5m/3m."""
    def __init__(self, config):
        self.cfg = config

    def _bias(self, ctx: _Context) -> Optional[Side]:
        # Sesgo: EMA_FAST vs EMA_SLOW en 1h con ADX
        adx_min = getattr(self.cfg, "ADX_MIN", 20)
        if ctx.adx_h1 < adx_min:
            return None
        if ctx.ema_fast_h1 > ctx.ema_slow_h1 * (1 + 0.001):
            return Side.LONG
        if ctx.ema_fast_h1 < ctx.ema_slow_h1 * (1 - 0.001):
            return Side.SHORT
        return None

    def _trigger(self, side: Side, m5: List[Candle], m3: List[Candle]) -> bool:
        # Pullback a EMA_FAST en 5m y reenganche con vela que cierre a favor.
        closes5 = get_closes(m5)
        closes3 = get_closes(m3)
        efast = getattr(self.cfg, "EMA_FAST", 7)
        ema_fast_5 = float(ema(closes5, efast)[-1])
        ema_fast_3 = float(ema(closes3, efast)[-1])
        c5 = float(closes5[-1])
        c3 = float(closes3[-1])

        # Evitar persecución: no exceder extensión desde EMA_FAST
        max_ext = getattr(self.cfg, "MAX_EXT_FROM_EMA_FAST", 0.015)  # 1.5%
        if _ext_from(c5, ema_fast_5) > max_ext or _ext_from(c3, ema_fast_3) > max_ext:
            return False

        # Confirmación con última vela 3m a favor
        if side == Side.LONG:
            cond = c5 >= ema_fast_5 and c3 > ema_fast_3 and m3[-1].close > m3[-1].open
        else:
            cond = c5 <= ema_fast_5 and c3 < ema_fast_3 and m3[-1].close < m3[-1].open
        return bool(cond)

    def _volume_ok(self, ctx: _Context, side: Side) -> bool:
        # Usa tu config VOL_FACTOR_ENTRY
        vf = getattr(self.cfg, "VOL_FACTOR_ENTRY", 1.1)
        return ctx.v_ratio_5m >= vf

    def evaluate(self, ctx: _Context) -> Tuple[str, str]:
        bias = self._bias(ctx)
        if not bias:
            return "IDLE", "momentum_no_bias"

        # Confirmación 15m: alineación con EMA_MID
        closes15 = get_closes(ctx.m15)
        emid = getattr(self.cfg, "EMA_MID", 25)
        ema_mid_15 = float(ema(closes15, emid)[-1])
        c15 = float(closes15[-1])

        if bias == Side.LONG and c15 < ema_mid_15:
            return "IDLE", "m15_not_aligned_long"
        if bias == Side.SHORT and c15 > ema_mid_15:
            return "IDLE", "m15_not_aligned_short"

        # Trigger 5m/3m + volumen
        if self._trigger(bias, ctx.m5, ctx.m3) and self._volume_ok(ctx, bias):
            act = "ENTER_LONG" if bias == Side.LONG else "ENTER_SHORT"
            return act, f"momentum_mtf_bias_{bias.value.lower()}"

        return "IDLE", "momentum_no_trigger"

    def exit_signal(self, side: Side, ctx: _Context) -> Tuple[str, str]:
        # Invalida si cierre 5m cruza la EMA_MID en contra
        closes5 = get_closes(ctx.m5)
        emid = getattr(self.cfg, "EMA_MID", 25)
        ema_mid_5 = float(ema(closes5, emid)[-1])
        c5 = float(closes5[-1])
        if (side == Side.LONG and c5 < ema_mid_5) or (side == Side.SHORT and c5 > ema_mid_5):
            return "EXIT", "momentum_invalidation"
        return "IDLE", "hold"


class _BreakoutPullback:
    """Ruptura del rango h1 con pullback en 3m/5m."""
    def __init__(self, config):
        self.cfg = config

    def _range(self, h1: List[Candle]) -> Tuple[float, float]:
        lb = getattr(self.cfg, "BREAKOUT_LOOKBACK", 50)
        recent = h1[-lb:]
        hi = max(c.high for c in recent)
        lo = min(c.low for c in recent)
        return lo, hi

    def _confirm_break(self, m15: List[Candle], level: float, up: bool) -> bool:
        k = getattr(self.cfg, "BREAKOUT_CONFIRM_BARS", 2)
        closes = get_closes(m15)
        if len(closes) < k:
            return False
        last_k = closes[-k:]
        return all(x > level for x in last_k) if up else all(x < level for x in last_k)

    def _pullback_hit(self, m3: List[Candle], level: float, eps: float) -> bool:
        # Toca +-eps% y regresa
        last = m3[-1]
        dist = abs(last.close - level) / level
        if dist <= eps:
            # vela de rechazo (cola) o cierre regresando al lado de la ruptura
            return True
        return False

    def evaluate(self, ctx: _Context) -> Tuple[str, str]:
        lo, hi = self._range(ctx.h1)
        eps = getattr(self.cfg, "MR_TOUCH_EPS", 0.001)  # 0.1%
        # ¿Se rompió por arriba/abajo en 15m?
        broke_up = self._confirm_break(ctx.m15, hi, up=True)
        broke_dn = self._confirm_break(ctx.m15, lo, up=False)
        # Volumen 5m decente
        if ctx.v_ratio_5m < getattr(self.cfg, "VOL_FACTOR_ENTRY", 1.1):
            return "IDLE", "bo_vol_low"
        if broke_up and self._pullback_hit(ctx.m3, hi, eps):
            return "ENTER_LONG", "breakout_pullback_up"
        if broke_dn and self._pullback_hit(ctx.m3, lo, eps):
            return "ENTER_SHORT", "breakout_pullback_down"
        return "IDLE", "bo_no_trigger"

    def exit_signal(self, side: Side, ctx: _Context) -> Tuple[str, str]:
        # Si regresa dentro del rango h1 -> salir
        lo, hi = self._range(ctx.h1)
        c15 = float(get_closes(ctx.m15)[-1])
        if side == Side.LONG and c15 < hi:
            return "EXIT", "bo_failed_reentry_in_range"
        if side == Side.SHORT and c15 > lo:
            return "EXIT", "bo_failed_reentry_in_range"
        return "IDLE", "hold"


class _MeanReversionBB:
    """Media reversión intradía cuando 1h está en rango (BB width baja)."""
    def __init__(self, config):
        self.cfg = config

    def _is_range_regime(self, ctx: _Context) -> bool:
        # Usa BBWidth 1h + ADX máximo
        bbw_lim = getattr(self.cfg, "MR_BB_WIDTH_MAX", 0.025)
        adx_max = getattr(self.cfg, "MR_ADX_MAX", 20)
        width = _bb_width(ctx.bb_lower_h1, ctx.bb_mid_h1, ctx.bb_upper_h1)
        return (width <= bbw_lim) and (ctx.adx_h1 <= adx_max)

    def evaluate(self, ctx: _Context) -> Tuple[str, str]:
        if not self._is_range_regime(ctx):
            return "IDLE", "mr_not_range"
        # Señales en 5m/3m con BB-20 y RSI filtros
        closes5 = get_closes(ctx.m5)
        rsiperiod = 14
        rsi5 = calculate_rsi(closes5, rsiperiod)
        lo5, mid5, up5 = _bbands(closes5, getattr(self.cfg, "BOLL_PERIOD", 20), getattr(self.cfg, "BOLL_STD", 2.0))
        last5 = float(closes5[-1])
        eps = getattr(self.cfg, "MR_TOUCH_EPS", 0.001)
        # Streak de velas para confianza
        streak_min = getattr(self.cfg, "MR_MIN_STREAK", 2)
        last_k = ctx.m5[-streak_min:]
        red_streak = sum(1 for c in last_k if is_red(c)) >= streak_min
        green_streak = sum(1 for c in last_k if is_green(c)) >= streak_min

        # LONG: toque a banda inferior + RSI bajo
        if (
            last5 <= float(lo5[-1]) * (1 + eps)
            and rsi5 <= getattr(self.cfg, "MR_RSI_LOW", 32)
            and red_streak
        ):
            return "ENTER_LONG", "mr_bb_long"

        # SHORT: toque a banda superior + RSI alto (+ confirmación 1h si así se pide)
        strict_shorts = getattr(self.cfg, "MR_STRICT_SHORTS", True)
        if (
            last5 >= float(up5[-1]) * (1 - eps)
            and rsi5 >= getattr(self.cfg, "MR_RSI_HIGH", 68)
            and green_streak
        ):
            if strict_shorts and getattr(self.cfg, "MR_SHORT_CONFIRM_1H", False):
                # Requiere 1h sesgo bajista suave: EMA_FAST <= EMA_MID
                if ctx.ema_fast_h1 <= ctx.ema_mid_h1:
                    return "ENTER_SHORT", "mr_bb_short_confirmed"
                else:
                    return "IDLE", "mr_short_denied_no_h1_confirm"
            return "ENTER_SHORT", "mr_bb_short"

        return "IDLE", "mr_no_trigger"

    def exit_signal(self, side: Side, ctx: _Context) -> Tuple[str, str]:
        # Salida en media 5m o si se rompe la banda en contra con cierre
        closes5 = get_closes(ctx.m5)
        lo5, mid5, up5 = _bbands(closes5, getattr(self.cfg, "BOLL_PERIOD", 20), getattr(self.cfg, "BOLL_STD", 2.0))
        last = float(closes5[-1])
        mid = float(mid5[-1])
        lo = float(lo5[-1])
        up = float(up5[-1])
        if side == Side.LONG:
            if last >= mid:
                return "EXIT", "mr_target_mid"
            if last < lo * (1 - getattr(self.cfg, "MR_TOUCH_EPS", 0.001)):
                return "EXIT", "mr_break_lower"
        else:
            if last <= mid:
                return "EXIT", "mr_target_mid"
            if last > up * (1 + getattr(self.cfg, "MR_TOUCH_EPS", 0.001)):
                return "EXIT", "mr_break_upper"
        return "IDLE", "hold"


# =====================
# RegimeSwitcher (público)
# =====================

class RegimeSwitcher(Strategy):
    name = "RegimeSwitcher-MTF"

    def __init__(self):
        self._momentum = None
        self._breakout = None
        self._mr = None
        # --- Estado interno para evitar salidas cruzadas y re-entradas inmediatas ---
        self._hold_strategy: Optional[str] = None  # "momentum" | "breakout" | "mr"
        self._block_reentry_until_ms: int = 0

    async def evaluate(self, *, symbol, exchange, config, state, now, logger) -> Signal:  # type: ignore[override]
        # Cargar contexto multi-timeframe
        ctx = await _load_context(symbol, exchange, config, logger)
        if ctx is None:
            return {"action": "IDLE", "reason": "insufficient_candles"}

        # Lazy init de sub-estrategias (para leer CONFIG)
        if self._momentum is None:
            self._momentum = _MomentumMTF(config)
        if self._breakout is None:
            self._breakout = _BreakoutPullback(config)
        if self._mr is None:
            self._mr = _MeanReversionBB(config)

        # -------- Cooldown de re-entrada tras EXIT --------
        if now < self._block_reentry_until_ms:
            return {"action": "IDLE", "reason": "cooldown_after_exit"}

        # Si ya hay posición, **solo** la sub-estrategia que la abrió decide la salida
        if state and state.mode in (BotMode.LONG_RIDE, BotMode.SHORT_RIDE) and state.last_side:
            side = state.last_side
            chosen = self._hold_strategy or "momentum"  # por defecto momentum si no hay tag
            if chosen == "momentum":
                act, rs = self._momentum.exit_signal(side, ctx)
            elif chosen == "breakout":
                act, rs = self._breakout.exit_signal(side, ctx)
            else:  # "mr"
                act, rs = self._mr.exit_signal(side, ctx)

            if act == "EXIT":
                # activa cooldown local para evitar flip-flop
                cooldown_ms = getattr(config, "REENTER_COOLDOWN_MS", 5000)
                self._block_reentry_until_ms = now + int(cooldown_ms)
                self._hold_strategy = None
                return {"action": act, "reason": rs}

            return {"action": "IDLE", "reason": "hold_position"}

        # Si NO hay posición, decide el **régimen** y elige la estrategia
        adx_min = getattr(config, "ADX_MIN", 20)
        trending_up = ctx.ema_fast_h1 > ctx.ema_slow_h1 * (1 + 0.001) and ctx.adx_h1 >= adx_min
        trending_dn = ctx.ema_fast_h1 < ctx.ema_slow_h1 * (1 - 0.001) and ctx.adx_h1 >= adx_min
        range_regime = _MeanReversionBB(config)._is_range_regime(ctx)

        # Probar Momentum primero si aplica
        if trending_up or trending_dn:
            act, rs = self._momentum.evaluate(ctx)
            if act != "IDLE":
                self._hold_strategy = "momentum"
                return {"action": act, "reason": rs}
            # Fallback a breakout si Momentum no gatilló
            act, rs = self._breakout.evaluate(ctx)
            if act != "IDLE":
                self._hold_strategy = "breakout"
                return {"action": act, "reason": rs}
            # Si tampoco, y el régimen luce lateral, intenta MR
            if range_regime:
                act, rs = self._mr.evaluate(ctx)
                if act != "IDLE":
                    self._hold_strategy = "mr"
                return {"action": act, "reason": rs}
            return {"action": "IDLE", "reason": "trend_no_trigger"}

        # Si no trending, intenta Breakout (pre-ruptura con volumen)
        act, rs = self._breakout.evaluate(ctx)
        if act != "IDLE":
            self._hold_strategy = "breakout"
            return {"action": act, "reason": rs}

        # Si todo está lateral, MR
        if range_regime:
            act, rs = self._mr.evaluate(ctx)
            if act != "IDLE":
                self._hold_strategy = "mr"
            return {"action": act, "reason": rs}

        return {"action": "IDLE", "reason": "no_regime_match"}