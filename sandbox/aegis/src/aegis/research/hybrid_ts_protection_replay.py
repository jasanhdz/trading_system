"""Causal OHLC replay of the price-dependent TypeScript protection rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..data import CanonicalBar
from ..training.hybrid_directional import DirectionalSide


class IntrabarPath(str, Enum):
    OPEN_HIGH_LOW_CLOSE = "OPEN_HIGH_LOW_CLOSE"
    OPEN_LOW_HIGH_CLOSE = "OPEN_LOW_HIGH_CLOSE"


class ProtectionExit(str, Enum):
    HARD_STOP = "HARD_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    BREAK_EVEN_STOP = "BREAK_EVEN_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    HORIZON_CLOSE = "HORIZON_CLOSE"


@dataclass(frozen=True)
class TsProtectionConfig:
    leverage: float = 15.0
    hard_stop_roe: float = -0.40
    take_profit_roe: float = 0.50
    break_even_trigger_roe: float = 0.08
    break_even_offset_fraction: float = 0.003
    trailing_activation_roe: float = 0.15
    trailing_callback_roe: float = 0.08
    use_atr_trailing: bool = True
    atr_period: int = 14
    atr_multiplier: float = 1.5
    round_trip_cost_fraction: float = 0.001

    def __post_init__(self) -> None:
        values = (
            self.leverage,
            self.hard_stop_roe,
            self.take_profit_roe,
            self.break_even_trigger_roe,
            self.break_even_offset_fraction,
            self.trailing_activation_roe,
            self.trailing_callback_roe,
            self.atr_multiplier,
            self.round_trip_cost_fraction,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or self.leverage <= 0.0
            or self.hard_stop_roe >= 0.0
            or min(
                self.take_profit_roe,
                self.break_even_trigger_roe,
                self.break_even_offset_fraction,
                self.trailing_activation_roe,
                self.trailing_callback_roe,
                self.atr_multiplier,
                self.round_trip_cost_fraction,
            )
            < 0.0
            or self.atr_period < 1
            or self.round_trip_cost_fraction >= 1.0
        ):
            raise ValueError("TypeScript protection replay config is invalid")


@dataclass(frozen=True)
class ProtectionReplayResult:
    side: DirectionalSide
    path: IntrabarPath
    entry_price: float
    exit_price: float
    exit_reason: ProtectionExit
    bars_held: int
    gross_return_fraction: float
    net_return_after_costs: float
    peak_roe: float
    lowest_roe: float
    break_even_armed: bool
    trailing_armed: bool
    atr_available: bool


def wilder_atr(candles: Sequence[CanonicalBar], period: int = 14) -> float | None:
    """Match TechnicalIndicators.calculateATR using finalized candles only."""

    if len(candles) < period + 1:
        return None
    true_ranges = []
    for previous, current in zip(candles, candles[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges[:period]) / period
    for value in true_ranges[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr


def _roe(side: DirectionalSide, entry: float, price: float, leverage: float) -> float:
    move = (price - entry) / entry
    return move * leverage * side.sign


def _gross_return(side: DirectionalSide, entry: float, exit_price: float) -> float:
    return ((exit_price - entry) / entry) * side.sign


def _better_stop(side: DirectionalSide, candidate: float, current: float) -> bool:
    return candidate > current if side is DirectionalSide.LONG else candidate < current


def _crossed(start: float, end: float, level: float) -> bool:
    return min(start, end) <= level <= max(start, end) and start != end


def _first_exit_level(
    start: float,
    end: float,
    stop: float,
    take_profit: float,
) -> tuple[float, ProtectionExit] | None:
    candidates: list[tuple[float, float, ProtectionExit]] = []
    if _crossed(start, end, stop):
        candidates.append((abs(stop - start), stop, ProtectionExit.HARD_STOP))
    if _crossed(start, end, take_profit):
        candidates.append(
            (abs(take_profit - start), take_profit, ProtectionExit.TAKE_PROFIT)
        )
    if not candidates:
        return None
    _, level, reason = min(candidates, key=lambda item: item[0])
    return level, reason


def replay_ts_price_protection(
    *,
    side: DirectionalSide,
    history: Sequence[CanonicalBar],
    future: Sequence[CanonicalBar],
    path: IntrabarPath,
    config: TsProtectionConfig,
) -> ProtectionReplayResult:
    """Replay bracket, break-even and ATR/fixed trailing without future signals.

    Exit Eye is intentionally excluded because it requires subsequent committee
    decisions that are not part of the directional H12 fixture.
    """

    if not future or len(history) < config.atr_period + 1:
        raise ValueError("protection replay requires history and future bars")
    entry = future[0].open
    if entry <= 0.0:
        raise ValueError("protection replay entry price is invalid")

    stop = entry * (1.0 + config.hard_stop_roe / config.leverage * side.sign)
    take_profit = entry * (1.0 + config.take_profit_roe / config.leverage * side.sign)
    current_stop_reason = ProtectionExit.HARD_STOP
    peak_price = entry
    peak_roe = 0.0
    lowest_roe = 0.0
    break_even_armed = False
    trailing_armed = False
    atr_available = False
    current = entry

    for bar_index, candle in enumerate(future, start=1):
        prior = tuple(history) + tuple(future[: bar_index - 1])
        atr = wilder_atr(prior[-(config.atr_period + 1) :], config.atr_period)
        atr_available = atr_available or atr is not None
        points = (
            (candle.open, candle.high, candle.low, candle.close)
            if path is IntrabarPath.OPEN_HIGH_LOW_CLOSE
            else (candle.open, candle.low, candle.high, candle.close)
        )
        current = points[0]

        for target in points[1:]:
            crossed = _first_exit_level(current, target, stop, take_profit)
            if crossed is not None:
                exit_price, reason = crossed
                if reason is ProtectionExit.HARD_STOP:
                    reason = current_stop_reason
                gross = _gross_return(side, entry, exit_price)
                return ProtectionReplayResult(
                    side,
                    path,
                    entry,
                    exit_price,
                    reason,
                    bar_index,
                    gross,
                    gross - config.round_trip_cost_fraction,
                    peak_roe,
                    lowest_roe,
                    break_even_armed,
                    trailing_armed,
                    atr_available,
                )

            current = target
            current_roe = _roe(side, entry, current, config.leverage)
            peak_roe = max(peak_roe, current_roe)
            lowest_roe = min(lowest_roe, current_roe)
            peak_price = (
                max(peak_price, current)
                if side is DirectionalSide.LONG
                else min(peak_price, current)
            )

            if peak_roe >= config.break_even_trigger_roe:
                break_even_armed = True
                break_even_stop = entry * (
                    1.0 + config.break_even_offset_fraction * side.sign
                )
                if _better_stop(side, break_even_stop, stop):
                    stop = break_even_stop
                    current_stop_reason = ProtectionExit.BREAK_EVEN_STOP

            if peak_roe >= config.trailing_activation_roe:
                trailing_armed = True
                if config.use_atr_trailing and atr is not None:
                    trailing = peak_price - atr * config.atr_multiplier * side.sign
                else:
                    trigger_roe = peak_roe * (1.0 - config.trailing_callback_roe)
                    trailing = entry * (1.0 + trigger_roe / config.leverage * side.sign)
                if _better_stop(side, trailing, stop):
                    stop = trailing
                    current_stop_reason = ProtectionExit.TRAILING_STOP

    gross = _gross_return(side, entry, future[-1].close)
    return ProtectionReplayResult(
        side,
        path,
        entry,
        future[-1].close,
        ProtectionExit.HORIZON_CLOSE,
        len(future),
        gross,
        gross - config.round_trip_cost_fraction,
        peak_roe,
        lowest_roe,
        break_even_armed,
        trailing_armed,
        atr_available,
    )
