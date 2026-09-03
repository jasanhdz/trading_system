"""Factorized, stateful regime analysis for offline and Shadow evaluation.

This module deliberately does not expose a single trade-authorization label.
Direction, volatility, and market structure are separate observations so a
high-volatility bear trend cannot be collapsed into one ambiguous category.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Deque


class DirectionRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


class VolatilityRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class StructureRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    TREND = "TREND"
    RANGE = "RANGE"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class RegimeV2Settings:
    schema_version: str
    history_window: int
    minimum_history: int
    low_volatility_quantile: float
    high_volatility_quantile: float
    trend_enter_fraction: float
    trend_exit_fraction: float
    trend_strength_enter: float
    trend_strength_exit: float
    chop_enter_fraction: float
    chop_exit_fraction: float
    high_expansion_ratio: float
    low_expansion_ratio: float
    minimum_state_bars: int

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-regime-v2-research-settings-v1":
            raise ValueError("unsupported regime V2 settings schema")
        if self.history_window < 3 or not 2 <= self.minimum_history <= self.history_window:
            raise ValueError("regime history settings are invalid")
        if not 0.0 < self.low_volatility_quantile < self.high_volatility_quantile < 1.0:
            raise ValueError("volatility quantiles are invalid")
        if not 0.0 <= self.trend_exit_fraction < self.trend_enter_fraction:
            raise ValueError("trend hysteresis is invalid")
        if not 0.0 <= self.trend_strength_exit < self.trend_strength_enter:
            raise ValueError("trend-strength hysteresis is invalid")
        if not 0.0 <= self.chop_exit_fraction < self.chop_enter_fraction <= 1.0:
            raise ValueError("structure hysteresis is invalid")
        if not self.low_expansion_ratio < self.high_expansion_ratio:
            raise ValueError("expansion thresholds are invalid")
        if self.minimum_state_bars < 1:
            raise ValueError("minimum state duration must be positive")


@dataclass(frozen=True)
class RegimeV2Observation:
    symbol: str
    timestamp: datetime
    market_direction_6: float
    range_mean_24: float
    range_expansion: float
    chop_12: float
    trend_strength_12: float

    def __post_init__(self) -> None:
        if not self.symbol or self.timestamp.tzinfo is None:
            raise ValueError("regime observation identity is invalid")
        values = (
            self.market_direction_6,
            self.range_mean_24,
            self.range_expansion,
            self.chop_12,
            self.trend_strength_12,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("regime observation contains a non-finite value")
        if self.range_mean_24 < 0.0 or not 0.0 <= self.chop_12 <= 1.0:
            raise ValueError("regime observation range is invalid")


@dataclass(frozen=True)
class RegimeV2Result:
    schema_version: str
    symbol: str
    timestamp: datetime
    direction: DirectionRegime
    volatility: VolatilityRegime
    structure: StructureRegime
    evidence_ready: bool
    history_count: int
    direction_stability_bars: int
    volatility_stability_bars: int
    structure_stability_bars: int
    low_volatility_boundary: float | None
    high_volatility_boundary: float | None

    @property
    def short_context(self) -> bool:
        """Observational context only; this is not an entry authorization."""
        return (
            self.evidence_ready
            and self.direction is DirectionRegime.BEARISH
            and self.structure is StructureRegime.TREND
        )


@dataclass
class _AxisState:
    value: Enum
    stable_bars: int = 0
    pending: Enum | None = None
    pending_bars: int = 0


@dataclass
class _SymbolState:
    ranges: Deque[float]
    direction: _AxisState
    volatility: _AxisState
    structure: _AxisState


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class FactorizedRegimeAnalyzer:
    """Per-symbol rolling regime analyzer with hysteresis and state persistence."""

    def __init__(self, settings: RegimeV2Settings) -> None:
        self.settings = settings
        self._states: dict[str, _SymbolState] = {}
        self._last_timestamp: dict[str, datetime] = {}

    def _state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(
                ranges=deque(maxlen=self.settings.history_window),
                direction=_AxisState(DirectionRegime.UNKNOWN),
                volatility=_AxisState(VolatilityRegime.UNKNOWN),
                structure=_AxisState(StructureRegime.UNKNOWN),
            )
        return self._states[symbol]

    def observe(self, observation: RegimeV2Observation) -> RegimeV2Result:
        previous_timestamp = self._last_timestamp.get(observation.symbol)
        if previous_timestamp is not None and observation.timestamp <= previous_timestamp:
            raise ValueError("regime observations must be strictly chronological per symbol")
        self._last_timestamp[observation.symbol] = observation.timestamp
        state = self._state(observation.symbol)
        state.ranges.append(observation.range_mean_24)
        ready = len(state.ranges) >= self.settings.minimum_history
        low_boundary = high_boundary = None

        direction = self._direction_candidate(observation, state.direction.value)
        structure = self._structure_candidate(observation, state.structure.value)
        if ready:
            values = tuple(state.ranges)
            low_boundary = _quantile(values, self.settings.low_volatility_quantile)
            high_boundary = _quantile(values, self.settings.high_volatility_quantile)
            volatility = self._volatility_candidate(observation, low_boundary, high_boundary)
        else:
            volatility = VolatilityRegime.UNKNOWN

        self._advance(state.direction, direction)
        self._advance(state.volatility, volatility)
        self._advance(state.structure, structure)
        return RegimeV2Result(
            schema_version="aegis-factorized-regime-v2-result-v1",
            symbol=observation.symbol,
            timestamp=observation.timestamp,
            direction=state.direction.value,  # type: ignore[arg-type]
            volatility=state.volatility.value,  # type: ignore[arg-type]
            structure=state.structure.value,  # type: ignore[arg-type]
            evidence_ready=ready,
            history_count=len(state.ranges),
            direction_stability_bars=state.direction.stable_bars,
            volatility_stability_bars=state.volatility.stable_bars,
            structure_stability_bars=state.structure.stable_bars,
            low_volatility_boundary=low_boundary,
            high_volatility_boundary=high_boundary,
        )

    def _direction_candidate(
        self, observation: RegimeV2Observation, previous: Enum
    ) -> DirectionRegime:
        value = observation.market_direction_6
        if previous is DirectionRegime.BEARISH and value < -self.settings.trend_exit_fraction:
            return DirectionRegime.BEARISH
        if previous is DirectionRegime.BULLISH and value > self.settings.trend_exit_fraction:
            return DirectionRegime.BULLISH
        if value <= -self.settings.trend_enter_fraction:
            return DirectionRegime.BEARISH
        if value >= self.settings.trend_enter_fraction:
            return DirectionRegime.BULLISH
        return DirectionRegime.NEUTRAL

    def _structure_candidate(
        self, observation: RegimeV2Observation, previous: Enum
    ) -> StructureRegime:
        if previous is StructureRegime.RANGE and observation.chop_12 > self.settings.chop_exit_fraction:
            return StructureRegime.RANGE
        if (
            previous is StructureRegime.TREND
            and observation.trend_strength_12 >= self.settings.trend_strength_exit
            and observation.chop_12 < self.settings.chop_enter_fraction
        ):
            return StructureRegime.TREND
        if observation.chop_12 >= self.settings.chop_enter_fraction:
            return StructureRegime.RANGE
        if (
            abs(observation.market_direction_6) >= self.settings.trend_enter_fraction
            and observation.trend_strength_12 >= self.settings.trend_strength_enter
        ):
            return StructureRegime.TREND
        return StructureRegime.TRANSITION

    def _volatility_candidate(
        self, observation: RegimeV2Observation, low_boundary: float, high_boundary: float
    ) -> VolatilityRegime:
        if (
            observation.range_mean_24 >= high_boundary
            or observation.range_expansion >= self.settings.high_expansion_ratio
        ):
            return VolatilityRegime.HIGH
        if (
            observation.range_mean_24 <= low_boundary
            and observation.range_expansion <= self.settings.low_expansion_ratio
        ):
            return VolatilityRegime.LOW
        return VolatilityRegime.NORMAL

    def _advance(self, state: _AxisState, candidate: Enum) -> None:
        if candidate == state.value:
            state.stable_bars += 1
            state.pending = None
            state.pending_bars = 0
            return
        if state.value.value == "UNKNOWN":
            state.value = candidate
            state.stable_bars = 1
            state.pending = None
            state.pending_bars = 0
            return
        if state.pending == candidate:
            state.pending_bars += 1
        else:
            state.pending = candidate
            state.pending_bars = 1
        if state.pending_bars >= self.settings.minimum_state_bars:
            state.value = candidate
            state.stable_bars = 1
            state.pending = None
            state.pending_bars = 0
