"""Pure predicates frozen for deterministic Phase 2 candidate generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aegis_strategy_router.domain.types import (
    ConfirmedPivot,
    DataStatus,
    FeatureObservation,
    MarketSnapshot,
    PivotKind,
    Side,
    StructuralLevel,
    Timeframe,
    TimeframeSnapshot,
)


BREAKOUT_PENETRATION_ATR = 0.10
RETEST_TOUCH_ATR = 0.20
COMMON_TARGET_ATR = 0.50
RANGE_MAX_PATH_EFFICIENCY = 0.35
RANGE_MAX_HTF_SLOPE_ATR = 0.05
RANGE_MIN_WIDTH_ATR = 1.00
RANGE_EDGE_ATR = 0.20


class FrozenRuleInputUnavailable(ValueError):
    """A frozen rule cannot be evaluated without inventing missing state."""


@dataclass(frozen=True, slots=True)
class RangeGeometry:
    support: StructuralLevel
    resistance: StructuralLevel
    midpoint: float
    width_atr: float
    distance_to_support_atr: float
    distance_to_resistance_atr: float


def direction(side: Side) -> float:
    return 1.0 if side is Side.LONG else -1.0


def timeframe_state(snapshot: MarketSnapshot, timeframe: Timeframe) -> TimeframeSnapshot:
    for state in snapshot.timeframes:
        if state.timeframe is timeframe:
            if state.status is not DataStatus.AVAILABLE:
                raise FrozenRuleInputUnavailable(f"{timeframe.value}:TIMEFRAME_UNAVAILABLE")
            return state
    raise FrozenRuleInputUnavailable(f"{timeframe.value}:TIMEFRAME_MISSING")


def feature_observation(
    snapshot: MarketSnapshot, timeframe: Timeframe, base_name: str
) -> FeatureObservation:
    state = timeframe_state(snapshot, timeframe)
    full_name = f"tf{timeframe.value}__{base_name}"
    for observation in state.features.observations:
        if observation.name == full_name:
            if observation.status is not DataStatus.AVAILABLE or observation.value is None:
                raise FrozenRuleInputUnavailable(f"{full_name}:FEATURE_UNAVAILABLE")
            observation.assert_available_by(snapshot.decision_at)
            return observation
    raise FrozenRuleInputUnavailable(f"{full_name}:FEATURE_MISSING")


def feature_value(snapshot: MarketSnapshot, timeframe: Timeframe, base_name: str) -> float:
    value = feature_observation(snapshot, timeframe, base_name).value
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise FrozenRuleInputUnavailable(f"tf{timeframe.value}__{base_name}:NOT_FINITE")
    return float(value)


def _structural(snapshot: MarketSnapshot, timeframe: Timeframe):
    state = timeframe_state(snapshot, timeframe)
    if state.structural is None or state.structural.status is not DataStatus.AVAILABLE:
        raise FrozenRuleInputUnavailable(f"{timeframe.value}:STRUCTURE_UNAVAILABLE")
    return state.structural


def latest_pivots(
    snapshot: MarketSnapshot, timeframe: Timeframe, kind: PivotKind, count: int = 2
) -> tuple[ConfirmedPivot, ...]:
    structural = _structural(snapshot, timeframe)
    values = tuple(sorted(
        (pivot for pivot in structural.pivots if pivot.kind is kind),
        key=lambda pivot: (pivot.available_at, pivot.pivot_at, pivot.bar_index, pivot.price),
    ))
    if len(values) < count:
        raise FrozenRuleInputUnavailable(f"{timeframe.value}:{kind.value}_PIVOTS_LT_{count}")
    return values[-count:]


def structure_aligned(snapshot: MarketSnapshot, timeframe: Timeframe, side: Side) -> bool:
    signed = direction(side)
    slope = feature_value(snapshot, timeframe, "ema25_slope_atr")
    return signed * slope > 0 and pivot_structure_aligned(snapshot, timeframe, side)


def pivot_structure_aligned(snapshot: MarketSnapshot, timeframe: Timeframe, side: Side) -> bool:
    signed = direction(side)
    highs = latest_pivots(snapshot, timeframe, PivotKind.HIGH)
    lows = latest_pivots(snapshot, timeframe, PivotKind.LOW)
    return (
        signed * (highs[-1].price - highs[-2].price) > 0
        and signed * (lows[-1].price - lows[-2].price) > 0
    )


def higher_timeframe_alignment(snapshot: MarketSnapshot, side: Side) -> bool:
    return all(structure_aligned(snapshot, timeframe, side) for timeframe in (Timeframe.H1, Timeframe.H4))


def most_recent_invalidation_pivot(
    snapshot: MarketSnapshot, timeframe: Timeframe, side: Side
) -> ConfirmedPivot:
    kind = PivotKind.LOW if side is Side.LONG else PivotKind.HIGH
    return latest_pivots(snapshot, timeframe, kind, count=1)[-1]


def closed_beyond_pivot(
    snapshot: MarketSnapshot,
    *,
    candle_timeframe: Timeframe,
    pivot_timeframe: Timeframe,
    side: Side,
) -> bool:
    candle = timeframe_state(snapshot, candle_timeframe).latest_candle
    if candle is None:
        raise FrozenRuleInputUnavailable(f"{candle_timeframe.value}:LATEST_CANDLE_MISSING")
    pivot = most_recent_invalidation_pivot(snapshot, pivot_timeframe, side)
    return candle.close < pivot.price if side is Side.LONG else candle.close > pivot.price


def trend_invalidation(snapshot: MarketSnapshot, side: Side) -> bool:
    return closed_beyond_pivot(
        snapshot, candle_timeframe=Timeframe.M15, pivot_timeframe=Timeframe.M15, side=side
    )


def pullback_invalidation(snapshot: MarketSnapshot, side: Side) -> bool:
    return closed_beyond_pivot(
        snapshot, candle_timeframe=Timeframe.M5, pivot_timeframe=Timeframe.H1, side=side
    )


def sustained_directional_move(snapshot: MarketSnapshot, side: Side) -> bool:
    signed = direction(side)
    return (
        signed * feature_value(snapshot, Timeframe.M5, "return_3_bps") > 0
        and signed * feature_value(snapshot, Timeframe.M15, "return_3_bps") > 0
        and feature_value(snapshot, Timeframe.M5, "path_efficiency_6") > 0
    )


def pullback_opposition(snapshot: MarketSnapshot, side: Side) -> bool:
    signed = direction(side)
    return all(
        signed * feature_value(snapshot, timeframe, "return_1_bps") < 0
        for timeframe in (Timeframe.M1, Timeframe.M5)
    )


def pullback_realigned(snapshot: MarketSnapshot, side: Side) -> bool:
    signed = direction(side)
    return (
        signed * feature_value(snapshot, Timeframe.M1, "return_1_bps") > 0
        and signed * feature_value(snapshot, Timeframe.M5, "return_1_bps") > 0
        and signed * feature_value(snapshot, Timeframe.M1, "taker_imbalance") > 0
    )


def _require_positive(atr: float, level_price: float) -> None:
    if not math.isfinite(atr) or atr <= 0 or not math.isfinite(level_price) or level_price <= 0:
        raise ValueError("ATR and level price must be finite and positive")


def breakout_close_confirmed(*, close: float, level_price: float, atr: float, side: Side) -> bool:
    _require_positive(atr, level_price)
    return direction(side) * (close - level_price) / atr >= BREAKOUT_PENETRATION_ATR


def retest_touch_and_close_confirmed(
    *, low: float, high: float, close: float, level_price: float, atr: float, side: Side
) -> bool:
    _require_positive(atr, level_price)
    if low > high:
        raise ValueError("low cannot exceed high")
    touched = low <= level_price + RETEST_TOUCH_ATR * atr and high >= level_price - RETEST_TOUCH_ATR * atr
    closes_on_breakout_side = close > level_price if side is Side.LONG else close < level_price
    return touched and closes_on_breakout_side


def closed_back_inside(*, close: float, level_price: float, side: Side) -> bool:
    return close < level_price if side is Side.LONG else close > level_price


def breakout_too_late(*, remaining_space_atr: float) -> bool:
    if not math.isfinite(remaining_space_atr) or remaining_space_atr < 0:
        raise ValueError("remaining_space_atr must be finite and non-negative")
    return remaining_space_atr < COMMON_TARGET_ATR


def breakout_levels(snapshot: MarketSnapshot, side: Side) -> tuple[StructuralLevel, ...]:
    state = timeframe_state(snapshot, Timeframe.M15)
    structural = _structural(snapshot, Timeframe.M15)
    candle = state.latest_candle
    if candle is None or structural.atr14 is None:
        raise FrozenRuleInputUnavailable("15m:BREAKOUT_INPUT_UNAVAILABLE")
    kind = PivotKind.HIGH if side is Side.LONG else PivotKind.LOW
    qualifying = [
        level for level in structural.levels
        if level.kind is kind
        and level.available_at <= candle.open_at
        and breakout_close_confirmed(
            close=candle.close, level_price=level.price, atr=structural.atr14, side=side
        )
    ]
    return tuple(sorted(
        qualifying,
        key=lambda level: (
            direction(side) * (candle.close - level.price) / structural.atr14,
            level.level_id,
        ),
    ))


def favorable_structural_space_atr(
    snapshot: MarketSnapshot, side: Side
) -> tuple[tuple[Timeframe, float], ...]:
    values = []
    for state in snapshot.timeframes:
        structural = state.structural
        if structural is None or structural.status is not DataStatus.AVAILABLE:
            continue
        distance = structural.nearest_above if side is Side.LONG else structural.nearest_below
        if distance is not None:
            values.append((state.timeframe, distance.distance_atr))
    return tuple(sorted(values, key=lambda item: item[0].minutes))


def common_target_space_available(snapshot: MarketSnapshot, side: Side) -> bool | None:
    values = favorable_structural_space_atr(snapshot, side)
    if not values:
        return None
    return min(distance for _, distance in values) >= COMMON_TARGET_ATR


def range_geometry(snapshot: MarketSnapshot) -> RangeGeometry | None:
    state = timeframe_state(snapshot, Timeframe.M15)
    structural = _structural(snapshot, Timeframe.M15)
    candle = state.latest_candle
    if candle is None or structural.atr14 is None:
        raise FrozenRuleInputUnavailable("15m:RANGE_INPUT_UNAVAILABLE")
    prior = tuple(level for level in structural.levels if level.available_at <= candle.open_at)
    supports = tuple(sorted(
        (level for level in prior if level.kind is PivotKind.LOW and level.price <= candle.close),
        key=lambda level: (-level.price, level.level_id),
    ))
    resistances = tuple(sorted(
        (level for level in prior if level.kind is PivotKind.HIGH and level.price >= candle.close),
        key=lambda level: (level.price, level.level_id),
    ))
    if not supports or not resistances:
        return None
    support, resistance = supports[0], resistances[0]
    width_atr = (resistance.price - support.price) / structural.atr14
    return RangeGeometry(
        support=support,
        resistance=resistance,
        midpoint=(support.price + resistance.price) / 2.0,
        width_atr=width_atr,
        distance_to_support_atr=(candle.close - support.price) / structural.atr14,
        distance_to_resistance_atr=(resistance.price - candle.close) / structural.atr14,
    )


def low_directional_efficiency(snapshot: MarketSnapshot) -> bool:
    return abs(feature_value(snapshot, Timeframe.M15, "path_efficiency_6")) <= RANGE_MAX_PATH_EFFICIENCY


def flat_higher_timeframe_slope(snapshot: MarketSnapshot) -> bool:
    return all(
        abs(feature_value(snapshot, timeframe, "ema25_slope_atr")) <= RANGE_MAX_HTF_SLOPE_ATR
        for timeframe in (Timeframe.H1, Timeframe.H4)
    )


def stable_range(snapshot: MarketSnapshot) -> tuple[bool, RangeGeometry | None]:
    geometry = range_geometry(snapshot)
    return (
        geometry is not None
        and geometry.width_atr >= RANGE_MIN_WIDTH_ATR
        and low_directional_efficiency(snapshot)
        and flat_higher_timeframe_slope(snapshot),
        geometry,
    )


def at_range_edge(geometry: RangeGeometry, side: Side) -> bool:
    distance = (
        geometry.distance_to_support_atr if side is Side.LONG
        else geometry.distance_to_resistance_atr
    )
    return 0 <= distance <= RANGE_EDGE_ATR


def range_rejection_confirmed(snapshot: MarketSnapshot, geometry: RangeGeometry, side: Side) -> bool:
    state = timeframe_state(snapshot, Timeframe.M15)
    structural = _structural(snapshot, Timeframe.M15)
    candle = state.latest_candle
    if candle is None or structural.atr14 is None:
        raise FrozenRuleInputUnavailable("15m:RANGE_REJECTION_INPUT_UNAVAILABLE")
    edge = geometry.support.price if side is Side.LONG else geometry.resistance.price
    touched = (
        candle.low <= edge + RANGE_EDGE_ATR * structural.atr14
        and candle.high >= edge - RANGE_EDGE_ATR * structural.atr14
    )
    closed_inward = candle.close > edge if side is Side.LONG else candle.close < edge
    return touched and closed_inward


def range_broken(snapshot: MarketSnapshot, geometry: RangeGeometry, side: Side) -> bool:
    candle = timeframe_state(snapshot, Timeframe.M15).latest_candle
    if candle is None:
        raise FrozenRuleInputUnavailable("15m:LATEST_CANDLE_MISSING")
    return (
        candle.close < geometry.support.price if side is Side.LONG
        else candle.close > geometry.resistance.price
    )


def new_structure(snapshot: MarketSnapshot, side: Side) -> bool:
    return pivot_structure_aligned(snapshot, Timeframe.M15, side)


def transition_slopes_confirm(snapshot: MarketSnapshot, side: Side) -> bool:
    signed = direction(side)
    return all(
        signed * feature_value(snapshot, timeframe, "ema25_slope_atr") > 0
        for timeframe in (Timeframe.M15, Timeframe.H1)
    )
