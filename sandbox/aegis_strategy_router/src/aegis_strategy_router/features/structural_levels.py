"""Causal pivots and deterministic complete-linkage structural levels."""

from __future__ import annotations

import math
from datetime import datetime
from statistics import median
from typing import Iterable

from aegis_strategy_router.domain.serialization import content_hash, utc_datetime
from aegis_strategy_router.domain.types import (
    Candle,
    ConfirmedPivot,
    DataStatus,
    LevelDistance,
    PivotKind,
    StructuralContext,
    StructuralLevel,
    Timeframe,
)


CLUSTERING_METHOD = "COMPLETE_LINKAGE_DETERMINISTIC"
CLUSTER_TOLERANCE_ATR = 0.20
ATR_PERIOD = 14


def extract_confirmed_pivots(
    candles: Iterable[Candle],
    *,
    decision_at: datetime,
    lookback_bars: int,
) -> tuple[ConfirmedPivot, ...]:
    """Extract strict L=R=2 pivots only after the second right bar closes."""
    boundary = utc_datetime(decision_at)
    closed = _causal_candles(candles, boundary)
    lower_index = max(2, len(closed) - lookback_bars)
    pivots = []
    for index in range(lower_index, len(closed) - 2):
        center = closed[index]
        neighbors = (closed[index - 2], closed[index - 1], closed[index + 1], closed[index + 2])
        available_at = max(center.available_at, closed[index + 2].available_at)
        if available_at > boundary:
            continue
        if all(center.high > candle.high for candle in neighbors):
            pivots.append(ConfirmedPivot(PivotKind.HIGH, center.high, index, center.close_at, available_at))
        if all(center.low < candle.low for candle in neighbors):
            pivots.append(ConfirmedPivot(PivotKind.LOW, center.low, index, center.close_at, available_at))
    return tuple(sorted(pivots, key=_pivot_identity))


def wilder_atr14(candles: Iterable[Candle], *, decision_at: datetime) -> float | None:
    """Match the frozen Wilder-style EWM alpha=1/14 used by existing features."""
    closed = _causal_candles(candles, utc_datetime(decision_at))
    if len(closed) < ATR_PERIOD:
        return None
    true_ranges = []
    previous_close = None
    for candle in closed:
        candidates = [candle.high - candle.low]
        if previous_close is not None:
            candidates.extend((abs(candle.high - previous_close), abs(candle.low - previous_close)))
        true_ranges.append(max(candidates))
        previous_close = candle.close
    alpha = 1.0 / ATR_PERIOD
    atr = true_ranges[0]
    for value in true_ranges[1:]:
        atr = (1.0 - alpha) * atr + alpha * value
    return float(atr) if math.isfinite(atr) and atr > 0 else None


def cluster_confirmed_pivots(
    pivots: Iterable[ConfirmedPivot],
    *,
    timeframe: Timeframe,
    tolerance: float,
) -> tuple[StructuralLevel, ...]:
    """Agglomerate causal pivots with frozen complete-linkage semantics."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("cluster tolerance must be finite and positive")
    values = tuple(sorted(pivots, key=_pivot_identity))
    levels = []
    for kind in (PivotKind.HIGH, PivotKind.LOW):
        clusters: list[tuple[ConfirmedPivot, ...]] = [
            (pivot,) for pivot in values if pivot.kind is kind
        ]
        while True:
            candidates = []
            for left_index in range(len(clusters)):
                for right_index in range(left_index + 1, len(clusters)):
                    merged = tuple(sorted((*clusters[left_index], *clusters[right_index]), key=_pivot_identity))
                    distance = _complete_linkage_distance(merged)
                    if distance <= tolerance and _touch_spacing_valid(merged):
                        key = (distance, tuple(_pivot_identity(pivot) for pivot in merged))
                        candidates.append((key, left_index, right_index, merged))
            if not candidates:
                break
            _, left_index, right_index, merged = min(candidates, key=lambda item: item[0])
            clusters = [
                cluster for index, cluster in enumerate(clusters)
                if index not in {left_index, right_index}
            ]
            clusters.append(merged)
            clusters.sort(key=lambda cluster: tuple(_pivot_identity(pivot) for pivot in cluster))
        levels.extend(
            _level_from_cluster(cluster, timeframe)
            for cluster in clusters
            if len(cluster) >= 2
        )
    return tuple(sorted(levels, key=lambda level: (level.price, level.kind.value, level.level_id)))


class StructuralLevelAdapter:
    """Build complete causal level context without outcome information."""

    def context(
        self,
        candles: Iterable[Candle],
        *,
        timeframe: Timeframe,
        decision_at: datetime,
        reference_price: float | None = None,
    ) -> StructuralContext:
        lookback = timeframe.structural_lookback
        if lookback is None:
            raise ValueError(f"{timeframe.value} has no frozen structural-level lookback")
        boundary = utc_datetime(decision_at)
        closed = _causal_candles(candles, boundary)
        pivots = extract_confirmed_pivots(closed, decision_at=boundary, lookback_bars=lookback)
        atr14 = wilder_atr14(closed, decision_at=boundary)
        if atr14 is None:
            return StructuralContext(
                status=DataStatus.UNKNOWN,
                pivots=pivots,
                reason="STRUCTURAL_ATR14_WARMUP_INCOMPLETE",
            )
        current_price = float(reference_price if reference_price is not None else closed[-1].close)
        if not math.isfinite(current_price) or current_price <= 0:
            return StructuralContext(
                status=DataStatus.INVALID,
                pivots=pivots,
                reason="STRUCTURAL_REFERENCE_PRICE_INVALID",
            )
        tolerance = CLUSTER_TOLERANCE_ATR * atr14
        levels = cluster_confirmed_pivots(pivots, timeframe=timeframe, tolerance=tolerance)
        below = sorted(
            (level for level in levels if level.price <= current_price),
            key=lambda level: (-level.price, level.level_id),
        )
        above = sorted(
            (level for level in levels if level.price >= current_price),
            key=lambda level: (level.price, level.level_id),
        )
        return StructuralContext(
            status=DataStatus.AVAILABLE,
            pivots=pivots,
            levels=levels,
            atr14=atr14,
            cluster_tolerance=tolerance,
            reference_price=current_price,
            nearest_below=_distance(below[0], current_price, atr14) if below else None,
            nearest_above=_distance(above[0], current_price, atr14) if above else None,
        )

    def clustered_levels(
        self,
        pivots: Iterable[ConfirmedPivot],
        *,
        timeframe: Timeframe,
        tolerance: float,
    ) -> tuple[StructuralLevel, ...]:
        return cluster_confirmed_pivots(pivots, timeframe=timeframe, tolerance=tolerance)


def _causal_candles(candles: Iterable[Candle], boundary: datetime) -> tuple[Candle, ...]:
    closed = tuple(sorted(
        (
            candle for candle in candles
            if candle.complete and candle.close_at <= boundary and candle.available_at <= boundary
        ),
        key=lambda candle: candle.close_at,
    ))
    if len({candle.open_at for candle in closed}) != len(closed):
        raise ValueError("duplicate candle timestamps prevent structural reconstruction")
    return closed


def _pivot_identity(pivot: ConfirmedPivot) -> tuple[datetime, datetime, int, float]:
    return (pivot.pivot_at, pivot.available_at, pivot.bar_index, pivot.price)


def _complete_linkage_distance(cluster: tuple[ConfirmedPivot, ...]) -> float:
    prices = tuple(pivot.price for pivot in cluster)
    return max(prices) - min(prices)


def _touch_spacing_valid(cluster: tuple[ConfirmedPivot, ...]) -> bool:
    indices = sorted(pivot.bar_index for pivot in cluster)
    return all(right - left >= 3 for left, right in zip(indices, indices[1:]))


def _level_from_cluster(cluster: tuple[ConfirmedPivot, ...], timeframe: Timeframe) -> StructuralLevel:
    ordered = tuple(sorted(cluster, key=lambda pivot: pivot.bar_index))
    level_id = content_hash({
        "timeframe": timeframe,
        "kind": ordered[0].kind,
        "pivots": [_pivot_identity(pivot) for pivot in ordered],
    })
    return StructuralLevel(
        level_id=level_id,
        timeframe=timeframe,
        kind=ordered[0].kind,
        price=float(median(pivot.price for pivot in ordered)),
        touch_count=len(ordered),
        pivot_indices=tuple(pivot.bar_index for pivot in ordered),
        pivot_prices=tuple(pivot.price for pivot in ordered),
        first_touch_at=min(pivot.pivot_at for pivot in ordered),
        last_touch_at=max(pivot.pivot_at for pivot in ordered),
        available_at=max(pivot.available_at for pivot in ordered),
    )


def _distance(level: StructuralLevel, reference_price: float, atr14: float) -> LevelDistance:
    absolute = abs(level.price - reference_price)
    return LevelDistance(
        level_id=level.level_id,
        price=level.price,
        distance_atr=absolute / atr14,
        distance_bps=absolute / reference_price * 10_000.0,
    )

