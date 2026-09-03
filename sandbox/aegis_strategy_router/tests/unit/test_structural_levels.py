from datetime import datetime, timedelta, timezone

from aegis_strategy_router.domain.types import Candle, ConfirmedPivot, DataStatus, PivotKind, Timeframe
from aegis_strategy_router.features.structural_levels import (
    StructuralLevelAdapter,
    cluster_confirmed_pivots,
    extract_confirmed_pivots,
)


START = datetime(2026, 8, 17, 0, 0, tzinfo=timezone.utc)


def _candles() -> tuple[Candle, ...]:
    highs = [10.0, 11.0, 15.0, 12.0, 11.0, 12.0, 11.0]
    lows = [8.0, 8.5, 9.0, 8.7, 8.6, 8.8, 8.7]
    return tuple(_candle(index, high, low) for index, (high, low) in enumerate(zip(highs, lows)))


def _candle(index: int, high: float, low: float) -> Candle:
    open_at = START + timedelta(minutes=15 * index)
    close_at = open_at + timedelta(minutes=15)
    middle = (high + low) / 2.0
    return Candle(open_at, close_at, middle, high, low, middle, 100.0, 50.0, close_at, str(index))


def _pivot(index: int, price: float, kind: PivotKind = PivotKind.HIGH) -> ConfirmedPivot:
    pivot_at = START + timedelta(minutes=15 * index)
    available_at = pivot_at + timedelta(minutes=30)
    return ConfirmedPivot(kind, price, index, pivot_at, available_at)


def test_pivot_is_unavailable_until_two_right_bars_close() -> None:
    candles = _candles()
    before = extract_confirmed_pivots(candles, decision_at=candles[3].close_at, lookback_bars=96)
    after = extract_confirmed_pivots(candles, decision_at=candles[4].close_at, lookback_bars=96)
    assert before == ()
    assert len(after) == 1
    assert after[0].kind is PivotKind.HIGH
    assert after[0].price == 15.0
    assert after[0].available_at == candles[4].close_at


def test_complete_linkage_does_not_chain_overlapping_clusters() -> None:
    pivots = (_pivot(2, 100.0), _pivot(5, 101.0), _pivot(8, 102.0))
    levels = cluster_confirmed_pivots(pivots, timeframe=Timeframe.M15, tolerance=1.1)
    assert len(levels) == 1
    assert levels[0].pivot_indices == (2, 5)
    assert levels[0].price == 100.5
    assert 8 not in levels[0].pivot_indices


def test_complete_linkage_is_deterministic_for_input_order_and_ties() -> None:
    pivots = (_pivot(2, 100.0), _pivot(5, 101.0), _pivot(8, 102.0))
    forward = cluster_confirmed_pivots(pivots, timeframe=Timeframe.M15, tolerance=1.1)
    reversed_input = cluster_confirmed_pivots(reversed(pivots), timeframe=Timeframe.M15, tolerance=1.1)
    assert forward == reversed_input
    assert forward[0].pivot_indices == (2, 5)


def test_touch_spacing_and_pivot_kind_are_enforced() -> None:
    too_close = (_pivot(2, 100.0), _pivot(4, 100.0))
    assert cluster_confirmed_pivots(too_close, timeframe=Timeframe.M15, tolerance=1.0) == ()

    separated_kinds = (_pivot(2, 100.0, PivotKind.HIGH), _pivot(5, 100.0, PivotKind.LOW))
    assert cluster_confirmed_pivots(separated_kinds, timeframe=Timeframe.M15, tolerance=1.0) == ()


def test_future_candles_cannot_rewrite_historical_structural_context() -> None:
    candles = tuple(
        _candle(
            index,
            101.0 + (3.0 if index % 6 == 2 else 0.2 * (index % 3)),
            99.0 - (2.0 if index % 7 == 3 else 0.1 * (index % 2)),
        )
        for index in range(30)
    )
    decision = candles[19].close_at
    adapter = StructuralLevelAdapter()
    prefix_context = adapter.context(
        candles[:20], timeframe=Timeframe.M15, decision_at=decision, reference_price=100.0
    )
    future_extended_context = adapter.context(
        candles, timeframe=Timeframe.M15, decision_at=decision, reference_price=100.0
    )
    assert prefix_context.status is DataStatus.AVAILABLE
    assert prefix_context == future_extended_context
    assert prefix_context.levels
    assert prefix_context.nearest_above is not None
    assert all(pivot.available_at <= decision for pivot in future_extended_context.pivots)
    assert all(level.available_at <= decision for level in future_extended_context.levels)
