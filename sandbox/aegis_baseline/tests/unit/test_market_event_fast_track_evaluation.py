from dataclasses import replace

import pytest

from aegis.domain import TradeSide
from aegis.research.market_event_fast_track_evaluation import (
    assess_pattern_gate,
    evaluate_candidate,
    summarize_economics,
)
from aegis.research.market_event_fast_track_m1a import (
    DirectionAxis,
    MicroPattern,
    PatternCandidate,
    VolatilityAxis,
)
from test_market_event_fast_track_m1a import _bars


def _candidate(timestamp):
    return PatternCandidate(
        MicroPattern.COMPRESSION_BREAKOUT,
        TradeSide.LONG,
        "ADAUSDT",
        timestamp,
        DirectionAxis.BULL,
        VolatilityAxis.EXPANDING,
        "f" * 64,
        ("BREAKOUT",),
    )


def test_evaluation_uses_next_minute_costs_and_side_correct_path() -> None:
    bars = _bars(20, drift=0.001)
    candidate = _candidate(bars[0].open_time_ms - 1)
    row = evaluate_candidate(candidate, bars, horizon_minutes=10)
    assert row.entry_timestamp_ms == candidate.timestamp_ms + 1
    assert row.gross_return_fraction > 0.0
    assert row.net_return_fraction < row.gross_return_fraction
    assert row.mfe_fraction > 0.0
    assert row.mae_fraction > 0.0
    short = evaluate_candidate(
        replace(candidate, side=TradeSide.SHORT), _bars(20, drift=-0.001), horizon_minutes=10
    )
    assert short.gross_return_fraction > 0.0


def test_evaluation_fails_closed_on_gap_and_incomplete_path() -> None:
    bars = _bars(20)
    candidate = _candidate(bars[0].open_time_ms - 1)
    with pytest.raises(Exception, match="PATH_INCOMPLETE"):
        evaluate_candidate(candidate, bars, horizon_minutes=21)
    with pytest.raises(Exception, match="PATH_GAP"):
        evaluate_candidate(candidate, bars[:5] + bars[6:], horizon_minutes=10)


def test_bootstrap_metrics_and_gate_require_real_stable_edge() -> None:
    source = _bars(20, drift=0.001)
    rows = []
    for index in range(120):
        candidate = replace(
            _candidate(source[0].open_time_ms - 1 + index * 86_400_000),
            symbol=("ADAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT")[index % 4],
        )
        future = tuple(replace(bar, open_time_ms=bar.open_time_ms + index * 86_400_000) for bar in source)
        rows.append(evaluate_candidate(candidate, future, horizon_minutes=10))
    metrics = summarize_economics(rows, bootstrap_repetitions=200)
    assert metrics.events == 120
    assert metrics.expectancy_ci_95[0] > 0.0
    assert metrics.profit_factor_ci_95[0] > 1.0
    assert assess_pattern_gate(
        metrics, matched_random_expectancy=0.0, stress_expectancy=0.001
    ).passed
    failed = assess_pattern_gate(
        replace(metrics, events=99), matched_random_expectancy=metrics.net_expectancy,
        stress_expectancy=-0.001,
    )
    assert not failed.passed
    assert "EVENT_COUNT_LT_100" in failed.blockers
