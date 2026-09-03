from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aegis.research.competing_barrier_v10_training import (
    fold_passes,
    select_cross_section,
    utility_metrics,
)


def candidate(symbol: str, utility: float, actual: float = 0.01) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "timestamp_value": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "symbol": symbol,
        "side": "LONG",
        "predicted_utility": utility,
        "direction_probability": 0.8,
        "unknown_probability": 0.1,
        "actual_utility": actual,
    }


def test_selection_abstains_and_limits_cross_section() -> None:
    policy = {
        "minimum_utility": 0.001,
        "minimum_direction_probability": 0.6,
        "maximum_unknown_probability": 0.3,
        "maximum_selected_per_timestamp": 1,
    }
    assert select_cross_section(
        [candidate("BTCUSDT", 0.01), candidate("ETHUSDT", 0.005)], policy
    ) == (True, False)
    assert select_cross_section([candidate("BTCUSDT", -0.01)], policy) == (False,)


def test_utility_metrics_include_tail_payoff_and_frequency() -> None:
    rows = [candidate("BTCUSDT", 0.01, value) for value in (0.02, 0.01, -0.005)]
    for index, row in enumerate(rows):
        row["timestamp_value"] += timedelta(hours=index)
    metrics = utility_metrics(rows)
    assert metrics["mean_utility"] > 0.0
    assert metrics["cvar"] == -0.005
    assert metrics["payoff_ratio"] == 3.0
    assert metrics["p95_gap_hours"] == 1.0


def test_fold_gate_requires_positive_incremental_tail_economics() -> None:
    selected = {
        "count": 20,
        "mean_utility": 0.002,
        "cvar": -0.004,
        "payoff_ratio": 1.2,
        "p95_gap_hours": 12.0,
    }
    control = {"mean_utility": -0.001, "cvar": -0.01}
    assert fold_passes(
        selected,
        control,
        minimum_count=15,
        minimum_payoff=1.0,
        maximum_p95_gap_hours=96.0,
    )
    selected["mean_utility"] = -0.0001
    assert not fold_passes(
        selected,
        control,
        minimum_count=15,
        minimum_payoff=1.0,
        maximum_p95_gap_hours=96.0,
    )
