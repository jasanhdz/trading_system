from __future__ import annotations

import pytest

from aegis.research.regime_aware_directional_v6_training import (
    ablation_score,
    bootstrap_mean_interval,
    quality_score,
    reliability_table,
    select_ablation_cross_section,
    select_cross_section,
    selection_metrics,
)


def row(symbol: str, score: float, mae: float = 0.004) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "quality_score": score,
        "mae_q90": mae,
        "time_to_advantage": 0.2,
        "early_reversal_probability": 0.1,
        "full_lifecycle_worst_net_return": 0.002,
        "mae_fraction": 0.003,
        "time_underwater_bars": 2,
        "protectable_advantage": True,
        "target_before_stop": True,
        "early_reversal": False,
        "full_lifecycle_worst_bars_held": 3,
    }


def test_quality_score_rewards_edge_and_penalizes_mae() -> None:
    clean = quality_score(
        protectable_probability=0.8,
        target_probability=0.7,
        early_reversal_probability=0.1,
        expected_protected_net=0.002,
        mae_q90=0.003,
        time_to_advantage=0.2,
    )
    risky = quality_score(
        protectable_probability=0.8,
        target_probability=0.7,
        early_reversal_probability=0.3,
        expected_protected_net=-0.001,
        mae_q90=0.012,
        time_to_advantage=0.8,
    )
    assert clean > risky


def test_ablation_scores_do_not_fabricate_directional_votes() -> None:
    candidate = {
        "protectable_probability": 0.8,
        "target_probability": 0.6,
        "early_reversal_probability": 0.1,
        "expected_protected_net": 0.002,
        "mae_q90": 0.003,
        "time_to_advantage": 0.2,
    }
    assert ablation_score(candidate, "PROTECTABLE_ONLY") == pytest.approx(0.8)
    assert ablation_score(candidate, "REVERSAL_ONLY") == pytest.approx(0.9)
    assert ablation_score(candidate, "FULL_COMMITTEE") == pytest.approx(
        quality_score(**candidate)
    )


def test_ablation_selection_uses_threshold_and_cross_sectional_limit() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "symbol": symbol,
            "protectable_probability": probability,
            "target_probability": 0.5,
            "early_reversal_probability": 0.2,
            "expected_protected_net": 0.001,
            "mae_q90": 0.004,
            "time_to_advantage": 0.3,
        }
        for symbol, probability in (("BTCUSDT", 0.7), ("ETHUSDT", 0.9))
    ]
    assert select_ablation_cross_section(
        rows,
        variant="PROTECTABLE_ONLY",
        minimum_score=0.6,
        maximum_selected_per_timestamp=1,
    ) == (False, True)


def test_selection_ranks_without_forcing_a_trade() -> None:
    policy = {
        "minimum_score": 0.5,
        "maximum_mae_q90": 0.01,
        "maximum_time_to_advantage": 0.5,
        "maximum_early_reversal_probability": 0.5,
        "maximum_selected_per_timestamp": 1,
    }
    assert select_cross_section([row("BTCUSDT", 0.8), row("ETHUSDT", 0.7)], policy) == (
        True,
        False,
    )
    assert select_cross_section([row("BTCUSDT", 0.2)], policy) == (False,)


def test_metrics_reliability_and_bootstrap_are_deterministic() -> None:
    rows = [row("BTCUSDT", 0.8), row("ETHUSDT", 0.7)]
    metrics = selection_metrics(rows)
    assert metrics["count"] == 2
    assert metrics["mean_protected_net"] == 0.002
    table = reliability_table([0.1, 0.9], [False, True])
    assert table[1]["observed_rate"] == 0.0
    assert table[9]["observed_rate"] == 1.0
    first = bootstrap_mean_interval([0.1, 0.2, 0.3], samples=100, seed=7)
    second = bootstrap_mean_interval([0.1, 0.2, 0.3], samples=100, seed=7)
    assert first == second


def test_metrics_measure_opportunity_gaps_on_unique_timestamps() -> None:
    first = row("BTCUSDT", 0.8)
    duplicate = row("ETHUSDT", 0.7)
    later = {
        **row("SOLUSDT", 0.9),
        "timestamp": "2026-01-01T05:00:00+00:00",
    }
    metrics = selection_metrics([first, duplicate, later])
    assert metrics["p95_gap_hours"] == 5.0
    assert metrics["maximum_gap_hours"] == 5.0
