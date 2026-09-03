from __future__ import annotations

import pytest

from aegis.research.regime_entry_exit_v7_training import (
    fold_passes,
    joint_quality_score,
    select_v7_cross_section,
    v7_ablation_score,
    v7_selection_metrics,
)


def row(symbol: str, score: float, net: float = 0.002) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "v7_quality_score": score,
        "late_probability": 0.1,
        "mae_q90": 0.003,
        "mae_fraction": 0.002,
        "selected_profile": "LOCK_AT_10_ROE",
        "selected_profile_net": net,
        "selected_capture_efficiency": 0.6,
        "time_underwater_bars": 2,
        "v7_archetype": "TREND_CONTINUATION",
        "trajectory_attribution": {"clean_entry": True, "late_entry": False},
    }


def test_joint_score_rewards_clean_fast_profitable_path() -> None:
    clean = joint_quality_score(
        clean_probability=0.8,
        positive_probability=0.8,
        late_probability=0.1,
        expected_profile_net=0.003,
        mae_q90=0.002,
        time_to_positive=0.2,
        capture_efficiency=0.7,
    )
    late = joint_quality_score(
        clean_probability=0.4,
        positive_probability=0.5,
        late_probability=0.8,
        expected_profile_net=-0.001,
        mae_q90=0.012,
        time_to_positive=0.9,
        capture_efficiency=0.1,
    )
    assert clean > late


def test_selection_does_not_force_a_candidate() -> None:
    policy = {
        "minimum_score": 0.5,
        "maximum_late_probability": 0.4,
        "maximum_mae_q90": 0.01,
        "maximum_selected_per_timestamp": 1,
    }
    assert select_v7_cross_section(
        [row("BTCUSDT", 0.8), row("ETHUSDT", 0.7)], policy
    ) == (
        True,
        False,
    )
    assert select_v7_cross_section([row("BTCUSDT", 0.2)], policy) == (False,)


def test_ablations_measure_components_without_becoming_votes() -> None:
    value = {
        "clean_probability": 0.8,
        "positive_probability": 0.7,
        "late_probability": 0.2,
        "expected_profile_net": 0.002,
        "mae_q90": 0.003,
        "predicted_time_to_positive": 0.2,
        "predicted_capture_efficiency": 0.6,
    }
    assert v7_ablation_score(value, "ENTRY_ONLY") == pytest.approx(0.56)
    assert v7_ablation_score(value, "NO_LATE_PENALTY") > v7_ablation_score(
        value, "FULL"
    )


def test_metrics_and_gate_require_net_mae_capture_and_frequency() -> None:
    values = [
        row("BTCUSDT", 0.8),
        {**row("ETHUSDT", 0.7), "timestamp": "2026-01-01T00:30:00+00:00"},
    ]
    metrics = v7_selection_metrics(values)
    control = {
        **metrics,
        "mean_net": 0.001,
        "mean_mae": 0.003,
        "mean_capture_efficiency": 0.5,
    }
    assert fold_passes(metrics, control, minimum_count=2, maximum_p95_gap_hours=1.0)
    assert not fold_passes(
        {**metrics, "mean_net": -0.001},
        control,
        minimum_count=2,
        maximum_p95_gap_hours=1.0,
    )
