from __future__ import annotations

from aegis.research.decomposed_entry_v9_training import (
    decomposed_quality_score,
    quantile_skill,
    regression_skill,
    select_decomposed_cross_section,
)


def candidate(symbol: str, score: float) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "v9_quality_score": score,
        "direction_probability": 0.8,
        "maximum_timing_risk": 0.1,
        "mae_q90": 0.003,
        "predicted_reward_risk": 2.0,
    }


def test_quality_requires_direction_timing_and_reward_risk() -> None:
    strong, strong_ratio = decomposed_quality_score(
        direction_probability=0.8,
        positive_probability=0.8,
        maximum_timing_risk=0.1,
        catastrophic_probability=0.05,
        expected_stress_net=0.003,
        mae_q90=0.003,
        mfe_q50=0.012,
        time_to_positive=0.2,
        stress_cost_fraction=0.0015,
    )
    weak, weak_ratio = decomposed_quality_score(
        direction_probability=0.4,
        positive_probability=0.5,
        maximum_timing_risk=0.8,
        catastrophic_probability=0.4,
        expected_stress_net=-0.002,
        mae_q90=0.012,
        mfe_q50=0.003,
        time_to_positive=0.8,
        stress_cost_fraction=0.0015,
    )
    assert strong > weak
    assert strong_ratio > weak_ratio


def test_selection_abstains_and_limits_each_timestamp() -> None:
    policy = {
        "minimum_score": 0.5,
        "minimum_direction_probability": 0.6,
        "maximum_timing_risk": 0.4,
        "maximum_mae_q90": 0.01,
        "minimum_reward_risk": 1.0,
        "maximum_selected_per_timestamp": 1,
    }
    assert select_decomposed_cross_section(
        [candidate("BTCUSDT", 0.8), candidate("ETHUSDT", 0.7)], policy
    ) == (True, False)
    assert select_decomposed_cross_section([candidate("BTCUSDT", 0.2)], policy) == (
        False,
    )


def test_regression_must_beat_training_constant() -> None:
    assert regression_skill([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.0)["passed"] is True
    assert regression_skill([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 1.0)["passed"] is False
    assert quantile_skill([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.0, 0.9)["passed"] is True
