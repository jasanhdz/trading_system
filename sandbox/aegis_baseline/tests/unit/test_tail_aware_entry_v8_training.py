from __future__ import annotations

from aegis.research.tail_aware_entry_v8_training import (
    binary_skill_metrics,
    fold_passes,
    select_tail_aware_cross_section,
    tail_aware_quality_score,
    tail_selection_metrics,
)


def row(symbol: str, score: float, stress: float = 0.003) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "v8_quality_score": score,
        "late_probability": 0.1,
        "catastrophic_probability": 0.05,
        "mae_q90": 0.003,
        "mae_fraction": 0.002,
        "time_underwater_bars": 2,
        "selected_profile": "STOP_15_LOCK_10",
        "selected_expected_net": stress + 0.0005,
        "selected_stress_net": stress,
        "selected_severe_net": stress - 0.0005,
    }


def test_tail_score_penalizes_late_and_catastrophic_paths() -> None:
    clean = tail_aware_quality_score(
        clean_probability=0.8,
        positive_probability=0.8,
        late_probability=0.1,
        catastrophic_probability=0.05,
        expected_stress_net=0.003,
        mae_q90=0.003,
        time_to_positive=0.2,
    )
    risky = tail_aware_quality_score(
        clean_probability=0.8,
        positive_probability=0.8,
        late_probability=0.7,
        catastrophic_probability=0.5,
        expected_stress_net=-0.002,
        mae_q90=0.012,
        time_to_positive=0.8,
    )
    assert clean > risky


def test_selection_can_abstain_and_limits_cross_section() -> None:
    policy = {
        "minimum_score": 0.5,
        "maximum_late_probability": 0.4,
        "maximum_catastrophic_probability": 0.3,
        "maximum_mae_q90": 0.01,
        "maximum_selected_per_timestamp": 1,
    }
    assert select_tail_aware_cross_section(
        [row("BTCUSDT", 0.8), row("ETHUSDT", 0.7)], policy
    ) == (True, False)
    assert select_tail_aware_cross_section([row("BTCUSDT", 0.2)], policy) == (False,)


def test_binary_skill_must_beat_prevalence_baselines() -> None:
    skilled = binary_skill_metrics([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    constant = binary_skill_metrics([0.5, 0.5, 0.5, 0.5], [False, False, True, True])
    assert skilled["passed"] is True
    assert constant["passed"] is False


def test_tail_metrics_and_gate_require_payoff_and_cvar() -> None:
    values = [
        row("BTCUSDT", 0.8, 0.004),
        {**row("ETHUSDT", 0.7, -0.002), "timestamp": "2026-01-01T00:30:00+00:00"},
    ]
    metrics = tail_selection_metrics(values, tail_quantile=0.10)
    control = {
        **metrics,
        "mean_stress_net": -0.001,
        "stress_cvar": -0.004,
        "mean_mae": 0.003,
    }
    assert fold_passes(
        metrics,
        control,
        minimum_count=2,
        minimum_payoff=1.0,
        maximum_p95_gap_hours=1.0,
    )
