from __future__ import annotations

from training.train_decomposed_entry_v9_research import _control_rows, _derive_policy


def row(index: int, score: float = 0.8) -> dict[str, object]:
    return {
        "timestamp": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00+00:00",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "entry_brain_action": "SHORT",
        "v9_quality_score": score,
        "direction_probability": 0.8,
        "maximum_timing_risk": 0.1,
        "mae_q90": 0.003,
        "predicted_reward_risk": 2.0,
        "mae_fraction": 0.002,
        "time_underwater_bars": 2,
        "trajectory_targets": {
            "current_ts_stress_net": 0.002,
            "mae_fraction": 0.002,
        },
        "selected_profile": "CURRENT_TS",
        "selected_expected_net": 0.0025,
        "selected_stress_net": 0.002,
        "selected_severe_net": 0.0015,
    }


def config() -> dict[str, object]:
    return {
        "trajectory": {"tail_quantile": 0.10},
        "validation": {
            "minimum_calibration_selections": 40,
            "score_quantiles": [0.5],
            "direction_probability_quantiles": [0.5],
            "maximum_timing_risk_quantiles": [0.5],
            "maximum_mae_quantiles": [0.5],
            "minimum_reward_risk_quantiles": [0.5],
            "maximum_selected_per_timestamp": 1,
        },
    }


def test_policy_is_calibration_only_and_requires_frequency() -> None:
    assert _derive_policy([row(index) for index in range(20)], config()) is None
    policy = _derive_policy([row(index) for index in range(80)], config())
    assert policy is not None
    assert policy["source"] == "CALIBRATION_ONLY"


def test_control_uses_current_ts_without_profile_optimization() -> None:
    values, identity = _control_rows([row(0)], "SHORT", 0.0015)
    assert identity == "CURRENT_BRAIN"
    assert values[0]["selected_profile"] == "CURRENT_TS"
    assert values[0]["selected_stress_net"] == 0.002
