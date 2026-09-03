from __future__ import annotations

from training.train_tail_aware_entry_v8_research import _control_rows, _derive_policy


def row(index: int, score: float = 0.8) -> dict[str, object]:
    profile = {"expected": 0.0025, "stress": 0.002, "severe": 0.0015}
    return {
        "timestamp": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00+00:00",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "entry_brain_action": "SHORT",
        "v8_quality_score": score,
        "late_probability": 0.1,
        "catastrophic_probability": 0.05,
        "mae_q90": 0.003,
        "mae_fraction": 0.002,
        "time_underwater_bars": 2,
        "profile_returns": {"CURRENT_TS": profile, "STOP_15_LOCK_10": profile},
        "selected_profile": "STOP_15_LOCK_10",
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
            "maximum_late_probability_quantiles": [0.5],
            "maximum_catastrophic_probability_quantiles": [0.5],
            "maximum_mae_quantiles": [0.5],
            "maximum_selected_per_timestamp": 1,
        },
    }


def test_v8_policy_is_calibration_only_and_requires_frequency() -> None:
    assert _derive_policy([row(index) for index in range(20)], config()) is None
    policy = _derive_policy([row(index) for index in range(80)], config())
    assert policy is not None
    assert policy["source"] == "CALIBRATION_ONLY"


def test_v8_control_uses_current_ts_with_all_cost_scenarios() -> None:
    values, identity = _control_rows([row(0)], "SHORT")
    assert identity == "CURRENT_BRAIN"
    assert values[0]["selected_profile"] == "CURRENT_TS"
    assert values[0]["selected_expected_net"] == 0.0025
    assert values[0]["selected_stress_net"] == 0.002
    assert values[0]["selected_severe_net"] == 0.0015
