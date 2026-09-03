from __future__ import annotations

from train_regime_entry_exit_v7_research import _control_rows, _derive_policy


def row(index: int, score: float = 0.8) -> dict[str, object]:
    return {
        "timestamp": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00+00:00",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "entry_brain_action": "SHORT",
        "v7_quality_score": score,
        "late_probability": 0.1,
        "mae_q90": 0.003,
        "mae_fraction": 0.002,
        "time_underwater_bars": 2,
        "v7_archetype": "TREND_CONTINUATION",
        "profile_returns": {"CURRENT_TS": 0.002},
        "trajectory_attribution": {
            "available_net_opportunity": 0.004,
            "clean_entry": True,
            "late_entry": False,
        },
        "selected_profile": "CURRENT_TS",
        "selected_profile_net": 0.002,
        "selected_capture_efficiency": 0.5,
    }


def test_policy_is_calibration_only_and_requires_frequency() -> None:
    config = {
        "validation": {
            "minimum_calibration_selections": 40,
            "score_quantiles": [0.5],
            "maximum_late_probability_quantiles": [0.5],
            "maximum_mae_quantiles": [0.5],
            "maximum_selected_per_timestamp": 1,
        }
    }
    assert _derive_policy([row(index) for index in range(20)], config) is None
    policy = _derive_policy([row(index) for index in range(80)], config)
    assert policy is not None
    assert policy["source"] == "CALIBRATION_ONLY"


def test_current_brain_control_uses_current_ts_protection() -> None:
    values, identity = _control_rows([row(0)], "SHORT")
    assert identity == "CURRENT_BRAIN"
    assert values[0]["selected_profile"] == "CURRENT_TS"
    assert values[0]["selected_capture_efficiency"] == 0.5
