from __future__ import annotations

from build_tail_aware_entry_v8_dataset import _cost_adjusted, _profile_grid


def config() -> dict[str, object]:
    return {
        "protection_grid": {
            "hard_stop_roe": [-0.15, -0.25, -0.40],
            "lock_trigger_roe": [0.05, 0.10, 0.20],
            "retained_trigger_fraction": 0.5,
            "trailing_callback_roe": 0.08,
            "take_profit_roe": 0.50,
            "leverage": 15.0,
            "use_atr_trailing": True,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "current_ts_control": {
                "hard_stop_roe": -0.40,
                "break_even_trigger_roe": 0.08,
                "break_even_offset_fraction": 0.003,
                "trailing_activation_roe": 0.15,
                "trailing_callback_roe": 0.08,
            },
        },
        "cost_sensitivity": {"replay_base_round_trip_fraction": 0.001},
    }


def test_profile_grid_contains_control_and_nine_counterfactuals() -> None:
    profiles = _profile_grid(config())
    assert len(profiles) == 10
    assert "CURRENT_TS" in profiles
    assert "STOP_15_LOCK_05" in profiles
    assert "STOP_40_LOCK_20" in profiles
    assert profiles["STOP_15_LOCK_05"].hard_stop_roe == -0.15


def test_cost_sensitivity_never_changes_gross_path() -> None:
    result = _cost_adjusted(
        0.002,
        base_cost=0.001,
        costs={"expected": 0.001, "stress": 0.0015, "severe": 0.002},
    )
    assert result == {"expected": 0.002, "stress": 0.0015, "severe": 0.001}
