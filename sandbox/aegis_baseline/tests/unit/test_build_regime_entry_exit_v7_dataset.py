from __future__ import annotations

from build_regime_entry_exit_v7_dataset import _audit_contract, _profiles


def config() -> dict[str, object]:
    return {
        "trajectory_audit": {
            "maximum_clean_mae_fraction": 0.006,
            "maximum_clean_positive_bar": 6,
            "late_entry_extension_atr": 2.0,
            "late_entry_positive_bar": 12,
            "minimum_available_net_fraction": 0.001,
        },
        "protection_profiles": {
            "CURRENT_TS": {
                "break_even_trigger_roe": 0.08,
                "break_even_offset_fraction": 0.003,
                "trailing_activation_roe": 0.15,
                "trailing_callback_roe": 0.08,
            },
            "LOCK_AT_5_ROE": {
                "break_even_trigger_roe": 0.05,
                "break_even_offset_fraction": 0.0016,
                "trailing_activation_roe": 0.05,
                "trailing_callback_roe": 0.08,
            },
            "LOCK_AT_10_ROE": {
                "break_even_trigger_roe": 0.10,
                "break_even_offset_fraction": 0.0033,
                "trailing_activation_roe": 0.10,
                "trailing_callback_roe": 0.08,
            },
            "LOCK_AT_20_ROE": {
                "break_even_trigger_roe": 0.20,
                "break_even_offset_fraction": 0.0066,
                "trailing_activation_roe": 0.20,
                "trailing_callback_roe": 0.08,
            },
            "shared": {
                "leverage": 15.0,
                "hard_stop_roe": -0.40,
                "take_profit_roe": 0.50,
                "use_atr_trailing": True,
                "atr_period": 14,
                "atr_multiplier": 1.5,
                "round_trip_cost_fraction": 0.001,
            },
            "profile_choice_source": "CALIBRATION_ONLY_MODEL_PREDICTION",
            "production_effect": "NONE",
        },
    }


def test_v7_contract_and_profiles_are_explicit() -> None:
    audit = _audit_contract(config())
    profiles = _profiles(config())
    assert audit.maximum_clean_positive_bar == 6
    assert set(profiles) == {
        "CURRENT_TS",
        "LOCK_AT_5_ROE",
        "LOCK_AT_10_ROE",
        "LOCK_AT_20_ROE",
    }
    assert profiles["LOCK_AT_5_ROE"].break_even_trigger_roe == 0.05
    assert profiles["LOCK_AT_20_ROE"].break_even_trigger_roe == 0.20
