from __future__ import annotations

from aegis.research.seven_point_entry_shadow import assess_seven_point_entry_shadow


def assessment(**overrides):
    values = {
        "side": "SHORT",
        "prediction": {
            "opportunity_probability": 0.60,
            "danger_probability": 0.40,
            "mae_q90": 0.004,
            "mfe_q50": 0.012,
            "net_return_mean": 0.003,
        },
        "confirmation": {"state": "CONFIRMED", "components_passed": 4},
        "confirmation_features": {
            "signed_ret_1_atr": 1.0,
            "signed_ret_3_atr": 1.0,
            "signed_ret_6_atr": 1.0,
            "signed_ret_12_atr": -1.0,
        },
        "current_layer": {"rv2_tail_risk": 0.20},
        "entry_intelligence": {
            "regime_v3_shadow": {"volatility": "NORMAL", "structure": "TREND"},
            "entry_timing_shadow": {"state": "TIMING_CONFIRMED"},
        },
        "confirmed_same_side": 2,
        "confirmed_total": 4,
    }
    values.update(overrides)
    return assess_seven_point_entry_shadow(**values)


def test_seven_point_shadow_retains_clean_candidate_without_execution_effect() -> None:
    result = assessment()
    assert result["disposition"] == "COUNTERFACTUAL_QUALITY_CANDIDATE"
    assert result["clean_path_quality"]["path_efficiency"] == 3.0
    assert result["selection_effect"] == "NONE"
    assert result["exchange_authority"] is False
    assert result["quality_sizing"]["numeric_fraction"] == "NOT_ASSIGNED"


def test_seven_point_shadow_abstains_only_on_complete_dangerous_confluence() -> None:
    result = assessment(
        confirmation={"state": "CONFIRMED", "components_passed": 3},
        current_layer={"rv2_tail_risk": 0.50},
        entry_intelligence={
            "regime_v3_shadow": {"volatility": "HIGH", "structure": "TREND"},
            "entry_timing_shadow": {"state": "WAITING_FOR_RETEST"},
        },
    )
    assert result["disposition"] == "COUNTERFACTUAL_ABSTAIN"
    assert result["dangerous_confluence"]["would_abstain"] is True
    assert result["quality_sizing"]["typescript_sizing_unchanged"] is True


def test_seven_point_shadow_waits_for_incomplete_multi_horizon_alignment() -> None:
    result = assessment(
        confirmation_features={
            "signed_ret_1_atr": 1.0,
            "signed_ret_3_atr": -1.0,
            "signed_ret_6_atr": -1.0,
            "signed_ret_12_atr": 1.0,
        }
    )
    assert result["disposition"] == "COUNTERFACTUAL_WAIT_CONFIRMATION"
    assert result["multi_horizon_context"]["aligned_horizons"] == 2
    assert result["selection_effect"] == "NONE"
