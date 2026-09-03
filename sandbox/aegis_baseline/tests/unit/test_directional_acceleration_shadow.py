from __future__ import annotations

from pathlib import Path

import pytest

from aegis.research.directional_acceleration_shadow import (
    AccelerationState,
    DirectionalAccelerationSettings,
    DirectionalAccelerationShadowError,
    evaluate_directional_acceleration_shadow,
)


def settings() -> DirectionalAccelerationSettings:
    return DirectionalAccelerationSettings(
        minimum_pressure_components=5,
        minimum_acceleration_components=7,
        minimum_persistence=1.0 / 3.0,
        minimum_impulse_atr_multiple=1.5,
        minimum_relative_atr_multiple=0.5,
        minimum_volume_zscore=0.5,
        minimum_volume_ratio=1.1,
        minimum_trend_strength=1.25,
    )


def features() -> dict[str, float]:
    return {
        "ret_3": 0.006,
        "ret_6": 0.012,
        "ret_12": 0.02,
        "atr_12": 0.006,
        "btc_divergence_6": 0.005,
        "persistence_6": 2.0 / 3.0,
        "ema_slope_6": 0.004,
        "ema_slope_24": 0.002,
        "trend_stack_long": 1.0,
        "trend_stack_short": 0.0,
        "distance_to_rolling_high_12": -0.001,
        "close_below_rolling_low_12": 0.0,
        "volume_zscore_24": 1.2,
        "volume_ratio_6_24": 1.3,
        "volume_spike_12": 0.0,
        "range_expansion": 0.3,
        "trend_strength_12": 2.0,
        "overextended_down_risk_proxy": 0.0,
        "low_room_to_fall_risk_proxy": 0.0,
        "squeeze_risk_proxy_causal": 0.0,
        "rebound_risk_proxy": 0.0,
        "failed_breakdown_proxy": 0.0,
        "high_wick_reclaim_risk_proxy": 0.0,
    }


def test_upward_acceleration_warns_short_without_exchange_authority() -> None:
    result = evaluate_directional_acceleration_shadow(features(), settings())

    assert result["state"] == AccelerationState.UPWARD_ACCELERATION.value
    assert result["short_adverse_risk"] == "HIGH"
    assert result["short_entry_disposition"] == "DO_NOT_ENTER_COUNTERFACTUAL"
    assert result["short_add_disposition"] == "DO_NOT_ADD_COUNTERFACTUAL"
    assert result["selection_effect"] == "NONE"
    assert result["exchange_authority"] is False
    assert result["exchange_mutations"] == 0
    assert result["online_learning"] is False


def test_downward_acceleration_does_not_authorize_averaging() -> None:
    values = features()
    directional = (
        "ret_3",
        "ret_6",
        "ret_12",
        "btc_divergence_6",
        "persistence_6",
        "ema_slope_6",
        "ema_slope_24",
    )
    for name in directional:
        values[name] *= -1.0
    values["trend_stack_long"] = 0.0
    values["trend_stack_short"] = 1.0
    values["distance_to_rolling_high_12"] = 0.02
    values["close_below_rolling_low_12"] = 1.0

    result = evaluate_directional_acceleration_shadow(values, settings())

    assert result["state"] == AccelerationState.DOWNWARD_ACCELERATION.value
    assert result["short_add_disposition"] == "INSUFFICIENT_EVIDENCE_TO_ADD"


def test_transition_or_chase_risk_requires_confirmation() -> None:
    values = features()
    values.update(
        {
            "ret_3": -0.002,
            "ret_6": -0.006,
            "ret_12": 0.01,
            "trend_stack_long": 0.0,
            "ema_slope_6": -0.001,
            "low_room_to_fall_risk_proxy": 1.0,
        }
    )

    result = evaluate_directional_acceleration_shadow(values, settings())

    assert result["state"] == AccelerationState.CONFLICTING_TRANSITION.value
    assert result["short_adverse_risk"] == "ELEVATED"
    assert result["short_entry_disposition"] == "WAIT_CONFIRMATION_COUNTERFACTUAL"
    assert result["short_add_disposition"] == "DO_NOT_ADD_COUNTERFACTUAL"


def test_missing_or_non_finite_features_fail_closed() -> None:
    missing = features()
    missing.pop("ret_3")
    with pytest.raises(
        DirectionalAccelerationShadowError,
        match="AEGIS_ACCELERATION_FEATURE_MISSING:ret_3",
    ):
        evaluate_directional_acceleration_shadow(missing, settings())

    non_finite = features()
    non_finite["ret_3"] = float("nan")
    with pytest.raises(
        DirectionalAccelerationShadowError,
        match="AEGIS_ACCELERATION_FEATURE_NON_FINITE:ret_3",
    ):
        evaluate_directional_acceleration_shadow(non_finite, settings())


def test_observer_has_no_exchange_mutation_surface() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/aegis/research/directional_acceleration_shadow.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "create_order",
        "cancel_order",
        "modify_order",
        "close_position",
        "BinanceAdapter",
        "api_secret",
    ):
        assert forbidden not in source
