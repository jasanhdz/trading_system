"""Causal directional-acceleration evidence with no trading authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DirectionalAccelerationShadowError(ValueError):
    pass


class AccelerationState(str, Enum):
    UPWARD_ACCELERATION = "UPWARD_ACCELERATION"
    UPWARD_PRESSURE = "UPWARD_PRESSURE"
    DOWNWARD_ACCELERATION = "DOWNWARD_ACCELERATION"
    DOWNWARD_PRESSURE = "DOWNWARD_PRESSURE"
    CONFLICTING_TRANSITION = "CONFLICTING_TRANSITION"
    BALANCED = "BALANCED"


@dataclass(frozen=True)
class DirectionalAccelerationSettings:
    minimum_pressure_components: int
    minimum_acceleration_components: int
    minimum_persistence: float
    minimum_impulse_atr_multiple: float
    minimum_relative_atr_multiple: float
    minimum_volume_zscore: float
    minimum_volume_ratio: float
    minimum_trend_strength: float

    def validate(self) -> None:
        if (
            not 1
            <= self.minimum_pressure_components
            < self.minimum_acceleration_components
            <= 9
        ):
            raise DirectionalAccelerationShadowError(
                "AEGIS_ACCELERATION_COMPONENT_LIMIT_INVALID"
            )
        bounded = (
            self.minimum_persistence,
            self.minimum_impulse_atr_multiple,
            self.minimum_relative_atr_multiple,
            self.minimum_volume_zscore,
            self.minimum_volume_ratio,
            self.minimum_trend_strength,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in bounded):
            raise DirectionalAccelerationShadowError(
                "AEGIS_ACCELERATION_THRESHOLD_INVALID"
            )
        if self.minimum_persistence > 1.0 or self.minimum_volume_ratio < 1.0:
            raise DirectionalAccelerationShadowError(
                "AEGIS_ACCELERATION_THRESHOLD_INVALID"
            )


def _number(features: Mapping[str, Any], name: str) -> float:
    try:
        value = float(features[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectionalAccelerationShadowError(
            f"AEGIS_ACCELERATION_FEATURE_MISSING:{name}"
        ) from exc
    if not math.isfinite(value):
        raise DirectionalAccelerationShadowError(
            f"AEGIS_ACCELERATION_FEATURE_NON_FINITE:{name}"
        )
    return value


def _directional_components(
    features: Mapping[str, Any],
    settings: DirectionalAccelerationSettings,
    *,
    sign: float,
) -> dict[str, bool]:
    ret_3 = sign * _number(features, "ret_3")
    ret_6 = sign * _number(features, "ret_6")
    ret_12 = sign * _number(features, "ret_12")
    atr_12 = max(_number(features, "atr_12"), 1e-15)
    relative = sign * _number(features, "btc_divergence_6")
    persistence = sign * _number(features, "persistence_6")
    ema_slope_6 = sign * _number(features, "ema_slope_6")
    ema_slope_24 = sign * _number(features, "ema_slope_24")
    distance_to_high = _number(features, "distance_to_rolling_high_12")
    below_low = _number(features, "close_below_rolling_low_12") > 0.0
    trend_stack = (
        _number(features, "trend_stack_long" if sign > 0.0 else "trend_stack_short")
        > 0.0
    )
    breakout = distance_to_high <= 0.0 if sign > 0.0 else below_low
    return {
        "multi_horizon_alignment": ret_3 > 0.0 and ret_6 > 0.0 and ret_12 > 0.0,
        "directional_persistence": persistence >= settings.minimum_persistence,
        "trend_stack": trend_stack,
        "ema_slope_alignment": ema_slope_6 > 0.0 and ema_slope_24 >= 0.0,
        "atr_scaled_impulse": ret_6 >= atr_12 * settings.minimum_impulse_atr_multiple,
        "btc_relative_divergence": relative
        >= atr_12 * settings.minimum_relative_atr_multiple,
        "volume_confirmation": (
            _number(features, "volume_zscore_24") >= settings.minimum_volume_zscore
            or _number(features, "volume_ratio_6_24") >= settings.minimum_volume_ratio
            or _number(features, "volume_spike_12") > 0.0
        ),
        "range_trend_expansion": (
            _number(features, "range_expansion") > 0.0
            and _number(features, "trend_strength_12")
            >= settings.minimum_trend_strength
        ),
        "rolling_level_break": breakout,
    }


def _strength(
    components: Mapping[str, bool], settings: DirectionalAccelerationSettings
) -> tuple[int, str]:
    count = sum(bool(value) for value in components.values())
    core = (
        components["multi_horizon_alignment"]
        and components["trend_stack"]
        and components["ema_slope_alignment"]
    )
    if count >= settings.minimum_acceleration_components and core:
        return count, "ACCELERATION"
    if count >= settings.minimum_pressure_components:
        return count, "PRESSURE"
    return count, "NONE"


def evaluate_directional_acceleration_shadow(
    features: Mapping[str, Any],
    settings: DirectionalAccelerationSettings,
) -> Mapping[str, Any]:
    """Classify current closed-candle evidence without predicting or trading."""

    settings.validate()
    upward = _directional_components(features, settings, sign=1.0)
    downward = _directional_components(features, settings, sign=-1.0)
    upward_count, upward_strength = _strength(upward, settings)
    downward_count, downward_strength = _strength(downward, settings)

    if upward_strength == "ACCELERATION" and downward_strength == "ACCELERATION":
        state = AccelerationState.CONFLICTING_TRANSITION
    elif upward_strength == "ACCELERATION":
        state = AccelerationState.UPWARD_ACCELERATION
    elif downward_strength == "ACCELERATION":
        state = AccelerationState.DOWNWARD_ACCELERATION
    elif _number(features, "ret_6") * _number(features, "ret_12") < 0.0:
        state = AccelerationState.CONFLICTING_TRANSITION
    elif upward_strength == "PRESSURE" and downward_strength == "PRESSURE":
        state = AccelerationState.CONFLICTING_TRANSITION
    elif upward_strength == "PRESSURE":
        state = AccelerationState.UPWARD_PRESSURE
    elif downward_strength == "PRESSURE":
        state = AccelerationState.DOWNWARD_PRESSURE
    else:
        state = AccelerationState.BALANCED

    short_chase_flags = {
        name: _number(features, name) > 0.0
        for name in (
            "overextended_down_risk_proxy",
            "low_room_to_fall_risk_proxy",
            "squeeze_risk_proxy_causal",
            "rebound_risk_proxy",
            "failed_breakdown_proxy",
            "high_wick_reclaim_risk_proxy",
        )
    }
    if state in {
        AccelerationState.UPWARD_ACCELERATION,
        AccelerationState.UPWARD_PRESSURE,
    }:
        short_entry_disposition = "DO_NOT_ENTER_COUNTERFACTUAL"
        short_add_disposition = "DO_NOT_ADD_COUNTERFACTUAL"
        short_adverse_risk = "HIGH"
    elif state is AccelerationState.CONFLICTING_TRANSITION or any(
        short_chase_flags.values()
    ):
        short_entry_disposition = "WAIT_CONFIRMATION_COUNTERFACTUAL"
        short_add_disposition = "DO_NOT_ADD_COUNTERFACTUAL"
        short_adverse_risk = "ELEVATED"
    else:
        short_entry_disposition = "NO_SHADOW_OBJECTION"
        short_add_disposition = "INSUFFICIENT_EVIDENCE_TO_ADD"
        short_adverse_risk = "NORMAL"

    return {
        "schema_id": "aegis-directional-acceleration-shadow-v1",
        "mode": "SHADOW",
        "state": state.value,
        "upward_component_count": upward_count,
        "downward_component_count": downward_count,
        "component_count": len(upward),
        "upward_components": upward,
        "downward_components": downward,
        "short_chase_flags": short_chase_flags,
        "short_adverse_risk": short_adverse_risk,
        "short_entry_disposition": short_entry_disposition,
        "short_add_disposition": short_add_disposition,
        "online_learning": False,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
