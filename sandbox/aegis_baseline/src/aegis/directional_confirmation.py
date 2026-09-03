"""Causal absolute-edge and entry-timing confirmation for directional selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .features import FEATURE_NAMES

CONFIRMATION_FEATURE_NAMES = (
    "ema6_distance_atr",
    "ema12_distance_atr",
    "ema24_distance_atr",
    "fast_ema_reclaim_strength",
    "directional_candle_body_atr",
    "directional_close_location",
    "rejection_wick_fraction",
    "signed_ret_1_atr",
    "signed_ret_3_atr",
    "signed_ret_6_atr",
    "signed_ret_12_atr",
    "momentum_acceleration_atr",
    "trend_slope_alignment_atr",
    "trend_stack_alignment",
    "volume_zscore_24",
    "volume_ratio_6_24",
    "volume_return_1",
    "pullback_depth_atr",
    "impulse_exhaustion_atr",
)


class ConfirmationState(str, Enum):
    CONFIRMED = "CONFIRMED"
    WAIT_CONFIRMATION = "WAIT_CONFIRMATION"
    ABSTAIN_WEAK_QUALITY = "ABSTAIN_WEAK_QUALITY"


@dataclass(frozen=True)
class DirectionalConfirmationPolicy:
    round_trip_cost_fraction: float
    minimum_opportunity_probability_long: float
    minimum_opportunity_probability_short: float
    maximum_danger_probability: float
    minimum_net_return_fraction: float
    minimum_opportunity_percentile: float
    minimum_danger_quality_percentile: float
    minimum_net_return_percentile: float
    minimum_path_efficiency_percentile: float
    minimum_confirmation_components: int
    minimum_close_location: float
    minimum_volume_zscore: float

    def __post_init__(self) -> None:
        numeric = (
            self.round_trip_cost_fraction,
            self.minimum_opportunity_probability_long,
            self.minimum_opportunity_probability_short,
            self.maximum_danger_probability,
            self.minimum_net_return_fraction,
            self.minimum_opportunity_percentile,
            self.minimum_danger_quality_percentile,
            self.minimum_net_return_percentile,
            self.minimum_path_efficiency_percentile,
            self.minimum_close_location,
            self.minimum_volume_zscore,
        )
        if (
            not all(math.isfinite(value) for value in numeric)
            or not 0.0 <= self.round_trip_cost_fraction < 1.0
            or not 0.0 <= self.minimum_opportunity_probability_long <= 1.0
            or not 0.0 <= self.minimum_opportunity_probability_short <= 1.0
            or not 0.0 <= self.maximum_danger_probability <= 1.0
            or not 0.0 <= self.minimum_close_location <= 1.0
            or not all(
                0.0 <= value <= 1.0
                for value in (
                    self.minimum_opportunity_percentile,
                    self.minimum_danger_quality_percentile,
                    self.minimum_net_return_percentile,
                    self.minimum_path_efficiency_percentile,
                )
            )
            or not 1 <= self.minimum_confirmation_components <= 5
        ):
            raise ValueError("directional confirmation policy is invalid")

    def minimum_opportunity_probability(self, side: str) -> float:
        if side == "LONG":
            return self.minimum_opportunity_probability_long
        if side == "SHORT":
            return self.minimum_opportunity_probability_short
        raise ValueError("directional confirmation side is invalid")


def directional_confirmation_features(
    features: Mapping[str, Any], side: str
) -> Mapping[str, float]:
    if side not in {"LONG", "SHORT"}:
        raise ValueError("directional confirmation side is invalid")
    try:
        values = {name: float(features[name]) for name in FEATURE_NAMES}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "directional confirmation feature contract is invalid"
        ) from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("directional confirmation features are non-finite")
    sign = 1.0 if side == "LONG" else -1.0
    atr = max(values["atr_12"], 1e-9)
    ema6 = sign * values["close_vs_ema_6"] / atr
    ema12 = sign * values["close_vs_ema_12"] / atr
    ema24 = sign * values["close_vs_ema_24"] / atr
    close_location = (
        values["close_position_in_range"]
        if side == "LONG"
        else 1.0 - values["close_position_in_range"]
    )
    rejection_wick = (
        values["lower_wick_fraction"]
        if side == "LONG"
        else values["upper_wick_fraction"]
    )
    trend_stack = (
        values["trend_stack_long"] if side == "LONG" else values["trend_stack_short"]
    )
    pullback = (
        max(0.0, values["distance_to_rolling_high_12"])
        if side == "LONG"
        else max(0.0, values["distance_to_rolling_low_12"])
    )
    derived = {
        "ema6_distance_atr": ema6,
        "ema12_distance_atr": ema12,
        "ema24_distance_atr": ema24,
        "fast_ema_reclaim_strength": min(ema6, ema12),
        "directional_candle_body_atr": sign * values["close_to_open_return"] / atr,
        "directional_close_location": close_location,
        "rejection_wick_fraction": rejection_wick,
        "signed_ret_1_atr": sign * values["ret_1"] / atr,
        "signed_ret_3_atr": sign * values["ret_3"] / atr,
        "signed_ret_6_atr": sign * values["ret_6"] / atr,
        "signed_ret_12_atr": sign * values["ret_12"] / atr,
        "momentum_acceleration_atr": (
            sign * values["momentum_acceleration_3_12"] / atr
        ),
        "trend_slope_alignment_atr": sign * values["ema_slope_6"] / atr,
        "trend_stack_alignment": trend_stack,
        "volume_zscore_24": values["volume_zscore_24"],
        "volume_ratio_6_24": values["volume_ratio_6_24"],
        "volume_return_1": values["volume_return_1"],
        "pullback_depth_atr": pullback / atr,
        "impulse_exhaustion_atr": max(0.0, -sign * values["ret_12"]) / atr,
    }
    if tuple(derived) != CONFIRMATION_FEATURE_NAMES or not all(
        math.isfinite(value) for value in derived.values()
    ):
        raise ValueError("directional confirmation derivation failed")
    return derived


def assess_directional_confirmation(
    prediction: Mapping[str, Any],
    confirmation: Mapping[str, float],
    relative_quality: Mapping[str, float],
    policy: DirectionalConfirmationPolicy,
) -> Mapping[str, Any]:
    try:
        side = str(prediction["side"])
        opportunity = float(prediction["opportunity_probability"])
        danger = float(prediction["danger_probability"])
        mae_q90 = float(prediction["mae_q90"])
        mfe_q50 = float(prediction["mfe_q50"])
        net_return = float(prediction["net_return_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("directional prediction contract is invalid") from exc
    if not all(
        math.isfinite(value)
        for value in (opportunity, danger, mae_q90, mfe_q50, net_return)
    ):
        raise ValueError("directional prediction is non-finite")

    minimum_opportunity = policy.minimum_opportunity_probability(side)
    quality_reasons = []
    if opportunity < minimum_opportunity:
        quality_reasons.append("OPPORTUNITY_PROBABILITY_BELOW_CALIBRATED_MINIMUM")
    if danger > policy.maximum_danger_probability:
        quality_reasons.append("DANGER_PROBABILITY_ABOVE_CALIBRATED_MAXIMUM")
    if net_return <= policy.minimum_net_return_fraction:
        quality_reasons.append("NET_EXPECTANCY_BELOW_TOLERANCE")
    quality_thresholds = {
        "opportunity_percentile": policy.minimum_opportunity_percentile,
        "danger_quality_percentile": policy.minimum_danger_quality_percentile,
        "net_return_percentile": policy.minimum_net_return_percentile,
        "path_efficiency_percentile": policy.minimum_path_efficiency_percentile,
    }
    for name, threshold in quality_thresholds.items():
        value = float(relative_quality[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("directional relative quality is invalid")
        if value < threshold:
            quality_reasons.append(f"{name.upper()}_BELOW_MINIMUM")

    components = {
        "fast_ema_reclaimed": confirmation["fast_ema_reclaim_strength"] > 0.0,
        "directional_reversal_candle": (
            confirmation["directional_candle_body_atr"] > 0.0
            and confirmation["directional_close_location"]
            >= policy.minimum_close_location
        ),
        "short_horizon_momentum_aligned": (
            confirmation["signed_ret_1_atr"] > 0.0
            and confirmation["signed_ret_3_atr"] > 0.0
        ),
        "volume_confirms": (
            confirmation["volume_zscore_24"] >= policy.minimum_volume_zscore
            and confirmation["volume_return_1"] > 0.0
        ),
        "trend_improving": (
            confirmation["trend_stack_alignment"] > 0.0
            or confirmation["trend_slope_alignment_atr"] > 0.0
        ),
    }
    passed = sum(components.values())
    if quality_reasons:
        state = ConfirmationState.ABSTAIN_WEAK_QUALITY
        reasons = quality_reasons
    elif passed < policy.minimum_confirmation_components:
        state = ConfirmationState.WAIT_CONFIRMATION
        reasons = [name.upper() for name, value in components.items() if not value]
    else:
        state = ConfirmationState.CONFIRMED
        reasons = ["RELATIVE_QUALITY_AND_TIMING_CONFIRMED"]
    return {
        "state": state.value,
        "reason_codes": reasons,
        "relative_quality": dict(relative_quality),
        "net_return_mean": net_return,
        "opportunity_probability": opportunity,
        "minimum_opportunity_probability": minimum_opportunity,
        "danger_probability": danger,
        "mae_q90": mae_q90,
        "mfe_q50": mfe_q50,
        "components": components,
        "components_passed": passed,
        "components_required": policy.minimum_confirmation_components,
    }


def directional_relative_quality(
    candidates: list[Mapping[str, Any]],
) -> list[Mapping[str, float]]:
    if not candidates:
        raise ValueError("directional candidate population is empty")

    def percentile(
        value: float, population: list[float], *, lower_is_better: bool = False
    ) -> float:
        if len(population) == 1:
            return 1.0
        better = (
            sum(candidate > value for candidate in population)
            if lower_is_better
            else sum(candidate < value for candidate in population)
        )
        equal = sum(candidate == value for candidate in population)
        return (better + (equal - 1) / 2.0) / (len(population) - 1)

    opportunity = [float(item["opportunity_probability"]) for item in candidates]
    danger = [float(item["danger_probability"]) for item in candidates]
    net_return = [float(item["net_return_mean"]) for item in candidates]
    path_efficiency = [
        float(item["mfe_q50"])
        / (float(item["mfe_q50"]) + float(item["mae_q90"]) + 0.001)
        for item in candidates
    ]
    populations = (*opportunity, *danger, *net_return, *path_efficiency)
    if not all(math.isfinite(value) for value in populations):
        raise ValueError("directional candidate quality is non-finite")
    return [
        {
            "opportunity_percentile": percentile(opportunity[index], opportunity),
            "danger_quality_percentile": percentile(
                danger[index], danger, lower_is_better=True
            ),
            "net_return_percentile": percentile(net_return[index], net_return),
            "path_efficiency_percentile": percentile(
                path_efficiency[index], path_efficiency
            ),
            "path_efficiency": path_efficiency[index],
        }
        for index in range(len(candidates))
    ]
