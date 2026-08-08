"""Counterfactual implementation of the seven-point entry-quality proposal.

This assessment is deliberately non-authoritative.  It records whether the
new policy would abstain, wait, or retain a candidate while the current Live
selector remains the sole decision authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SevenPointEntryShadowPolicy:
    high_tail_risk_threshold: float = 0.45
    minimum_timing_components: int = 4
    minimum_aligned_horizons: int = 3
    crowded_side_fraction: float = 0.70

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.high_tail_risk_threshold)
            or not 0.0 <= self.high_tail_risk_threshold <= 1.0
            or not 1 <= self.minimum_timing_components <= 5
            or not 1 <= self.minimum_aligned_horizons <= 4
            or not math.isfinite(self.crowded_side_fraction)
            or not 0.0 <= self.crowded_side_fraction <= 1.0
        ):
            raise ValueError("seven-point entry Shadow policy is invalid")


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def assess_seven_point_entry_shadow(
    *,
    side: str,
    prediction: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    confirmation_features: Mapping[str, Any],
    current_layer: Mapping[str, Any],
    entry_intelligence: Mapping[str, Any],
    confirmed_same_side: int,
    confirmed_total: int,
    policy: SevenPointEntryShadowPolicy | None = None,
) -> Mapping[str, Any]:
    config = policy or SevenPointEntryShadowPolicy()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("seven-point entry side is invalid")

    state = str(confirmation.get("state", "UNKNOWN"))
    components_passed = int(confirmation.get("components_passed", 0))
    tail_risk = _finite(current_layer.get("rv2_tail_risk"))
    regime = entry_intelligence.get("regime_v3_shadow", {})
    if not isinstance(regime, Mapping):
        regime = {}
    volatility = str(regime.get("volatility", "UNKNOWN"))
    structure = str(regime.get("structure", "UNKNOWN"))

    signed_horizons = {
        name: _finite(confirmation_features.get(name))
        for name in (
            "signed_ret_1_atr",
            "signed_ret_3_atr",
            "signed_ret_6_atr",
            "signed_ret_12_atr",
        )
    }
    aligned_horizons = sum(
        value is not None and value > 0.0 for value in signed_horizons.values()
    )
    marginal_timing = components_passed < config.minimum_timing_components
    high_tail_risk = (
        tail_risk is not None and tail_risk >= config.high_tail_risk_threshold
    )
    dangerous_confluence = marginal_timing and volatility == "HIGH" and high_tail_risk

    crowded_fraction = (
        confirmed_same_side / confirmed_total if confirmed_total > 0 else 0.0
    )
    crowded = confirmed_total >= 3 and crowded_fraction >= config.crowded_side_fraction
    absolute_quality_pass = state != "ABSTAIN_WEAK_QUALITY"
    timing_pass = not marginal_timing
    multi_horizon_pass = aligned_horizons >= config.minimum_aligned_horizons

    if not absolute_quality_pass or dangerous_confluence:
        disposition = "COUNTERFACTUAL_ABSTAIN"
        sizing = "NO_ALLOCATION_COUNTERFACTUAL"
    elif not timing_pass or not multi_horizon_pass:
        disposition = "COUNTERFACTUAL_WAIT_CONFIRMATION"
        sizing = "WAIT_NO_ALLOCATION_COUNTERFACTUAL"
    else:
        disposition = "COUNTERFACTUAL_QUALITY_CANDIDATE"
        sizing = "ORIGINAL_SIZING_UNCHANGED_IF_EVENTUALLY_PROMOTED"

    timing_context = entry_intelligence.get("entry_timing_shadow", {})
    if not isinstance(timing_context, Mapping):
        timing_context = {}
    mae_q90 = _finite(prediction.get("mae_q90"))
    mfe_q50 = _finite(prediction.get("mfe_q50"))
    path_efficiency = (
        mfe_q50 / max(mae_q90, 1e-12)
        if mae_q90 is not None and mfe_q50 is not None
        else None
    )

    return {
        "schema_id": "aegis-seven-point-entry-shadow-v1",
        "mode": "SHADOW",
        "disposition": disposition,
        "clean_path_quality": {
            "opportunity_probability": _finite(
                prediction.get("opportunity_probability")
            ),
            "danger_probability": _finite(prediction.get("danger_probability")),
            "mae_q90": mae_q90,
            "mfe_q50": mfe_q50,
            "net_return_mean": _finite(prediction.get("net_return_mean")),
            "path_efficiency": path_efficiency,
            "absolute_quality_pass": absolute_quality_pass,
        },
        "dangerous_confluence": {
            "marginal_timing": marginal_timing,
            "volatility": volatility,
            "tail_risk": tail_risk,
            "high_tail_risk_threshold": config.high_tail_risk_threshold,
            "would_abstain": dangerous_confluence,
        },
        "temporal_confirmation": {
            "components_passed": components_passed,
            "minimum_components": config.minimum_timing_components,
            "same_bar_pass": timing_pass,
            "sequential_state": timing_context.get("state", "NOT_AVAILABLE"),
            "sequential_state_effect": "OBSERVATIONAL_CONTEXT_ONLY",
        },
        "multi_horizon_context": {
            "signed_horizons": signed_horizons,
            "aligned_horizons": aligned_horizons,
            "minimum_aligned_horizons": config.minimum_aligned_horizons,
            "pass": multi_horizon_pass,
            "structure": structure,
        },
        "correlation_context": {
            "confirmed_same_side": confirmed_same_side,
            "confirmed_total": confirmed_total,
            "same_side_fraction": crowded_fraction,
            "crowded": crowded,
            "selection_effect": "NONE_PENDING_VALIDATION",
        },
        "quality_sizing": {
            "recommendation": sizing,
            "numeric_fraction": "NOT_ASSIGNED",
            "typescript_sizing_unchanged": True,
        },
        "feedback_contract": {
            "required_outcomes": [
                "MFE_BEFORE_MAE",
                "MAE_FRACTION",
                "MFE_FRACTION",
                "TIME_TO_MFE",
                "NET_RETURN_AFTER_COSTS",
                "MANUAL_SIZE_CHANGE",
            ],
            "online_weight_updates": False,
        },
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
