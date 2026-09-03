"""Sequential entry-quality methodology with observational authority only.

The methodology separates directional opportunity from entry-path quality. It
never changes the canonical selector: grades and actions are counterfactual,
and clean-path labels are produced only after the future horizon has matured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from .seven_point_entry_shadow import assess_seven_point_entry_shadow

ENTRY_PATH_MODEL_FEATURE_NAMES = (
    "opportunity_probability",
    "danger_probability",
    "mae_q50",
    "mae_q90",
    "mfe_q50",
    "net_return_mean",
    "shadow_rank_score",
    "components_passed_fraction",
    "opportunity_percentile",
    "danger_quality_percentile",
    "net_return_percentile",
    "path_efficiency_percentile",
    "path_efficiency",
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
    "global_direction_alignment",
    "symbol_direction_alignment",
    "volatility_code",
    "structure_code",
    "regime_alignment_code",
    "extension_code",
    "regime_evidence_ready",
    "global_stability_log",
    "symbol_stability_log",
    "volatility_stability_log",
    "structure_stability_log",
    "reversal_flag_fraction",
    "directional_acceleration_fraction",
    "opposite_acceleration_fraction",
)


class EntryMethodologyTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class CleanPathClassification(str, Enum):
    CLEAN_FAST_SUCCESS = "CLEAN_FAST_SUCCESS"
    CLEAN_SLOW_SUCCESS = "CLEAN_SLOW_SUCCESS"
    ADVERSE_FIRST = "ADVERSE_FIRST"
    AMBIGUOUS_SAME_BAR = "AMBIGUOUS_SAME_BAR"
    NO_DECISIVE_EDGE = "NO_DECISIVE_EDGE"


@dataclass(frozen=True)
class EntryMethodologyV2Policy:
    """Prospective research policy; none of these values authorize execution."""

    horizon_bars: int = 12
    fast_edge_bars: int = 6
    favorable_barrier_fraction: float = 0.003
    adverse_barrier_fraction: float = 0.003
    round_trip_cost_fraction: float = 0.001
    maximum_wait_bars: int = 3

    def __post_init__(self) -> None:
        numeric = (
            self.favorable_barrier_fraction,
            self.adverse_barrier_fraction,
            self.round_trip_cost_fraction,
        )
        if (
            self.horizon_bars <= 0
            or not 1 <= self.fast_edge_bars <= self.horizon_bars
            or not 1 <= self.maximum_wait_bars <= self.horizon_bars
            or not all(math.isfinite(value) for value in numeric)
            or self.favorable_barrier_fraction <= 0.0
            or self.adverse_barrier_fraction <= 0.0
            or not 0.0 <= self.round_trip_cost_fraction < 1.0
        ):
            raise ValueError("entry methodology v2 policy is invalid")


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("entry methodology timestamp is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("entry methodology timestamp is invalid")
    return parsed


def entry_path_model_features(
    *,
    side: str,
    prediction: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    confirmation_features: Mapping[str, Any],
    entry_intelligence: Mapping[str, Any],
) -> tuple[float, ...]:
    """Build causal meta-model inputs without outcomes or execution state."""

    if side not in {"LONG", "SHORT"}:
        raise ValueError("entry path model side is invalid")
    relative = confirmation.get("relative_quality", {})
    regime = entry_intelligence.get("regime_v3_shadow", {})
    timing = entry_intelligence.get("entry_timing_shadow", {})
    acceleration = entry_intelligence.get("directional_acceleration_shadow", {})
    if not all(
        isinstance(value, Mapping) for value in (relative, regime, timing, acceleration)
    ):
        raise ValueError("entry path model context is invalid")

    desired_direction = "BULLISH" if side == "LONG" else "BEARISH"
    opposite_direction = "BEARISH" if side == "LONG" else "BULLISH"

    def alignment(value: object) -> float:
        text = str(value)
        if text == desired_direction:
            return 1.0
        if text == opposite_direction:
            return -1.0
        return 0.0

    volatility_code = {"LOW": -1.0, "NORMAL": 0.0, "HIGH": 1.0}.get(
        str(regime.get("volatility")), 0.0
    )
    structure_code = {"CHOP": -1.0, "TRANSITION": 0.0, "TREND": 1.0}.get(
        str(regime.get("structure")), 0.0
    )
    regime_alignment = str(regime.get("alignment", "UNKNOWN"))
    regime_alignment_code = (
        1.0
        if regime_alignment == f"ALIGNED_{desired_direction}"
        else (-1.0 if regime_alignment == "DIVERGENT" else 0.0)
    )
    extension = str(regime.get("extension", "NORMAL"))
    extension_code = {
        "EXTENDED_UP": 1.0 if side == "LONG" else -1.0,
        "EXTENDED_DOWN": 1.0 if side == "SHORT" else -1.0,
        "NORMAL": 0.0,
    }.get(extension, 0.0)
    reversal_flags = timing.get("reversal_flags", {})
    if not isinstance(reversal_flags, Mapping):
        reversal_flags = {}
    directional_key = (
        "upward_component_count" if side == "LONG" else "downward_component_count"
    )
    opposite_key = (
        "downward_component_count" if side == "LONG" else "upward_component_count"
    )
    component_count = max(_finite(acceleration.get("component_count")) or 0.0, 1.0)
    raw = {
        "opportunity_probability": prediction.get("opportunity_probability"),
        "danger_probability": prediction.get("danger_probability"),
        "mae_q50": prediction.get("mae_q50"),
        "mae_q90": prediction.get("mae_q90"),
        "mfe_q50": prediction.get("mfe_q50"),
        "net_return_mean": prediction.get("net_return_mean"),
        "shadow_rank_score": prediction.get("shadow_rank_score"),
        "components_passed_fraction": (
            _finite(confirmation.get("components_passed")) or 0.0
        )
        / 5.0,
        "opportunity_percentile": relative.get("opportunity_percentile"),
        "danger_quality_percentile": relative.get("danger_quality_percentile"),
        "net_return_percentile": relative.get("net_return_percentile"),
        "path_efficiency_percentile": relative.get("path_efficiency_percentile"),
        "path_efficiency": relative.get("path_efficiency"),
        **{
            name: confirmation_features.get(name)
            for name in ENTRY_PATH_MODEL_FEATURE_NAMES[13:32]
        },
        "global_direction_alignment": alignment(regime.get("global_direction")),
        "symbol_direction_alignment": alignment(regime.get("symbol_direction")),
        "volatility_code": volatility_code,
        "structure_code": structure_code,
        "regime_alignment_code": regime_alignment_code,
        "extension_code": extension_code,
        "regime_evidence_ready": 1.0 if regime.get("evidence_ready") is True else 0.0,
        "global_stability_log": math.log1p(
            max(_finite(regime.get("global_stability_bars")) or 0.0, 0.0)
        ),
        "symbol_stability_log": math.log1p(
            max(_finite(regime.get("symbol_stability_bars")) or 0.0, 0.0)
        ),
        "volatility_stability_log": math.log1p(
            max(_finite(regime.get("volatility_stability_bars")) or 0.0, 0.0)
        ),
        "structure_stability_log": math.log1p(
            max(_finite(regime.get("structure_stability_bars")) or 0.0, 0.0)
        ),
        "reversal_flag_fraction": (
            sum(value is True for value in reversal_flags.values())
            / max(len(reversal_flags), 1)
        ),
        "directional_acceleration_fraction": (
            (_finite(acceleration.get(directional_key)) or 0.0) / component_count
        ),
        "opposite_acceleration_fraction": (
            (_finite(acceleration.get(opposite_key)) or 0.0) / component_count
        ),
    }
    if tuple(raw) != ENTRY_PATH_MODEL_FEATURE_NAMES:
        raise ValueError("entry path model feature order is invalid")
    values = tuple(_finite(raw[name]) for name in ENTRY_PATH_MODEL_FEATURE_NAMES)
    if any(value is None for value in values):
        raise ValueError("entry path model features are incomplete")
    result = tuple(float(value) for value in values if value is not None)
    if len(result) != len(ENTRY_PATH_MODEL_FEATURE_NAMES):
        raise ValueError("entry path model features are invalid")
    return result


def assess_entry_methodology_v2_shadow(
    *,
    market_timestamp: str,
    side: str,
    prediction: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    confirmation_features: Mapping[str, Any],
    current_layer: Mapping[str, Any],
    entry_intelligence: Mapping[str, Any],
    confirmed_same_side: int,
    confirmed_total: int,
    previous: Mapping[str, Any] | None = None,
    policy: EntryMethodologyV2Policy | None = None,
) -> Mapping[str, Any]:
    """Grade one candidate while preserving the current selector unchanged."""

    config = policy or EntryMethodologyV2Policy()
    current_time = _timestamp(market_timestamp)
    seven_point = assess_seven_point_entry_shadow(
        side=side,
        prediction=prediction,
        confirmation=confirmation,
        confirmation_features=confirmation_features,
        current_layer=current_layer,
        entry_intelligence=entry_intelligence,
        confirmed_same_side=confirmed_same_side,
        confirmed_total=confirmed_total,
    )
    base_disposition = str(seven_point["disposition"])
    previous_sequential = (
        previous.get("sequential_confirmation", {})
        if isinstance(previous, Mapping)
        else {}
    )
    if not isinstance(previous_sequential, Mapping):
        previous_sequential = {}
    previous_state = str(previous_sequential.get("state", "NONE"))
    previous_side = (
        str(previous.get("side", "")) if isinstance(previous, Mapping) else ""
    )
    previous_time_value = (
        previous.get("market_timestamp") if isinstance(previous, Mapping) else None
    )
    previous_time = (
        _timestamp(previous_time_value)
        if isinstance(previous_time_value, str)
        else None
    )
    contiguous = (
        previous_time is not None
        and previous_side == side
        and 0.0 < (current_time - previous_time).total_seconds() <= 5.0 * 60.0 * 1.5
    )

    waiting_before = previous_state == "WAITING" and contiguous
    prior_age = int(previous_sequential.get("age_bars", 0)) if waiting_before else 0
    age_bars = prior_age + 1 if waiting_before else 0
    origin_timestamp = (
        str(previous_sequential.get("origin_timestamp"))
        if waiting_before
        else market_timestamp
    )

    if base_disposition == "COUNTERFACTUAL_QUALITY_CANDIDATE":
        tier = EntryMethodologyTier.A
        action = (
            "COUNTERFACTUAL_ENTER_AFTER_CONFIRMATION"
            if waiting_before
            else "COUNTERFACTUAL_ENTER_NOW"
        )
        sequential_state = "CONFIRMED"
    elif base_disposition == "COUNTERFACTUAL_WAIT_CONFIRMATION":
        if age_bars >= config.maximum_wait_bars:
            tier = EntryMethodologyTier.C
            action = "COUNTERFACTUAL_ABSTAIN_EXPIRED"
            sequential_state = "EXPIRED"
        else:
            tier = EntryMethodologyTier.B
            action = "COUNTERFACTUAL_WAIT_NEXT_CLOSED_BAR"
            sequential_state = "WAITING"
    else:
        tier = EntryMethodologyTier.C
        action = "COUNTERFACTUAL_ABSTAIN"
        sequential_state = "ABSTAINED"

    quality = seven_point["clean_path_quality"]
    timing = seven_point["temporal_confirmation"]
    horizons = seven_point["multi_horizon_context"]
    danger = seven_point["dangerous_confluence"]
    evidence = {
        "absolute_quality": bool(quality["absolute_quality_pass"]),
        "same_bar_timing": bool(timing["same_bar_pass"]),
        "multi_horizon_alignment": bool(horizons["pass"]),
        "dangerous_confluence_absent": not bool(danger["would_abstain"]),
        "positive_net_forecast": (_finite(prediction.get("net_return_mean")) or 0.0)
        > 0.0,
    }
    evidence_passed = sum(evidence.values())
    return {
        "schema_id": "aegis-entry-methodology-v2-shadow-v1",
        "mode": "SHADOW",
        "market_timestamp": market_timestamp,
        "side": side,
        "tier": tier.value,
        "counterfactual_action": action,
        "evidence": evidence,
        "evidence_passed": evidence_passed,
        "evidence_total": len(evidence),
        "sequential_confirmation": {
            "state": sequential_state,
            "origin_timestamp": origin_timestamp,
            "age_bars": age_bars,
            "maximum_wait_bars": config.maximum_wait_bars,
            "confirmed_after_wait": waiting_before and tier is EntryMethodologyTier.A,
        },
        "directional_prediction": {
            "opportunity_probability": _finite(
                prediction.get("opportunity_probability")
            ),
            "danger_probability": _finite(prediction.get("danger_probability")),
            "mae_q90": _finite(prediction.get("mae_q90")),
            "mfe_q50": _finite(prediction.get("mfe_q50")),
            "net_return_mean": _finite(prediction.get("net_return_mean")),
        },
        "seven_point_assessment": seven_point,
        "label_contract": {
            "horizon_bars": config.horizon_bars,
            "fast_edge_bars": config.fast_edge_bars,
            "favorable_barrier_fraction": config.favorable_barrier_fraction,
            "adverse_barrier_fraction": config.adverse_barrier_fraction,
            "round_trip_cost_fraction": config.round_trip_cost_fraction,
            "leverage_independent": True,
            "outcome_blind_at_decision_time": True,
        },
        "selection_effect": "NONE",
        "typescript_sizing_unchanged": True,
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def label_clean_entry_path(
    *,
    side: str,
    entry_price: float,
    future_bars: Sequence[Mapping[str, Any]],
    policy: EntryMethodologyV2Policy | None = None,
) -> Mapping[str, Any]:
    """Create an outcome label after a complete, causal future horizon."""

    config = policy or EntryMethodologyV2Policy()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("clean entry path side is invalid")
    if (
        not math.isfinite(entry_price)
        or entry_price <= 0.0
        or len(future_bars) != config.horizon_bars
    ):
        raise ValueError("clean entry path input is invalid")

    sign = 1.0 if side == "LONG" else -1.0
    favorable_path: list[float] = []
    adverse_path: list[float] = []
    close_path: list[float] = []
    for bar in future_bars:
        try:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("clean entry path bar is invalid") from exc
        if not all(
            math.isfinite(value) and value > 0.0 for value in (high, low, close)
        ):
            raise ValueError("clean entry path bar is invalid")
        favorable = (
            (high - entry_price) / entry_price
            if side == "LONG"
            else (entry_price - low) / entry_price
        )
        adverse = (
            (entry_price - low) / entry_price
            if side == "LONG"
            else (high - entry_price) / entry_price
        )
        favorable_path.append(max(0.0, favorable))
        adverse_path.append(max(0.0, adverse))
        close_path.append(sign * (close / entry_price - 1.0))

    favorable_bar = next(
        (
            index
            for index, value in enumerate(favorable_path, start=1)
            if value >= config.favorable_barrier_fraction
        ),
        None,
    )
    adverse_bar = next(
        (
            index
            for index, value in enumerate(adverse_path, start=1)
            if value >= config.adverse_barrier_fraction
        ),
        None,
    )
    same_bar_ambiguous = (
        favorable_bar is not None
        and adverse_bar is not None
        and favorable_bar == adverse_bar
    )
    favorable_first = favorable_bar is not None and (
        adverse_bar is None or favorable_bar < adverse_bar
    )
    adverse_first = adverse_bar is not None and (
        favorable_bar is None or adverse_bar < favorable_bar
    )
    if same_bar_ambiguous:
        classification = CleanPathClassification.AMBIGUOUS_SAME_BAR
    elif favorable_first and favorable_bar is not None:
        classification = (
            CleanPathClassification.CLEAN_FAST_SUCCESS
            if favorable_bar <= config.fast_edge_bars
            else CleanPathClassification.CLEAN_SLOW_SUCCESS
        )
    elif adverse_first:
        classification = CleanPathClassification.ADVERSE_FIRST
    else:
        classification = CleanPathClassification.NO_DECISIVE_EDGE

    mfe = max(favorable_path)
    mae = max(adverse_path)
    terminal_return = close_path[-1]
    net_return = terminal_return - config.round_trip_cost_fraction
    mfe_peak_bar = favorable_path.index(mfe) + 1
    mae_peak_bar = adverse_path.index(mae) + 1
    return {
        "schema_id": "aegis-clean-entry-path-label-v1",
        "side": side,
        "classification": classification.value,
        "clean_path_success": classification
        in {
            CleanPathClassification.CLEAN_FAST_SUCCESS,
            CleanPathClassification.CLEAN_SLOW_SUCCESS,
        },
        "fast_edge_success": classification
        is CleanPathClassification.CLEAN_FAST_SUCCESS,
        "favorable_barrier_bar": favorable_bar,
        "adverse_barrier_bar": adverse_bar,
        "same_bar_order_ambiguous": same_bar_ambiguous,
        "mfe_fraction": mfe,
        "mae_fraction": mae,
        "mfe_peak_bar": mfe_peak_bar,
        "mae_peak_bar": mae_peak_bar,
        "mfe_before_mae_peak": mfe_peak_bar < mae_peak_bar,
        "terminal_return_fraction": terminal_return,
        "net_return_after_costs": net_return,
        "underwater_bars": sum(value < 0.0 for value in close_path),
        "horizon_bars": config.horizon_bars,
        "outcome_available": True,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
