"""Causal entry/exit attribution and context for directional V7 research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ..data import CanonicalBar
from ..training.hybrid_directional import DirectionalSide
from .hybrid_ts_protection_replay import (
    IntrabarPath,
    TsProtectionConfig,
    replay_ts_price_protection,
)
from .regime_aware_directional_v6 import REGIME_AWARE_V6_FEATURE_NAMES


class EntryArchetype(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    CONFIRMED_BREAKOUT = "CONFIRMED_BREAKOUT"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    RANGE_REVERSION = "RANGE_REVERSION"
    TRANSITION_SELECTIVE = "TRANSITION_SELECTIVE"


class TrajectoryResponsibility(str, Enum):
    CLEAN_REALIZED_WIN = "CLEAN_REALIZED_WIN"
    GOOD_ENTRY_POOR_CAPTURE = "GOOD_ENTRY_POOR_CAPTURE"
    LATE_OR_ADVERSE_ENTRY = "LATE_OR_ADVERSE_ENTRY"
    NO_DIRECTIONAL_EDGE = "NO_DIRECTIONAL_EDGE"
    AMBIGUOUS_PATH = "AMBIGUOUS_PATH"


V7_CONTEXT_FEATURE_NAMES = (
    "side_extension_atr",
    "side_acceleration",
    "volume_direction_impulse",
    "btc_side_alignment",
    "relative_side_strength",
    "m15_side_alignment",
    "h1_side_alignment",
    "favorable_close_location",
    "exhaustion_pressure",
    "trend_agreement_score",
)
V7_FEATURE_NAMES = (
    *REGIME_AWARE_V6_FEATURE_NAMES,
    *V7_CONTEXT_FEATURE_NAMES,
    *(f"archetype_{value.value}" for value in EntryArchetype),
)


class RegimeEntryExitV7Error(ValueError):
    pass


@dataclass(frozen=True)
class TrajectoryAuditContract:
    round_trip_cost_fraction: float
    maximum_clean_mae_fraction: float
    maximum_clean_positive_bar: int
    late_entry_extension_atr: float
    late_entry_positive_bar: int
    minimum_available_net_fraction: float

    def __post_init__(self) -> None:
        numeric = (
            self.round_trip_cost_fraction,
            self.maximum_clean_mae_fraction,
            self.late_entry_extension_atr,
            self.minimum_available_net_fraction,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in numeric)
            or self.round_trip_cost_fraction >= 1.0
            or self.maximum_clean_positive_bar <= 0
            or self.late_entry_positive_bar < self.maximum_clean_positive_bar
        ):
            raise RegimeEntryExitV7Error("invalid V7 trajectory audit contract")


def named_v6_features(values: Sequence[float]) -> Mapping[str, float]:
    if len(values) != len(REGIME_AWARE_V6_FEATURE_NAMES):
        raise RegimeEntryExitV7Error("V7 source feature count is invalid")
    result = {
        name: float(value) for name, value in zip(REGIME_AWARE_V6_FEATURE_NAMES, values)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RegimeEntryExitV7Error("V7 source features are non-finite")
    return result


def causal_entry_context(
    values: Mapping[str, float], side: DirectionalSide
) -> Mapping[str, float]:
    """Build side-relative context using only information known at signal time."""

    required = (
        "atr_12",
        "side_ret_1",
        "side_ret_3",
        "side_ret_12",
        "volume_ratio_6_24",
        "btc_trend_proxy",
        "relative_return_6",
        "15m_ret_3",
        "1h_ret_3",
        "side_close_position_in_range",
        "adverse_wick_fraction",
        "side_trend_stack",
        "15m_trend_stack_long",
        "15m_trend_stack_short",
        "1h_trend_stack_long",
        "1h_trend_stack_short",
    )
    try:
        source = {name: float(values[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeEntryExitV7Error("V7 context features are incomplete") from exc
    if not all(math.isfinite(value) for value in source.values()):
        raise RegimeEntryExitV7Error("V7 context features are non-finite")
    atr = max(source["atr_12"], 1e-8)
    side_sign = side.sign
    m15_stack = (
        source["15m_trend_stack_long"]
        if side_sign > 0.0
        else source["15m_trend_stack_short"]
    )
    h1_stack = (
        source["1h_trend_stack_long"]
        if side_sign > 0.0
        else source["1h_trend_stack_short"]
    )
    m15_alignment = source["15m_ret_3"] * side_sign
    h1_alignment = source["1h_ret_3"] * side_sign
    extension = source["side_ret_12"] / atr
    impulse = max(0.0, source["side_ret_3"] / atr) * max(
        0.0, source["volume_ratio_6_24"]
    )
    exhaustion = max(0.0, extension - 1.0) * (
        1.0 + max(0.0, source["adverse_wick_fraction"])
    )
    trend_agreement = (
        sum(
            (
                source["side_trend_stack"] > 0.5,
                m15_stack > 0.5,
                h1_stack > 0.5,
                m15_alignment > 0.0,
                h1_alignment > 0.0,
            )
        )
        / 5.0
    )
    result = {
        "side_extension_atr": extension,
        "side_acceleration": source["side_ret_3"] - source["side_ret_12"] / 4.0,
        "volume_direction_impulse": impulse,
        "btc_side_alignment": source["btc_trend_proxy"] * side_sign,
        "relative_side_strength": source["relative_return_6"] * side_sign,
        "m15_side_alignment": m15_alignment,
        "h1_side_alignment": h1_alignment,
        "favorable_close_location": source["side_close_position_in_range"],
        "exhaustion_pressure": exhaustion,
        "trend_agreement_score": trend_agreement,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RegimeEntryExitV7Error("V7 derived context is non-finite")
    return result


def classify_entry_archetype(
    row: Mapping[str, Any], features: Mapping[str, float], context: Mapping[str, float]
) -> EntryArchetype:
    """Route by causal structure; no outcome field participates."""

    regime = row.get("regime")
    if not isinstance(regime, Mapping):
        raise RegimeEntryExitV7Error("V7 regime is missing")
    structure = str(regime.get("structure"))
    phase = str(regime.get("phase"))
    role = str(row.get("directional_role"))
    side_ret_1 = float(features["side_ret_1"])
    side_ret_3 = float(features["side_ret_3"])
    side_ret_12 = float(features["side_ret_12"])
    volume = float(features["volume_ratio_6_24"])
    close_location = float(features["side_close_position_in_range"])
    favorable_wick = float(features["favorable_wick_fraction"])
    if (
        structure == "TREND"
        and role == "PRIMARY_TREND"
        and phase in {"CONTINUATION", "PULLBACK"}
        and side_ret_12 > 0.0
        and context["trend_agreement_score"] >= 0.6
    ):
        return EntryArchetype.TREND_CONTINUATION
    if (
        side_ret_3 > 0.0
        and side_ret_12 > 0.0
        and volume >= 1.0
        and close_location >= 0.65
        and context["trend_agreement_score"] >= 0.4
    ):
        return EntryArchetype.CONFIRMED_BREAKOUT
    if (
        role == "TACTICAL_COUNTERTREND"
        and phase in {"EXHAUSTION", "PULLBACK", "TRANSITION"}
        and side_ret_1 > 0.0
        and side_ret_3 > 0.0
        and favorable_wick > float(features["adverse_wick_fraction"])
    ):
        return EntryArchetype.CONFIRMED_REVERSAL
    if (
        structure == "RANGE"
        and side_ret_1 > 0.0
        and favorable_wick >= 0.25
        and 0.25 <= close_location <= 0.85
    ):
        return EntryArchetype.RANGE_REVERSION
    return EntryArchetype.TRANSITION_SELECTIVE


def v7_feature_vector(
    row: Mapping[str, Any],
) -> tuple[tuple[float, ...], EntryArchetype, Mapping[str, float]]:
    source = named_v6_features(row["features"])
    try:
        side = DirectionalSide(str(row["side"]))
    except (KeyError, ValueError) as exc:
        raise RegimeEntryExitV7Error("V7 side is invalid") from exc
    context = causal_entry_context(source, side)
    archetype = classify_entry_archetype(row, source, context)
    result = (
        *(float(value) for value in row["features"]),
        *(context[name] for name in V7_CONTEXT_FEATURE_NAMES),
        *(1.0 if archetype is value else 0.0 for value in EntryArchetype),
    )
    if len(result) != len(V7_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in result
    ):
        raise RegimeEntryExitV7Error("V7 feature vector is invalid")
    return tuple(result), archetype, context


def trajectory_attribution(
    row: Mapping[str, Any],
    context: Mapping[str, float],
    contract: TrajectoryAuditContract,
) -> Mapping[str, Any]:
    """Separate entry quality from the amount of opportunity captured on exit."""

    try:
        mfe = float(row["mfe_fraction"])
        mae = float(row["mae_fraction"])
        realized = float(row["full_lifecycle_worst_net_return"])
        first_positive_raw = row.get("first_positive_after_cost_bar")
        first_positive = (
            int(first_positive_raw) if first_positive_raw is not None else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeEntryExitV7Error("V7 trajectory row is incomplete") from exc
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (mfe, mae)
    ) or not math.isfinite(realized):
        raise RegimeEntryExitV7Error("V7 trajectory values are invalid")
    available_net = max(0.0, mfe - contract.round_trip_cost_fraction)
    positive_realized = max(0.0, realized)
    capture_efficiency = (
        min(1.0, positive_realized / available_net) if available_net > 0.0 else 0.0
    )
    missed_capture = max(0.0, available_net - positive_realized)
    slow = first_positive is None or first_positive > contract.late_entry_positive_bar
    extended_failure = context[
        "side_extension_atr"
    ] >= contract.late_entry_extension_atr and not bool(row["protectable_advantage"])
    late = bool(row["early_reversal"]) or extended_failure or (slow and realized <= 0.0)
    clean = bool(
        not row["same_bar_ambiguity"]
        and bool(row["protectable_advantage"])
        and mae <= contract.maximum_clean_mae_fraction
        and first_positive is not None
        and first_positive <= contract.maximum_clean_positive_bar
        and realized > 0.0
    )
    if bool(row["same_bar_ambiguity"]):
        responsibility = TrajectoryResponsibility.AMBIGUOUS_PATH
    elif clean:
        responsibility = TrajectoryResponsibility.CLEAN_REALIZED_WIN
    elif late:
        responsibility = TrajectoryResponsibility.LATE_OR_ADVERSE_ENTRY
    elif (
        bool(row["protectable_advantage"])
        or available_net >= contract.minimum_available_net_fraction
    ) and realized <= 0.0:
        responsibility = TrajectoryResponsibility.GOOD_ENTRY_POOR_CAPTURE
    else:
        responsibility = TrajectoryResponsibility.NO_DIRECTIONAL_EDGE
    return {
        "clean_entry": clean,
        "positive_protected_net": realized > 0.0,
        "late_entry": late,
        "responsibility": responsibility.value,
        "available_net_opportunity": available_net,
        "realized_protected_net": realized,
        "capture_efficiency": capture_efficiency,
        "missed_capture_fraction": missed_capture,
        "slow_to_positive": slow,
        "extended_entry_failure": extended_failure,
    }


def replay_protection_profiles(
    *,
    side: DirectionalSide,
    history: Sequence[CanonicalBar],
    future: Sequence[CanonicalBar],
    profiles: Mapping[str, TsProtectionConfig],
) -> Mapping[str, Mapping[str, Any]]:
    if not profiles:
        raise RegimeEntryExitV7Error("V7 protection profiles are empty")
    result: dict[str, Mapping[str, Any]] = {}
    for name, config in profiles.items():
        paths = [
            replay_ts_price_protection(
                side=side,
                history=history,
                future=future,
                path=path,
                config=config,
            )
            for path in IntrabarPath
        ]
        worst = min(paths, key=lambda value: value.net_return_after_costs)
        result[name] = {
            "worst_net_return": worst.net_return_after_costs,
            "worst_exit_reason": worst.exit_reason.value,
            "worst_bars_held": worst.bars_held,
            "break_even_armed": worst.break_even_armed,
            "trailing_armed": worst.trailing_armed,
            "path_spread": max(value.net_return_after_costs for value in paths)
            - min(value.net_return_after_costs for value in paths),
        }
    return result
