"""Causal features and frozen labels for decomposed V9 research."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .tail_aware_entry_v8 import V8_FEATURE_NAMES, named_v7_features


class DirectionClass(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    ABSTAIN = "ABSTAIN"


class TimingFailure(str, Enum):
    EXHAUSTED_MOVE = "EXHAUSTED_MOVE"
    COUNTER_TREND_FAILURE = "COUNTER_TREND_FAILURE"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    WEAK_VOLUME_IMPULSE = "WEAK_VOLUME_IMPULSE"
    OVEREXTENSION_FAILURE = "OVEREXTENSION_FAILURE"
    TRANSITION_FAILURE = "TRANSITION_FAILURE"
    ADVERSE_CONTINUATION = "ADVERSE_CONTINUATION"


ROLLING_CONTEXT_FEATURE_NAMES = (
    "rolling_4h_return",
    "rolling_12h_return",
    "rolling_4h_volatility",
    "rolling_4h_volume_ratio",
    "rolling_12h_close_location",
)
SIDE_CONTEXT_FEATURE_NAMES = (
    "side_rolling_4h_return",
    "side_rolling_12h_return",
    "timeframe_alignment_score",
    "timeframe_conflict_score",
)
V9_FEATURE_NAMES = (
    *V8_FEATURE_NAMES,
    *ROLLING_CONTEXT_FEATURE_NAMES,
    *SIDE_CONTEXT_FEATURE_NAMES,
)
SIDE_NEUTRAL_V8_FEATURE_COUNT = V8_FEATURE_NAMES.index("side_ret_1")
V9_DIRECTION_FEATURE_NAMES = (
    *V8_FEATURE_NAMES[:SIDE_NEUTRAL_V8_FEATURE_COUNT],
    *ROLLING_CONTEXT_FEATURE_NAMES,
)


class DecomposedEntryV9Error(ValueError):
    pass


@dataclass(frozen=True)
class DirectionLabelContract:
    source_base_cost_fraction: float
    stress_cost_fraction: float
    minimum_edge_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.source_base_cost_fraction,
            self.stress_cost_fraction,
            self.minimum_edge_fraction,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in values)
            or self.stress_cost_fraction < self.source_base_cost_fraction
            or self.minimum_edge_fraction <= 0.0
        ):
            raise DecomposedEntryV9Error("invalid V9 direction label contract")


@dataclass(frozen=True)
class TimingLabelContract:
    clean_mae_fraction: float
    clean_positive_bar: int
    overextension_atr: float
    exhaustion_atr: float
    weak_volume_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.clean_mae_fraction,
            self.overextension_atr,
            self.exhaustion_atr,
            self.weak_volume_ratio,
        )
        if (
            not all(math.isfinite(value) and value > 0.0 for value in values)
            or self.clean_positive_bar <= 0
            or self.overextension_atr < self.exhaustion_atr
        ):
            raise DecomposedEntryV9Error("invalid V9 timing label contract")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DecomposedEntryV9Error(f"non-finite V9 value: {name}")
    return result


def rolling_four_hour_context(history: Sequence[Any]) -> Mapping[str, float]:
    """Build 4h/12h context from closed 5m bars only."""

    if len(history) < 145:
        raise DecomposedEntryV9Error("V9 rolling context requires 145 closed bars")
    closes = [_finite(value.close, "close") for value in history]
    highs = [_finite(value.high, "high") for value in history]
    lows = [_finite(value.low, "low") for value in history]
    volumes = [_finite(value.volume, "volume") for value in history]
    if min(closes) <= 0.0 or min(volumes) < 0.0:
        raise DecomposedEntryV9Error("V9 rolling context contains invalid market data")
    returns = [right / left - 1.0 for left, right in zip(closes[-49:-1], closes[-48:])]
    prior_volume = sum(volumes[-96:-48])
    current_volume = sum(volumes[-48:])
    low, high = min(lows[-144:]), max(highs[-144:])
    context = {
        "rolling_4h_return": closes[-1] / closes[-49] - 1.0,
        "rolling_12h_return": closes[-1] / closes[-145] - 1.0,
        "rolling_4h_volatility": statistics.pstdev(returns),
        "rolling_4h_volume_ratio": (
            current_volume / prior_volume if prior_volume > 0.0 else 1.0
        ),
        "rolling_12h_close_location": (
            (closes[-1] - low) / (high - low) if high > low else 0.5
        ),
    }
    if not all(math.isfinite(value) for value in context.values()):
        raise DecomposedEntryV9Error("V9 rolling context is invalid")
    return context


def v9_feature_vectors(
    row: Mapping[str, Any], rolling: Mapping[str, float]
) -> tuple[tuple[float, ...], tuple[float, ...], Mapping[str, float]]:
    source = tuple(_finite(value, "v8_feature") for value in row["v8_features"])
    if len(source) != len(V8_FEATURE_NAMES):
        raise DecomposedEntryV9Error("V9 source feature count is invalid")
    try:
        side_sign = 1.0 if str(row["side"]) == "LONG" else -1.0
        if str(row["side"]) not in {"LONG", "SHORT"}:
            raise KeyError("side")
        v7 = named_v7_features(row["v7_features"])
        context = {
            name: _finite(rolling[name], name) for name in ROLLING_CONTEXT_FEATURE_NAMES
        }
    except (KeyError, TypeError) as exc:
        raise DecomposedEntryV9Error("V9 feature row is incomplete") from exc
    alignments = (
        float(v7["side_ret_3"] > 0.0),
        float(v7["m15_side_alignment"] > 0.0),
        float(v7["h1_side_alignment"] > 0.0),
        float(context["rolling_4h_return"] * side_sign > 0.0),
    )
    side_context = {
        "side_rolling_4h_return": context["rolling_4h_return"] * side_sign,
        "side_rolling_12h_return": context["rolling_12h_return"] * side_sign,
        "timeframe_alignment_score": sum(alignments) / len(alignments),
        "timeframe_conflict_score": 1.0 - sum(alignments) / len(alignments),
    }
    side_features = (
        *source,
        *(context[name] for name in ROLLING_CONTEXT_FEATURE_NAMES),
        *(side_context[name] for name in SIDE_CONTEXT_FEATURE_NAMES),
    )
    direction_features = (
        *source[:SIDE_NEUTRAL_V8_FEATURE_COUNT],
        *(context[name] for name in ROLLING_CONTEXT_FEATURE_NAMES),
    )
    if (
        len(side_features) != len(V9_FEATURE_NAMES)
        or len(direction_features) != len(V9_DIRECTION_FEATURE_NAMES)
        or not all(
            math.isfinite(value) for value in (*side_features, *direction_features)
        )
    ):
        raise DecomposedEntryV9Error("V9 feature vector is invalid")
    return tuple(side_features), tuple(direction_features), {**context, **side_context}


def direction_label(
    long_row: Mapping[str, Any],
    short_row: Mapping[str, Any],
    contract: DirectionLabelContract,
) -> Mapping[str, Any]:
    if str(long_row.get("side")) != "LONG" or str(short_row.get("side")) != "SHORT":
        raise DecomposedEntryV9Error("V9 direction pair is invalid")
    adjustment = contract.stress_cost_fraction - contract.source_base_cost_fraction
    long_net = (
        _finite(long_row["terminal_return_after_costs"], "long_terminal") - adjustment
    )
    short_net = (
        _finite(short_row["terminal_return_after_costs"], "short_terminal") - adjustment
    )
    best = DirectionClass.LONG if long_net >= short_net else DirectionClass.SHORT
    best_net, opposite_net = (
        (long_net, short_net) if best is DirectionClass.LONG else (short_net, long_net)
    )
    label = (
        best
        if best_net > 0.0 and best_net - opposite_net >= contract.minimum_edge_fraction
        else DirectionClass.ABSTAIN
    )
    return {
        "label": label.value,
        "long_terminal_stress_net": long_net,
        "short_terminal_stress_net": short_net,
        "edge_fraction": best_net - opposite_net,
        "horizon_bars": 24,
    }


def timing_labels(
    row: Mapping[str, Any],
    context: Mapping[str, float],
    contract: TimingLabelContract,
) -> Mapping[str, Any]:
    v7 = named_v7_features(row["v7_features"])
    current_stress = _finite(
        row["v8_profile_cost_returns"]["CURRENT_TS"]["stress"], "current_stress"
    )
    mae = _finite(row["mae_fraction"], "mae")
    first_positive_raw = row.get("first_positive_after_cost_bar")
    first_positive = int(first_positive_raw) if first_positive_raw is not None else None
    first_adverse_raw = row.get("first_adverse_bar")
    first_favorable_raw = row.get("first_favorable_bar")
    first_adverse = int(first_adverse_raw) if first_adverse_raw is not None else None
    first_favorable = (
        int(first_favorable_raw) if first_favorable_raw is not None else None
    )
    adverse_first = first_adverse is not None and (
        first_favorable is None or first_adverse <= first_favorable
    )
    slow = first_positive is None or first_positive > contract.clean_positive_bar
    failed = current_stress <= 0.0
    extension = _finite(v7["side_extension_atr"], "extension")
    agreement = _finite(v7["trend_agreement_score"], "agreement")
    volume = _finite(v7["volume_ratio_6_24"], "volume")
    impulse = _finite(v7["side_ret_3"], "side_ret_3") > 0.0
    memberships = row["soft_archetype_memberships"]
    breakout = _finite(memberships["BREAKOUT"], "breakout_membership")
    maximum_membership = max(
        _finite(value, "membership") for value in memberships.values()
    )
    labels = {
        TimingFailure.EXHAUSTED_MOVE.value: bool(
            extension >= contract.exhaustion_atr and bool(row["early_reversal"])
        ),
        TimingFailure.COUNTER_TREND_FAILURE.value: bool(
            agreement < 0.5 and context["timeframe_conflict_score"] >= 0.5 and failed
        ),
        TimingFailure.FALSE_BREAKOUT.value: bool(
            breakout >= maximum_membership and adverse_first and slow
        ),
        TimingFailure.WEAK_VOLUME_IMPULSE.value: bool(
            impulse and volume < contract.weak_volume_ratio and failed
        ),
        TimingFailure.OVEREXTENSION_FAILURE.value: bool(
            extension >= contract.overextension_atr and failed
        ),
        TimingFailure.TRANSITION_FAILURE.value: bool(
            str(row["forward_regime_multihorizon"]["label"]) == "TRANSITION" and failed
        ),
        TimingFailure.ADVERSE_CONTINUATION.value: bool(
            adverse_first and mae > contract.clean_mae_fraction and slow
        ),
    }
    labels["CLEAN_TIMING"] = bool(
        current_stress > 0.0
        and mae <= contract.clean_mae_fraction
        and first_positive is not None
        and first_positive <= contract.clean_positive_bar
        and not adverse_first
    )
    labels["ANY_FAILURE"] = any(labels[name.value] for name in TimingFailure)
    return labels


def trajectory_targets(
    row: Mapping[str, Any], catastrophic_fraction: float
) -> Mapping[str, Any]:
    current = row["v8_profile_cost_returns"]["CURRENT_TS"]
    stress = _finite(current["stress"], "stress")
    first_positive_raw = row.get("first_positive_after_cost_bar")
    first_positive = int(first_positive_raw) if first_positive_raw is not None else None
    return {
        "positive_current_ts_stress": stress > 0.0,
        "catastrophic_current_ts_stress": stress <= catastrophic_fraction,
        "current_ts_expected_net": _finite(current["expected"], "expected"),
        "current_ts_stress_net": stress,
        "current_ts_severe_net": _finite(current["severe"], "severe"),
        "mae_fraction": _finite(row["mae_fraction"], "mae"),
        "mfe_fraction": _finite(row["mfe_fraction"], "mfe"),
        "time_to_positive": (
            min(1.0, first_positive / 24.0) if first_positive is not None else 1.0
        ),
    }
