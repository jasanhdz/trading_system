"""Causal soft routing and tail-aware labels for V8 research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .regime_entry_exit_v7 import V7_FEATURE_NAMES


class SoftArchetype(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    RANGE_REVERSION = "RANGE_REVERSION"
    EXHAUSTION_RISK = "EXHAUSTION_RISK"


FORWARD_REGIMES = ("BULLISH", "BEARISH", "RANGE", "TRANSITION")
SOFT_ARCHETYPE_FEATURE_NAMES = tuple(
    f"soft_archetype_{value.value}" for value in SoftArchetype
)
V8_FEATURE_NAMES = (*V7_FEATURE_NAMES, *SOFT_ARCHETYPE_FEATURE_NAMES)


class TailAwareV8Error(ValueError):
    pass


@dataclass(frozen=True)
class TailLabelContract:
    clean_mae_fraction: float
    clean_positive_bar: int
    late_positive_bar: int
    catastrophic_net_fraction: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.clean_mae_fraction)
            or self.clean_mae_fraction <= 0.0
            or self.clean_positive_bar <= 0
            or self.late_positive_bar < self.clean_positive_bar
            or not math.isfinite(self.catastrophic_net_fraction)
            or self.catastrophic_net_fraction >= 0.0
        ):
            raise TailAwareV8Error("invalid V8 tail label contract")


def named_v7_features(values: Sequence[float]) -> Mapping[str, float]:
    if len(values) != len(V7_FEATURE_NAMES):
        raise TailAwareV8Error("V8 source feature count is invalid")
    result = {name: float(value) for name, value in zip(V7_FEATURE_NAMES, values)}
    if not all(math.isfinite(value) for value in result.values()):
        raise TailAwareV8Error("V8 source features are non-finite")
    return result


def _softmax(scores: Mapping[SoftArchetype, float]) -> Mapping[str, float]:
    maximum = max(scores.values())
    exponentials = {name: math.exp(value - maximum) for name, value in scores.items()}
    denominator = sum(exponentials.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise TailAwareV8Error("V8 archetype normalization failed")
    return {name.value: value / denominator for name, value in exponentials.items()}


def soft_archetype_memberships(
    row: Mapping[str, Any], features: Mapping[str, float]
) -> Mapping[str, float]:
    """Return overlapping causal setup strengths instead of one hard fallback."""

    regime = row.get("regime")
    if not isinstance(regime, Mapping):
        raise TailAwareV8Error("V8 regime is missing")
    structure = str(regime.get("structure"))
    phase = str(regime.get("phase"))
    role = str(row.get("directional_role"))
    side_ret_1 = features["side_ret_1"]
    side_ret_3 = features["side_ret_3"]
    side_ret_12 = features["side_ret_12"]
    atr = max(features["atr_12"], 1e-8)
    extension = side_ret_12 / atr
    volume = max(0.0, features["volume_ratio_6_24"])
    location = features["side_close_position_in_range"]
    favorable_wick = features["favorable_wick_fraction"]
    adverse_wick = features["adverse_wick_fraction"]
    agreement = features["trend_agreement_score"]
    scores = {
        SoftArchetype.TREND_CONTINUATION: (
            1.2 * agreement
            + 0.8 * float(role == "PRIMARY_TREND")
            + 0.7 * float(structure == "TREND")
            + 0.5 * float(side_ret_12 > 0.0)
            + 0.3 * float(phase in {"CONTINUATION", "PULLBACK"})
        ),
        SoftArchetype.BREAKOUT: (
            0.8 * min(volume, 2.0)
            + 0.8 * location
            + 0.5 * float(side_ret_3 > 0.0)
            + 0.4 * float(features["side_acceleration"] > 0.0)
            + 0.3 * float(features["range_expansion"] > 0.0)
        ),
        SoftArchetype.REVERSAL: (
            0.9 * float(side_ret_12 < 0.0)
            + 0.7 * float(side_ret_1 > 0.0)
            + 0.7 * float(side_ret_3 > 0.0)
            + 0.6 * max(0.0, favorable_wick - adverse_wick)
            + 0.4 * float(phase in {"EXHAUSTION", "PULLBACK"})
        ),
        SoftArchetype.RANGE_REVERSION: (
            1.0 * float(structure == "RANGE")
            + 0.6 * favorable_wick
            + 0.5 * float(0.25 <= location <= 0.85)
            + 0.4 * float(abs(extension) <= 1.5)
        ),
        SoftArchetype.EXHAUSTION_RISK: (
            0.8 * max(0.0, abs(extension) - 1.0)
            + 0.7 * adverse_wick
            + 0.5 * max(0.0, volume - 1.0)
            + 0.6 * float(phase == "EXHAUSTION")
            + 0.4 * float(location >= 0.85)
        ),
    }
    return _softmax(scores)


def v8_feature_vector(
    row: Mapping[str, Any],
) -> tuple[tuple[float, ...], Mapping[str, float]]:
    source = named_v7_features(row["v7_features"])
    memberships = soft_archetype_memberships(row, source)
    result = (
        *(float(value) for value in row["v7_features"]),
        *(memberships[value.value] for value in SoftArchetype),
    )
    if len(result) != len(V8_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in result
    ):
        raise TailAwareV8Error("V8 feature vector is invalid")
    return tuple(result), memberships


def classify_forward_regime(
    returns_by_horizon: Mapping[int, Mapping[str, float]],
    *,
    btc_threshold_at_24_fraction: float,
    breadth_threshold: float,
    range_breadth_band: tuple[float, float],
    consensus_horizons: int,
) -> Mapping[str, Any]:
    """Label the future environment; this label is never an input feature."""

    if (
        not returns_by_horizon
        or not 0.5 < breadth_threshold < 1.0
        or not 0.0 <= range_breadth_band[0] < range_breadth_band[1] <= 1.0
        or consensus_horizons <= 0
    ):
        raise TailAwareV8Error("invalid V8 forward regime contract")
    labels = {}
    diagnostics = {}
    for horizon, returns in sorted(returns_by_horizon.items()):
        if "BTCUSDT" not in returns or not returns:
            raise TailAwareV8Error("V8 forward regime requires BTC")
        values = tuple(float(value) for value in returns.values())
        if not all(math.isfinite(value) for value in values):
            raise TailAwareV8Error("V8 forward returns are non-finite")
        threshold = btc_threshold_at_24_fraction * math.sqrt(horizon / 24.0)
        btc = float(returns["BTCUSDT"])
        breadth = sum(value > 0.0 for value in values) / len(values)
        if btc >= threshold and breadth >= breadth_threshold:
            label = "BULLISH"
        elif btc <= -threshold and breadth <= 1.0 - breadth_threshold:
            label = "BEARISH"
        elif (
            abs(btc) < threshold
            and range_breadth_band[0] <= breadth <= range_breadth_band[1]
        ):
            label = "RANGE"
        else:
            label = "TRANSITION"
        labels[horizon] = label
        diagnostics[horizon] = {
            "btc_return": btc,
            "breadth": breadth,
            "threshold": threshold,
            "label": label,
        }
    counts = {
        name: sum(value == name for value in labels.values())
        for name in FORWARD_REGIMES
    }
    directional = max(("BULLISH", "BEARISH", "RANGE"), key=lambda name: counts[name])
    consensus = (
        directional if counts[directional] >= consensus_horizons else "TRANSITION"
    )
    return {
        "label": consensus,
        "horizon_labels": {str(key): value for key, value in labels.items()},
        "diagnostics": {str(key): value for key, value in diagnostics.items()},
    }


def tail_labels(
    row: Mapping[str, Any],
    profile_stress_returns: Mapping[str, float],
    contract: TailLabelContract,
) -> Mapping[str, Any]:
    try:
        mae = float(row["mae_fraction"])
        first_positive_raw = row.get("first_positive_after_cost_bar")
        first_positive = (
            int(first_positive_raw) if first_positive_raw is not None else None
        )
        first_adverse_raw = row.get("first_adverse_bar")
        first_favorable_raw = row.get("first_favorable_bar")
        first_adverse = (
            int(first_adverse_raw) if first_adverse_raw is not None else None
        )
        first_favorable = (
            int(first_favorable_raw) if first_favorable_raw is not None else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TailAwareV8Error("V8 tail label row is incomplete") from exc
    if not profile_stress_returns or not all(
        math.isfinite(float(value)) for value in profile_stress_returns.values()
    ):
        raise TailAwareV8Error("V8 profile returns are invalid")
    early_adverse = first_adverse is not None and (
        first_favorable is None or first_adverse <= first_favorable
    )
    slow = first_positive is None or first_positive > contract.late_positive_bar
    late = bool(row["early_reversal"]) or early_adverse or slow
    clean = bool(
        not row["same_bar_ambiguity"]
        and bool(row["target_before_stop"])
        and mae <= contract.clean_mae_fraction
        and first_positive is not None
        and first_positive <= contract.clean_positive_bar
    )
    best_profile = max(profile_stress_returns, key=profile_stress_returns.get)
    best_net = float(profile_stress_returns[best_profile])
    return {
        "clean_entry": clean,
        "late_entry": late,
        "early_adverse": early_adverse,
        "slow_to_positive": slow,
        "hindsight_best_profile": best_profile,
        "hindsight_best_stress_net": best_net,
        "positive_stress_net": best_net > 0.0,
        "catastrophic_stress_loss": best_net <= contract.catastrophic_net_fraction,
    }
