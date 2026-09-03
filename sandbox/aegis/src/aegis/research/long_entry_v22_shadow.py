"""Causal LONG v2.2 specialist committee and cross-sectional ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .long_entry_v21_shadow import LONG_V21_FEATURE_NAMES

LONG_V22_INTERACTION_NAMES = (
    "directional_volume_pressure",
    "volume_acceleration_pressure",
    "breakout_conviction",
    "pullback_reclaim_quality",
    "cross_market_long_support",
    "higher_timeframe_alignment",
    "falling_knife_risk",
    "volatility_structure_interaction",
)
REGIME_DIRECTIONS = ("BULLISH", "NEUTRAL", "BEARISH")
REGIME_VOLATILITIES = ("LOW", "NORMAL", "HIGH")
REGIME_STRUCTURES = ("TREND", "RANGE", "TRANSITION")
REGIME_IDENTITIES = tuple(
    f"{direction}::{volatility}::{structure}"
    for direction in REGIME_DIRECTIONS
    for volatility in REGIME_VOLATILITIES
    for structure in REGIME_STRUCTURES
)
LONG_V22_FEATURE_NAMES = (
    *LONG_V21_FEATURE_NAMES,
    *LONG_V22_INTERACTION_NAMES,
    *(f"regime_{identity}" for identity in REGIME_IDENTITIES),
)


class LongV22ShadowError(ValueError):
    pass


def _finite(values: Mapping[str, float], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongV22ShadowError(f"missing LONG v2.2 input: {name}") from exc
    if not math.isfinite(value):
        raise LongV22ShadowError(f"non-finite LONG v2.2 input: {name}")
    return value


def _clip(value: float, lower: float = -20.0, upper: float = 20.0) -> float:
    return min(upper, max(lower, value))


def long_v22_feature_vector(
    v21_vector: Sequence[float], regime_axes: Mapping[str, str]
) -> tuple[float, ...]:
    """Add preregistered causal interactions and regime identity."""

    if len(v21_vector) != len(LONG_V21_FEATURE_NAMES):
        raise LongV22ShadowError("LONG v2.1 vector length is invalid")
    values = {
        name: float(value) for name, value in zip(LONG_V21_FEATURE_NAMES, v21_vector)
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise LongV22ShadowError("LONG v2.1 vector contains a non-finite value")
    atr = max(_finite(values, "atr_12"), 1e-8)
    ret_1 = _finite(values, "ret_1")
    ret_3 = _finite(values, "ret_3")
    ret_12 = _finite(values, "ret_12")
    volume_ratio = max(0.0, _finite(values, "volume_ratio_6_24"))
    volume_trend = _finite(values, "volume_trend_12")
    volume_return = _finite(values, "volume_return_1")
    close_location = min(1.0, max(0.0, _finite(values, "close_position_in_range")))
    body = min(1.0, max(0.0, _finite(values, "body_to_range")))
    distance_high = max(0.0, _finite(values, "distance_to_rolling_high_12"))
    long_stack = min(1.0, max(0.0, _finite(values, "trend_stack_long")))
    lower_wick = min(1.0, max(0.0, _finite(values, "lower_wick_fraction")))
    breadth = min(1.0, max(0.0, _finite(values, "market_breadth_6")))
    btc_support = 1.0 if _finite(values, "btc_trend_proxy") > 0.0 else 0.0
    eth_support = 1.0 if _finite(values, "eth_trend_proxy") > 0.0 else 0.0
    trend_15m = min(1.0, max(0.0, _finite(values, "15m_trend_stack_long")))
    trend_1h = min(1.0, max(0.0, _finite(values, "1h_trend_stack_long")))
    ret_15m = _finite(values, "15m_ret_3")
    ret_1h = _finite(values, "1h_ret_3")
    chop_1h = min(1.5, max(0.0, _finite(values, "1h_chop_12")))
    vol_ratio_15m = max(0.0, _finite(values, "15m_volatility_ratio_6_24"))

    interactions = (
        _clip(math.tanh(ret_3 / atr) * math.log1p(volume_ratio)),
        _clip(math.tanh(volume_return + volume_trend) * math.log1p(volume_ratio)),
        _clip(
            max(0.0, 1.0 - distance_high / atr)
            * close_location
            * body
            * math.log1p(volume_ratio)
        ),
        _clip(
            long_stack
            * lower_wick
            * max(0.0, math.tanh(ret_1 / atr))
            * (1.0 if _finite(values, "close_vs_ema_6") > 0.0 else 0.0)
        ),
        _clip(breadth * (btc_support + eth_support) / 2.0),
        _clip(
            (trend_15m + trend_1h)
            / 2.0
            * max(0.0, math.tanh(ret_15m / atr))
            * max(0.0, math.tanh(ret_1h / atr))
        ),
        _clip(
            max(0.0, -ret_12 / atr)
            * (1.0 - max(0.0, math.tanh(ret_1 / atr)))
            * (1.0 + max(0.0, _finite(values, "downside_momentum_6")))
        ),
        _clip(vol_ratio_15m * max(0.0, 1.0 - chop_1h)),
    )
    identity = "::".join(
        (
            str(regime_axes.get("direction")),
            str(regime_axes.get("volatility")),
            str(regime_axes.get("structure")),
        )
    )
    if identity not in REGIME_IDENTITIES:
        raise LongV22ShadowError("LONG v2.2 regime identity is invalid")
    one_hot = tuple(
        1.0 if candidate == identity else 0.0 for candidate in REGIME_IDENTITIES
    )
    result = (*values.values(), *interactions, *one_hot)
    if len(result) != len(LONG_V22_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in result
    ):
        raise LongV22ShadowError("LONG v2.2 feature vector is invalid")
    return result


def specialist_committee_score(
    direction_probability: float,
    timing_probability: float,
    path_risk_probability: float,
) -> float:
    """Combine calibrated specialists without manufacturing votes."""

    values = (direction_probability, timing_probability, path_risk_probability)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise LongV22ShadowError("LONG v2.2 probability is invalid")
    return direction_probability * timing_probability * (1.0 - path_risk_probability)


def select_cross_section(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_score: float,
    maximum_path_risk: float,
    maximum_selected_per_timestamp: int,
) -> tuple[bool, ...]:
    """Rank current candidates against each other using predictions only."""

    if (
        not math.isfinite(minimum_score)
        or not math.isfinite(maximum_path_risk)
        or not 0.0 <= minimum_score <= 1.0
        or not 0.0 <= maximum_path_risk <= 1.0
        or maximum_selected_per_timestamp <= 0
    ):
        raise LongV22ShadowError("LONG v2.2 ranking policy is invalid")
    groups: dict[Any, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        score = float(row["committee_score"])
        risk = float(row["path_risk_probability"])
        timing = float(row["timing_probability"])
        if not all(math.isfinite(value) for value in (score, risk, timing)):
            raise LongV22ShadowError("LONG v2.2 ranking input is invalid")
        if score >= minimum_score and risk <= maximum_path_risk:
            groups[row["timestamp"]].append((index, row))
    selected = [False] * len(rows)
    for candidates in groups.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["committee_score"]),
                float(item[1]["path_risk_probability"]),
                -float(item[1]["timing_probability"]),
                str(item[1]["symbol"]),
            ),
        )
        for index, _ in ordered[:maximum_selected_per_timestamp]:
            selected[index] = True
    return tuple(selected)
