"""Feature families, causal taker flow, and quality metrics for V14."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from .competing_barrier_v10 import BarrierResearchError
from .decomposed_entry_v9 import V9_FEATURE_NAMES

TAKER_FLOW_FEATURE_NAMES = (
    "taker_imbalance_1",
    "taker_imbalance_3",
    "taker_imbalance_6",
    "taker_imbalance_12",
    "taker_imbalance_24",
    "taker_imbalance_acceleration_3_12",
    "market_taker_imbalance_6",
    "market_taker_breadth_6",
    "btc_taker_imbalance_6",
    "relative_taker_imbalance_6",
)


def positional_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    counts = {name: names.count(name) for name in set(names)}
    return tuple(
        f"{name}__index_{index}" if counts[name] > 1 else name
        for index, name in enumerate(names)
    )


def feature_families() -> Mapping[str, tuple[str, ...]]:
    names = tuple(V9_FEATURE_NAMES)
    assigned: dict[str, list[str]] = {
        "MARKET_CROSS_SECTION": [],
        "LOCAL_MOMENTUM_TREND": [],
        "VOLATILITY_RANGE": [],
        "VOLUME_ACTIVITY": [],
        "CANDLE_STRUCTURE_REVERSAL": [],
        "MULTI_TIMEFRAME": [],
        "REGIME_SIDE_ARCHETYPE": [],
        "ROLLING_CONTEXT": [],
    }
    for name in names:
        if name.startswith(("15m_", "1h_")) or name in {
            "m15_side_alignment",
            "h1_side_alignment",
            "timeframe_alignment_score",
            "timeframe_conflict_score",
        }:
            family = "MULTI_TIMEFRAME"
        elif name.startswith("rolling_") or name.startswith("side_rolling_"):
            family = "ROLLING_CONTEXT"
        elif name.startswith(("regime_", "archetype_", "soft_archetype_")) or name in {
            "directional_role_PRIMARY_TREND",
            "directional_role_TACTICAL_COUNTERTREND",
            "directional_role_SELECTIVE",
            "side_extension_atr",
            "side_acceleration",
            "btc_side_alignment",
            "relative_side_strength",
            "exhaustion_pressure",
            "trend_agreement_score",
        }:
            family = "REGIME_SIDE_ARCHETYPE"
        elif "volume" in name or name == "volume_direction_impulse":
            family = "VOLUME_ACTIVITY"
        elif any(
            token in name for token in ("volatility", "atr_", "range_", "compression")
        ):
            family = "VOLATILITY_RANGE"
        elif any(
            token in name
            for token in (
                "wick",
                "position_in_range",
                "body_to_range",
                "breakdown",
                "reversal",
                "rebound",
                "exhaustion",
                "distance_to_rolling",
            )
        ):
            family = "CANDLE_STRUCTURE_REVERSAL"
        elif name.startswith(("market_", "cross_", "btc_", "eth_", "relative_return")):
            family = "MARKET_CROSS_SECTION"
        else:
            family = "LOCAL_MOMENTUM_TREND"
        assigned[family].append(name)
    flattened = [name for values in assigned.values() for name in values]
    if len(flattened) != len(names) or set(flattened) != set(names):
        raise BarrierResearchError("V14 feature family partition is incomplete")
    return {name: tuple(values) for name, values in assigned.items()}


def taker_imbalance(volume: float, buy_volume: float) -> float:
    total = float(volume)
    bought = float(buy_volume)
    if (
        not math.isfinite(total)
        or not math.isfinite(bought)
        or total < 0.0
        or bought < 0.0
        or bought > total + 1e-9
    ):
        raise BarrierResearchError("invalid V14 taker volume")
    return 0.0 if total == 0.0 else (2.0 * bought - total) / total


def local_taker_flow(history: Sequence[float]) -> Mapping[str, float]:
    if len(history) < 24 or not all(math.isfinite(float(value)) for value in history):
        raise BarrierResearchError("V14 taker flow requires 24 finite closed bars")
    values = tuple(float(value) for value in history[-24:])
    mean = lambda size: math.fsum(values[-size:]) / size
    return {
        "taker_imbalance_1": values[-1],
        "taker_imbalance_3": mean(3),
        "taker_imbalance_6": mean(6),
        "taker_imbalance_12": mean(12),
        "taker_imbalance_24": mean(24),
        "taker_imbalance_acceleration_3_12": mean(3) - mean(12),
    }


def market_taker_flow(
    local: Mapping[str, Mapping[str, float]], *, symbol: str
) -> Mapping[str, float]:
    if symbol not in local or "BTCUSDT" not in local or len(local) != 11:
        raise BarrierResearchError("V14 market flow requires eleven symbols and BTC")
    values = [float(item["taker_imbalance_6"]) for item in local.values()]
    market = math.fsum(values) / len(values)
    own = float(local[symbol]["taker_imbalance_6"])
    return {
        "market_taker_imbalance_6": market,
        "market_taker_breadth_6": sum(value > 0.0 for value in values) / len(values),
        "btc_taker_imbalance_6": float(local["BTCUSDT"]["taker_imbalance_6"]),
        "relative_taker_imbalance_6": own - market,
    }


def quality_profile(
    matrix: Sequence[Sequence[float]], names: Sequence[str], *, near_constant_std: float
) -> Mapping[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or len(set(names)) != len(names)
        or not np.isfinite(values).all()
    ):
        raise BarrierResearchError("invalid V14 feature quality matrix")
    standard_deviation = np.std(values, axis=0)
    unique_counts = [
        len(np.unique(values[:, index])) for index in range(values.shape[1])
    ]
    return {
        "rows": len(values),
        "features": len(names),
        "non_finite_values": 0,
        "near_constant": [
            name
            for name, value in zip(names, standard_deviation)
            if value <= near_constant_std
        ],
        "single_value": [
            name for name, count in zip(names, unique_counts) if count == 1
        ],
        "minimum_unique_values": min(unique_counts),
        "median_unique_values": float(np.median(unique_counts)),
    }


def robust_shift(
    earlier: Sequence[Sequence[float]],
    later: Sequence[Sequence[float]],
    names: Sequence[str],
) -> Mapping[str, float]:
    left = np.asarray(earlier, dtype=np.float64)
    right = np.asarray(later, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[1]
        or left.shape[1] != len(names)
    ):
        raise BarrierResearchError("invalid V14 drift matrices")
    lower, upper = np.quantile(left, [0.25, 0.75], axis=0)
    scale = np.maximum(upper - lower, 1e-12)
    shift = np.abs(np.median(right, axis=0) - np.median(left, axis=0)) / scale
    return {name: float(value) for name, value in zip(names, shift)}


def binary_probability_metrics(
    actual: Sequence[int], predicted: Sequence[float]
) -> Mapping[str, float]:
    labels = np.asarray(actual, dtype=np.int8)
    probabilities = np.asarray(predicted, dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or not len(labels)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or len(np.unique(labels)) != 2
    ):
        raise BarrierResearchError("invalid V14 binary probability observations")
    return {
        "rows": int(len(labels)),
        "prevalence": float(np.mean(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def quantile_pinball_loss(
    actual: Sequence[float], predicted: Sequence[float], *, quantile: float
) -> float:
    labels = np.asarray(actual, dtype=np.float64)
    estimates = np.asarray(predicted, dtype=np.float64)
    if (
        labels.ndim != 1
        or estimates.ndim != 1
        or labels.shape != estimates.shape
        or not len(labels)
        or not np.isfinite(labels).all()
        or not np.isfinite(estimates).all()
        or not 0.0 < quantile < 1.0
    ):
        raise BarrierResearchError("invalid V14 quantile observations")
    residual = labels - estimates
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))
