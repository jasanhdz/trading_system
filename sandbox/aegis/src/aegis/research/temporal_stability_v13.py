"""Temporal consensus, distribution, and MAE gates for V13 research."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


def jensen_shannon_divergence(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    if set(left) != set(right) or not left:
        raise BarrierResearchError("incompatible V13 probability vectors")
    p = np.asarray([float(left[name]) for name in sorted(left)], dtype=np.float64)
    q = np.asarray([float(right[name]) for name in sorted(left)], dtype=np.float64)
    if (
        not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or np.any(p < 0.0)
        or np.any(q < 0.0)
        or not math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-6)
        or not math.isclose(float(np.sum(q)), 1.0, abs_tol=1e-6)
    ):
        raise BarrierResearchError("invalid V13 probability vector")
    midpoint = 0.5 * (p + q)
    epsilon = 1e-12
    left_kl = np.sum(p * np.log((p + epsilon) / (midpoint + epsilon)))
    right_kl = np.sum(q * np.log((q + epsilon) / (midpoint + epsilon)))
    return float(0.5 * (left_kl + right_kl))


def consensus_probabilities(
    historical: Mapping[str, float],
    recent: Mapping[str, float],
    *,
    regime_expert: Mapping[str, float] | None,
    maximum_divergence: float,
) -> Mapping[str, Any]:
    if not math.isfinite(maximum_divergence) or maximum_divergence < 0.0:
        raise BarrierResearchError("invalid V13 divergence threshold")
    vectors = [historical, recent]
    historical_state = max(historical, key=historical.get)
    recent_state = max(recent, key=recent.get)
    dominant_match = historical_state == recent_state
    expert_match = True
    if regime_expert is not None:
        vectors.append(regime_expert)
        expert_match = max(regime_expert, key=regime_expert.get) == historical_state
    divergences = [
        jensen_shannon_divergence(left, right)
        for index, left in enumerate(vectors)
        for right in vectors[index + 1 :]
    ]
    maximum_observed = max(divergences)
    names = set(historical)
    if any(set(vector) != names for vector in vectors):
        raise BarrierResearchError("incompatible V13 consensus classes")
    blended = {
        name: sum(float(vector[name]) for vector in vectors) / len(vectors)
        for name in names
    }
    return {
        "probabilities": blended,
        "historical_recent_match": dominant_match,
        "regime_expert_match": expert_match,
        "maximum_divergence": maximum_observed,
        "eligible": bool(
            dominant_match and expert_match and maximum_observed <= maximum_divergence
        ),
        "predictor_count": len(vectors),
    }


def fit_robust_distribution(
    features: Sequence[Sequence[float]], *, minimum_scale: float
) -> Mapping[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    if (
        matrix.ndim != 2
        or not len(matrix)
        or not np.isfinite(matrix).all()
        or not math.isfinite(minimum_scale)
        or minimum_scale <= 0.0
    ):
        raise BarrierResearchError("invalid V13 distribution reference")
    lower = np.quantile(matrix, 0.25, axis=0)
    upper = np.quantile(matrix, 0.75, axis=0)
    return {
        "center": np.median(matrix, axis=0),
        "scale": np.maximum(upper - lower, minimum_scale),
    }


def distribution_scores(
    features: Sequence[Sequence[float]], reference: Mapping[str, Any]
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    center = np.asarray(reference["center"], dtype=np.float64)
    scale = np.asarray(reference["scale"], dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1:] != center.shape
        or center.shape != scale.shape
    ):
        raise BarrierResearchError("incompatible V13 distribution features")
    values = np.median(np.abs((matrix - center) / scale), axis=1)
    if not np.isfinite(values).all():
        raise BarrierResearchError("non-finite V13 distribution score")
    return values


def conservative_mae(
    historical: Sequence[float], recent: Sequence[float]
) -> np.ndarray:
    left = np.asarray(historical, dtype=np.float64)
    right = np.asarray(recent, dtype=np.float64)
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise BarrierResearchError("invalid V13 MAE predictions")
    return np.maximum(0.0, np.maximum(left, right))


def select_temporal_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    required = (
        "minimum_utility",
        "minimum_coherent_probability",
        "maximum_adverse_probability",
        "maximum_unknown_probability",
        "maximum_predicted_mae",
    )
    try:
        limits = {name: float(policy[name]) for name in required}
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BarrierResearchError("incomplete V13 policy") from exc
    if maximum <= 0 or any(not math.isfinite(value) for value in limits.values()):
        raise BarrierResearchError("invalid V13 policy")
    candidates: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if (
            bool(row["temporal_consensus"])
            and bool(row["in_distribution"])
            and float(row["predicted_utility"]) >= limits["minimum_utility"]
            and float(row["coherent_probability"])
            >= limits["minimum_coherent_probability"]
            and float(row["adverse_probability"])
            <= limits["maximum_adverse_probability"]
            and float(row["unknown_probability"])
            <= limits["maximum_unknown_probability"]
            and float(row["predicted_mae_q90"]) <= limits["maximum_predicted_mae"]
        ):
            candidates.setdefault(str(row["timestamp"]), []).append((index, row))
    selected = [False] * len(rows)
    for values in candidates.values():
        values.sort(
            key=lambda item: (
                -float(item[1]["predicted_utility"]),
                float(item[1]["predicted_mae_q90"]),
                float(item[1]["consensus_divergence"]),
                str(item[1]["symbol"]),
                str(item[1]["side"]),
            )
        )
        for index, _ in values[:maximum]:
            selected[index] = True
    return tuple(selected)
