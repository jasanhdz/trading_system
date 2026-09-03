"""Calibration and selection primitives for V11 research."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


def multiclass_ece(
    probabilities: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    *,
    bins: int,
) -> float:
    if len(probabilities) != len(labels) or not probabilities or bins <= 1:
        raise BarrierResearchError("invalid V11 ECE inputs")
    confidence = np.asarray([max(values.values()) for values in probabilities])
    predicted = [max(values, key=values.get) for values in probabilities]
    correct = np.asarray([left == right for left, right in zip(predicted, labels)])
    if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise BarrierResearchError("invalid V11 probability confidence")
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
    return result


def shrink_group_probabilities(
    global_values: Mapping[str, float],
    group_values: Sequence[tuple[Mapping[str, float], int]],
    *,
    shrinkage_rows: int,
) -> Mapping[str, float]:
    if shrinkage_rows <= 0 or not global_values:
        raise BarrierResearchError("invalid V11 shrinkage inputs")
    classes = set(global_values)
    accum = {name: float(value) for name, value in global_values.items()}
    total_weight = 1.0
    for values, rows in group_values:
        if set(values) != classes or rows <= 0:
            raise BarrierResearchError("invalid V11 calibration group")
        weight = rows / (rows + shrinkage_rows)
        for name in classes:
            accum[name] += weight * float(values[name])
        total_weight += weight
    normalized = {name: max(1e-9, value / total_weight) for name, value in accum.items()}
    denominator = sum(normalized.values())
    return {name: value / denominator for name, value in normalized.items()}


def select_v11_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    required = (
        "minimum_utility",
        "minimum_direction_probability",
        "minimum_clean_probability",
        "maximum_unknown_probability",
    )
    try:
        thresholds = {name: float(policy[name]) for name in required}
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BarrierResearchError("incomplete V11 selection policy") from exc
    if (
        not all(math.isfinite(value) for value in thresholds.values())
        or thresholds["minimum_utility"] < 0.0
        or not all(0.0 <= thresholds[name] <= 1.0 for name in required[1:])
        or maximum <= 0
    ):
        raise BarrierResearchError("invalid V11 selection policy")
    eligible: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            float(row["predicted_utility"]) >= thresholds["minimum_utility"]
            and float(row["direction_probability"])
            >= thresholds["minimum_direction_probability"]
            and float(row["clean_probability"])
            >= thresholds["minimum_clean_probability"]
            and float(row["unknown_probability"])
            <= thresholds["maximum_unknown_probability"]
        ):
            eligible[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in eligible.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["predicted_utility"]),
                -float(item[1]["clean_probability"]),
                -float(item[1]["direction_probability"]),
                float(item[1]["unknown_probability"]),
                str(item[1]["symbol"]),
                str(item[1]["side"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def attribution_means(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float | None]:
    names = (
        "favorable_value",
        "adverse_value",
        "cost",
        "ambiguous_penalty",
        "unresolved_penalty",
        "clean_entry_bonus",
        "base_utility",
        "total_utility",
    )
    return {
        name: (
            float(np.mean([float(row["utility_attribution"][name]) for row in rows]))
            if rows
            else None
        )
        for name in names
    }
