"""Selection and fail-closed metrics for competing-barrier V10 research."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


def select_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    try:
        minimum_utility = float(policy["minimum_utility"])
        minimum_direction = float(policy["minimum_direction_probability"])
        maximum_unknown = float(policy["maximum_unknown_probability"])
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BarrierResearchError("incomplete V10 selection policy") from exc
    if (
        not all(math.isfinite(value) for value in (minimum_utility, minimum_direction, maximum_unknown))
        or minimum_utility < 0.0
        or not 0.0 <= minimum_direction <= 1.0
        or not 0.0 <= maximum_unknown <= 1.0
        or maximum <= 0
    ):
        raise BarrierResearchError("invalid V10 selection policy")
    eligible: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            float(row["predicted_utility"]) >= minimum_utility
            and float(row["direction_probability"]) >= minimum_direction
            and float(row["unknown_probability"]) <= maximum_unknown
        ):
            eligible[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in eligible.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["predicted_utility"]),
                -float(item[1]["direction_probability"]),
                float(item[1]["unknown_probability"]),
                str(item[1]["symbol"]),
                str(item[1]["side"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def utility_metrics(
    rows: Sequence[Mapping[str, Any]], *, tail_quantile: float = 0.10
) -> Mapping[str, Any]:
    if not 0.0 < tail_quantile < 1.0:
        raise BarrierResearchError("invalid utility tail quantile")
    if not rows:
        return {
            "count": 0,
            "mean_utility": None,
            "cvar": None,
            "positive_rate": None,
            "payoff_ratio": None,
            "p95_gap_hours": None,
        }
    values = np.asarray([float(row["actual_utility"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise BarrierResearchError("non-finite realized utility")
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    tail_count = max(1, int(math.ceil(len(values) * tail_quantile)))
    times = sorted({row["timestamp_value"] for row in rows})
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(times, times[1:])
    ]
    return {
        "count": len(values),
        "mean_utility": float(np.mean(values)),
        "cvar": float(np.mean(np.sort(values)[:tail_count])),
        "positive_rate": float(np.mean(values > 0.0)),
        "payoff_ratio": (
            float(np.mean(wins) / abs(np.mean(losses)))
            if len(wins) and len(losses)
            else None
        ),
        "p95_gap_hours": float(np.quantile(gaps, 0.95)) if gaps else 0.0,
    }


def fold_passes(
    selected: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    minimum_count: int,
    minimum_payoff: float,
    maximum_p95_gap_hours: float,
) -> bool:
    required = (
        selected["mean_utility"],
        selected["cvar"],
        selected["payoff_ratio"],
        selected["p95_gap_hours"],
        control["mean_utility"],
        control["cvar"],
    )
    return bool(
        int(selected["count"]) >= minimum_count
        and all(value is not None and math.isfinite(float(value)) for value in required)
        and float(selected["mean_utility"]) > 0.0
        and float(selected["mean_utility"]) > float(control["mean_utility"])
        and float(selected["cvar"]) > float(control["cvar"])
        and float(selected["payoff_ratio"]) >= minimum_payoff
        and float(selected["p95_gap_hours"]) <= maximum_p95_gap_hours
    )
