"""Fail-closed policy selection and metrics for V7 research."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from .regime_entry_exit_v7 import RegimeEntryExitV7Error

V7_ABLATIONS = (
    "ENTRY_ONLY",
    "MAE_AND_SPEED",
    "NO_LATE_PENALTY",
    "NO_CAPTURE_WEIGHT",
    "FULL",
)


def joint_quality_score(
    *,
    clean_probability: float,
    positive_probability: float,
    late_probability: float,
    expected_profile_net: float,
    mae_q90: float,
    time_to_positive: float,
    capture_efficiency: float,
) -> float:
    values = (
        clean_probability,
        positive_probability,
        late_probability,
        expected_profile_net,
        mae_q90,
        time_to_positive,
        capture_efficiency,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not all(
            0.0 <= value <= 1.0
            for value in (
                clean_probability,
                positive_probability,
                late_probability,
                time_to_positive,
                capture_efficiency,
            )
        )
        or mae_q90 < 0.0
    ):
        raise RegimeEntryExitV7Error("invalid V7 quality inputs")
    net_quality = 1.0 / (1.0 + math.exp(-expected_profile_net / 0.001))
    mae_quality = math.exp(-mae_q90 / 0.01)
    speed_quality = max(0.0, 1.0 - time_to_positive)
    return float(
        math.prod(
            (
                clean_probability,
                positive_probability,
                1.0 - late_probability,
                net_quality,
                mae_quality,
                speed_quality,
                max(0.05, capture_efficiency),
            )
        )
    )


def v7_ablation_score(row: Mapping[str, Any], variant: str) -> float:
    if variant not in V7_ABLATIONS:
        raise RegimeEntryExitV7Error(f"unknown V7 ablation: {variant}")
    clean = float(row["clean_probability"])
    positive = float(row["positive_probability"])
    late = float(row["late_probability"])
    expected = float(row["expected_profile_net"])
    mae = float(row["mae_q90"])
    timing = float(row["predicted_time_to_positive"])
    capture = float(row["predicted_capture_efficiency"])
    if variant == "ENTRY_ONLY":
        return clean * positive
    if variant == "MAE_AND_SPEED":
        return math.exp(-mae / 0.01) * max(0.0, 1.0 - timing)
    if variant == "NO_LATE_PENALTY":
        late = 0.0
    if variant == "NO_CAPTURE_WEIGHT":
        capture = 1.0
    return joint_quality_score(
        clean_probability=clean,
        positive_probability=positive,
        late_probability=late,
        expected_profile_net=expected,
        mae_q90=mae,
        time_to_positive=timing,
        capture_efficiency=capture,
    )


def select_v7_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    try:
        minimum_score = float(policy["minimum_score"])
        maximum_late = float(policy["maximum_late_probability"])
        maximum_mae = float(policy["maximum_mae_q90"])
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeEntryExitV7Error("incomplete V7 policy") from exc
    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in (minimum_score, maximum_late, maximum_mae)
        )
        or maximum <= 0
    ):
        raise RegimeEntryExitV7Error("invalid V7 policy")
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        score = float(row["v7_quality_score"])
        late = float(row["late_probability"])
        mae = float(row["mae_q90"])
        if not all(math.isfinite(value) for value in (score, late, mae)):
            raise RegimeEntryExitV7Error("non-finite V7 selection input")
        if score >= minimum_score and late <= maximum_late and mae <= maximum_mae:
            groups[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in groups.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["v7_quality_score"]),
                float(item[1]["late_probability"]),
                float(item[1]["mae_q90"]),
                str(item[1]["symbol"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def v7_selection_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_net": None,
            "positive_rate": None,
            "mean_mae": None,
            "mae_q90": None,
            "mean_capture_efficiency": None,
            "mean_time_underwater_bars": None,
            "clean_entry_rate": None,
            "late_entry_rate": None,
            "p95_gap_hours": None,
            "maximum_gap_hours": None,
            "profile_counts": {},
            "archetype_counts": {},
        }
    net = np.asarray([float(row["selected_profile_net"]) for row in rows])
    mae = np.asarray([float(row["mae_fraction"]) for row in rows])
    capture = np.asarray([float(row["selected_capture_efficiency"]) for row in rows])
    timestamps = sorted({datetime.fromisoformat(str(row["timestamp"])) for row in rows})
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "count": len(rows),
        "mean_net": float(np.mean(net)),
        "positive_rate": float(np.mean(net > 0.0)),
        "mean_mae": float(np.mean(mae)),
        "mae_q90": float(np.quantile(mae, 0.90)),
        "mean_capture_efficiency": float(np.mean(capture)),
        "mean_time_underwater_bars": float(
            np.mean([float(row["time_underwater_bars"]) for row in rows])
        ),
        "clean_entry_rate": float(
            np.mean(
                [bool(row["trajectory_attribution"]["clean_entry"]) for row in rows]
            )
        ),
        "late_entry_rate": float(
            np.mean([bool(row["trajectory_attribution"]["late_entry"]) for row in rows])
        ),
        "p95_gap_hours": float(np.quantile(gaps, 0.95)) if gaps else None,
        "maximum_gap_hours": max(gaps) if gaps else None,
        "profile_counts": dict(
            sorted(Counter(str(row["selected_profile"]) for row in rows).items())
        ),
        "archetype_counts": dict(
            sorted(Counter(str(row["v7_archetype"]) for row in rows).items())
        ),
    }


def fold_passes(
    metrics: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    minimum_count: int,
    maximum_p95_gap_hours: float,
) -> bool:
    if int(metrics["count"]) < minimum_count:
        return False
    values = (
        metrics["mean_net"],
        metrics["mean_mae"],
        metrics["mean_capture_efficiency"],
        control["mean_net"],
        control["mean_mae"],
        control["mean_capture_efficiency"],
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return False
    gap = metrics["p95_gap_hours"]
    return bool(
        float(metrics["mean_net"]) > 0.0
        and float(metrics["mean_net"]) > float(control["mean_net"])
        and float(metrics["mean_mae"]) <= float(control["mean_mae"])
        and float(metrics["mean_capture_efficiency"])
        >= float(control["mean_capture_efficiency"])
        and gap is not None
        and float(gap) <= maximum_p95_gap_hours
    )
