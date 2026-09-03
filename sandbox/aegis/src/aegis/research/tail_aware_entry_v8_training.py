"""Tail-aware scoring, diagnostics and fail-closed metrics for V8."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from .tail_aware_entry_v8 import TailAwareV8Error


def tail_aware_quality_score(
    *,
    clean_probability: float,
    positive_probability: float,
    late_probability: float,
    catastrophic_probability: float,
    expected_stress_net: float,
    mae_q90: float,
    time_to_positive: float,
) -> float:
    values = (
        clean_probability,
        positive_probability,
        late_probability,
        catastrophic_probability,
        expected_stress_net,
        mae_q90,
        time_to_positive,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not all(
            0.0 <= value <= 1.0
            for value in (
                clean_probability,
                positive_probability,
                late_probability,
                catastrophic_probability,
                time_to_positive,
            )
        )
        or mae_q90 < 0.0
    ):
        raise TailAwareV8Error("invalid V8 quality inputs")
    net_quality = 1.0 / (1.0 + math.exp(-expected_stress_net / 0.001))
    mae_quality = math.exp(-mae_q90 / 0.01)
    speed = max(0.0, 1.0 - time_to_positive)
    return float(
        math.prod(
            (
                clean_probability,
                positive_probability,
                1.0 - late_probability,
                (1.0 - catastrophic_probability) ** 2,
                net_quality,
                mae_quality,
                speed,
            )
        )
    )


def select_tail_aware_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    try:
        minimum_score = float(policy["minimum_score"])
        maximum_late = float(policy["maximum_late_probability"])
        maximum_catastrophic = float(policy["maximum_catastrophic_probability"])
        maximum_mae = float(policy["maximum_mae_q90"])
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TailAwareV8Error("incomplete V8 policy") from exc
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        values = (
            float(row["v8_quality_score"]),
            float(row["late_probability"]),
            float(row["catastrophic_probability"]),
            float(row["mae_q90"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise TailAwareV8Error("non-finite V8 selection input")
        score, late, catastrophic, mae = values
        if (
            score >= minimum_score
            and late <= maximum_late
            and catastrophic <= maximum_catastrophic
            and mae <= maximum_mae
        ):
            groups[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in groups.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["v8_quality_score"]),
                float(item[1]["catastrophic_probability"]),
                float(item[1]["mae_q90"]),
                str(item[1]["symbol"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def _cvar(values: np.ndarray, quantile: float) -> float:
    count = max(1, int(math.ceil(len(values) * quantile)))
    return float(np.mean(np.sort(values)[:count]))


def tail_selection_metrics(
    rows: Sequence[Mapping[str, Any]], *, tail_quantile: float
) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_expected_net": None,
            "mean_stress_net": None,
            "mean_severe_net": None,
            "stress_positive_rate": None,
            "stress_cvar": None,
            "payoff_ratio": None,
            "mean_mae": None,
            "mae_q90": None,
            "mean_underwater_bars": None,
            "p95_gap_hours": None,
            "maximum_gap_hours": None,
            "profile_counts": {},
        }
    if not 0.0 < tail_quantile < 0.5:
        raise TailAwareV8Error("invalid V8 tail quantile")
    expected = np.asarray([float(row["selected_expected_net"]) for row in rows])
    stress = np.asarray([float(row["selected_stress_net"]) for row in rows])
    severe = np.asarray([float(row["selected_severe_net"]) for row in rows])
    mae = np.asarray([float(row["mae_fraction"]) for row in rows])
    wins = stress[stress > 0.0]
    losses = stress[stress < 0.0]
    payoff = (
        float(np.mean(wins) / abs(np.mean(losses)))
        if len(wins) and len(losses)
        else None
    )
    timestamps = sorted({datetime.fromisoformat(str(row["timestamp"])) for row in rows})
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "count": len(rows),
        "mean_expected_net": float(np.mean(expected)),
        "mean_stress_net": float(np.mean(stress)),
        "mean_severe_net": float(np.mean(severe)),
        "stress_positive_rate": float(np.mean(stress > 0.0)),
        "stress_cvar": _cvar(stress, tail_quantile),
        "payoff_ratio": payoff,
        "mean_mae": float(np.mean(mae)),
        "mae_q90": float(np.quantile(mae, 0.90)),
        "mean_underwater_bars": float(
            np.mean([float(row["time_underwater_bars"]) for row in rows])
        ),
        "p95_gap_hours": float(np.quantile(gaps, 0.95)) if gaps else None,
        "maximum_gap_hours": max(gaps) if gaps else None,
        "profile_counts": dict(
            sorted(Counter(str(row["selected_profile"]) for row in rows).items())
        ),
    }


def binary_skill_metrics(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> Mapping[str, Any]:
    if len(probabilities) != len(labels) or not probabilities:
        raise TailAwareV8Error("invalid V8 binary skill inputs")
    predicted = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.int8)
    if not np.isfinite(predicted).all() or np.any(
        (predicted < 0.0) | (predicted > 1.0)
    ):
        raise TailAwareV8Error("invalid V8 binary probabilities")
    prevalence = float(np.mean(actual))
    brier = float(np.mean((predicted - actual) ** 2))
    baseline_brier = prevalence * (1.0 - prevalence)
    average_precision = float(average_precision_score(actual, predicted))
    return {
        "count": len(actual),
        "prevalence": prevalence,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "average_precision": average_precision,
        "passed": brier < baseline_brier and average_precision > prevalence,
    }


def fold_passes(
    metrics: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    minimum_count: int,
    minimum_payoff: float,
    maximum_p95_gap_hours: float,
) -> bool:
    if int(metrics["count"]) < minimum_count:
        return False
    required = (
        metrics["mean_expected_net"],
        metrics["mean_stress_net"],
        metrics["stress_cvar"],
        metrics["payoff_ratio"],
        metrics["mean_mae"],
        control["mean_stress_net"],
        control["stress_cvar"],
        control["mean_mae"],
    )
    if any(value is None or not math.isfinite(float(value)) for value in required):
        return False
    gap = metrics["p95_gap_hours"]
    return bool(
        float(metrics["mean_expected_net"]) > 0.0
        and float(metrics["mean_stress_net"]) > 0.0
        and float(metrics["mean_stress_net"]) > float(control["mean_stress_net"])
        and float(metrics["stress_cvar"]) > float(control["stress_cvar"])
        and float(metrics["mean_mae"]) <= float(control["mean_mae"])
        and float(metrics["payoff_ratio"]) >= minimum_payoff
        and gap is not None
        and float(gap) <= maximum_p95_gap_hours
    )
