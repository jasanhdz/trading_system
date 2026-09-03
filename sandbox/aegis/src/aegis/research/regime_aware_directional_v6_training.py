"""Leakage-safe selection and evaluation helpers for directional v6."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from .regime_aware_directional_v6 import RegimeAwareV6Error


def quality_score(
    *,
    protectable_probability: float,
    target_probability: float,
    early_reversal_probability: float,
    expected_protected_net: float,
    mae_q90: float,
    time_to_advantage: float,
) -> float:
    """Combine complementary questions without treating them as fake votes."""

    components = quality_score_components(
        protectable_probability=protectable_probability,
        target_probability=target_probability,
        early_reversal_probability=early_reversal_probability,
        expected_protected_net=expected_protected_net,
        mae_q90=mae_q90,
        time_to_advantage=time_to_advantage,
    )
    return float(math.prod(components.values()))


def quality_score_components(
    *,
    protectable_probability: float,
    target_probability: float,
    early_reversal_probability: float,
    expected_protected_net: float,
    mae_q90: float,
    time_to_advantage: float,
) -> Mapping[str, float]:
    """Return normalized specialist evidence without manufacturing votes."""

    values = (
        protectable_probability,
        target_probability,
        early_reversal_probability,
        expected_protected_net,
        mae_q90,
        time_to_advantage,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not all(
            0.0 <= value <= 1.0
            for value in (
                protectable_probability,
                target_probability,
                early_reversal_probability,
                time_to_advantage,
            )
        )
        or mae_q90 < 0.0
    ):
        raise RegimeAwareV6Error("v6 quality score inputs are invalid")
    net_quality = 1.0 / (1.0 + math.exp(-expected_protected_net / 0.001))
    mae_quality = math.exp(-mae_q90 / 0.01)
    speed_quality = max(0.0, 1.0 - time_to_advantage)
    return {
        "PROTECTABLE": protectable_probability,
        "TARGET": target_probability,
        "REVERSAL": 1.0 - early_reversal_probability,
        "PROTECTED_NET": net_quality,
        "MAE": mae_quality,
        "SPEED": speed_quality,
    }


ABLATION_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "PROTECTABLE_ONLY": ("PROTECTABLE",),
    "TARGET_ONLY": ("TARGET",),
    "REVERSAL_ONLY": ("REVERSAL",),
    "PROTECTED_NET_ONLY": ("PROTECTED_NET",),
    "MAE_ONLY": ("MAE",),
    "SPEED_ONLY": ("SPEED",),
    "PROTECTABLE_AND_NET": ("PROTECTABLE", "PROTECTED_NET"),
    "MAE_AND_SPEED": ("MAE", "SPEED"),
    "FULL_COMMITTEE": (
        "PROTECTABLE",
        "TARGET",
        "REVERSAL",
        "PROTECTED_NET",
        "MAE",
        "SPEED",
    ),
}


def ablation_score(row: Mapping[str, Any], variant: str) -> float:
    """Score a fixed specialist subset for diagnostic comparison only."""

    try:
        selected = ABLATION_COMPONENTS[variant]
    except KeyError as exc:
        raise RegimeAwareV6Error(f"unknown v6 ablation: {variant}") from exc
    components = quality_score_components(
        protectable_probability=float(row["protectable_probability"]),
        target_probability=float(row["target_probability"]),
        early_reversal_probability=float(row["early_reversal_probability"]),
        expected_protected_net=float(row["expected_protected_net"]),
        mae_q90=float(row["mae_q90"]),
        time_to_advantage=float(row["time_to_advantage"]),
    )
    return float(math.prod(components[name] for name in selected))


def select_ablation_cross_section(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    minimum_score: float,
    maximum_selected_per_timestamp: int,
) -> tuple[bool, ...]:
    """Apply a fixed calibration threshold without peeking at test outcomes."""

    if (
        not math.isfinite(minimum_score)
        or minimum_score < 0.0
        or maximum_selected_per_timestamp <= 0
    ):
        raise RegimeAwareV6Error("v6 ablation policy is invalid")
    groups: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        score = ablation_score(row, variant)
        if score >= minimum_score:
            groups[str(row["timestamp"])].append((index, score, str(row["symbol"])))
    selected = [False] * len(rows)
    for candidates in groups.values():
        ordered = sorted(candidates, key=lambda item: (-item[1], item[2]))
        for index, _, _ in ordered[:maximum_selected_per_timestamp]:
            selected[index] = True
    return tuple(selected)


def select_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    required = (
        "minimum_score",
        "maximum_mae_q90",
        "maximum_time_to_advantage",
        "maximum_early_reversal_probability",
        "maximum_selected_per_timestamp",
    )
    try:
        minimum_score = float(policy[required[0]])
        maximum_mae = float(policy[required[1]])
        maximum_time = float(policy[required[2]])
        maximum_reversal = float(policy[required[3]])
        maximum_selected = int(policy[required[4]])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeAwareV6Error("v6 policy is incomplete") from exc
    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in (minimum_score, maximum_mae, maximum_time, maximum_reversal)
        )
        or maximum_selected <= 0
    ):
        raise RegimeAwareV6Error("v6 policy is invalid")
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        score = float(row["quality_score"])
        mae = float(row["mae_q90"])
        timing = float(row["time_to_advantage"])
        reversal = float(row["early_reversal_probability"])
        if not all(math.isfinite(value) for value in (score, mae, timing, reversal)):
            raise RegimeAwareV6Error("v6 selection input is non-finite")
        if (
            score >= minimum_score
            and mae <= maximum_mae
            and timing <= maximum_time
            and reversal <= maximum_reversal
        ):
            groups[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in groups.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["quality_score"]),
                float(item[1]["mae_q90"]),
                str(item[1]["symbol"]),
            ),
        )
        for index, _ in ordered[:maximum_selected]:
            selected[index] = True
    return tuple(selected)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def selection_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_protected_net": None,
            "positive_net_rate": None,
            "mean_mae": None,
            "mae_q90": None,
            "mean_underwater_bars": None,
            "protectable_rate": None,
            "target_before_stop_rate": None,
            "early_reversal_rate": None,
            "mean_bars_held": None,
            "p95_gap_hours": None,
            "maximum_gap_hours": None,
        }
    net = [float(row["full_lifecycle_worst_net_return"]) for row in rows]
    mae = [float(row["mae_fraction"]) for row in rows]
    timestamps = sorted({datetime.fromisoformat(str(row["timestamp"])) for row in rows})
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "count": len(rows),
        "mean_protected_net": float(np.mean(net)),
        "positive_net_rate": float(np.mean([value > 0.0 for value in net])),
        "mean_mae": float(np.mean(mae)),
        "mae_q90": _quantile(mae, 0.90),
        "mean_underwater_bars": float(
            np.mean([float(row["time_underwater_bars"]) for row in rows])
        ),
        "protectable_rate": float(
            np.mean([bool(row["protectable_advantage"]) for row in rows])
        ),
        "target_before_stop_rate": float(
            np.mean([bool(row["target_before_stop"]) for row in rows])
        ),
        "early_reversal_rate": float(
            np.mean([bool(row["early_reversal"]) for row in rows])
        ),
        "mean_bars_held": float(
            np.mean([float(row["full_lifecycle_worst_bars_held"]) for row in rows])
        ),
        "p95_gap_hours": _quantile(gaps, 0.95),
        "maximum_gap_hours": max(gaps) if gaps else None,
    }


def reliability_table(
    probabilities: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> tuple[Mapping[str, Any], ...]:
    if len(probabilities) != len(labels) or bins <= 1:
        raise RegimeAwareV6Error("reliability inputs are invalid")
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RegimeAwareV6Error("reliability probability is invalid")
        groups[min(bins - 1, int(value * bins))].append((value, bool(label)))
    return tuple(
        {
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(group),
            "mean_probability": (
                float(np.mean([value for value, _ in group])) if group else None
            ),
            "observed_rate": (
                float(np.mean([label for _, label in group])) if group else None
            ),
        }
        for index, group in enumerate(groups)
    )


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> Mapping[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "lower_95": None, "upper_95": None}
    if samples < 100:
        raise RegimeAwareV6Error("bootstrap sample count is too small")
    source = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source).all():
        raise RegimeAwareV6Error("bootstrap values are non-finite")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(np.mean(rng.choice(source, size=len(source), replace=True)))
            for _ in range(samples)
        ]
    )
    return {
        "samples": samples,
        "mean": float(np.mean(source)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }
