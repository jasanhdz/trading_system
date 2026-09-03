"""Scoring, selection, and component metrics for V9 research."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .decomposed_entry_v9 import DecomposedEntryV9Error
from .tail_aware_entry_v8_training import tail_selection_metrics


def decomposed_quality_score(
    *,
    direction_probability: float,
    positive_probability: float,
    maximum_timing_risk: float,
    catastrophic_probability: float,
    expected_stress_net: float,
    mae_q90: float,
    mfe_q50: float,
    time_to_positive: float,
    stress_cost_fraction: float,
) -> tuple[float, float]:
    values = (
        direction_probability,
        positive_probability,
        maximum_timing_risk,
        catastrophic_probability,
        expected_stress_net,
        mae_q90,
        mfe_q50,
        time_to_positive,
        stress_cost_fraction,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not all(
            0.0 <= value <= 1.0
            for value in (
                direction_probability,
                positive_probability,
                maximum_timing_risk,
                catastrophic_probability,
                time_to_positive,
            )
        )
        or min(mae_q90, mfe_q50, stress_cost_fraction) < 0.0
    ):
        raise DecomposedEntryV9Error("invalid V9 quality inputs")
    reward_risk = mfe_q50 / max(mae_q90 + stress_cost_fraction, 1e-8)
    reward_quality = min(1.0, reward_risk / 3.0)
    net_quality = 1.0 / (
        1.0 + math.exp(-max(-0.5, min(0.5, expected_stress_net)) / 0.001)
    )
    score = math.prod(
        (
            direction_probability,
            positive_probability,
            1.0 - maximum_timing_risk,
            (1.0 - catastrophic_probability) ** 2,
            net_quality,
            reward_quality,
            max(0.0, 1.0 - time_to_positive),
        )
    )
    return float(score), float(reward_risk)


def select_decomposed_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    required = (
        "minimum_score",
        "minimum_direction_probability",
        "maximum_timing_risk",
        "maximum_mae_q90",
        "minimum_reward_risk",
        "maximum_selected_per_timestamp",
    )
    try:
        values = {name: float(policy[name]) for name in required[:-1]}
        maximum = int(policy[required[-1]])
    except (KeyError, TypeError, ValueError) as exc:
        raise DecomposedEntryV9Error("incomplete V9 selection policy") from exc
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in values.values())
        or maximum <= 0
    ):
        raise DecomposedEntryV9Error("invalid V9 selection policy")
    eligible: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if (
            float(row["v9_quality_score"]) >= values["minimum_score"]
            and float(row["direction_probability"])
            >= values["minimum_direction_probability"]
            and float(row["maximum_timing_risk"]) <= values["maximum_timing_risk"]
            and float(row["mae_q90"]) <= values["maximum_mae_q90"]
            and float(row["predicted_reward_risk"]) >= values["minimum_reward_risk"]
        ):
            eligible.setdefault(str(row["timestamp"]), []).append((index, row))
    selected = [False] * len(rows)
    for candidates in eligible.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["v9_quality_score"]),
                float(item[1]["maximum_timing_risk"]),
                float(item[1]["mae_q90"]),
                str(item[1]["symbol"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def regression_skill(
    predicted: Sequence[float], actual: Sequence[float], baseline_value: float
) -> Mapping[str, Any]:
    if len(predicted) != len(actual) or not predicted:
        raise DecomposedEntryV9Error("invalid V9 regression skill inputs")
    prediction = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(observed).all()
        or not math.isfinite(baseline_value)
    ):
        raise DecomposedEntryV9Error("non-finite V9 regression skill input")
    mse = float(np.mean((prediction - observed) ** 2))
    baseline_mse = float(np.mean((baseline_value - observed) ** 2))
    return {
        "count": len(observed),
        "mse": mse,
        "constant_baseline_mse": baseline_mse,
        "passed": mse < baseline_mse,
    }


def quantile_skill(
    predicted: Sequence[float],
    actual: Sequence[float],
    baseline_value: float,
    quantile: float,
) -> Mapping[str, Any]:
    if len(predicted) != len(actual) or not predicted or not 0.0 < quantile < 1.0:
        raise DecomposedEntryV9Error("invalid V9 quantile skill inputs")
    prediction = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if not np.isfinite(prediction).all() or not np.isfinite(observed).all():
        raise DecomposedEntryV9Error("non-finite V9 quantile skill input")

    def loss(values: np.ndarray) -> float:
        residual = observed - values
        return float(
            np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual))
        )

    model_loss = loss(prediction)
    baseline_loss = loss(np.full(len(observed), baseline_value, dtype=np.float64))
    return {
        "count": len(observed),
        "quantile": quantile,
        "pinball_loss": model_loss,
        "constant_baseline_pinball_loss": baseline_loss,
        "passed": model_loss < baseline_loss,
    }


def v9_selection_metrics(
    rows: Sequence[Mapping[str, Any]], *, tail_quantile: float
) -> Mapping[str, Any]:
    base = dict(tail_selection_metrics(rows, tail_quantile=tail_quantile))
    base["mean_direction_probability"] = (
        float(np.mean([row["direction_probability"] for row in rows])) if rows else None
    )
    base["mean_maximum_timing_risk"] = (
        float(np.mean([row["maximum_timing_risk"] for row in rows])) if rows else None
    )
    base["mean_predicted_reward_risk"] = (
        float(np.mean([row["predicted_reward_risk"] for row in rows])) if rows else None
    )
    return base


def v9_fold_passes(
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
