"""Frozen selection and economic evaluation primitives for V18 research."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


V18_PREDICTION_FIELDS = (
    "clean_probability",
    "danger_probability",
    "mae_q90",
    "expected_utility",
)


def select_candidates(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    thresholds = {
        "clean_probability": float(policy["minimum_clean_probability"]),
        "danger_probability": float(policy["maximum_danger_probability"]),
        "mae_q90": float(policy["maximum_mae_fraction"]),
        "expected_utility": float(policy["minimum_expected_utility"]),
    }
    if (
        not all(math.isfinite(value) for value in thresholds.values())
        or not 0.0 <= thresholds["clean_probability"] <= 1.0
        or not 0.0 <= thresholds["danger_probability"] <= 1.0
        or thresholds["mae_q90"] < 0.0
    ):
        raise BarrierResearchError("invalid V18 frozen policy")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values = {name: float(row[name]) for name in V18_PREDICTION_FIELDS}
        if not all(math.isfinite(value) for value in values.values()):
            raise BarrierResearchError("non-finite V18 prediction")
        if (
            values["clean_probability"] >= thresholds["clean_probability"]
            and values["danger_probability"] <= thresholds["danger_probability"]
            and values["mae_q90"] <= thresholds["mae_q90"]
            and values["expected_utility"] > thresholds["expected_utility"]
        ):
            grouped[str(row["timestamp"])].append(row)
    selected = []
    for timestamp in sorted(grouped):
        selected.append(
            min(
                grouped[timestamp],
                key=lambda row: (
                    -float(row["expected_utility"]),
                    float(row["danger_probability"]),
                    float(row["mae_q90"]),
                    str(row["symbol"]),
                ),
            )
        )
    return selected


def economic_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "net_expectancy": None,
            "total_pnl_fraction": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "mean_mae_fraction": None,
            "mean_mfe_fraction": None,
            "maximum_drawdown_fraction": None,
            "cvar_05": None,
        }
    ordered = sorted(rows, key=lambda row: (str(row["timestamp"]), str(row["symbol"])))
    utility = np.asarray([float(row["actual_utility"]) for row in ordered], dtype=np.float64)
    mae = np.asarray([float(row["mae_fraction"]) for row in ordered], dtype=np.float64)
    mfe = np.asarray(
        [float(row.get("mfe_fraction", 0.0)) for row in ordered], dtype=np.float64
    )
    if not all(np.isfinite(values).all() for values in (utility, mae, mfe)):
        raise BarrierResearchError("non-finite V18 economic evidence")
    gains = utility[utility > 0.0].sum()
    losses = -utility[utility < 0.0].sum()
    equity = np.cumsum(utility)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdown = running_peak[1:] - equity
    tail_count = max(1, int(math.ceil(len(utility) * 0.05)))
    return {
        "count": len(ordered),
        "net_expectancy": float(utility.mean()),
        "total_pnl_fraction": float(utility.sum()),
        "profit_factor": float(gains / losses) if losses > 0.0 else None,
        "win_rate": float(np.mean(utility > 0.0)),
        "mean_mae_fraction": float(mae.mean()),
        "mean_mfe_fraction": float(mfe.mean()),
        "maximum_drawdown_fraction": float(drawdown.max(initial=0.0)),
        "cvar_05": float(np.sort(utility)[:tail_count].mean()),
    }


def moving_block_intervals(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int, block_size: int = 12
) -> Mapping[str, Any]:
    if not rows:
        return {"net_expectancy_95": None, "profit_factor_95": None}
    ordered = sorted(rows, key=lambda row: (str(row["timestamp"]), str(row["symbol"])))
    count = len(ordered)
    size = min(block_size, count)
    starts = np.arange(max(1, count - size + 1))
    rng = np.random.default_rng(seed)
    means = []
    factors = []
    for _ in range(resamples):
        sample: list[Mapping[str, Any]] = []
        while len(sample) < count:
            start = int(rng.choice(starts))
            sample.extend(ordered[start : start + size])
        metrics = economic_metrics(sample[:count])
        means.append(float(metrics["net_expectancy"]))
        factor = metrics["profit_factor"]
        if factor is not None and math.isfinite(float(factor)):
            factors.append(float(factor))
    quantiles = lambda values: [float(value) for value in np.quantile(values, [0.025, 0.975])]
    return {
        "net_expectancy_95": quantiles(means),
        "profit_factor_95": quantiles(factors) if factors else None,
        "resamples": resamples,
        "block_size": size,
    }


def temporal_thirds(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    timestamps = sorted({str(row["timestamp"]) for row in rows})
    if not timestamps:
        return []
    chunks = np.array_split(np.asarray(timestamps, dtype=object), 3)
    reports = []
    for index, chunk in enumerate(chunks, start=1):
        allowed = set(str(value) for value in chunk.tolist())
        metrics = economic_metrics([row for row in rows if str(row["timestamp"]) in allowed])
        reports.append({"third": index, "metrics": metrics})
    return reports


def offline_side_gate(
    metrics: Mapping[str, Any],
    intervals: Mapping[str, Any],
    thirds: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    *,
    random_expectancy: float | None,
) -> Mapping[str, Any]:
    mean_interval = intervals.get("net_expectancy_95")
    factor_interval = intervals.get("profit_factor_95")
    positive_thirds = sum(
        report["metrics"]["net_expectancy"] is not None
        and float(report["metrics"]["net_expectancy"]) > 0.0
        for report in thirds
    )
    checks = {
        "minimum_selected": int(metrics["count"]) >= int(gate["minimum_selected_per_direction"]),
        "expectancy_ci": mean_interval is not None
        and float(mean_interval[0]) > float(gate["require_mean_net_expectancy_ci_lower_gt"]),
        "profit_factor_ci": factor_interval is not None
        and float(factor_interval[0]) > float(gate["require_profit_factor_ci_lower_gt"]),
        "cvar": metrics["cvar_05"] is not None
        and float(metrics["cvar_05"]) > float(gate["require_cvar_05_gt"]),
        "mae": metrics["mean_mae_fraction"] is not None
        and float(metrics["mean_mae_fraction"]) <= float(gate["require_mean_mae_lte"]),
        "temporal_stability": positive_thirds >= int(gate["require_positive_validation_thirds"]),
        "better_than_random": random_expectancy is not None
        and metrics["net_expectancy"] is not None
        and float(metrics["net_expectancy"]) > float(random_expectancy),
    }
    return {"passed": all(checks.values()), "checks": checks, "positive_thirds": positive_thirds}
