"""Calibration and feasibility primitives for V17 offline research."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


def split_calibration(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    timestamps = sorted({str(row["timestamp"]) for row in rows})
    if len(timestamps) < 2:
        raise BarrierResearchError("V17 calibration has insufficient timestamps")
    boundary = timestamps[len(timestamps) // 2 - 1]
    gate = [row for row in rows if str(row["timestamp"]) <= boundary]
    rank = [row for row in rows if str(row["timestamp"]) > boundary]
    if not gate or not rank:
        raise BarrierResearchError("V17 calibration split is empty")
    return gate, rank


def gate_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    clean_quantile: float,
    danger_quantile: float,
    mae_quantile: float,
) -> Mapping[str, float]:
    if not rows:
        raise BarrierResearchError("V17 cannot calibrate an empty gate")
    quantiles = (clean_quantile, danger_quantile, mae_quantile)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in quantiles):
        raise BarrierResearchError("invalid V17 gate quantile")
    clean = np.asarray([float(row["clean_probability"]) for row in rows])
    danger = np.asarray([float(row["danger_probability"]) for row in rows])
    mae = np.asarray([float(row["mae_q90"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (clean, danger, mae)):
        raise BarrierResearchError("non-finite V17 safety prediction")
    return {
        "minimum_clean_probability": float(np.quantile(clean, clean_quantile)),
        "maximum_danger_probability": float(np.quantile(danger, danger_quantile)),
        "maximum_mae_q90": float(np.quantile(mae, mae_quantile)),
    }


def gate_survivors(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> list[Mapping[str, Any]]:
    minimum_clean = float(thresholds["minimum_clean_probability"])
    maximum_danger = float(thresholds["maximum_danger_probability"])
    maximum_mae = float(thresholds["maximum_mae_q90"])
    if (
        not all(math.isfinite(value) for value in thresholds.values())
        or not 0.0 <= minimum_clean <= 1.0
        or not 0.0 <= maximum_danger <= 1.0
        or maximum_mae < 0.0
    ):
        raise BarrierResearchError("invalid V17 gate thresholds")
    survivors = []
    for row in rows:
        clean = float(row["clean_probability"])
        danger = float(row["danger_probability"])
        mae = float(row["mae_q90"])
        if not all(math.isfinite(value) for value in (clean, danger, mae)):
            raise BarrierResearchError("non-finite V17 gate input")
        if clean >= minimum_clean and danger <= maximum_danger and mae <= maximum_mae:
            survivors.append(row)
    return survivors
