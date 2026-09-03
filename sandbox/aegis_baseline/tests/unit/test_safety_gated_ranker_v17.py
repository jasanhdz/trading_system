from __future__ import annotations

import pytest

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.safety_gated_ranker_v17 import (
    gate_survivors,
    gate_thresholds,
    split_calibration,
)


def _row(
    timestamp: str,
    *,
    clean: float,
    danger: float,
    mae: float,
):
    return {
        "timestamp": timestamp,
        "clean_probability": clean,
        "danger_probability": danger,
        "mae_q90": mae,
    }


def test_v17_calibration_split_never_splits_a_timestamp() -> None:
    rows = [
        _row("2026-01-01T00:00:00+00:00", clean=0.1, danger=0.9, mae=0.03),
        _row("2026-01-01T00:00:00+00:00", clean=0.2, danger=0.8, mae=0.02),
        _row("2026-01-01T01:00:00+00:00", clean=0.8, danger=0.2, mae=0.01),
        _row("2026-01-01T02:00:00+00:00", clean=0.9, danger=0.1, mae=0.005),
    ]
    gate, rank = split_calibration(rows)
    assert {row["timestamp"] for row in gate}.isdisjoint(
        {row["timestamp"] for row in rank}
    )
    assert max(row["timestamp"] for row in gate) < min(row["timestamp"] for row in rank)


def test_v17_gate_requires_every_safety_condition() -> None:
    thresholds = {
        "minimum_clean_probability": 0.7,
        "maximum_danger_probability": 0.3,
        "maximum_mae_q90": 0.01,
    }
    safe = _row("1", clean=0.8, danger=0.2, mae=0.005)
    low_clean = _row("2", clean=0.6, danger=0.2, mae=0.005)
    high_danger = _row("3", clean=0.8, danger=0.4, mae=0.005)
    high_mae = _row("4", clean=0.8, danger=0.2, mae=0.02)
    assert gate_survivors([safe, low_clean, high_danger, high_mae], thresholds) == [
        safe
    ]


def test_v17_thresholds_are_derived_only_from_supplied_rows() -> None:
    calibration = [
        _row("1", clean=0.2, danger=0.8, mae=0.03),
        _row("2", clean=0.8, danger=0.2, mae=0.01),
    ]
    test_only_extreme = _row("3", clean=1.0, danger=0.0, mae=0.0)
    before = gate_thresholds(
        calibration, clean_quantile=0.5, danger_quantile=0.5, mae_quantile=0.5
    )
    after = gate_thresholds(
        calibration, clean_quantile=0.5, danger_quantile=0.5, mae_quantile=0.5
    )
    assert before == after
    assert test_only_extreme not in calibration


def test_v17_rejects_invalid_thresholds() -> None:
    row = _row("1", clean=0.5, danger=0.5, mae=0.01)
    with pytest.raises(BarrierResearchError):
        gate_survivors(
            [row],
            {
                "minimum_clean_probability": 0.5,
                "maximum_danger_probability": 0.5,
                "maximum_mae_q90": -0.01,
            },
        )
