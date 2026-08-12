from __future__ import annotations

import yaml
from datetime import datetime

from aegis.research.directional_contract_v15 import (
    contract_indices,
    entry_quality_score,
    select_at_most_one_per_timestamp,
)
from train_directional_contract_v15_research import _partition


def _config():
    with open(
        "config/experiments/aegis_directional_contract_v15_research.yaml"
    ) as handle:
        return yaml.safe_load(handle)


def test_v15_contracts_are_frozen_and_direction_specific() -> None:
    config = _config()
    baseline = contract_indices(config, "BASELINE")
    long = contract_indices(config, "LONG")
    short = contract_indices(config, "SHORT")
    assert len(baseline) == 176
    assert len(short) == 168
    assert len(long) < len(short)
    assert set(long) < set(short) < set(baseline)


def test_v15_score_rewards_clean_and_penalizes_danger_and_mae() -> None:
    strong = entry_quality_score(
        clean_probability=0.8, danger_probability=0.1, mae_q90=0.001, adverse=0.01
    )
    weak = entry_quality_score(
        clean_probability=0.4, danger_probability=0.3, mae_q90=0.004, adverse=0.01
    )
    assert strong > weak


def test_v15_selection_keeps_best_symbol_per_timestamp() -> None:
    rows = [
        {
            "timestamp": "t",
            "symbol": "A",
            "score": 0.2,
            "danger_probability": 0.1,
            "mae_q90": 0.01,
        },
        {
            "timestamp": "t",
            "symbol": "B",
            "score": 0.3,
            "danger_probability": 0.2,
            "mae_q90": 0.01,
        },
        {
            "timestamp": "u",
            "symbol": "C",
            "score": 0.1,
            "danger_probability": 0.1,
            "mae_q90": 0.01,
        },
    ]
    assert select_at_most_one_per_timestamp(rows, minimum_score=0.15) == (
        False,
        True,
        False,
    )


def test_v15_partition_enforces_embargo_and_independent_scoring() -> None:
    def row(timestamp: str, independent: bool = True):
        return {
            "timestamp_value": datetime.fromisoformat(timestamp),
            "independent": independent,
        }

    rows = [
        row("2026-01-01T00:00:00+00:00"),
        row("2026-01-01T02:00:00+00:00"),
        row("2026-01-01T04:01:00+00:00", False),
        row("2026-01-01T04:02:00+00:00"),
        row("2026-01-01T08:01:00+00:00"),
    ]
    fold = {
        "train_end": "2026-01-01T02:00:00+00:00",
        "calibration_end": "2026-01-01T06:00:00+00:00",
        "test_end": "2026-01-01T10:00:00+00:00",
    }
    train, calibration, test = _partition(rows, fold, 120)
    assert len(train) == 2
    assert calibration == [rows[3]]
    assert test == [rows[4]]
