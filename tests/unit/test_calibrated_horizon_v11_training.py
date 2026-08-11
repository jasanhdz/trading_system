from __future__ import annotations

import pytest

from aegis.research.calibrated_horizon_v11_training import (
    multiclass_ece,
    select_v11_cross_section,
    shrink_group_probabilities,
)


def test_ece_is_zero_for_correct_certain_predictions() -> None:
    probabilities = [{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}]
    assert multiclass_ece(probabilities, ["A", "B"], bins=10) == 0.0


def test_group_calibration_is_shrunk_and_normalized() -> None:
    result = shrink_group_probabilities(
        {"A": 0.6, "B": 0.4},
        [({"A": 0.9, "B": 0.1}, 500)],
        shrinkage_rows=500,
    )
    assert sum(result.values()) == pytest.approx(1.0)
    assert 0.6 < result["A"] < 0.9


def test_v11_selection_requires_clean_probability() -> None:
    policy = {
        "minimum_utility": 0.001,
        "minimum_direction_probability": 0.6,
        "minimum_clean_probability": 0.7,
        "maximum_unknown_probability": 0.3,
        "maximum_selected_per_timestamp": 1,
    }
    base = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "side": "LONG",
        "predicted_utility": 0.01,
        "direction_probability": 0.8,
        "unknown_probability": 0.1,
    }
    rows = [
        {**base, "symbol": "BTCUSDT", "clean_probability": 0.8},
        {**base, "symbol": "ETHUSDT", "clean_probability": 0.6},
    ]
    assert select_v11_cross_section(rows, policy) == (True, False)
