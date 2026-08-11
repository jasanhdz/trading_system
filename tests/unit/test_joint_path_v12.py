from __future__ import annotations

import pytest

from aegis.research.competing_barrier_v10 import BarrierContract
from aegis.research.joint_path_v12 import (
    JointPathState,
    joint_path_state,
    joint_state_utility,
    path_quality_metrics,
    select_joint_cross_section,
)


def test_joint_state_requires_direction_and_clean_path() -> None:
    assert (
        joint_path_state(
            side="LONG",
            direction_label="LONG",
            clean_entry=True,
            outcome="FAVORABLE_FIRST",
        )
        == JointPathState.COHERENT_CLEAN_FAVORABLE.value
    )
    assert (
        joint_path_state(
            side="LONG",
            direction_label="SHORT",
            clean_entry=True,
            outcome="FAVORABLE_FIRST",
        )
        == JointPathState.UNRESOLVED_OR_DIRECTION_MISMATCH.value
    )


def test_joint_utility_does_not_treat_dirty_path_as_clean() -> None:
    contract = BarrierContract("TEST", 0.01, 0.01, 12, 0.002)
    config = {
        "clean_favorable_discount": 1.0,
        "dirty_favorable_discount": 0.5,
        "ambiguous_penalty_fraction_of_adverse": 1.0,
        "unresolved_penalty_fraction_of_adverse": 0.25,
    }
    clean = joint_state_utility(
        {
            "COHERENT_CLEAN_FAVORABLE": 1.0,
            "COHERENT_DIRTY_FAVORABLE": 0.0,
            "ADVERSE_FIRST": 0.0,
            "SAME_BAR_AMBIGUOUS": 0.0,
            "UNRESOLVED_OR_DIRECTION_MISMATCH": 0.0,
        },
        contract,
        config,
    )
    dirty = joint_state_utility(
        {
            "COHERENT_CLEAN_FAVORABLE": 0.0,
            "COHERENT_DIRTY_FAVORABLE": 1.0,
            "ADVERSE_FIRST": 0.0,
            "SAME_BAR_AMBIGUOUS": 0.0,
            "UNRESOLVED_OR_DIRECTION_MISMATCH": 0.0,
        },
        contract,
        config,
    )
    assert clean["total_utility"] == pytest.approx(0.008)
    assert dirty["total_utility"] == pytest.approx(0.003)


def test_selection_ranks_joint_candidates_without_forcing_trade() -> None:
    policy = {
        "minimum_utility": 0.001,
        "minimum_coherent_probability": 0.6,
        "maximum_adverse_probability": 0.3,
        "maximum_unknown_probability": 0.3,
        "maximum_selected_per_timestamp": 1,
    }
    base = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "side": "LONG",
        "coherent_probability": 0.7,
        "adverse_probability": 0.2,
        "unknown_probability": 0.1,
    }
    rows = [
        {**base, "symbol": "BTCUSDT", "predicted_utility": 0.002},
        {**base, "symbol": "ETHUSDT", "predicted_utility": 0.003},
        {**base, "symbol": "SOLUSDT", "predicted_utility": -0.001},
    ]
    assert select_joint_cross_section(rows, policy) == (False, True, False)


def test_path_metrics_report_mae_and_time_to_positive() -> None:
    rows = [
        {
            "v11_clean_entry_label": True,
            "actual_outcome": "FAVORABLE_FIRST",
            "mae_fraction": 0.001,
            "v11_path_diagnostics": {
                "first_positive_after_severe_cost_bar": 2,
                "maximum_favorable_excursion_fraction": 0.01,
            },
        },
        {
            "v11_clean_entry_label": False,
            "actual_outcome": "ADVERSE_FIRST",
            "mae_fraction": 0.005,
            "v11_path_diagnostics": {
                "first_positive_after_severe_cost_bar": None,
                "maximum_favorable_excursion_fraction": 0.002,
            },
        },
    ]
    result = path_quality_metrics(rows)
    assert result["clean_rate"] == 0.5
    assert result["adverse_first_rate"] == 0.5
    assert result["mean_mae_fraction"] == pytest.approx(0.003)
