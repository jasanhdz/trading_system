from __future__ import annotations

from dataclasses import dataclass

import pytest

from aegis.research.calibrated_horizon_v11 import (
    attributed_utility,
    causal_regime,
    clean_entry_diagnostics,
)
from aegis.research.competing_barrier_v10 import BarrierContract


@dataclass(frozen=True)
class Bar:
    high: float
    low: float
    close: float


REGIME = {
    "range_absolute_4h_return": 0.003,
    "range_absolute_12h_return": 0.006,
    "high_volatility_threshold": 0.003,
    "identities": [
        "RANGE_LOW_VOL",
        "RANGE_HIGH_VOL",
        "TREND_UP_LOW_VOL",
        "TREND_UP_HIGH_VOL",
        "TREND_DOWN_LOW_VOL",
        "TREND_DOWN_HIGH_VOL",
        "TRANSITION_LOW_VOL",
        "TRANSITION_HIGH_VOL",
    ],
}


def test_regime_uses_only_causal_context() -> None:
    context = {
        "rolling_4h_return": 0.01,
        "rolling_12h_return": 0.02,
        "rolling_4h_volatility": 0.004,
    }
    assert causal_regime(context, REGIME) == "TREND_UP_HIGH_VOL"
    context["rolling_4h_return"] = 0.001
    context["rolling_12h_return"] = 0.002
    assert causal_regime(context, REGIME) == "RANGE_HIGH_VOL"


def test_clean_entry_requires_fast_favorable_path_with_small_mae() -> None:
    outcome = {
        "outcome": "FAVORABLE_FIRST",
        "event_bar": 2,
        "horizon_bars": 4,
        "adverse_fraction": 0.01,
        "terminal_side_return": 0.012,
    }
    result = clean_entry_diagnostics(
        side="LONG",
        entry_price=100.0,
        future_bars=[
            Bar(100.7, 99.8, 100.5),
            Bar(101.2, 100.1, 101.0),
            Bar(101.3, 100.7, 101.2),
            Bar(101.4, 101.0, 101.2),
        ],
        primary_outcome=outcome,
        maximum_mae_fraction_of_barrier=0.5,
        maximum_event_bar=2,
        severe_cost_fraction=0.002,
    )
    assert result["clean_entry"] is True
    assert result["pre_event_mae_fraction"] == pytest.approx(0.002)
    assert result["first_positive_after_severe_cost_bar"] == 1


def test_clean_bonus_cannot_rescue_negative_base_utility() -> None:
    contract = BarrierContract("TEST", 0.01, 0.01, 6, 0.002)
    result = attributed_utility(
        {
            "FAVORABLE_FIRST": 0.4,
            "ADVERSE_FIRST": 0.4,
            "SAME_BAR_AMBIGUOUS": 0.1,
            "NEITHER_REACHED": 0.1,
        },
        contract,
        clean_probability=1.0,
        unknown_penalty_fraction=0.25,
        clean_bonus_fraction=0.10,
    )
    assert result["base_utility"] < 0.0
    assert result["clean_entry_bonus"] == 0.0
    assert result["total_utility"] == result["base_utility"]
