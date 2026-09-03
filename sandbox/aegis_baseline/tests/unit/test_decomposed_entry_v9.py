from __future__ import annotations

from dataclasses import dataclass

from aegis.research.decomposed_entry_v9 import (
    DirectionLabelContract,
    TimingLabelContract,
    direction_label,
    rolling_four_hour_context,
    timing_labels,
)


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def test_rolling_context_uses_closed_history() -> None:
    bars = [Bar(100 + i, 101 + i, 99 + i, 100.5 + i, 10 + i) for i in range(145)]
    result = rolling_four_hour_context(bars)
    assert result["rolling_4h_return"] > 0.0
    assert result["rolling_12h_return"] > result["rolling_4h_return"]
    assert 0.0 <= result["rolling_12h_close_location"] <= 1.0


def test_direction_label_requires_positive_edge_over_opposite() -> None:
    contract = DirectionLabelContract(0.001, 0.0015, 0.001)
    long_row = {"side": "LONG", "terminal_return_after_costs": 0.01}
    short_row = {"side": "SHORT", "terminal_return_after_costs": -0.012}
    assert direction_label(long_row, short_row, contract)["label"] == "LONG"
    flat_long = {"side": "LONG", "terminal_return_after_costs": 0.0007}
    flat_short = {"side": "SHORT", "terminal_return_after_costs": 0.0006}
    assert direction_label(flat_long, flat_short, contract)["label"] == "ABSTAIN"


def test_timing_failures_are_separate_and_can_overlap() -> None:
    row = {
        "v7_features": [0.0] * 162,
        "v8_profile_cost_returns": {"CURRENT_TS": {"stress": -0.01}},
        "mae_fraction": 0.02,
        "first_positive_after_cost_bar": None,
        "first_adverse_bar": 1,
        "first_favorable_bar": None,
        "early_reversal": True,
        "soft_archetype_memberships": {
            "TREND_CONTINUATION": 0.1,
            "BREAKOUT": 0.6,
            "REVERSAL": 0.1,
            "RANGE_REVERSION": 0.1,
            "EXHAUSTION_RISK": 0.1,
        },
        "forward_regime_multihorizon": {"label": "TRANSITION"},
    }
    # V7 derived fields occupy the final context slots.
    row["v7_features"][123] = 0.01
    row["v7_features"][126] = 0.5
    row["v7_features"][147] = 2.5
    row["v7_features"][156] = 0.2
    labels = timing_labels(
        row,
        {"timeframe_conflict_score": 0.75},
        TimingLabelContract(0.006, 6, 2.0, 1.5, 0.9),
    )
    assert labels["EXHAUSTED_MOVE"] is True
    assert labels["FALSE_BREAKOUT"] is True
    assert labels["TRANSITION_FAILURE"] is True
    assert labels["ADVERSE_CONTINUATION"] is True
    assert labels["ANY_FAILURE"] is True
