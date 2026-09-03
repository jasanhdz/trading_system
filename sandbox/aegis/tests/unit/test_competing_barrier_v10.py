from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from aegis.research.competing_barrier_v10 import (
    BarrierContract,
    BarrierOutcome,
    BarrierResearchError,
    conservative_utility,
    deterministic_episode_mask,
    evaluate_barrier_path,
    primary_direction_label,
)


@dataclass(frozen=True)
class Bar:
    high: float
    low: float
    close: float


CONTRACT = BarrierContract("TEST", 0.01, 0.01, 2, 0.002)


def test_barrier_outcome_is_future_ohlc_only_and_side_aware() -> None:
    bars = [Bar(101.1, 99.5, 100.8), Bar(101.2, 100.0, 101.0)]
    long = evaluate_barrier_path(
        side="LONG", entry_price=100.0, future_bars=bars, contract=CONTRACT
    )
    short = evaluate_barrier_path(
        side="SHORT", entry_price=100.0, future_bars=bars, contract=CONTRACT
    )
    assert long["outcome"] == BarrierOutcome.FAVORABLE_FIRST.value
    assert long["event_bar"] == 1
    assert long["realized_utility"] == pytest.approx(0.008)
    assert short["outcome"] == BarrierOutcome.ADVERSE_FIRST.value


def test_same_bar_is_ambiguous_and_valued_as_loss() -> None:
    result = evaluate_barrier_path(
        side="LONG",
        entry_price=100.0,
        future_bars=[Bar(101.1, 98.9, 100.0), Bar(100.0, 100.0, 100.0)],
        contract=CONTRACT,
    )
    assert result["outcome"] == BarrierOutcome.SAME_BAR_AMBIGUOUS.value
    assert result["realized_utility"] == pytest.approx(-0.012)


def test_neither_uses_terminal_side_return_after_severe_cost() -> None:
    result = evaluate_barrier_path(
        side="SHORT",
        entry_price=100.0,
        future_bars=[Bar(100.5, 99.5, 100.2), Bar(100.4, 99.4, 99.8)],
        contract=CONTRACT,
    )
    assert result["outcome"] == BarrierOutcome.NEITHER_REACHED.value
    assert result["realized_utility"] == pytest.approx(0.0)


def test_direction_abstains_unless_exactly_one_side_wins() -> None:
    favorable = {"outcome": BarrierOutcome.FAVORABLE_FIRST.value}
    adverse = {"outcome": BarrierOutcome.ADVERSE_FIRST.value}
    assert primary_direction_label(favorable, adverse) == "LONG"
    assert primary_direction_label(adverse, favorable) == "SHORT"
    assert primary_direction_label(adverse, adverse) == "ABSTAIN"


def test_conservative_utility_penalizes_unknown_probability() -> None:
    strong = conservative_utility(
        {
            "FAVORABLE_FIRST": 0.8,
            "ADVERSE_FIRST": 0.1,
            "SAME_BAR_AMBIGUOUS": 0.05,
            "NEITHER_REACHED": 0.05,
        },
        CONTRACT,
        unknown_penalty_fraction=0.25,
    )
    weak = conservative_utility(
        {
            "FAVORABLE_FIRST": 0.4,
            "ADVERSE_FIRST": 0.1,
            "SAME_BAR_AMBIGUOUS": 0.25,
            "NEITHER_REACHED": 0.25,
        },
        CONTRACT,
        unknown_penalty_fraction=0.25,
    )
    assert strong > weak
    with pytest.raises(BarrierResearchError):
        conservative_utility({"FAVORABLE_FIRST": 1.0}, CONTRACT, unknown_penalty_fraction=0.25)


def test_episode_selection_is_deterministic_and_outcome_independent() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"symbol": "BTCUSDT", "timestamp_value": start},
        {"symbol": "ETHUSDT", "timestamp_value": start + timedelta(minutes=5)},
        {"symbol": "BTCUSDT", "timestamp_value": start + timedelta(minutes=60)},
        {"symbol": "BTCUSDT", "timestamp_value": start + timedelta(minutes=120)},
    ]
    assert deterministic_episode_mask(rows, spacing_minutes=120) == (
        True,
        True,
        False,
        True,
    )
