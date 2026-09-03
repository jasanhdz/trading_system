from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.domain import Candle
from aegis.research.long_entry_v31_shadow import (
    entry_confirmation,
    family_horizon,
    global_context_gate,
    specialist_committee_score_v31,
)
from aegis.research.long_entry_v3_shadow import LongCandidateFamily


def _config() -> dict[str, object]:
    root = Path(__file__).parents[2]
    return yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v31_shadow.yaml").read_text()
    )


def _candle(index: int, *, open_price: float, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return Candle(
        open_time=start,
        close_time=start + timedelta(minutes=5),
        open=open_price,
        high=max(open_price, close) + 0.2,
        low=min(open_price, close) - 0.2,
        close=close,
        volume=1000.0,
        is_closed=True,
        source="TEST",
    )


def test_global_context_requires_breadth_cross_market_and_higher_timeframe() -> None:
    config = _config()["global_context_gate"]
    passed = global_context_gate(
        base={"market_breadth_6": 0.7, "btc_trend_proxy": 0.1, "eth_trend_proxy": 0.1},
        context={"1h_trend_stack_long": 1.0, "15m_trend_stack_long": 0.0},
        regime={"direction": "BULLISH"},
        config=config,
    )
    assert passed["passed"] is True
    failed = global_context_gate(
        base={"market_breadth_6": 0.4, "btc_trend_proxy": 0.1, "eth_trend_proxy": 0.1},
        context={"1h_trend_stack_long": 1.0, "15m_trend_stack_long": 0.0},
        regime={"direction": "BULLISH"},
        config=config,
    )
    assert failed["passed"] is False


def test_breakout_confirmation_waits_for_closed_follow_through_bar() -> None:
    config = _config()["entry_timing_stage"]
    result = entry_confirmation(
        family=LongCandidateFamily.BREAKOUT_EXPANSION.value,
        signal=_candle(0, open_price=99.5, close=100.0),
        confirmation=_candle(1, open_price=100.0, close=100.8),
        confirmation_micro={"taker_buy_ratio_1": 0.60},
        config=config,
    )
    assert result["passed"] is True
    assert result["entry_offset_bars"] == 2


def test_failed_confirmation_never_becomes_execution_candidate() -> None:
    config = _config()["entry_timing_stage"]
    result = entry_confirmation(
        family=LongCandidateFamily.CONFIRMED_REVERSAL.value,
        signal=_candle(0, open_price=99.5, close=100.0),
        confirmation=_candle(1, open_price=100.0, close=99.0),
        confirmation_micro={"taker_buy_ratio_1": 0.40},
        config=config,
    )
    assert result["passed"] is False


def test_retest_uses_immediate_entry_without_reading_future_confirmation() -> None:
    config = _config()["entry_timing_stage"]
    result = entry_confirmation(
        family=LongCandidateFamily.BREAKOUT_RETEST.value,
        signal=_candle(0, open_price=99.5, close=100.0),
        confirmation=None,
        confirmation_micro=None,
        config=config,
    )
    assert result["passed"] is True
    assert result["entry_offset_bars"] == 1


def test_family_horizons_are_fixed_before_validation() -> None:
    stage = _config()["opportunity_stage"]
    assert family_horizon(LongCandidateFamily.CONFIRMED_REVERSAL.value, stage) == 12
    assert family_horizon(LongCandidateFamily.BREAKOUT_RETEST.value, stage) == 48


def test_v31_committee_preserves_all_four_specialists() -> None:
    assert specialist_committee_score_v31(0.8, 0.6, 0.25, 0.2) == pytest.approx(0.288)
    with pytest.raises(ValueError):
        specialist_committee_score_v31(0.8, 0.6, -0.1, 0.2)


def test_v31_preregistration_cannot_select_live_or_force_frequency() -> None:
    payload = _config()
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["ranking"]["force_minimum_frequency"] is False
    assert payload["family_retirement"]["family_fallback"] == "PROHIBITED"
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0
