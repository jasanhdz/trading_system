from __future__ import annotations

import pytest

from aegis.research.execution_timing_w4 import (
    ExecutionDataCapabilities,
    adverse_selection_bps,
    assess_data_quality,
    assert_causal_snapshot,
    implementation_shortfall_bps,
    microprice,
    midprice,
    stable_execution_intent_id,
    timestamp_diagnostics,
    total_cost_per_intent_bps,
)


def test_execution_intent_identity_is_stable_and_side_specific() -> None:
    first = stable_execution_intent_id("signal-1", "BTCUSDT", "LONG", 123)
    assert first == stable_execution_intent_id("signal-1", "BTCUSDT", "LONG", 123)
    assert first != stable_execution_intent_id("signal-1", "BTCUSDT", "SHORT", 123)


def test_mid_and_microprice_contract() -> None:
    assert midprice(99, 101) == pytest.approx(100)
    assert microprice(99, 101, 3, 1) == pytest.approx(100.5)
    with pytest.raises(ValueError, match="AEGIS_W4_BBO_INVALID"):
        midprice(101, 100)


def test_implementation_shortfall_positive_always_means_worse() -> None:
    assert implementation_shortfall_bps("LONG", 100.1, 100) == pytest.approx(10)
    assert implementation_shortfall_bps("SHORT", 99.9, 100) == pytest.approx(10)
    assert implementation_shortfall_bps("LONG", 99.9, 100) == pytest.approx(-10)
    assert implementation_shortfall_bps("SHORT", 100.1, 100) == pytest.approx(-10)


def test_adverse_selection_is_directionally_symmetric() -> None:
    assert adverse_selection_bps("LONG", 100, 99.9) == pytest.approx(10)
    assert adverse_selection_bps("SHORT", 100, 100.1) == pytest.approx(10)
    assert adverse_selection_bps("LONG", 100, 100.1) == pytest.approx(-10)


def test_total_cost_keeps_components_explicit() -> None:
    assert total_cost_per_intent_bps(
        implementation_shortfall=2, fee=5, delay_cost=1, missed_opportunity_cost=3
    ) == pytest.approx(11)


def test_timestamp_audit_detects_duplicates_and_reordering() -> None:
    result = timestamp_diagnostics([100, 90, 110], [1, 1, 2])
    assert result["out_of_order_events"] == 1
    assert result["duplicate_identities"] == 1
    assert result["passes"] is False


def test_data_quality_fails_without_synchronized_bbo_and_receive_time() -> None:
    result = assess_data_quality(
        ExecutionDataCapabilities(True, False, False, False, True, True)
    )
    assert result["sufficient_for_w4a"] is False
    assert result["sufficient_for_passive_limit"] is False
    assert "HISTORICAL_SYNCHRONIZED_BBO_MISSING" in result["blockers"]


def test_passive_limit_requires_l2_beyond_market_now() -> None:
    result = assess_data_quality(
        ExecutionDataCapabilities(True, True, False, True, True, True)
    )
    assert result["sufficient_for_w4a"] is True
    assert result["sufficient_for_market_now"] is True
    assert result["sufficient_for_passive_limit"] is False


def test_causal_snapshot_rejects_future_book() -> None:
    snapshot = {
        "exchange_timestamp_ms": 101,
        "best_bid": 99,
        "best_ask": 101,
        "best_bid_quantity": 2,
        "best_ask_quantity": 3,
    }
    with pytest.raises(ValueError, match="AEGIS_W4_SNAPSHOT_LOOKAHEAD"):
        assert_causal_snapshot(snapshot, 100)
