from __future__ import annotations

from datetime import timedelta

import pytest

from aegis_range_v1.costs import BASELINE, adverse_fill
from aegis_range_v1.lifecycle import RangeLifecycleV1, StateInvariantViolation
from aegis_range_v1.models import PendingEntry, Position

from conftest import make_5m


def pending(origin, side="LONG", *, support=90.0, resistance=110.0, midpoint=100.0, atr=2.0):
    return PendingEntry(
        "BTCUSDT",
        side,
        origin,
        origin + timedelta(minutes=5),
        "episode",
        "range",
        origin - timedelta(hours=1),
        support,
        resistance,
        midpoint,
        atr,
        "ACCUMULATION_RANGE",
        70.0,
        None,
    )


def enter(lifecycle, origin, item=None, raw_open=94.0):
    item = item or pending(origin)
    lifecycle.schedule_entry(item)
    return lifecycle.consume_pending_entry(
        open_at=item.entry_available_at,
        raw_open=raw_open,
        same_split=True,
        episode_active=True,
    )


def test_pending_entry_invariant_and_next_bar_only(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    item = pending(origin)
    lifecycle.schedule_entry(item)
    lifecycle.position = Position("BTCUSDT", "LONG", origin, 95, "x", "x", origin, 90, 110, 100, 2, 89, 100, "{}", "hash")
    with pytest.raises(StateInvariantViolation, match="STATE_INVARIANT_VIOLATION"):
        lifecycle.consume_pending_entry(open_at=item.entry_available_at, raw_open=94, same_split=True, episode_active=True)
    lifecycle.position = None
    lifecycle.pending_entry = item
    assert lifecycle.consume_pending_entry(open_at=origin, raw_open=94, same_split=True, episode_active=True) is None
    assert lifecycle.trade_counts == {}
    assert lifecycle.last_entry_cancel_reason == "NOT_NEXT_BAR_OPEN"


def test_entry_42bps_reward_risk_and_quota_after_success(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    assert enter(lifecycle, origin, pending(origin, midpoint=94.3), raw_open=94.0) is None
    assert lifecycle.trade_counts == {}
    assert lifecycle.last_entry_cancel_reason == "TARGET_DISTANCE_LT_42_BPS"
    assert enter(lifecycle, origin + timedelta(hours=1), pending(origin + timedelta(hours=1), support=90, midpoint=98), raw_open=97.0) is None
    assert lifecycle.trade_counts == {}
    assert lifecycle.last_entry_cancel_reason == "REWARD_RISK_LT_1"
    position = enter(lifecycle, origin + timedelta(hours=2), raw_open=94.0)
    assert position is not None
    assert position.entry_fill == adverse_fill(94.0, "LONG", BASELINE.slippage_bps_per_side)
    assert lifecycle.trade_counts["episode"] == 1
    assert position.thesis_feature_hash


def test_thesis_tp_sl_are_frozen(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    position = enter(lifecycle, origin)
    assert position is not None
    original = (position.stop_at_entry, position.target_at_entry, position.thesis_serialized)
    lifecycle.process_close(108.0)
    assert (position.stop_at_entry, position.target_at_entry, position.thesis_serialized) == original


def test_entry_bar_resting_orders_are_active_and_adverse_first(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    position = enter(lifecycle, origin)
    assert position is not None
    candle = make_5m(
        1,
        origin,
        open_=94.0,
        high=position.target_at_entry + 1.0,
        low=position.stop_at_entry - 1.0,
        close=95.0,
    )
    event = lifecycle.process_position_open_and_intrabar(candle, include_open_gaps=False)
    assert event.reason == "STOP"
    assert event.fill_price == adverse_fill(position.stop_at_entry, "SHORT", 2)


def position(origin, side="LONG"):
    return Position("BTCUSDT", side, origin, 94, "episode", "range", origin, 90, 110, 100, 2, 89, 100, "{}", "hash")


def test_adverse_first_and_gap_policies(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    lifecycle.position = position(origin)
    both = make_5m(0, origin, open_=95, high=101, low=88, close=96)
    assert lifecycle.process_position_open_and_intrabar(both).reason == "STOP"
    lifecycle.position = position(origin)
    gap_stop = make_5m(1, origin, open_=88, high=95, low=87, close=94)
    event = lifecycle.process_position_open_and_intrabar(gap_stop)
    assert event.reason == "STOP_GAP"
    assert event.fill_price == adverse_fill(88, "SHORT", 2)
    lifecycle.position = position(origin)
    gap_target = make_5m(2, origin, open_=105, high=106, low=99, close=102)
    event = lifecycle.process_position_open_and_intrabar(gap_target)
    assert event.reason == "TARGET_GAP"
    assert event.fill_price == adverse_fill(100, "SHORT", 2)


def test_trade_breakout_and_max_hold_exit_next_open(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    lifecycle.position = position(origin)
    lifecycle.process_close(89.7)
    lifecycle.process_close(89.7)
    assert lifecycle.position.pending_exit_reason == "TRADE_BREAKOUT"
    breakout_bar = make_5m(2, origin, open_=89, high=90, low=88, close=89)
    event = lifecycle.process_position_open_and_intrabar(breakout_bar)
    assert event.reason == "TRADE_BREAKOUT"
    assert event.fill_at == breakout_bar.open_time
    assert event.fill_price == adverse_fill(89, "SHORT", 2)
    lifecycle.position = position(origin)
    lifecycle.position.closed_bars = 143
    lifecycle.process_close(95)
    assert lifecycle.position.pending_exit_reason == "MAX_HOLD"
    max_hold_bar = make_5m(3, origin, open_=95)
    event = lifecycle.process_position_open_and_intrabar(max_hold_bar)
    assert event.reason == "MAX_HOLD"
    assert event.fill_at == max_hold_bar.open_time
    assert event.fill_price == adverse_fill(95, "SHORT", 2)


def test_cooldown_close_12_and_side_episode_quotas(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    lifecycle.position = position(origin)
    lifecycle.position.pending_exit_reason = "MAX_HOLD"
    lifecycle.process_position_open_and_intrabar(make_5m(0, origin, open_=95))
    for _ in range(11):
        lifecycle.process_close(95)
    assert not lifecycle.cooldown_ready()
    lifecycle.process_close(95)
    assert lifecycle.cooldown_ready()
    lifecycle.trade_counts["episode"] = 1
    lifecycle.traded_sides["episode"] = {"LONG"}
    assert not lifecycle.quota_ready("episode", "LONG")
    assert lifecycle.quota_ready("episode", "SHORT")
    lifecycle.trade_counts["episode"] = 2
    assert not lifecycle.quota_ready("episode", "SHORT")


def test_invariant_checked_before_pending_market_exit(origin, candidate):
    lifecycle = RangeLifecycleV1(candidate)
    lifecycle.pending_entry = pending(origin)
    lifecycle.position = position(origin)
    lifecycle.position.pending_exit_reason = "TRADE_BREAKOUT"
    with pytest.raises(StateInvariantViolation, match="STATE_INVARIANT_VIOLATION"):
        lifecycle.assert_open_invariants()
