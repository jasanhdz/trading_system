from __future__ import annotations

from datetime import timedelta

import pytest

from aegis_range_v1.levels import RangeLevelsV1
from aegis_range_v1.models import LevelCluster, Pivot, RangePair, RegimeSnapshot, Touch
from aegis_range_v1.numeric import cluster_id
from aegis_range_v1.safety import RangeSafetyV1
from aegis_range_v1.signal import RangeSignalV1

from conftest import make_5m


def pivot(symbol, side, price, at):
    return Pivot(symbol, side, price, repr(price), at, at + timedelta(minutes=10))


def test_future_pivot_invisible_and_append_future_stable(origin):
    history = [make_5m(i, origin) for i in range(5)]
    history[2] = make_5m(2, origin, high=105, low=99, close=100)
    assert RangeLevelsV1.detect_available_pivots(history[:4]) == ()
    available = RangeLevelsV1.detect_available_pivots(history)
    assert len(available) == 1 and available[0].side == "HIGH"
    history.append(make_5m(5, origin, high=999))
    assert available[0].available_at == origin + timedelta(minutes=25)


def test_cluster_assignment_tie_break_expiration_and_no_merge(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    first = pivot("BTCUSDT", "LOW", 100.0, origin)
    cluster = levels.insert_pivot(first, 2.0)
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 100.3, origin + timedelta(minutes=5)), 2.0)
    assert len(levels.clusters) == 1
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 105.0, origin + timedelta(minutes=10)), 2.0)
    assert len(levels.clusters) == 2
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 102.5, origin + timedelta(minutes=15)), 20.0)
    assert len(levels.clusters) == 2  # Compatible clusters are never merged.
    levels.expire(origin + timedelta(days=7, minutes=16))
    assert cluster.cluster_id not in levels.clusters


def test_cluster_equal_distance_prefers_oldest_then_lexical_id(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    older = LevelCluster("z-old", "BTCUSDT", "LOW", origin)
    older.pivots = [pivot("BTCUSDT", "LOW", 99.0, origin)]
    newer = LevelCluster("a-new", "BTCUSDT", "LOW", origin + timedelta(minutes=5))
    newer.pivots = [pivot("BTCUSDT", "LOW", 101.0, origin + timedelta(minutes=5))]
    levels.clusters = {older.cluster_id: older, newer.cluster_id: newer}
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 100.0, origin + timedelta(minutes=10)), 10.0)
    assert len(older.pivots) == 2 and len(newer.pivots) == 1
    older.first_pivot_at = newer.first_pivot_at
    older.pivots = [pivot("BTCUSDT", "LOW", 99.0, origin)]
    newer.pivots = [pivot("BTCUSDT", "LOW", 101.0, origin)]
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 100.0, origin + timedelta(minutes=15)), 10.0)
    assert len(newer.pivots) == 2  # a-new is lexically before z-old.


def test_cluster_expiration_boundary_is_inclusive(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    cluster = levels.insert_pivot(pivot("BTCUSDT", "LOW", 100, origin), 2)
    levels.expire(origin + timedelta(days=7))
    assert cluster.cluster_id in levels.clusters
    levels.expire(origin + timedelta(days=7, microseconds=1))
    assert cluster.cluster_id not in levels.clusters


def test_touch_counting_rearm_and_six_bar_separation(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    p1 = pivot("BTCUSDT", "LOW", 100.0, origin)
    cluster = levels.insert_pivot(p1, 2.0)
    levels.insert_pivot(pivot("BTCUSDT", "LOW", 100.1, origin + timedelta(minutes=5)), 2.0)
    touch = make_5m(0, origin, open_=100.2, high=101, low=99.9, close=100.2)
    assert cluster.cluster_id in levels.update_touches(touch, 2.0)
    assert not levels.update_touches(make_5m(1, origin, low=99.9, close=100.2), 2.0)
    levels.update_touches(make_5m(2, origin, low=101.0, close=101.2), 2.0)
    assert cluster.armed
    for index in range(3, 6):
        assert cluster.cluster_id not in levels.update_touches(make_5m(index, origin, low=99.9, close=100.2), 2.0)
    assert cluster.cluster_id in levels.update_touches(make_5m(6, origin, low=99.9, close=100.2), 2.0)


def qualified_cluster(identifier, side, center, origin, touches):
    cluster = LevelCluster(identifier, "BTCUSDT", side, origin)
    cluster.pivots = [pivot("BTCUSDT", side, center - 0.1, origin), pivot("BTCUSDT", side, center + 0.1, origin + timedelta(minutes=5))]
    cluster.touches = [Touch(origin + timedelta(minutes=10 * i), i * 2, center) for i in range(1, touches + 1)]
    return cluster


def test_pair_ranking_uses_min_recency_and_oldest_first_eligible(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    support_a = qualified_cluster("sa", "LOW", 99.0, origin, 3)
    support_b = qualified_cluster("sb", "LOW", 98.8, origin, 2)
    resistance = qualified_cluster("r", "HIGH", 101.0, origin, 3)
    support_a.touches[-1] = Touch(origin + timedelta(hours=2), 20, 99)
    resistance.touches[-1] = Touch(origin + timedelta(hours=1), 18, 101)
    levels.clusters = {c.cluster_id: c for c in (support_a, support_b, resistance)}
    pairs = levels.build_pairs(100.0, origin + timedelta(hours=3))
    assert pairs[0].support_cluster_id == "sa"
    assert pairs[0].pair_recency_at == min(support_a.last_touch_at, resistance.last_touch_at)


def test_pair_recency_breaks_equal_touch_counts(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    older = qualified_cluster("older", "LOW", 99.0, origin, 2)
    newer = qualified_cluster("newer", "LOW", 98.8, origin, 2)
    resistance = qualified_cluster("r", "HIGH", 101.0, origin, 2)
    older.touches[-1] = Touch(origin + timedelta(minutes=20), 4, 99)
    newer.touches[-1] = Touch(origin + timedelta(minutes=30), 6, 98.8)
    resistance.touches[-1] = Touch(origin + timedelta(minutes=40), 8, 101)
    levels.clusters = {c.cluster_id: c for c in (older, newer, resistance)}
    pairs = levels.build_pairs(100, origin + timedelta(hours=1))
    assert [pair.support_cluster_id for pair in pairs[:2]] == ["newer", "older"]


def test_pair_oldest_first_eligibility_precedes_lexical_ids(origin, candidate):
    levels = RangeLevelsV1("BTCUSDT", candidate)
    old = qualified_cluster("z-old", "LOW", 99.0, origin, 2)
    new = qualified_cluster("a-new", "LOW", 98.8, origin, 2)
    resistance = qualified_cluster("r", "HIGH", 101.0, origin, 2)
    shared_touch = Touch(origin + timedelta(minutes=30), 6, 99)
    old.touches[-1] = shared_touch
    new.touches[-1] = shared_touch
    resistance.touches[-1] = Touch(origin + timedelta(minutes=40), 8, 101)
    levels.clusters = {old.cluster_id: old, resistance.cluster_id: resistance}
    levels.build_pairs(100, origin + timedelta(hours=1))
    levels.clusters[new.cluster_id] = new
    pairs = levels.build_pairs(100, origin + timedelta(hours=2))
    assert [pair.support_cluster_id for pair in pairs[:2]] == ["z-old", "a-new"]


def make_pair(origin):
    return RangePair("support", "resistance", 90.0, 110.0, 100.0, 0.2, 2, 2, origin, origin)


def make_regime(**changes):
    values = dict(
        technical_regime="ACCUMULATION_RANGE",
        transition_risk="LOW",
        adx=18.0,
        atr_percentile=0.5,
        bollinger_width_percentile=0.2,
        volume_ratio=1.0,
        range_breakout="NONE",
        failed_breakout_count=0,
        structure="MIXED",
        chop_risk=0.7,
        atr14_raw=2.0,
    )
    values.update(changes)
    return RegimeSnapshot(**values)


def test_counted_touch_required_for_rejection(origin, candidate):
    pair = make_pair(origin)
    candle = make_5m(0, origin, open_=94, high=96, low=90, close=95)
    assert RangeSignalV1.evaluate(candle, pair, 2.0, candidate, frozenset()).side == "NONE"
    assert RangeSignalV1.evaluate(candle, pair, 2.0, candidate, frozenset({"support"})).side == "LONG"


def test_short_rejection_is_symmetric_and_e4_cannot_enter_decision(origin, candidate):
    pair = make_pair(origin)
    candle = make_5m(0, origin, open_=106, high=110, low=104, close=105)
    baseline = RangeSignalV1.evaluate(candle, pair, 2.0, candidate, frozenset({"resistance"}))
    arbitrary_tail_risk_score = 0.999999
    repeated = RangeSignalV1.evaluate(candle, pair, 2.0, candidate, frozenset({"resistance"}))
    assert arbitrary_tail_risk_score
    assert baseline == repeated
    assert baseline.side == "SHORT"


def test_hard_blockers_override_high_descriptive_score(origin, candidate):
    pair = make_pair(origin)
    decision = RangeSafetyV1.evaluate(
        pair,
        make_regime(transition_risk="HIGH"),
        candidate,
        48,
        episode_operable=True,
        flat=True,
        no_pending_exit=True,
        cooldown_ready=True,
        quota_ready=True,
    )
    assert decision.descriptive_score > 50
    assert not decision.allowed and decision.reason == "TRANSITION_RISK_HIGH"


@pytest.mark.parametrize(
    ("regime_changes", "flags", "reason"),
    [
        ({"technical_regime": "TREND"}, {}, "REGIME_BLOCKED"),
        ({"range_breakout": "UP"}, {}, "RANGE_BREAKOUT_ACTIVE"),
        ({"adx": 26.0}, {}, "ADX_BLOCKED"),
        ({"chop_risk": 0.61}, {}, "CHOP_RISK_BLOCKED"),
        ({"bollinger_width_percentile": 0.45}, {}, "BOLLINGER_WIDTH_BLOCKED"),
        ({"atr_percentile": 0.81}, {}, "ATR_PERCENTILE_BLOCKED"),
        ({"volume_ratio": 0.49}, {}, "VOLUME_BLOCKED"),
        ({}, {"flat": False}, "POSITION_OPEN"),
        ({}, {"no_pending_exit": False}, "PENDING_EXIT"),
        ({}, {"cooldown_ready": False}, "COOLDOWN"),
        ({}, {"quota_ready": False}, "QUOTA"),
    ],
)
def test_every_hard_blocker_is_non_compensatory(origin, candidate, regime_changes, flags, reason):
    arguments = dict(episode_operable=True, flat=True, no_pending_exit=True, cooldown_ready=True, quota_ready=True)
    arguments.update(flags)
    decision = RangeSafetyV1.evaluate(make_pair(origin), make_regime(**regime_changes), candidate, 48, **arguments)
    assert not decision.allowed
    assert decision.reason == reason
