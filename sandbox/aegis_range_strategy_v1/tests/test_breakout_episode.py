from __future__ import annotations

from datetime import timedelta

from aegis_range_v1.breakout import RangeBreakoutV1
from aegis_range_v1.models import Episode, LevelSnapshot, RangePair


def snapshot(origin, support=90.0, resistance=110.0, atr=2.0):
    pair = RangePair("s", "r", support, resistance, (support + resistance) / 2, (resistance - support) / 100, 2, 2, origin, origin)
    return LevelSnapshot(origin, pair, atr, "episode", "range")


def test_episode_breakout_uses_previous_snapshot(origin):
    previous = snapshot(origin, resistance=110)
    episode = Episode("BTCUSDT", "episode", origin, "s", "r", previous)
    assert not RangeBreakoutV1.update_episode(episode, 110.21)
    # Publishing wider levels after close one must not rewrite the prior decision.
    RangeBreakoutV1.publish_snapshot(episode, snapshot(origin + timedelta(minutes=5), resistance=120))
    assert not RangeBreakoutV1.update_episode(episode, 110.21)
    assert episode.outside_count == 0
    episode.previous_snapshot = previous
    assert not RangeBreakoutV1.update_episode(episode, 110.21)
    assert RangeBreakoutV1.update_episode(episode, 110.21)
