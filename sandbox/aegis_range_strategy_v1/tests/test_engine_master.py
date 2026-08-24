from __future__ import annotations

from datetime import timedelta

from aegis_range_v1.engine import RangeEngineV1
from aegis_range_v1.models import Episode, LevelSnapshot, RangePair
from aegis_range_v1.models import PendingEntry
from aegis_range_v1.numeric import range_episode_id, range_id
from aegis_range_v1.regime import RangeRegimeAdapter

from conftest import FakeRegimeEvaluator, make_5m
from test_levels_signal_safety import qualified_cluster


def make_engine(origin, candidate):
    return RangeEngineV1("BTCUSDT", candidate, RangeRegimeAdapter(FakeRegimeEvaluator()))


def install_episode(engine, origin, *, confirmed_at=None):
    support = qualified_cluster("old-s", "LOW", 98.0, origin + timedelta(hours=1), 2)
    resistance = qualified_cluster("old-r", "HIGH", 102.0, origin + timedelta(hours=1), 2)
    engine.levels.clusters = {support.cluster_id: support, resistance.cluster_id: resistance}
    pair = RangePair("old-s", "old-r", 98.0, 102.0, 100.0, 0.04, 2, 2, origin + timedelta(hours=2), origin + timedelta(hours=2))
    snapshot = LevelSnapshot(origin + timedelta(hours=2), pair, 2.0, "episode", "range")
    engine.episode = Episode(
        "BTCUSDT",
        "episode",
        confirmed_at or origin + timedelta(hours=2),
        "old-s",
        "old-r",
        snapshot,
    )


def test_pair_replaced_same_close_resets_and_cannot_rebirth(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    install_episode(engine, origin)
    new_support = qualified_cluster("new-s", "LOW", 97.5, origin + timedelta(hours=1), 3)
    new_resistance = qualified_cluster("new-r", "HIGH", 102.5, origin + timedelta(hours=1), 3)
    engine.levels.clusters.update({new_support.cluster_id: new_support, new_resistance.cluster_id: new_resistance})
    output = engine.process(make_5m(159, origin))
    assert output["episode_event"] == "PAIR_REPLACED"
    assert engine.episode is None
    assert engine.levels.clusters == {}
    assert engine.range_history == []
    assert output["signal"] == "NONE"


def test_episode_confirmation_ids_and_structure_loss_reset(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    support = qualified_cluster("s", "LOW", 98.0, origin + timedelta(hours=1), 2)
    resistance = qualified_cluster("r", "HIGH", 102.0, origin + timedelta(hours=1), 2)
    engine.levels.clusters = {"s": support, "r": resistance}
    candle = make_5m(159, origin)
    output = engine.process(candle)
    expected_episode = range_episode_id("BTCUSDT", candle.available_at, "s", "r")
    assert output["episode_event"] == "CONFIRMED"
    assert engine.episode.range_episode_id == expected_episode
    assert output["range_id"] == range_id(expected_episode, candle.available_at, 98.0, 102.0, 100.0)

    engine.lifecycle.pending_entry = PendingEntry(
        "BTCUSDT", "LONG", candle.available_at, candle.available_at + timedelta(minutes=5),
        expected_episode, output["range_id"], candle.available_at, 98, 102, 100, 2,
        "ACCUMULATION_RANGE", 70, None,
    )
    support.touches.pop()
    ending = engine.process(make_5m(160, origin))
    assert ending["episode_event"] == "STRUCTURE_LOST"
    assert engine.episode is None and engine.levels.clusters == {}
    assert engine.lifecycle.pending_entry is None


def test_engine_breakout_reads_previous_levels_before_publish(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    install_episode(engine, origin)
    resistance = engine.levels.clusters["old-r"]
    resistance.pivots[0] = resistance.pivots[0].__class__("BTCUSDT", "HIGH", 102.9, "102.9", resistance.pivots[0].pivot_at, resistance.pivots[0].available_at)
    resistance.pivots[1] = resistance.pivots[1].__class__("BTCUSDT", "HIGH", 103.1, "103.1", resistance.pivots[1].pivot_at, resistance.pivots[1].available_at)
    candle = make_5m(159, origin, open_=102.0, high=103.0, low=99.0, close=102.5)
    engine.process(candle)
    assert engine.episode.outside_direction == "UP"
    assert engine.episode.outside_count == 1
    assert engine.episode.previous_snapshot.pair.resistance == 103.0


def test_confirmed_episode_breakout_resets_on_second_close(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    install_episode(engine, origin)
    first = make_5m(159, origin, open_=102.0, high=103.0, low=99.0, close=102.3)
    assert engine.process(first)["episode_event"] is None
    second = make_5m(160, origin, open_=102.0, high=103.0, low=99.0, close=102.3)
    output = engine.process(second)
    assert output["episode_event"] == "CONFIRMED_BREAKOUT"
    assert engine.episode is None and engine.levels.clusters == {}


def test_episode_expires_at_exact_48_hours(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    current = make_5m(159, origin)
    install_episode(engine, origin, confirmed_at=current.available_at - timedelta(hours=48))
    output = engine.process(current)
    assert output["episode_event"] == "EXPIRED_48H"
    assert engine.episode is None


def test_exhausted_side_quota_suppresses_signal(origin, candidate):
    engine = make_engine(origin, candidate)
    engine.history = [make_5m(i, origin) for i in range(159)]
    install_episode(engine, origin)
    engine.levels._bar_index = 10
    engine.lifecycle.trade_counts["episode"] = 1
    engine.lifecycle.traded_sides["episode"] = {"LONG"}
    candle = make_5m(159, origin, open_=98.5, high=99.5, low=97.9, close=99.0)
    output = engine.process(candle)
    assert output["status"] == "RANGE_CANDIDATE"
    assert output["blocker_reason"] == "QUOTA"
    assert output["signal"] == "NONE"
    assert engine.lifecycle.pending_entry is None


def synthetic_history(origin, count):
    candles = []
    for index in range(count):
        phase = index % 8
        close = 100.0 + (phase - 3.5) * 0.15
        candles.append(make_5m(index, origin, open_=close - 0.05, high=close + 1.0, low=close - 1.0, close=close))
    return candles


def run(origin, candidate, candles):
    engine = make_engine(origin, candidate)
    for candle in candles:
        engine.process(candle)
    return engine.deterministic_outputs()


def test_no_lookahead_master_structural_equivalence(origin, candidate):
    history = synthetic_history(origin, 180)
    run_a = run(origin, candidate, history)
    future = synthetic_history(origin, 190)
    future[185] = make_5m(185, origin, open_=150, high=200, low=140, close=180)
    run_b = run(origin, candidate, future)
    assert run_a == run_b[: len(run_a)]
    unperturbed = run(origin, candidate, synthetic_history(origin, 190))
    assert run_b[185:] != unperturbed[185:]


def test_determinism_master_repeated_runs(origin, candidate):
    fixture = synthetic_history(origin, 190)
    baseline = run(origin, candidate, fixture)
    assert baseline == run(origin, candidate, fixture)
    assert baseline == run(origin, candidate, fixture)


def test_data_integrity_resets_range_and_warmup(origin, candidate):
    engine = make_engine(origin, candidate)
    for candle in synthetic_history(origin, 160):
        engine.process(candle)
    engine.process(make_5m(161, origin, segment_id=1))
    assert len(engine.history) == 1
    assert engine.outputs[-1]["status"] == "INSUFFICIENT_HISTORY"
    assert engine.outputs[-1]["data_integrity_reset"] is True


def test_split_resets_range_but_preserves_regime_warmup(origin, candidate):
    engine = make_engine(origin, candidate)
    for candle in synthetic_history(origin, 160):
        engine.process(candle)
    engine.on_split_boundary()
    assert len(engine.history) == 160
    assert engine.range_history == []
    assert engine.levels.clusters == {}
    next_candle = make_5m(160, origin)
    assert engine.process(next_candle, embargo=True)["status"] != "INSUFFICIENT_HISTORY"
    assert engine.levels.detect_available_pivots(engine.range_history) == ()
