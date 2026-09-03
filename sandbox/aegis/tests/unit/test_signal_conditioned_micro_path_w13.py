from __future__ import annotations

from aegis.research.signal_conditioned_micro_path_w13 import (
    PassiveMicroPathCollector,
    W13SampleRequirements,
    assess_sample_gate,
    stable_signal_episode_id,
)


def _event(identity: str, timestamp: int, *, symbol: str = "BTCUSDT") -> dict:
    return {
        "event_id": identity,
        "event_type": "TRADE",
        "symbol": symbol,
        "exchange_timestamp_us": timestamp,
        "local_receive_timestamp_us": timestamp + 100,
        "price": 100.0,
    }


def test_w13_episode_identity_includes_frozen_side() -> None:
    short = stable_signal_episode_id("btcusdt", "SHORT", 1_000_000)
    assert short == stable_signal_episode_id("BTCUSDT", "SHORT", 1_000_000)
    assert short != stable_signal_episode_id("BTCUSDT", "LONG", 1_000_000)


def test_w13_sample_gate_fails_closed_without_independent_evidence() -> None:
    rows = [{"symbol": "BTCUSDT", "side": "SHORT", "date": "2026-01-01"}]
    result = assess_sample_gate(
        {"W13_TRAIN": rows, "W13_VALIDATION": rows},
        W13SampleRequirements(
            minimum_train_episodes=2,
            minimum_validation_episodes=2,
            minimum_symbols_per_partition=1,
            minimum_temporal_days_per_partition=1,
            minimum_directions_per_partition=1,
        ),
    )
    assert result["passes"] is False
    assert "INSUFFICIENT_W13_TRAIN_EPISODES:1<2" in result["blockers"]


def test_passive_collector_captures_pre_and_post_signal_without_execution_authority() -> None:
    collector = PassiveMicroPathCollector(pre_signal_seconds=30, post_signal_seconds=180)
    base = 1_000_000_000
    collector.observe_market_event(_event("pre", base - 10_000_000))
    episode_id = collector.observe_signal(
        symbol="BTCUSDT", side="SHORT",
        signal_exchange_timestamp_us=base, signal_local_timestamp_us=base + 200,
    )
    collector.observe_market_event(_event("post", base + 1_000_000))
    collector.observe_market_event(_event("post", base + 1_000_000))
    completed = collector.finalize(through_exchange_timestamp_us=base + 180_000_000)
    assert completed[0]["episode_id"] == episode_id
    assert completed[0]["side"] == "SHORT"
    assert completed[0]["event_count"] == 2
    assert completed[0]["capture_only"] is True
    assert completed[0]["execution_authority"] is False
