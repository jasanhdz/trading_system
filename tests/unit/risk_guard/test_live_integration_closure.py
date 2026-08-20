"""Executable closure tests for the E4 live integration boundary."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aegis.live_api import _build_e4_config, create_app
from aegis.risk_guard.domain import FROZEN_TAIL_RISK_THRESHOLD
from aegis.risk_guard.feature_bridge import FROZEN_E4_UNIVERSE
from aegis.risk_guard.feature_bridge import FeatureRow
from aegis.risk_guard.market_snapshot import (
    RollingCandleCache,
    _compute_snapshot_hash,
    fetch_snapshot,
)
from aegis.risk_guard.observability import E4EvidenceRecorder
from aegis.risk_guard.precompute import E4PrecomputeService


DECISION_AT = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _candles(symbol_index: int, minutes: int = 24_000, future: int = 5) -> pd.DataFrame:
    start = DECISION_AT - timedelta(minutes=minutes)
    count = minutes + future
    open_time = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    index = np.arange(count, dtype=float)
    base = 10.0 + symbol_index * 7.0
    drift = 0.00003 + symbol_index * 0.000001
    close = base * np.exp(index * drift + np.sin(index / 31.0) * 0.0004)
    open_ = close * (1.0 + np.sin(index / 17.0) * 0.0001)
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = 1_000.0 + (index % 53) * (2.0 + symbol_index / 10.0)
    taker = volume * (0.45 + (index % 11) / 100.0)
    return pd.DataFrame({
        "open_time_ms": (open_time.astype("int64") // 1_000_000).astype("int64"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "taker_buy_volume": taker,
        "quote_volume": volume * close,
        "open_time": open_time,
        "close_time": open_time + pd.Timedelta(minutes=1),
    })


def _small_candles(start: datetime, count: int) -> pd.DataFrame:
    open_time = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    values = np.arange(count, dtype=float) + 1.0
    return pd.DataFrame({
        "open_time_ms": (open_time.astype("int64") // 1_000_000).astype("int64"),
        "open": values,
        "high": values + 1,
        "low": values - 1,
        "close": values + 0.5,
        "volume": values * 10,
        "taker_buy_volume": values * 5,
        "quote_volume": values * 100,
        "open_time": open_time,
        "close_time": open_time + pd.Timedelta(minutes=1),
    })


def test_rolling_cache_cold_and_incremental_do_not_deadlock(monkeypatch):
    cache = RollingCandleCache()
    cold = _small_candles(DECISION_AT - timedelta(minutes=3), 3)
    monkeypatch.setattr(
        "aegis.risk_guard.market_snapshot._fetch_klines_paginated",
        lambda *args, **kwargs: cold.copy(),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(cache.get_or_fetch, "BTCUSDT", DECISION_AT, 3).result(1.0)
    assert len(result) == 3

    next_cycle = DECISION_AT + timedelta(minutes=5)
    incremental = _small_candles(DECISION_AT, 5)
    monkeypatch.setattr(
        "aegis.risk_guard.market_snapshot._fetch_klines_forward",
        lambda *args, **kwargs: incremental.copy(),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(cache.get_or_fetch, "BTCUSDT", next_cycle, 3).result(1.0)
    assert result["close_time"].max() <= pd.Timestamp(next_cycle)
    assert int(result["open_time_ms"].iloc[-1]) == int(next_cycle.timestamp() * 1000) - 60_000


def test_snapshot_hash_covers_old_history_timestamp_and_taker_volume():
    frame = _small_candles(DECISION_AT - timedelta(minutes=200), 200)
    baseline = _compute_snapshot_hash({"BTCUSDT": frame}, DECISION_AT)
    for column in ("open_time_ms", "open", "taker_buy_volume"):
        changed = frame.copy()
        changed.loc[0, column] += 1
        if column == "open_time_ms":
            changed["open_time"] = pd.to_datetime(changed.open_time_ms, unit="ms", utc=True)
        assert _compute_snapshot_hash({"BTCUSDT": changed}, DECISION_AT) != baseline


class _Runtime:
    def health(self):
        return {"ready": True}

    def predict(self, symbol, _trace):
        return {"aegis": {"turbo": {"action": "LONG"}}, "symbol": symbol}


@pytest.fixture(scope="module")
def real_e4_cycle(tmp_path_factory):
    import aegis.risk_guard.market_snapshot as market_snapshot

    source = {
        symbol: _candles(index)
        for index, symbol in enumerate(sorted(FROZEN_E4_UNIVERSE))
    }
    original_page = market_snapshot._fetch_klines_page
    original_cache = market_snapshot._rolling_cache
    original_gap = market_snapshot.INTER_REQUEST_GAP_MS

    def fake_page(
        symbol,
        interval="1m",
        limit=1500,
        start_time_ms=None,
        end_time_ms=None,
    ):
        frame = source[symbol]
        selected = frame
        if start_time_ms is not None:
            selected = selected[selected.open_time_ms >= start_time_ms]
        if end_time_ms is not None:
            selected = selected[selected.open_time_ms <= end_time_ms]
        if start_time_ms is None:
            selected = selected.tail(limit)
        else:
            selected = selected.head(limit)
        return selected.copy().reset_index(drop=True)

    market_snapshot._fetch_klines_page = fake_page
    market_snapshot._rolling_cache = RollingCandleCache()
    market_snapshot.INTER_REQUEST_GAP_MS = 0
    evidence_path = tmp_path_factory.mktemp("e4") / "precompute.jsonl"
    now = DECISION_AT + timedelta(seconds=10)
    metrics: dict[str, float] = {}
    try:
        cold_start = time.monotonic()
        snapshot = fetch_snapshot(DECISION_AT)
        metrics["cold_snapshot_ms"] = (time.monotonic() - cold_start) * 1000

        recorder = E4EvidenceRecorder(evidence_path)
        service = E4PrecomputeService(
            _build_e4_config(),
            evidence_recorder=recorder,
            now_fn=lambda: now,
            snapshot_provider=lambda _decision_at: snapshot,
        )
        service.initialize()
        metrics["cold_start_ms"] = (time.monotonic() - cold_start) * 1000
        cycle = service.last_cycle
        assert cycle is not None
        metrics["panel_build_ms"] = cycle.feature_build_latency_ms
        metrics["score_22_ms"] = cycle.score_latency_ms
        incremental_start = time.monotonic()
        incremental_snapshot = fetch_snapshot(DECISION_AT + timedelta(minutes=5))
        metrics["incremental_snapshot_ms"] = (
            time.monotonic() - incremental_start
        ) * 1000
        assert all(
            pd.to_datetime(frame.close_time, utc=True).max()
            <= pd.Timestamp(DECISION_AT + timedelta(minutes=5))
            for frame in incremental_snapshot.candles_by_symbol.values()
        )

        from aegis.live_api import _lookup_e4_response
        lookup_start = time.monotonic()
        for _ in range(100):
            assert _lookup_e4_response(
                service, "BTCUSDT", "LONG", DECISION_AT, now
            )["available"] is True
        metrics["api_cache_lookup_ms"] = (time.monotonic() - lookup_start) * 10

        yield service, snapshot, evidence_path, source, metrics
    finally:
        market_snapshot._fetch_klines_page = original_page
        market_snapshot._rolling_cache = original_cache
        market_snapshot.INTER_REQUEST_GAP_MS = original_gap


def test_real_cold_start_batch_scores_evidence_and_api(real_e4_cycle):
    service, snapshot, evidence_path, _source, _metrics = real_e4_cycle
    assert all(
        pd.to_datetime(frame.close_time, utc=True).max() <= pd.Timestamp(DECISION_AT)
        for frame in snapshot.candles_by_symbol.values()
    )
    cycle = service.last_cycle
    assert cycle is not None
    assert cycle.score_count == 22
    scores = [score for sides in cycle.scores.values() for score in sides.values()]
    assert len(scores) == 22
    assert all(np.isfinite(item.score) and 0 <= item.score <= 1 for item in scores)
    assert all(item.threshold == FROZEN_TAIL_RISK_THRESHOLD for item in scores)
    assert all(
        item.risk_decision == ("BLOCK" if item.score >= item.threshold else "ALLOW")
        for item in scores
    )
    entries = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert len(entries) == 22
    assert all(entry["event"] == "e4_precompute_score" for entry in entries)
    assert all("signal_id" not in entry for entry in entries)

    async def request():
        app = create_app(
            service=_Runtime(),
            e4_service=service,
            now_fn=lambda: DECISION_AT + timedelta(minutes=4, seconds=50),
        )
        endpoint = next(
            route.endpoint for route in app.routes
            if getattr(route, "path", None) == "/ml-v2/e4_tail_risk"
        )
        from aegis.live_api import E4TailRiskRequest
        return await endpoint(E4TailRiskRequest(
            symbol="BTCUSDT",
            side="LONG",
            decision_at=DECISION_AT.isoformat(),
        ))

    response = asyncio.run(request())
    payload = response
    assert payload["available"] is True
    assert payload["decision_at"] == DECISION_AT.isoformat()


@pytest.mark.parametrize("offset", [10, 90, 290])
def test_cycle_t_valid_through_its_five_minute_window(real_e4_cycle, offset):
    service = real_e4_cycle[0]
    from aegis.live_api import _lookup_e4_response

    payload = _lookup_e4_response(
        service,
        "BTCUSDT",
        "LONG",
        DECISION_AT,
        DECISION_AT + timedelta(seconds=offset),
    )
    assert payload["available"] is True


def test_new_boundary_requires_new_cycle(real_e4_cycle):
    service = real_e4_cycle[0]
    from aegis.live_api import _lookup_e4_response

    next_cycle = DECISION_AT + timedelta(minutes=5)
    payload = _lookup_e4_response(
        service, "BTCUSDT", "LONG", next_cycle, next_cycle
    )
    assert payload["available"] is False
    assert payload["decision"] == "BLOCK"
    assert payload["reason"] == "E4_EXPECTED_CYCLE_UNAVAILABLE"


def test_future_data_does_not_change_snapshot_features_or_scores(real_e4_cycle):
    service, snapshot, _path, source, _metrics = real_e4_cycle
    baseline_hash = snapshot.snapshot_hash
    baseline_rows = service._guard.bridge.from_market_candles_batch(
        snapshot.candles_by_symbol, DECISION_AT
    )
    baseline_scores = {
        key: service._guard.score(row.to_dataframe()) for key, row in baseline_rows.items()
    }

    changed = {symbol: frame.copy() for symbol, frame in source.items()}
    for frame in changed.values():
        future = frame.close_time > pd.Timestamp(DECISION_AT)
        frame.loc[future, "close"] *= 100
        frame.loc[future, "taker_buy_volume"] = 0
    causal = {
        symbol: frame[frame.close_time <= pd.Timestamp(DECISION_AT)].copy()
        for symbol, frame in changed.items()
    }
    assert _compute_snapshot_hash(causal, DECISION_AT) == baseline_hash
    changed_rows = service._guard.bridge.from_market_candles_batch(causal, DECISION_AT)
    assert all(
        changed_rows[key].feature_hash == baseline_rows[key].feature_hash
        for key in baseline_rows
    )
    assert all(
        service._guard.score(changed_rows[key].to_dataframe()) == baseline_scores[key]
        for key in baseline_scores
    )


def test_batch_feature_score_and_decision_parity(real_e4_cycle):
    service, snapshot, _path, _source, _metrics = real_e4_cycle
    batch = service._guard.bridge.from_market_candles_batch(
        snapshot.candles_by_symbol, DECISION_AT
    )
    single = service._guard.bridge.from_market_candles(
        snapshot.candles_by_symbol, "BTCUSDT", "LONG", DECISION_AT
    )
    batch_row = batch[("BTCUSDT", "LONG")]
    assert batch_row.feature_hash == single.feature_hash
    assert batch_row.features == single.features
    batch_score = service._guard.score(batch_row.to_dataframe())
    single_score = service._guard.score(single.to_dataframe())
    assert batch_score == single_score
    assert (batch_score >= FROZEN_TAIL_RISK_THRESHOLD) == (
        single_score >= FROZEN_TAIL_RISK_THRESHOLD
    )


def test_initialize_then_background_does_not_recompute_same_cycle(real_e4_cycle):
    service = real_e4_cycle[0]
    calls = 0
    original = service._run_cycle

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    service._run_cycle = counted
    service._running = True
    service._sleep_fn = lambda _seconds: setattr(service, "_running", False)
    service._background_loop()
    service._run_cycle = original
    assert calls == 0


def test_one_score_failure_does_not_publish_or_emit_evidence():
    class Bridge:
        def from_market_candles_batch(self, _candles, decision_at):
            return {
                (symbol, side): FeatureRow(
                    features={"feature": 1.0},
                    symbol=symbol,
                    side=side,
                    timestamp=decision_at,
                    feature_hash=f"{symbol}-{side}",
                )
                for symbol in sorted(FROZEN_E4_UNIVERSE)
                for side in ("LONG", "SHORT")
            }

    class Guard:
        bridge = Bridge()
        calls = 0

        def score(self, _features):
            self.calls += 1
            if self.calls == 7:
                raise RuntimeError("score failure")
            return 0.2

        def version(self):
            return "E4_TAIL_RISK_GUARD_V1"

        def is_available(self):
            return True

    class Evidence:
        entries = []

        def record_precompute(self, **entry):
            self.entries.append(entry)

    snapshot = type("Snapshot", (), {
        "snapshot_id": "snapshot",
        "snapshot_hash": "hash",
        "candles_by_symbol": {symbol: pd.DataFrame() for symbol in FROZEN_E4_UNIVERSE},
    })()
    evidence = Evidence()
    service = E4PrecomputeService(
        _build_e4_config(),
        evidence_recorder=evidence,
        now_fn=lambda: DECISION_AT + timedelta(seconds=10),
        snapshot_provider=lambda _decision_at: snapshot,
    )
    service._guard = Guard()
    result = service._run_cycle(DECISION_AT)
    assert result.error == "score failure"
    assert service.last_cycle is None
    assert service.lookup("BTCUSDT", "LONG", DECISION_AT) is None
    assert evidence.entries == []


def test_latency_metrics_are_measured(real_e4_cycle):
    metrics = real_e4_cycle[4]
    assert metrics["cold_start_ms"] > 0
    assert metrics["incremental_snapshot_ms"] > 0
    assert metrics["panel_build_ms"] > 0
    assert metrics["score_22_ms"] > 0
    assert metrics["api_cache_lookup_ms"] > 0
    print(json.dumps(metrics, sort_keys=True))
