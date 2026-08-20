from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

from aegis.risk_guard import bootstrap


class Response:
    def __init__(self, status: int, payload=None, headers=None):
        self.status_code = status
        self._payload = [] if payload is None else payload
        self.headers = headers or {}

    def json(self):
        return self._payload


def candles(start_ms: int, count: int) -> pd.DataFrame:
    timestamps = [start_ms + index * 60_000 for index in range(count)]
    return pd.DataFrame({
        "open_time_ms": timestamps,
        "open": [10.0] * count,
        "high": [11.0] * count,
        "low": [9.0] * count,
        "close": [10.5] * count,
        "volume": [2.0] * count,
        "taker_buy_volume": [1.0] * count,
    })


@pytest.mark.parametrize("status,counter", [(429, "429_count"), (500, None)])
def test_transient_http_retries(status, counter):
    responses = iter([Response(status), Response(200, [[0] * 12])])
    sleeps = []
    client = bootstrap.RateLimitedRestClient(
        http_get=lambda *_args, **_kwargs: next(responses), sleep=sleeps.append,
        jitter=lambda: 0.0, now=lambda: 0.0, gap_seconds=0.0,
    )
    assert client.get_json("https://example.test", params={}) == [[0] * 12]
    assert client.counters["retries"] == 1
    if counter:
        assert client.counters[counter] == 1


def test_418_uses_conservative_delay_and_retry_after_case_insensitive():
    responses = iter([
        Response(418, headers={"rEtRy-AfTeR": "180"}), Response(200),
    ])
    sleeps = []
    client = bootstrap.RateLimitedRestClient(
        http_get=lambda *_args, **_kwargs: next(responses), sleep=sleeps.append,
        jitter=lambda: 0.0, now=lambda: 0.0, gap_seconds=0.0, ban_delay=120,
    )
    client.get_json("https://example.test", params={})
    assert sleeps == [180.0]
    assert client.counters["418_count"] == 1


@pytest.mark.parametrize("exception", [requests.Timeout(), requests.ConnectionError()])
def test_network_errors_retry(exception):
    calls = iter([exception, Response(200)])

    def get(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    client = bootstrap.RateLimitedRestClient(
        http_get=get, sleep=lambda _delay: None, jitter=lambda: 0.0,
        now=lambda: 0.0, gap_seconds=0.0,
    )
    client.get_json("https://example.test", params={})
    assert client.counters == {"requests": 2, "retries": 1, "429_count": 0, "418_count": 0}


def test_fatal_4xx_is_not_retried():
    client = bootstrap.RateLimitedRestClient(
        http_get=lambda *_args, **_kwargs: Response(400), sleep=lambda _delay: None,
        gap_seconds=0.0,
    )
    with pytest.raises(bootstrap.BootstrapError, match="REST_FATAL_HTTP:400"):
        client.get_json("https://example.test", params={})
    assert client.counters["retries"] == 0


def test_retry_exhaustion_is_bounded():
    client = bootstrap.RateLimitedRestClient(
        http_get=lambda *_args, **_kwargs: Response(503), sleep=lambda _delay: None,
        jitter=lambda: 0.0, now=lambda: 0.0, max_retries=2, gap_seconds=0.0,
    )
    with pytest.raises(bootstrap.BootstrapError, match="REST_RETRY_EXHAUSTED:503"):
        client.get_json("https://example.test", params={})
    assert client.counters["requests"] == 3
    assert client.counters["retries"] == 2


def test_conflicting_overlap_rejected_and_identical_is_idempotent():
    original = candles(0, 2)
    merged, duplicates = bootstrap.merge_candles(original, original.copy())
    assert len(merged) == 2
    assert duplicates == 2
    changed = original.copy()
    changed.loc[0, "close"] = 99
    with pytest.raises(bootstrap.BootstrapError, match="CONFLICTING_CANDLE"):
        bootstrap.merge_candles(original, changed)


def test_atomic_write_cleans_temp_on_failure(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        bootstrap.atomic_write_parquet(tmp_path / "month.parquet", candles(0, 1))
    assert list(tmp_path.iterdir()) == []


def write_universe(seed_root: Path, durable_root: Path, target_ms: int) -> None:
    seed_root.mkdir()
    for symbol in bootstrap.UNIVERSE:
        candles(target_ms - 120_000, 2).to_parquet(seed_root / f"{symbol}_1m.parquet", index=False)
        symbol_root = durable_root / symbol
        symbol_root.mkdir(parents=True)
        candles(target_ms, 1).to_parquet(symbol_root / "2026-01.parquet", index=False)


def test_global_validation_requires_all_11_and_detects_gap_duplicate_and_target(tmp_path):
    seed = tmp_path / "seed"
    durable = tmp_path / "durable"
    decision = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    target_ms = int(decision.timestamp() * 1000) - 60_000
    write_universe(seed, durable, target_ms)
    assert bootstrap.validate_bootstrap(seed, durable, decision)["ready"] is True

    missing = seed / f"{bootstrap.UNIVERSE[0]}_1m.parquet"
    missing.unlink()
    result = bootstrap.validate_bootstrap(seed, durable, decision)
    assert result["ready"] is False
    assert "SEED_MISSING" in result["errors"][bootstrap.UNIVERSE[0]]


@pytest.mark.parametrize("defect", ["gap", "duplicate", "past_target"])
def test_validation_rejects_corrupt_cache(defect, tmp_path):
    seed = tmp_path / "seed"
    durable = tmp_path / "durable"
    decision = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    target_ms = int(decision.timestamp() * 1000) - 60_000
    write_universe(seed, durable, target_ms)
    symbol = bootstrap.UNIVERSE[0]
    path = durable / symbol / "2026-01.parquet"
    frame = pd.read_parquet(path)
    if defect == "gap":
        frame["open_time_ms"] += 60_000
    elif defect == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    else:
        frame = pd.concat([frame, candles(target_ms + 60_000, 1)], ignore_index=True)
    frame.to_parquet(path, index=False)
    result = bootstrap.validate_bootstrap(seed, durable, decision)
    assert result["ready"] is False
    assert symbol in result["errors"]


def test_validate_only_never_constructs_network_clients(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "RateLimitedRestClient", lambda: pytest.fail("network client constructed"))
    monkeypatch.setattr(bootstrap, "BinancePublicArchiveClient", lambda: pytest.fail("archive client constructed"))
    assert bootstrap.main(["--validate-only", "--seed-root", str(tmp_path), "--durable-root", str(tmp_path)]) == 1
    assert '"ready": false' in capsys.readouterr().out


def test_status_does_not_modify_root(tmp_path, capsys):
    root = tmp_path / "absent"
    assert bootstrap.main(["--status", "--durable-root", str(root)]) == 1
    assert not root.exists()
    assert "NOT_STARTED" in capsys.readouterr().out


def test_target_alignment():
    now = datetime(2026, 8, 20, 12, 17, 59, 999, tzinfo=timezone.utc)
    assert bootstrap.target_decision_at(now) == datetime(2026, 8, 20, 12, 15, tzinfo=timezone.utc)


def test_resume_and_partial_month_persist_each_page(tmp_path, monkeypatch):
    seed_root = tmp_path / "seed"
    durable_root = tmp_path / "durable"
    seed_root.mkdir()
    symbol = "BTCUSDT"
    start = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles(start, 1).to_parquet(seed_root / f"{symbol}_1m.parquet", index=False)
    target = datetime.fromtimestamp((start + 5 * 60_000) / 1000, tz=timezone.utc)
    raw = []
    for row in candles(start + 60_000, 4).itertuples(index=False):
        raw.append([row.open_time_ms, row.open, row.high, row.low, row.close, row.volume, 0, 0, 0, row.taker_buy_volume, 0, 0])
    rest = bootstrap.RateLimitedRestClient(
        http_get=lambda *_args, **_kwargs: Response(200, raw), sleep=lambda _delay: None,
        now=lambda: 0.0, gap_seconds=0.0,
    )
    runner = bootstrap.Bootstrapper(seed_root, durable_root, target=target, rest=rest)
    runner.sync_symbol(symbol)
    assert len(pd.read_parquet(durable_root / symbol / "2026-08.parquet")) == 4
    first_requests = rest.counters["requests"]
    state = runner.manifest["symbols"][symbol]
    assert state["requests"] == 1
    assert state["updated_at"]
    assert state["target_timestamp"]
    runner.sync_symbol(symbol)
    assert rest.counters["requests"] == first_requests
    assert runner.manifest["symbols"][symbol]["status"] == "COMPLETE"
