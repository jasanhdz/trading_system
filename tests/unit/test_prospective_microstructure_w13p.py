import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from aegis.research.prospective_microstructure_w13p import (
    CollectorConfig,
    JournalTail,
    LocalOrderBook,
    MarketEvent,
    ParquetBatchWriter,
    PassiveCaptureCore,
    W13PSidecar,
    _write_parquet_part,
    progress_report,
)


def config(tmp_path: Path, **overrides) -> CollectorConfig:
    values = dict(
        symbols=("BTCUSDT",),
        signal_journal=tmp_path / "signals.jsonl",
        storage_root=tmp_path / "capture",
        public_websocket_url="wss://fstream.binance.com/public/stream",
        public_snapshot_url="https://fapi.binance.com/fapi/v1/depth",
        pre_signal_seconds=30,
        post_signal_seconds=180,
        ring_max_events_per_symbol=1000,
        market_queue_max_events=2,
        disk_queue_max_records=100,
        parquet_batch_rows=10,
        parquet_flush_seconds=1,
        minimum_free_disk_gb=0,
        maximum_collection_gb=1,
    )
    values.update(overrides)
    return CollectorConfig(**values)


def event(stamp_us: int, kind: str = "QUOTE", suffix: str = "0", valid: bool = True) -> MarketEvent:
    payload = {"b": "100", "a": "101"} if kind == "QUOTE" else {"x": suffix}
    return MarketEvent(
        f"{kind}:{stamp_us}:{suffix}", kind, "BTCUSDT", stamp_us, stamp_us,
        stamp_us + 100, stamp_us * 1000, payload, valid, 1,
    )


def envelope(t0_iso: str = "2026-08-15T12:00:00.000Z") -> dict:
    return {
        "schema_id": "aegis-prospective-signal-evidence-v1",
        "prospective_signal_id": "a" * 64,
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "signal_timestamp_utc": t0_iso,
        "model_identity": "frozen-model",
        "model_artifact_hash": "b" * 64,
        "configuration_hash": "c" * 64,
        "source_python_commit": "d" * 40,
        "source_typescript_commit": "e" * 40,
        "upstream_model": {"short_probability": 0.8},
        "component_evidence": {},
        "final_decision": {"action": "ENTER_NOW", "reason_codes": ["TEST"]},
    }


def test_local_book_snapshot_sequence_duplicate_gap_cross_and_resync():
    book = LocalOrderBook("BTCUSDT")
    book.buffer({"U": 100, "u": 101, "pu": 99, "b": [["100", "2"]], "a": []})
    assert book.install_snapshot({"lastUpdateId": 100, "bids": [["99", "1"]], "asks": [["101", "1"]]})
    assert book.apply({"U": 102, "u": 102, "pu": 101, "b": [], "a": [["101", "2"]]})
    assert book.apply({"U": 102, "u": 102, "pu": 101, "b": [], "a": []})
    assert book.integrity.duplicates == 1
    assert not book.apply({"U": 104, "u": 104, "pu": 103, "b": [], "a": []})
    assert book.integrity.gaps == 1
    book.buffer({"U": 200, "u": 201, "pu": 199, "b": [["102", "1"]], "a": [["101", "1"]]})
    assert not book.install_snapshot({"lastUpdateId": 200, "bids": [["100", "1"]], "asks": [["101", "1"]]})
    assert book.integrity.crossed_books == 1
    assert not book.integrity.valid


def test_signal_snapshot_ring_window_overlap_and_quality(tmp_path: Path, monkeypatch):
    emitted = []
    cfg = config(tmp_path, ring_max_events_per_symbol=5000)
    core = PassiveCaptureCore(cfg, lambda kind, row: emitted.append((kind, row)) or True)
    t0 = 1_765_800_000_000_000
    monkeypatch.setattr("aegis.research.prospective_microstructure_w13p._utc_us", lambda _: t0)
    for second in range(-30, 1):
        for kind in ("BOOK", "QUOTE", "TRADE"):
            core.observe_event(event(t0 + second * 1_000_000, kind, str(second)))
    first = envelope()
    assert core.observe_signal(first) == "a" * 64
    second = envelope()
    second["prospective_signal_id"] = "f" * 64
    assert core.observe_signal(second) == "f" * 64
    for second in range(1, 181):
        for kind in ("BOOK", "QUOTE", "TRADE"):
            core.observe_event(event(t0 + second * 1_000_000, kind, str(second)))
    quality = core.finalize(t0 + 180_000_000)
    assert len(quality) == 2
    assert all(row["W13_ELIGIBLE"] for row in quality)
    event_rows = [row for kind, row in emitted if kind == "EVENT"]
    assert len(event_rows) == 211 * 3  # shared once despite two logical windows
    snapshots = [row for kind, row in emitted if kind == "SIGNAL"]
    assert snapshots[0]["reference_mid"] == 100.5
    assert snapshots[0]["financial_mutation_capability"] is False
    assert snapshots[0]["authenticated_exchange_access"] is False
    assert snapshots[0]["open_position_state"] == "NOT_COLLECTED_PUBLIC_ONLY"
    without_hash = dict(snapshots[0])
    digest = without_hash.pop("signal_snapshot_hash")
    from aegis.research.prospective_microstructure_w13p import _sha256
    assert digest == _sha256(without_hash)


def test_drop_reconnect_invalidates_without_blocking(tmp_path: Path, monkeypatch):
    cfg = config(tmp_path)
    core = PassiveCaptureCore(cfg, lambda _kind, _row: False)
    t0 = 1_765_800_000_000_000
    monkeypatch.setattr("aegis.research.prospective_microstructure_w13p._utc_us", lambda _: t0)
    core.observe_event(event(t0 - 30_000_000, "QUOTE"))
    core.observe_signal(envelope())
    core.observe_event(event(t0, "BOOK"))
    core.note_reconnect()
    assert core.drop_count > 0
    assert next(iter(core.active.values())).invalid_book_seen


def test_journal_tail_starts_at_end_and_handles_partial_line(tmp_path: Path):
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps({"old": True}) + "\n")
    tail = JournalTail(path, start_at_end=True)
    with path.open("a") as handle:
        handle.write('{"new":')
    assert tail.poll() == []
    with path.open("a") as handle:
        handle.write("true}\n")
    assert tail.poll() == [{"new": True}]


def test_journal_checkpoint_resumes_at_completed_line(tmp_path: Path):
    path = tmp_path / "signals.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    path.write_text('{"one":1}\n')
    tail = JournalTail(path, start_at_end=False, checkpoint_path=checkpoint)
    assert tail.poll() == [{"one": 1}]
    with path.open("a") as handle:
        handle.write('{"two":2}\n')
    resumed = JournalTail(path, start_at_end=True, checkpoint_path=checkpoint)
    assert resumed.poll() == [{"two": 2}]


def test_parquet_writer_bounded_queue_and_progress(tmp_path: Path):
    asyncio.run(_exercise_parquet_writer(tmp_path))


async def _exercise_parquet_writer(tmp_path: Path):
    writer = ParquetBatchWriter(tmp_path, max_queue=1, batch_rows=1, flush_seconds=1)
    assert writer.submit("QUALITY", {"signal_id": "a", "symbol": "BTCUSDT", "side": "SHORT", "signal_timestamp_us": 1_765_800_000_000_000, "W13_ELIGIBLE": True})
    assert not writer.submit("QUALITY", {"signal_id": "b"})
    assert writer.dropped == 1
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.05)
    await writer.queue.put(None)
    await task
    report = progress_report(tmp_path)
    assert report["eligible_signals"] == 1
    assert report["total_signals_captured"] == 0
    assert report["direction_counts"] == {}
    assert report["final_holdout"] == "SEALED_NOT_OPENED"


def test_parquet_parts_are_partitioned_by_symbol(tmp_path: Path):
    rows = [
        {"symbol": "BTCUSDT", "signal_timestamp_us": 1_765_800_000_000_000, "value": 1},
        {"symbol": "ETHUSDT", "signal_timestamp_us": 1_765_800_000_000_000, "value": 2},
    ]
    assert _write_parquet_part(tmp_path, "SIGNAL", rows) == 2
    assert len(list((tmp_path / "signal").rglob("*.parquet"))) == 2


def test_disk_pressure_stops_collector_only(tmp_path: Path):
    sidecar = W13PSidecar(config(tmp_path, minimum_free_disk_gb=10**9), consume_signals=False)
    with pytest.raises(RuntimeError, match="W13P_DISK_SAFETY_STOP"):
        sidecar._assert_disk_safe()
    assert sidecar.stop_event.is_set()


def test_sidecar_crash_is_process_isolated(tmp_path: Path):
    marker = tmp_path / "trading-process-still-running"
    marker.write_text("alive")
    result = subprocess.run(
        [sys.executable, "-c", "raise SystemExit(27)"], capture_output=True, check=False
    )
    assert result.returncode == 27
    assert marker.read_text() == "alive"


def test_source_has_public_allowlist_and_no_financial_client():
    source_path = Path("src/aegis/research/prospective_microstructure_w13p.py")
    source = source_path.read_text().lower()
    assert "fstream.binance.com" in source
    assert '"/fapi/v1/depth"' in source
    prohibited = (
        "createorder", "cancelorder", "cancel_all_orders", "fapi/v1/order",
        "fapi/v2/account", "fapi/v2/positionrisk", "api_secret", "secret_key",
    )
    assert not any(token in source for token in prohibited)


def test_non_public_urls_and_unregistered_window_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="NON_PUBLIC"):
        config(tmp_path, public_websocket_url="wss://example.com/ws").validate()
    with pytest.raises(ValueError, match="NON_PUBLIC"):
        config(tmp_path, public_websocket_url="wss://fstream.binance.com.evil.invalid/public/stream").validate()
    with pytest.raises(ValueError, match="PREREGISTERED"):
        config(tmp_path, pre_signal_seconds=29).validate()
