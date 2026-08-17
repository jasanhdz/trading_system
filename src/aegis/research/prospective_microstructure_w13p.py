"""Passive public-market collector for W13-P.

This module has no trading imports, credentials, private endpoints, or financial
mutation capability. It tails an append-only evidence journal from a separate
process and fails by losing/invalidating research data, never by blocking Aegis.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import signal
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCHEMA_VERSION = "aegis-w13p-v1"
SIGNAL_SCHEMA = "aegis-prospective-signal-evidence-v1"
CURRENT_QUALITY_GATE_VERSION = "w13p-quality-v2-l2-base-snapshot"
PUBLIC_WS_HOST = "fstream.binance.com"
PUBLIC_SNAPSHOT_PATH = "/fapi/v1/depth"
EVENT_TYPES = frozenset({"BOOK", "QUOTE", "TRADE"})


def _rate_limit_delay_seconds(headers: Mapping[str, Any], body: str, now_ms: int) -> float:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    delays = [60.0]
    try:
        delays.append(float(retry_after))
    except (TypeError, ValueError):
        pass
    match = re.search(r"banned until\s+(\d+)", body)
    if match:
        delays.append(max(0.0, (int(match.group(1)) - now_ms) / 1000.0))
    return max(delays)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _utc_us(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000)


def _now_wall_us() -> int:
    return time.time_ns() // 1_000


@dataclass(frozen=True)
class CollectorConfig:
    symbols: tuple[str, ...]
    signal_journal: Path
    storage_root: Path
    public_websocket_url: str
    public_snapshot_url: str
    depth_snapshot_limit: int = 1000
    depth_stream_interval: str = "100ms"
    pre_signal_seconds: int = 30
    post_signal_seconds: int = 180
    ring_retention_seconds: int = 90
    ring_max_events_per_symbol: int = 500_000
    market_queue_max_events: int = 250_000
    disk_queue_max_records: int = 250_000
    parquet_batch_rows: int = 50_000
    parquet_flush_seconds: int = 30
    minimum_free_disk_gb: float = 100.0
    maximum_collection_gb: float = 100.0
    maximum_quality_gap_ms: int = 1000
    enabled: bool = False
    phase: str = "OFFLINE_SYNTHETIC"

    @classmethod
    def load(cls, path: Path) -> "CollectorConfig":
        raw = yaml.safe_load(path.read_text())
        resolved = path.resolve()
        repo = resolved.parents[2] if len(resolved.parents) > 2 else Path.cwd()
        values = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        values["symbols"] = tuple(str(x).upper() for x in values["symbols"])
        for key in ("signal_journal", "storage_root"):
            candidate = Path(values[key])
            values[key] = candidate if candidate.is_absolute() else repo / candidate
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("W13P_SYMBOL_UNIVERSE_INVALID")
        if self.pre_signal_seconds != 30 or self.post_signal_seconds != 180:
            raise ValueError("W13P_CAPTURE_WINDOW_MUST_REMAIN_PREREGISTERED")
        if self.ring_retention_seconds < self.pre_signal_seconds + 30:
            raise ValueError("W13P_RING_RETENTION_INSUFFICIENT_FOR_JOURNAL_DELAY")
        websocket = urlparse(self.public_websocket_url)
        snapshot = urlparse(self.public_snapshot_url)
        if websocket.scheme != "wss" or websocket.hostname != PUBLIC_WS_HOST or websocket.path != "/public/stream":
            raise ValueError("W13P_NON_PUBLIC_WEBSOCKET_PROHIBITED")
        if snapshot.scheme != "https" or snapshot.hostname != "fapi.binance.com" or snapshot.path != PUBLIC_SNAPSHOT_PATH:
            raise ValueError("W13P_NON_PUBLIC_SNAPSHOT_PROHIBITED")
        if min(self.market_queue_max_events, self.disk_queue_max_records) <= 0:
            raise ValueError("W13P_BOUNDED_QUEUE_REQUIRED")


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    event_type: str
    symbol: str
    exchange_event_timestamp_us: int
    exchange_trade_timestamp_us: int | None
    local_receive_wall_timestamp_us: int
    local_receive_monotonic_ns: int
    payload: Mapping[str, Any]
    book_valid: bool
    book_generation: int

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES or self.symbol != self.symbol.upper():
            raise ValueError("W13P_MARKET_EVENT_INVALID")
        if self.exchange_event_timestamp_us <= 0 or self.local_receive_wall_timestamp_us <= 0:
            raise ValueError("W13P_MARKET_TIMESTAMP_INVALID")

    def storage_row(self, capture_segment_id: str) -> dict[str, Any]:
        return {
            "schema_id": f"{SCHEMA_VERSION}-market-event",
            "capture_segment_id": capture_segment_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "symbol": self.symbol,
            "exchange_event_timestamp_us": self.exchange_event_timestamp_us,
            "exchange_trade_timestamp_us": self.exchange_trade_timestamp_us,
            "local_receive_wall_timestamp_us": self.local_receive_wall_timestamp_us,
            "local_receive_monotonic_ns": self.local_receive_monotonic_ns,
            "collector_write_timestamp_us": _now_wall_us(),
            "book_valid": self.book_valid,
            "book_generation": self.book_generation,
            "payload_json": _canonical_json(self.payload),
        }


@dataclass
class BookIntegrity:
    generation: int = 0
    valid: bool = False
    last_update_id: int | None = None
    gaps: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    crossed_books: int = 0
    resyncs: int = 0


class LocalOrderBook:
    """Binance USD-M diff-book reconstruction with fail-closed continuity."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.bids: dict[str, str] = {}
        self.asks: dict[str, str] = {}
        self.integrity = BookIntegrity()
        self._buffer: deque[Mapping[str, Any]] = deque(maxlen=200_000)
        self._snapshot_ready = False

    def buffer(self, event: Mapping[str, Any]) -> None:
        self._buffer.append(dict(event))

    @property
    def needs_snapshot(self) -> bool:
        return not self.integrity.valid and not self._snapshot_ready

    def install_snapshot(self, snapshot: Mapping[str, Any]) -> bool:
        last = int(snapshot["lastUpdateId"])
        self.bids = {str(p): str(q) for p, q in snapshot.get("bids", ()) if float(q) != 0}
        self.asks = {str(p): str(q) for p, q in snapshot.get("asks", ()) if float(q) != 0}
        self.integrity.generation += 1
        self.integrity.resyncs += 1
        self.integrity.valid = False
        self.integrity.last_update_id = last
        self._snapshot_ready = True
        pending = [event for event in self._buffer if int(event["u"]) >= last]
        self._buffer.clear()
        started = False
        for event in pending:
            if not started:
                if int(event["U"]) <= last + 1 <= int(event["u"]):
                    self._apply_levels(event)
                    self.integrity.last_update_id = int(event["u"])
                    started = True
                    crossed = self._crossed()
                    if crossed:
                        self.integrity.crossed_books += 1
                    self.integrity.valid = not crossed
                continue
            if not self.apply(event):
                return False
        return self.integrity.valid

    def apply(self, event: Mapping[str, Any]) -> bool:
        update_id = int(event["u"])
        previous = self.integrity.last_update_id
        if previous is not None and not self.integrity.valid and self._snapshot_ready:
            if update_id < previous:
                self.integrity.duplicates += 1
                return False
            if int(event["U"]) <= previous + 1 <= update_id:
                self._apply_levels(event)
                self.integrity.last_update_id = update_id
                self.integrity.valid = not self._crossed()
                return self.integrity.valid
            if int(event["U"]) > previous:
                self._snapshot_ready = False
        if previous is None or not self.integrity.valid:
            self.buffer(event)
            return False
        if update_id <= previous:
            self.integrity.duplicates += 1
            return True
        pu = int(event.get("pu", -1))
        if pu != previous:
            if update_id < previous:
                self.integrity.out_of_order += 1
            else:
                self.integrity.gaps += 1
            self.invalidate()
            self.buffer(event)
            return False
        self._apply_levels(event)
        self.integrity.last_update_id = update_id
        if self._crossed():
            self.integrity.crossed_books += 1
            self.invalidate()
            return False
        return True

    def invalidate(self) -> None:
        self.integrity.valid = False
        self._snapshot_ready = False

    def _apply_levels(self, event: Mapping[str, Any]) -> None:
        for side, key in ((self.bids, "b"), (self.asks, "a")):
            for price, quantity in event.get(key, ()):
                if float(quantity) == 0:
                    side.pop(str(price), None)
                else:
                    side[str(price)] = str(quantity)

    def _crossed(self) -> bool:
        return bool(self.bids and self.asks and max(map(float, self.bids)) >= min(map(float, self.asks)))

    def checkpoint(self, exchange_timestamp_us: int) -> dict[str, Any]:
        return {
            "exchange_timestamp_us": exchange_timestamp_us,
            "last_update_id": self.integrity.last_update_id,
            "generation": self.integrity.generation,
            "valid": self.integrity.valid,
            "bids": sorted(((price, quantity) for price, quantity in self.bids.items()), key=lambda x: float(x[0]), reverse=True),
            "asks": sorted(((price, quantity) for price, quantity in self.asks.items()), key=lambda x: float(x[0])),
        }


@dataclass
class SignalWindow:
    signal_id: str
    symbol: str
    side: str
    t0_us: int
    start_us: int
    end_us: int
    snapshot_hash: str
    capture_segment_id: str
    metadata: dict[str, Any]
    capture_start_us: int
    l2_base_snapshot_valid: bool
    event_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_event_us: int | None = None
    last_event_us: int | None = None
    max_gap_us: int = 0
    reconnects: int = 0
    drops_at_start: int = 0
    invalid_book_seen: bool = False


class PassiveCaptureCore:
    """Deterministic, bounded capture core with shared streams for overlaps."""

    def __init__(self, config: CollectorConfig, emit: Callable[[str, dict[str, Any]], bool]) -> None:
        self.config = config
        self.emit = emit
        self.rings: dict[str, deque[MarketEvent]] = {
            symbol: deque() for symbol in config.symbols
        }
        self.ring_type_counts: dict[str, dict[str, int]] = {
            symbol: defaultdict(int) for symbol in config.symbols
        }
        self.active: dict[str, SignalWindow] = {}
        self.persisted: dict[str, int] = {}
        self.persisted_order: deque[tuple[int, str]] = deque()
        self.drop_count = 0
        self.reconnect_count = 0

    def observe_event(self, event: MarketEvent) -> None:
        ring = self.rings.get(event.symbol)
        if ring is None:
            return
        if len(ring) >= self.config.ring_max_events_per_symbol:
            evicted = ring.popleft()
            self.ring_type_counts[event.symbol][evicted.event_type] -= 1
        ring.append(event)
        self.ring_type_counts[event.symbol][event.event_type] += 1
        cutoff = event.exchange_event_timestamp_us - self.config.ring_retention_seconds * 1_000_000
        while ring and ring[0].exchange_event_timestamp_us < cutoff:
            evicted = ring.popleft()
            self.ring_type_counts[event.symbol][evicted.event_type] -= 1
        self._prune_persisted(cutoff)
        active = [w for w in self.active.values() if w.symbol == event.symbol and w.start_us <= event.exchange_event_timestamp_us <= w.end_us]
        if not active:
            return
        self._persist_event_once(event, min(active, key=lambda w: w.t0_us).capture_segment_id)
        for window in active:
            self._account(window, event)

    def observe_signal(
        self,
        envelope: Mapping[str, Any],
        *,
        l2_base_snapshot: Mapping[str, Any] | None = None,
    ) -> str | None:
        if envelope.get("schema_id") != SIGNAL_SCHEMA:
            raise ValueError("W13P_SIGNAL_SCHEMA_INVALID")
        if envelope.get("final_decision", {}).get("action") != "ENTER_NOW":
            return None
        side = str(envelope.get("side", "")).upper()
        symbol = str(envelope.get("symbol", "")).upper()
        if side not in {"LONG", "SHORT"} or symbol not in self.rings:
            raise ValueError("W13P_SIGNAL_DIRECTION_OR_SYMBOL_INVALID")
        signal_id = str(envelope["prospective_signal_id"])
        if signal_id in self.active:
            return signal_id
        t0 = _utc_us(str(envelope["signal_timestamp_utc"]))
        observed_wall = _now_wall_us()
        snapshot = {
            "schema_id": f"{SCHEMA_VERSION}-signal-snapshot",
            "signal_id": signal_id,
            "signal_timestamp_us": t0,
            "collector_observed_wall_timestamp_us": observed_wall,
            "collector_observed_monotonic_ns": time.monotonic_ns(),
            "collector_observation_delay_us": observed_wall - t0,
            "symbol": symbol,
            "side": side,
            "source_envelope_json": _canonical_json(envelope),
            "model_identity": str(envelope.get("model_identity", "")),
            "model_artifact_hash": str(envelope.get("model_artifact_hash", "")),
            "configuration_hash": str(envelope.get("configuration_hash", "")),
            "source_python_commit": str(envelope.get("source_python_commit", "")),
            "source_typescript_commit": str(envelope.get("source_typescript_commit", "")),
            "reason_codes_json": _canonical_json(envelope.get("final_decision", {}).get("reason_codes", [])),
            "upstream_model_json": _canonical_json(envelope.get("upstream_model", {})),
            "component_evidence_json": _canonical_json(envelope.get("component_evidence", {})),
            "open_position_state": "NOT_COLLECTED_PUBLIC_ONLY",
            "pending_order_state": "NOT_COLLECTED_PUBLIC_ONLY",
            "collector_schema_version": SCHEMA_VERSION,
            "capture_only": True,
            "financial_mutation_capability": False,
            "authenticated_exchange_access": False,
            "l2_base_snapshot_timestamp_us": (
                int(l2_base_snapshot["exchange_timestamp_us"])
                if l2_base_snapshot is not None else None
            ),
            "l2_base_snapshot_last_update_id": (
                int(l2_base_snapshot["last_update_id"])
                if l2_base_snapshot is not None and l2_base_snapshot.get("last_update_id") is not None else None
            ),
            "l2_base_snapshot_generation": (
                int(l2_base_snapshot["generation"])
                if l2_base_snapshot is not None else None
            ),
            "l2_base_snapshot_valid": bool(
                l2_base_snapshot is not None and l2_base_snapshot.get("valid")
            ),
            "l2_base_bids_json": _canonical_json(l2_base_snapshot.get("bids", [])) if l2_base_snapshot else "[]",
            "l2_base_asks_json": _canonical_json(l2_base_snapshot.get("asks", [])) if l2_base_snapshot else "[]",
        }
        quote = next(
            (
                event for event in reversed(self.rings[symbol])
                if event.event_type == "QUOTE" and event.exchange_event_timestamp_us <= t0
            ),
            None,
        )
        if quote is None:
            snapshot.update({"reference_bid": None, "reference_ask": None, "reference_mid": None})
        else:
            bid = float(quote.payload["b"])
            ask = float(quote.payload["a"])
            snapshot.update({"reference_bid": bid, "reference_ask": ask, "reference_mid": (bid + ask) / 2})
        snapshot_hash = _sha256(snapshot)
        snapshot["signal_snapshot_hash"] = snapshot_hash
        segment = f"seg-{symbol}-{t0}-{uuid.uuid4().hex[:12]}"
        logical_start = t0 - self.config.pre_signal_seconds * 1_000_000
        snapshot_timestamp = (
            int(l2_base_snapshot["exchange_timestamp_us"])
            if l2_base_snapshot is not None else logical_start
        )
        snapshot_valid = bool(
            l2_base_snapshot is not None
            and l2_base_snapshot.get("valid")
            and snapshot_timestamp <= logical_start
        )
        window = SignalWindow(
            signal_id, symbol, side, t0,
            logical_start,
            t0 + self.config.post_signal_seconds * 1_000_000,
            snapshot_hash, segment, snapshot, snapshot_timestamp, snapshot_valid,
            drops_at_start=self.drop_count,
        )
        self.active[signal_id] = window
        if not self.emit("SIGNAL", snapshot):
            self.note_drop()
            window.invalid_book_seen = True
        for event in self.rings[symbol]:
            if window.capture_start_us <= event.exchange_event_timestamp_us <= t0:
                self._persist_event_once(event, segment)
            if window.start_us <= event.exchange_event_timestamp_us <= t0:
                self._account(window, event)
        return signal_id

    def finalize(self, through_us: int) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for signal_id, window in list(self.active.items()):
            if through_us < window.end_us:
                continue
            pre_complete = window.first_event_us is not None and window.first_event_us <= window.start_us + 1_500_000
            # Streams are event-driven: a quiet interval is not data loss. The
            # window is complete when capture survived through its deadline;
            # reconnects, sequence loss and queue drops are checked separately.
            post_complete = window.last_event_us is not None and window.last_event_us > window.t0_us
            quote_ok = window.event_counts.get("QUOTE", 0) > 0
            trade_ok = window.event_counts.get("TRADE", 0) > 0
            eligible = all((pre_complete, post_complete, quote_ok, trade_ok, window.l2_base_snapshot_valid, not window.invalid_book_seen, self.drop_count == window.drops_at_start))
            quality = {
                "schema_id": f"{SCHEMA_VERSION}-signal-quality",
                "signal_id": signal_id,
                "symbol": window.symbol,
                "side": window.side,
                "window_start_us": window.start_us,
                "signal_timestamp_us": window.t0_us,
                "window_end_us": window.end_us,
                "pre_window_complete": pre_complete,
                "post_window_complete": post_complete,
                "l2_sequence_valid": not window.invalid_book_seen,
                "quote_coverage": quote_ok,
                "trade_coverage": trade_ok,
                "l2_base_snapshot_valid": window.l2_base_snapshot_valid,
                "l2_base_snapshot_timestamp_us": window.capture_start_us,
                "max_gap_ms": window.max_gap_us / 1000,
                "event_counts_json": _canonical_json(dict(window.event_counts)),
                "reconnect_during_window": window.reconnects > 0,
                "disk_drop_count": self.drop_count - window.drops_at_start,
                "W13_ELIGIBLE": eligible,
                "quality_gate_version": CURRENT_QUALITY_GATE_VERSION,
            }
            if not self.emit("QUALITY", quality):
                self.note_drop()
            completed.append(quality)
            del self.active[signal_id]
        return completed

    def note_reconnect(self) -> None:
        self.reconnect_count += 1
        for window in self.active.values():
            window.reconnects += 1
            window.invalid_book_seen = True

    def note_drop(self, count: int = 1) -> None:
        self.drop_count += count

    def _account(self, window: SignalWindow, event: MarketEvent) -> None:
        stamp = event.exchange_event_timestamp_us
        if window.last_event_us is not None:
            window.max_gap_us = max(window.max_gap_us, stamp - window.last_event_us)
        window.first_event_us = stamp if window.first_event_us is None else min(window.first_event_us, stamp)
        window.last_event_us = stamp if window.last_event_us is None else max(window.last_event_us, stamp)
        window.event_counts[event.event_type] += 1
        if event.event_type == "BOOK" and not event.book_valid:
            window.invalid_book_seen = True

    def _persist_event_once(self, event: MarketEvent, segment: str) -> None:
        if event.event_id in self.persisted:
            return
        if not self.emit("EVENT", event.storage_row(segment)):
            self.note_drop()
            return
        stamp = event.exchange_event_timestamp_us
        self.persisted[event.event_id] = stamp
        self.persisted_order.append((stamp, event.event_id))

    def _prune_persisted(self, cutoff: int) -> None:
        while self.persisted_order and self.persisted_order[0][0] < cutoff:
            stamp, event_id = self.persisted_order.popleft()
            if self.persisted.get(event_id) == stamp:
                del self.persisted[event_id]


class ParquetBatchWriter:
    """Bounded asynchronous sink; each flush creates a coarse compressed part."""

    def __init__(self, root: Path, max_queue: int, batch_rows: int, flush_seconds: int) -> None:
        self.root = root
        self.queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue(maxsize=max_queue)
        self.batch_rows = batch_rows
        self.flush_seconds = flush_seconds
        self.dropped = 0
        self.written = 0

    def submit(self, kind: str, row: dict[str, Any]) -> bool:
        try:
            self.queue.put_nowait((kind, row))
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def run(self) -> None:
        batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        last_flush = time.monotonic()
        while True:
            timeout = max(0.1, self.flush_seconds - (time.monotonic() - last_flush))
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except TimeoutError:
                item = None if False else ("__FLUSH__", {})
            if item is None:
                await self._flush_all(batches)
                return
            kind, row = item
            if kind != "__FLUSH__":
                batches[kind].append(row)
            if kind == "__FLUSH__" or sum(map(len, batches.values())) >= self.batch_rows:
                await self._flush_all(batches)
                last_flush = time.monotonic()

    async def _flush_all(self, batches: dict[str, list[dict[str, Any]]]) -> None:
        pending = [(kind, rows[:]) for kind, rows in batches.items() if rows]
        batches.clear()
        for kind, rows in pending:
            # The bounded queue decouples capture from storage. This writer runs
            # in the isolated sidecar process; a slow flush may lose research
            # events but cannot delay or stop the trading process.
            count = _write_parquet_part(self.root, kind, rows)
            self.written += count


def _write_parquet_part(root: Path, kind: str, rows: list[dict[str, Any]]) -> int:
    partitions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stamp = row.get("exchange_event_timestamp_us") or row.get("signal_timestamp_us") or _now_wall_us()
        day = datetime.fromtimestamp(int(stamp) / 1_000_000, UTC).date().isoformat()
        partitions[(day, str(row.get("symbol", "ALL")))].append(row)
    for (day, symbol), partition_rows in partitions.items():
        directory = root / kind.lower() / f"date={day}" / f"symbol={symbol}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{int(time.time_ns())}-{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(pa.Table.from_pylist(partition_rows), path, compression="zstd")
    return len(rows)


class JournalTail:
    def __init__(self, path: Path, start_at_end: bool = True, checkpoint_path: Path | None = None) -> None:
        self.path = path
        self.checkpoint_path = checkpoint_path
        checkpoint = self._load_checkpoint()
        self.offset = checkpoint if checkpoint is not None else (path.stat().st_size if start_at_end and path.exists() else 0)
        self.partial = ""

    def poll(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = ""
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return []
        text = self.partial + chunk
        lines = text.split("\n")
        self.partial = lines.pop()
        rows = [json.loads(line) for line in lines if line]
        self._save_checkpoint()
        return rows

    def _load_checkpoint(self) -> int | None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        try:
            return max(0, int(json.loads(self.checkpoint_path.read_text())["offset"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".tmp")
        completed_offset = self.offset - len(self.partial.encode("utf-8"))
        temporary.write_text(_canonical_json({"schema_id": "aegis-w13p-journal-checkpoint-v1", "offset": completed_offset, "updated_at_us": _now_wall_us()}))
        temporary.replace(self.checkpoint_path)


class W13PSidecar:
    def __init__(self, config: CollectorConfig, *, consume_signals: bool = True) -> None:
        self.config = config
        self.writer = ParquetBatchWriter(config.storage_root, config.disk_queue_max_records, config.parquet_batch_rows, config.parquet_flush_seconds)
        self.core = PassiveCaptureCore(config, self.writer.submit)
        self.books = {symbol: LocalOrderBook(symbol) for symbol in config.symbols}
        self.market_queue: asyncio.Queue[tuple[dict[str, Any], int, int]] = asyncio.Queue(maxsize=config.market_queue_max_events)
        self.consume_signals = consume_signals
        self.tail = JournalTail(
            config.signal_journal,
            start_at_end=True,
            checkpoint_path=config.storage_root / "checkpoints" / "signal_journal_offset.json",
        )
        self.stop_event = asyncio.Event()
        self.snapshot_tasks: dict[str, asyncio.Task[None]] = {}
        self.resync_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max(1, len(config.symbols) * 4))
        self.resync_pending: set[str] = set()
        self.started_wall_us = _now_wall_us()
        self.book_checkpoints: dict[str, deque[dict[str, Any]]] = {
            symbol: deque() for symbol in config.symbols
        }
        self.last_book_checkpoint_us: dict[str, int] = defaultdict(int)
        self.snapshot_semaphore = asyncio.Semaphore(1)
        self.snapshot_backoff_until_monotonic = 0.0
        self.snapshot_rate_limit_count = 0

    async def run(self, duration_seconds: float | None = None) -> None:
        self._assert_disk_safe()
        writer_task = asyncio.create_task(self.writer.run())
        tasks = [
            asyncio.create_task(self._guard_task(self._market_worker())),
            asyncio.create_task(self._guard_task(self._websocket_loop())),
            asyncio.create_task(self._guard_task(self._resync_loop())),
            asyncio.create_task(self._guard_task(self._health_loop())),
        ]
        if self.consume_signals:
            tasks.append(asyncio.create_task(self._guard_task(self._signal_loop())))
        if duration_seconds is not None:
            tasks.append(asyncio.create_task(self._stop_after(duration_seconds)))
        try:
            await self.stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in self.snapshot_tasks.values():
                task.cancel()
            await self.writer.queue.put(None)
            await writer_task

    async def _guard_task(self, coroutine: Any) -> None:
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stop_event.set()
            raise

    async def _stop_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.stop_event.set()

    async def _signal_loop(self) -> None:
        while not self.stop_event.is_set():
            for row in self.tail.poll():
                symbol = str(row.get("symbol", "")).upper()
                t0 = _utc_us(str(row["signal_timestamp_utc"])) if row.get("signal_timestamp_utc") else 0
                logical_start = t0 - self.config.pre_signal_seconds * 1_000_000
                checkpoints = self.book_checkpoints.get(symbol, ())
                base = next(
                    (checkpoint for checkpoint in reversed(checkpoints) if int(checkpoint["exchange_timestamp_us"]) <= logical_start),
                    None,
                )
                self.core.observe_signal(row, l2_base_snapshot=base)
            self.core.finalize(_now_wall_us())
            self._assert_disk_safe()
            await asyncio.sleep(0.2)

    async def _health_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._assert_disk_safe()
                payload = {
                    "schema_id": "aegis-w13p-runtime-health-v1",
                    "state": "RUNNING",
                    "capture_only": True,
                    "financial_mutation_capability": False,
                    "authenticated_exchange_access": False,
                    "started_wall_timestamp_us": self.started_wall_us,
                    "updated_wall_timestamp_us": _now_wall_us(),
                    "active_signal_windows": len(self.core.active),
                    "market_queue_size": self.market_queue.qsize(),
                    "disk_queue_size": self.writer.queue.qsize(),
                    "collector_drop_count": self.core.drop_count,
                    "disk_drop_count": self.writer.dropped,
                    "reconnect_count": self.core.reconnect_count,
                    "snapshot_rate_limit_count": self.snapshot_rate_limit_count,
                    "snapshot_backoff_remaining_seconds": max(
                        0.0, self.snapshot_backoff_until_monotonic - time.monotonic()
                    ),
                    "ring_event_counts": {
                        symbol: dict(counts)
                        for symbol, counts in self.core.ring_type_counts.items()
                    },
                    "books": {symbol: asdict(book.integrity) for symbol, book in self.books.items()},
                }
                path = self.config.storage_root / "runtime" / "health.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(_canonical_json(payload))
                temporary.replace(path)
            except Exception:
                self.stop_event.set()
                raise
            await asyncio.sleep(5)

    async def _websocket_loop(self) -> None:
        streams = []
        for symbol in self.config.symbols:
            lower = symbol.lower()
            streams.extend((f"{lower}@depth@{self.config.depth_stream_interval}", f"{lower}@bookTicker", f"{lower}@trade"))
        url = f"{self.config.public_websocket_url}?streams={'/'.join(streams)}"
        timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=30)
        while not self.stop_event.is_set():
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(url, heartbeat=15, max_msg_size=8 * 1024 * 1024) as ws:
                        for symbol in self.config.symbols:
                            self._request_resync(symbol)
                        async for message in ws:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            wall, mono = _now_wall_us(), time.monotonic_ns()
                            try:
                                self.market_queue.put_nowait((json.loads(message.data)["data"], wall, mono))
                            except asyncio.QueueFull:
                                self.core.note_drop()
                                for book in self.books.values():
                                    book.invalidate()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.core.note_reconnect()
                for book in self.books.values():
                    book.invalidate()
                await asyncio.sleep(1)

    def _request_resync(self, symbol: str) -> None:
        if time.monotonic() < self.snapshot_backoff_until_monotonic:
            return
        if symbol in self.resync_pending or (symbol in self.snapshot_tasks and not self.snapshot_tasks[symbol].done()):
            return
        try:
            self.resync_queue.put_nowait(symbol)
            self.resync_pending.add(symbol)
        except asyncio.QueueFull:
            self.core.note_drop()

    async def _resync_loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self.stop_event.is_set():
                symbol = await self.resync_queue.get()
                self.resync_pending.discard(symbol)
                task = asyncio.create_task(self._resync_book(session, symbol))
                self.snapshot_tasks[symbol] = task
                await asyncio.sleep(0)

    async def _resync_book(self, session: aiohttp.ClientSession, symbol: str) -> None:
        params = {"symbol": symbol, "limit": self.config.depth_snapshot_limit}
        async with self.snapshot_semaphore:
            if time.monotonic() < self.snapshot_backoff_until_monotonic:
                return
            for attempt in range(5):
                try:
                    async with session.get(self.config.public_snapshot_url, params=params) as response:
                        if response.status in {418, 429}:
                            body = await response.text()
                            delay = _rate_limit_delay_seconds(
                                response.headers, body, int(time.time() * 1000)
                            )
                            self.snapshot_backoff_until_monotonic = time.monotonic() + delay
                            self.snapshot_rate_limit_count += 1
                            self.books[symbol].invalidate()
                            return
                        response.raise_for_status()
                        book = self.books[symbol]
                        if book.install_snapshot(await response.json()) or not book.needs_snapshot:
                            return
                except Exception:
                    self.books[symbol].invalidate()
                await asyncio.sleep(0.5 * (attempt + 1))
        self.books[symbol].invalidate()

    async def _market_worker(self) -> None:
        processed_since_yield = 0
        while not self.stop_event.is_set():
            data, wall, mono = await self.market_queue.get()
            event = self._normalize(data, wall, mono)
            if event is not None:
                self.core.observe_event(event)
            processed_since_yield += 1
            if processed_since_yield >= 1000:
                processed_since_yield = 0
                await asyncio.sleep(0)

    def _normalize(self, data: Mapping[str, Any], wall: int, mono: int) -> MarketEvent | None:
        kind = str(data.get("e", ""))
        symbol = str(data.get("s", "")).upper()
        if symbol not in self.books:
            return None
        book = self.books[symbol]
        if kind == "depthUpdate":
            before = book.integrity.valid
            valid = book.apply(data)
            if not valid:
                if before:
                    for window in self.core.active.values():
                        if window.symbol == symbol:
                            window.invalid_book_seen = True
                if book.needs_snapshot:
                    self._request_resync(symbol)
            payload = {key: data.get(key) for key in ("E", "T", "U", "u", "pu", "b", "a")}
            exchange_us = int(data["E"]) * 1000
            if valid and exchange_us - self.last_book_checkpoint_us[symbol] >= 5_000_000:
                checkpoints = self.book_checkpoints[symbol]
                checkpoints.append(book.checkpoint(exchange_us))
                self.last_book_checkpoint_us[symbol] = exchange_us
                cutoff = exchange_us - self.config.ring_retention_seconds * 1_000_000
                while checkpoints and int(checkpoints[0]["exchange_timestamp_us"]) < cutoff:
                    checkpoints.popleft()
            return MarketEvent(f"D:{symbol}:{data['U']}:{data['u']}", "BOOK", symbol, exchange_us, int(data.get("T", data["E"])) * 1000, wall, mono, payload, valid, book.integrity.generation)
        if kind == "bookTicker":
            payload = {key: data.get(key) for key in ("E", "T", "u", "b", "B", "a", "A")}
            return MarketEvent(f"Q:{symbol}:{data.get('u')}:{data.get('E')}", "QUOTE", symbol, int(data["E"]) * 1000, int(data.get("T", data["E"])) * 1000, wall, mono, payload, book.integrity.valid, book.integrity.generation)
        if kind in {"trade", "aggTrade"}:
            trade_id = data.get("t", data.get("a"))
            payload = {key: data.get(key) for key in ("E", "t", "a", "p", "q", "f", "l", "T", "m", "X")}
            return MarketEvent(f"T:{symbol}:{trade_id}", "TRADE", symbol, int(data["E"]) * 1000, int(data["T"]) * 1000, wall, mono, payload, book.integrity.valid, book.integrity.generation)
        return None

    def _assert_disk_safe(self) -> None:
        target = self.config.storage_root
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free / 1e9
        used = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) / 1e9
        if free < self.config.minimum_free_disk_gb or used >= self.config.maximum_collection_gb:
            self.stop_event.set()
            raise RuntimeError("W13P_DISK_SAFETY_STOP")


def progress_report(root: Path) -> dict[str, Any]:
    quality_files = list((root / "quality").rglob("*.parquet")) if (root / "quality").exists() else []
    signal_files = list((root / "signal").rglob("*.parquet")) if (root / "signal").exists() else []
    rows = [row for path in quality_files for row in pq.read_table(path).to_pylist()]
    signals = [row for path in signal_files for row in pq.read_table(path).to_pylist()]
    signals_by_id = {str(row["signal_id"]): row for row in signals}
    quality_by_id = {str(row["signal_id"]): row for row in rows}
    legacy_eligible = [row for row in rows if row.get("W13_ELIGIBLE")]
    eligible = [
        row for row in legacy_eligible
        if row.get("quality_gate_version") == CURRENT_QUALITY_GATE_VERSION
    ]
    by_symbol: dict[str, int] = defaultdict(int)
    by_day: dict[str, int] = defaultdict(int)
    directions: dict[str, int] = defaultdict(int)
    eligible_directions: dict[str, int] = defaultdict(int)
    for row in signals_by_id.values():
        by_symbol[str(row["symbol"])] += 1
        directions[str(row["side"])] += 1
        by_day[datetime.fromtimestamp(int(row["signal_timestamp_us"]) / 1e6, UTC).date().isoformat()] += 1
    for row in eligible:
        eligible_directions[str(row["side"])] += 1
    return {
        "schema_id": "aegis-w13p-progress-v1",
        "total_signals_captured": len(signals_by_id),
        "eligible_signals": len(eligible),
        "legacy_core_path_eligible_signals": len(legacy_eligible) - len(eligible),
        "ineligible_signals": len(rows) - len(eligible),
        "pending_signal_windows": len(set(signals_by_id) - set(quality_by_id)),
        "direction_counts": dict(sorted(directions.items())),
        "eligible_direction_counts": dict(sorted(eligible_directions.items())),
        "per_symbol": dict(sorted(by_symbol.items())),
        "per_day": dict(sorted(by_day.items())),
        "train_minimum": 1000,
        "validation_minimum": 500,
        "eligible_progress_to_total_minimum_pct": round(100 * len(eligible) / 1500, 2),
        "split_assignment": "NOT_ASSIGNED_OUTCOME_BLIND_TEMPORAL_SPLIT_REQUIRED",
        "final_holdout": "SEALED_NOT_OPENED",
    }


async def _main_async(args: argparse.Namespace) -> None:
    config = CollectorConfig.load(Path(args.config))
    if args.command == "progress":
        print(json.dumps(progress_report(config.storage_root), indent=2, sort_keys=True))
        return
    if args.command == "collect" and not config.enabled and not args.allow_disabled_dry_run:
        raise RuntimeError("W13P_CONFIG_DISABLED")
    sidecar = W13PSidecar(config, consume_signals=not args.market_only)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, sidecar.stop_event.set)
    await sidecar.run(args.duration_seconds)
    print(json.dumps({
        "written": sidecar.writer.written,
        "disk_drops": sidecar.writer.dropped,
        "collector_drops": sidecar.core.drop_count,
        "reconnects": sidecar.core.reconnect_count,
        "books": {symbol: asdict(book.integrity) for symbol, book in sidecar.books.items()},
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Passive public-only W13-P collector")
    parser.add_argument("command", choices=("collect", "progress"))
    parser.add_argument("--config", default="config/experiments/aegis_w13p_prospective_collection.yaml")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--allow-disabled-dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
