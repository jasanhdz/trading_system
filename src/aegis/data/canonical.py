"""Read-only adapter for immutable, final D3 canonical candle series."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping

from ..config import CANONICAL_SYMBOLS


class CanonicalDataError(RuntimeError):
    pass


class DataPurpose(str, Enum):
    TRAINING = "TRAINING"
    BACKTEST = "BACKTEST"
    REPLAY = "REPLAY"
    BENCHMARK = "BENCHMARK"


@dataclass(frozen=True)
class CanonicalBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class CanonicalSeriesAudit:
    artifact_id: str
    manifest_sha256: str
    content_sha256: Mapping[str, str]
    symbols: tuple[str, ...]
    timeframe: str
    purpose: DataPurpose
    read_only: bool
    finality_verified: bool
    gap_free: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CanonicalSeriesSource:
    """Verify and consume a finalized D3 series without any write capability."""

    root: Path
    purpose: DataPurpose
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve(strict=True))

    @property
    def manifest_path(self) -> Path:
        return self.root / "series_manifest.json"

    def audit(self, *, verify_content: bool = True) -> CanonicalSeriesAudit:
        if not self.manifest_path.is_file():
            raise CanonicalDataError("canonical series manifest is missing")
        manifest_hash = _sha256(self.manifest_path)
        sidecar = self.root / "series_manifest.sha256"
        if not sidecar.is_file():
            raise CanonicalDataError("canonical hash sidecar is missing")
        declared: dict[str, str] = {}
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2 or Path(parts[1]).name != parts[1]:
                raise CanonicalDataError("canonical hash sidecar is malformed")
            declared[parts[1]] = parts[0]
        if declared.get("series_manifest.json") != manifest_hash:
            raise CanonicalDataError("canonical manifest sidecar hash mismatch")
        if self.expected_manifest_sha256 is not None and manifest_hash != self.expected_manifest_sha256:
            raise CanonicalDataError("canonical manifest hash mismatch")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "gen2_d3_series_v1" or payload.get("status") != "OK":
            raise CanonicalDataError("canonical series is not final and approved")
        included = payload.get("included_symbols")
        if not isinstance(included, Mapping) or tuple(sorted(included)) != tuple(sorted(CANONICAL_SYMBOLS)):
            raise CanonicalDataError("canonical series universe mismatch")
        if payload.get("excluded_symbols"):
            raise CanonicalDataError("canonical series contains excluded symbols")
        gaps = payload.get("gates", {}).get("g4_gaps", [])
        coverage = payload.get("gates", {}).get("g5_coverage", [])
        if len(gaps) != len(CANONICAL_SYMBOLS) or not all(bool(item.get("passes")) for item in gaps):
            raise CanonicalDataError("canonical gap gate failed")
        if len(coverage) != len(CANONICAL_SYMBOLS) or not all(bool(item.get("passes")) for item in coverage):
            raise CanonicalDataError("canonical coverage gate failed")
        hashes: dict[str, str] = {}
        for symbol in CANONICAL_SYMBOLS:
            path = self.root / f"{symbol}_5m.csv"
            expected = str(included[symbol].get("sha256", ""))
            if not path.is_file() or not expected or declared.get(path.name) != expected:
                raise CanonicalDataError(f"canonical file contract missing for {symbol}")
            actual = _sha256(path) if verify_content else expected
            if actual != expected:
                raise CanonicalDataError(f"canonical file hash mismatch for {symbol}")
            hashes[path.name] = actual
        return CanonicalSeriesAudit(
            artifact_id=str(payload["artifact_id"]), manifest_sha256=manifest_hash,
            content_sha256=hashes, symbols=CANONICAL_SYMBOLS, timeframe="5m", purpose=self.purpose,
            read_only=True, finality_verified=True, gap_free=True,
        )

    def load(self, *, start: datetime, end: datetime) -> Mapping[str, tuple[CanonicalBar, ...]]:
        """Load a half-open UTC interval after complete content verification."""
        audit = self.audit(verify_content=True)
        if not audit.finality_verified or end <= start:
            raise CanonicalDataError("invalid canonical load interval")
        expected_step = timedelta(minutes=5)
        result: dict[str, tuple[CanonicalBar, ...]] = {}
        for symbol in CANONICAL_SYMBOLS:
            rows: list[CanonicalBar] = []
            with (self.root / f"{symbol}_5m.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"timestamp", "open", "high", "low", "close", "volume"}
                if reader.fieldnames is None or not required <= set(reader.fieldnames):
                    raise CanonicalDataError(f"canonical schema mismatch for {symbol}")
                for item in reader:
                    timestamp = _utc(item["timestamp"])
                    if timestamp < start:
                        continue
                    if timestamp >= end:
                        break
                    bar = CanonicalBar(timestamp, *(float(item[name]) for name in ("open", "high", "low", "close", "volume")))
                    values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
                    if not all(math.isfinite(value) for value in values):
                        raise CanonicalDataError(f"non-finite canonical row for {symbol}")
                    if min(values[:4]) <= 0 or bar.volume < 0 or bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                        raise CanonicalDataError(f"invalid canonical OHLCV for {symbol}")
                    if rows and bar.timestamp - rows[-1].timestamp != expected_step:
                        raise CanonicalDataError(f"canonical interval gap for {symbol}")
                    rows.append(bar)
            if not rows:
                raise CanonicalDataError(f"canonical interval is empty for {symbol}")
            result[symbol] = tuple(rows)
        return result
