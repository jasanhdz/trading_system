from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .data_adapter import FIVE_MINUTES, RangeDataAdapter
from .models import Candle1m, Candle5m

SOURCE_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
SOURCE_END = datetime(2026, 8, 1, tzinfo=timezone.utc)
SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)
MONTHS = tuple(
    f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"
    for ordinal in range(2024 * 12, 2026 * 12 + 7)
)
PARTITIONS = {
    "TRAIN": (SOURCE_START, datetime(2025, 1, 1, tzinfo=timezone.utc)),
    "CALIBRATION": (
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
    ),
    "VALIDATION": (
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
    "HOLDOUT": (datetime(2026, 1, 1, tzinfo=timezone.utc), SOURCE_END),
}


class SourceIntegrityError(RuntimeError):
    pass


class PartitionAccessError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ArchiveAudit:
    data_type: str
    expected: int
    verified: int
    absent: tuple[str, ...]
    conflicts: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def verify_r1_immutable(repo_root: Path) -> dict[str, str]:
    manifest_path = repo_root / "docs/aegis-range-v1/r1_implementation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: dict[str, str] = {}
    for section in ("authority_artifacts", "source_files", "test_files", "support_files"):
        for relative, expected in manifest[section].items():
            path = repo_root / relative
            actual = _sha256_file(path) if path.is_file() else "ABSENT"
            if actual != expected:
                mismatches[relative] = actual
    if mismatches:
        raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_R1_DRIFT")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_files": str(
            sum(len(manifest[name]) for name in ("authority_artifacts", "source_files", "test_files", "support_files"))
        ),
    }


def _load_manifest(path: Path) -> dict[tuple[str, str, str], dict]:
    records: dict[tuple[str, str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        request = row["request"]
        if request["market"] != "futures/um":
            continue
        key = (request["data_type"], request["symbol"], request["month"])
        if key in records:
            raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
        if request["data_type"] in {"klines", "markPriceKlines"} and request["interval"] != "1m":
            continue
        records[key] = row
    return records


def _expected_member(data_type: str, symbol: str, month: str) -> str:
    if data_type in {"klines", "markPriceKlines"}:
        return f"{symbol}-1m-{month}.csv"
    return f"{symbol}-fundingRate-{month}.csv"


def audit_sources(repo_root: Path) -> tuple[dict[str, ArchiveAudit], dict[tuple[str, str, str], dict]]:
    source_authority_path = repo_root / "docs/aegis-range-v1/r0_source_manifest.json"
    authority = json.loads(source_authority_path.read_text(encoding="utf-8"))
    manifests = {
        "klines": Path(authority["datasets"]["primary_ohlcv"]["manifest_path"]),
        "m1b": Path(authority["datasets"]["funding_and_mark_price"]["manifest_path"]),
    }
    pins = {
        "klines": authority["datasets"]["primary_ohlcv"]["manifest_sha256"],
        "m1b": authority["datasets"]["funding_and_mark_price"]["manifest_sha256"],
    }
    if any(_sha256_file(manifests[name]) != pins[name] for name in manifests):
        raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")

    all_records = _load_manifest(manifests["klines"])
    all_records.update(_load_manifest(manifests["m1b"]))
    audits: dict[str, ArchiveAudit] = {}
    for data_type in ("klines", "fundingRate", "markPriceKlines"):
        absent: list[str] = []
        conflicts: list[str] = []
        verified = 0
        for symbol in SYMBOLS:
            for month in MONTHS:
                key = (data_type, symbol, month)
                row = all_records.get(key)
                if row is None:
                    absent.append("/".join(key))
                    continue
                path = Path(row["file"])
                if not path.is_file():
                    absent.append(row["url"])
                    continue
                try:
                    actual = _sha256_file(path)
                    with zipfile.ZipFile(path) as archive:
                        corrupt = archive.testzip()
                        members = tuple(item.filename for item in archive.infolist() if not item.is_dir())
                except (OSError, zipfile.BadZipFile):
                    conflicts.append(row["url"])
                    continue
                expected_member = _expected_member(data_type, symbol, month)
                valid = (
                    actual == row.get("expected_sha256") == row.get("actual_sha256")
                    and path.stat().st_size == row.get("byte_size")
                    and corrupt is None
                    and members == (expected_member,)
                    and tuple(row.get("zip_members", ())) == members
                )
                if valid:
                    verified += 1
                else:
                    conflicts.append(row["url"])
        audits[data_type] = ArchiveAudit(data_type, 341, verified, tuple(absent), tuple(conflicts))
    if any(item.verified != item.expected or item.absent or item.conflicts for item in audits.values()):
        raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
    return audits, all_records


class SealedPartitionGuard:
    @staticmethod
    def access_flags(environment: Mapping[str, str] | None = None) -> dict[str, bool]:
        values = os.environ if environment is None else environment
        return {name: values.get(f"{name}_ACCESS", "false").lower() == "true" for name in PARTITIONS}

    @classmethod
    def require(cls, partition: str, environment: Mapping[str, str] | None = None) -> None:
        if partition not in PARTITIONS:
            raise PartitionAccessError("AEGIS_RANGE_R2_PARTITION_UNKNOWN")
        if not cls.access_flags(environment)[partition]:
            raise PartitionAccessError(f"AEGIS_RANGE_R2_{partition}_SEALED")


def _from_epoch(value: str) -> datetime:
    timestamp = int(value)
    if timestamp >= 10**15:
        timestamp //= 1000
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _csv_rows(row: dict, expected_header: tuple[str, ...]) -> Iterable[dict[str, str]]:
    path = Path(row["file"])
    member = row["zip_members"][0]
    with zipfile.ZipFile(path) as archive, archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
        reader = csv.DictReader(text)
        if reader.fieldnames is None or not set(expected_header).issubset(reader.fieldnames):
            raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
        yield from reader


def _load_ohlcv(row: dict, symbol: str) -> list[Candle1m]:
    result = []
    for item in _csv_rows(row, ("open_time", "open", "high", "low", "close", "volume")):
        result.append(
            RangeDataAdapter.candle_1m_from_source(
                symbol,
                _from_epoch(item["open_time"]),
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
            )
        )
    return result


def _deterministic_gzip_csv(path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="ascii", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
                    count += 1
    return count, _sha256_file(path)


def _funding_events(row: dict) -> list[tuple[datetime, datetime, str]]:
    events = []
    canonical_times: set[datetime] = set()
    for item in _csv_rows(row, ("calc_time", "last_funding_rate")):
        source_calc_time = _from_epoch(item["calc_time"])
        if source_calc_time.second != 0:
            raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
        funding_at = source_calc_time.replace(second=0, microsecond=0)
        if funding_at in canonical_times:
            raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
        canonical_times.add(funding_at)
        if SOURCE_START < funding_at < SOURCE_END:
            events.append((source_calc_time, funding_at, item["last_funding_rate"]))
    return events


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    ordinal = start.year * 12 + start.month
    end = datetime(ordinal // 12, ordinal % 12 + 1, 1, tzinfo=timezone.utc)
    return start, end


def _mark_closes(row: dict, required: set[datetime]) -> tuple[dict[datetime, str], int]:
    closes = {}
    seen: set[datetime] = set()
    for item in _csv_rows(row, ("open_time", "close")):
        open_time = _from_epoch(item["open_time"])
        if open_time.second or open_time.microsecond or open_time in seen:
            raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
        seen.add(open_time)
        if open_time in required:
            closes[open_time] = item["close"]
    return closes, len(seen)


def _artifact(relative: Path, rows: int, digest: str, source_rows: list[dict]) -> dict:
    return {
        "path": relative.as_posix(),
        "rows": rows,
        "sha256": digest,
        "source_sha256": sorted(item["actual_sha256"] for item in source_rows),
    }


def build_derived_dataset(repo_root: Path, output_root: Path) -> dict:
    r1 = verify_r1_immutable(repo_root)
    audits, records = audit_sources(repo_root)
    artifacts: list[dict] = []
    symbols: dict[str, dict] = {}
    missing_mark = 0
    for symbol in SYMBOLS:
        previous: Candle5m | None = None
        segment_id = -1
        symbol_rows = 0
        symbol_gaps = 0
        symbol_funding = 0
        symbol_missing_funding_mark = 0
        symbol_missing_mark_minutes = 0
        segment_ids: set[int] = set()
        for month_index, month in enumerate(MONTHS):
            ohlcv_source = records[("klines", symbol, month)]
            source_candles = _load_ohlcv(ohlcv_source, symbol)
            aggregation = RangeDataAdapter.aggregate_1m_to_5m(source_candles)
            normalized: list[Candle5m] = []
            for candle in aggregation.candles:
                if previous is None or candle.open_time != previous.open_time + FIVE_MINUTES:
                    segment_id += 1
                current = replace(candle, segment_id=segment_id)
                normalized.append(current)
                segment_ids.add(segment_id)
                previous = current
            relative = Path("ohlcv_5m") / symbol / f"{month}.csv.gz"
            rows, digest = _deterministic_gzip_csv(
                output_root / relative,
                ("symbol", "open_time", "available_at", "open", "high", "low", "close", "volume", "segment_id", "high_source", "low_source"),
                (
                    (
                        candle.symbol,
                        _iso(candle.open_time),
                        _iso(candle.available_at),
                        repr(candle.open),
                        repr(candle.high),
                        repr(candle.low),
                        repr(candle.close),
                        repr(candle.volume),
                        candle.segment_id,
                        candle.high_source,
                        candle.low_source,
                    )
                    for candle in normalized
                ),
            )
            artifacts.append(_artifact(relative, rows, digest, [ohlcv_source]))
            symbol_rows += rows
            month_start, month_end = _month_bounds(month)
            symbol_gaps += int((month_end - month_start) / FIVE_MINUTES) - rows

            funding_source = records[("fundingRate", symbol, month)]
            mark_source = records[("markPriceKlines", symbol, month)]
            funding = _funding_events(funding_source)
            needed = {funding_at - timedelta(minutes=1) for _, funding_at, _ in funding}
            marks, mark_rows = _mark_closes(mark_source, needed)
            symbol_missing_mark_minutes += int((month_end - month_start) / timedelta(minutes=1)) - mark_rows
            mark_sources = [mark_source]
            if month_index:
                previous_mark = records[("markPriceKlines", symbol, MONTHS[month_index - 1])]
                previous_marks, _ = _mark_closes(previous_mark, needed)
                marks.update(previous_marks)
                mark_sources.append(previous_mark)
            funding_rows = []
            for source_calc_time, funding_at, rate in funding:
                mark_open = funding_at - timedelta(minutes=1)
                mark_close = marks.get(mark_open)
                if mark_close is None:
                    missing_mark += 1
                    symbol_missing_funding_mark += 1
                    continue
                funding_rows.append((symbol, _iso(source_calc_time), _iso(funding_at), _iso(funding_at), rate, _iso(mark_open), mark_close))
            relative = Path("funding_mark") / symbol / f"{month}.csv.gz"
            rows, digest = _deterministic_gzip_csv(
                output_root / relative,
                ("symbol", "source_calc_time", "funding_at", "available_at", "funding_rate", "mark_open_time", "mark_close"),
                funding_rows,
            )
            artifacts.append(_artifact(relative, rows, digest, [funding_source, *mark_sources]))
            symbol_funding += rows
        symbols[symbol] = {
            "ohlcv_5m_rows": symbol_rows,
            "segments": len(segment_ids),
            "integrity_gap_blocks": symbol_gaps,
            "funding_events_mapped": symbol_funding,
            "funding_events_missing_mark_price": symbol_missing_funding_mark,
            "mark_price_missing_minutes": symbol_missing_mark_minutes,
        }
    logical = {
        "schema_version": "aegis-range-r2-derived-dataset-v1",
        "source_interval": {"start_inclusive": _iso(SOURCE_START), "end_exclusive": _iso(SOURCE_END)},
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "symbols": symbols,
        "funding_events_missing_mark_price": missing_mark,
        "partition_access_defaults": SealedPartitionGuard.access_flags({}),
        "r1_manifest_sha256": r1["manifest_sha256"],
        "source_coverage": {name: asdict(audit) for name, audit in audits.items()},
        "status": (
            "AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY"
            if missing_mark
            else "AEGIS_RANGE_R2_DATA_READINESS_READY_FOR_REVIEW"
        ),
    }
    logical_sha256 = hashlib.sha256(_canonical_json(logical).encode("ascii")).hexdigest()
    manifest = {**logical, "logical_sha256": logical_sha256}
    manifest_path = output_root / "derived_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    if missing_mark:
        raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
    return manifest
