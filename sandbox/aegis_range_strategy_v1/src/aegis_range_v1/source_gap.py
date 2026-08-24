from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from .readiness import (
    MONTHS,
    SOURCE_END,
    SOURCE_START,
    SYMBOLS,
    SealedPartitionGuard,
    SourceIntegrityError,
    _artifact,
    _canonical_json,
    _deterministic_gzip_csv,
    _from_epoch,
    _funding_events,
    _iso,
    _mark_closes,
    _month_bounds,
    _sha256_file,
    audit_sources,
    verify_r1_immutable,
)

DAILY_DATES = ("2024-08-12", "2026-06-29")
POLICY = "MONTHLY_PRIMARY_DAILY_GAP_FILL_V1"
ARCHIVE_HOST = "data.binance.vision"
DAILY_ROOT = f"https://{ARCHIVE_HOST}/data/futures/um/daily/markPriceKlines"
MARK_FIELDS = ("open_time", "open", "high", "low", "close")


class SourceConflictError(RuntimeError):
    pass


class OfficialSourceGapError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DailyRequest:
    symbol: str
    date: str

    @property
    def filename(self) -> str:
        return f"{self.symbol}-1m-{self.date}.zip"

    @property
    def member(self) -> str:
        return self.filename.removesuffix(".zip") + ".csv"

    @property
    def official_path(self) -> str:
        return f"data/futures/um/daily/markPriceKlines/{self.symbol}/1m/{self.filename}"

    @property
    def url(self) -> str:
        return f"{DAILY_ROOT}/{self.symbol}/1m/{self.filename}"


@dataclass(frozen=True, slots=True)
class DailyAudit:
    symbol: str
    date: str
    daily_exists: bool
    official_path: str
    local_file: str
    sha256: str
    byte_size: int
    csv_member: str
    crc: str
    crc_ok: bool
    rows: int
    minutes: int
    coverage_start: str | None
    coverage_end: str | None
    duplicate_count: int
    invalid_rows: int
    overlap_rows_compared: int
    exact_matches: int
    mismatches: int
    monthly_missing_minutes: int
    daily_recovered_minutes: int
    remaining_missing_minutes: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_REDIRECT_PROHIBITED")


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect())


def _read_official(opener: urllib.request.OpenerDirector, url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ARCHIVE_HOST:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_HOST_INVALID")
    with opener.open(urllib.request.Request(url, method="GET"), timeout=60) as response:
        resolved = urllib.parse.urlparse(response.geturl())
        if resolved.scheme != "https" or resolved.hostname != ARCHIVE_HOST:
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_HOST_INVALID")
        return response.read()


def _checksum(payload: bytes, filename: str) -> str:
    try:
        parts = payload.decode("ascii").strip().split()
    except UnicodeDecodeError as exc:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_CHECKSUM_INVALID") from exc
    if len(parts) != 2 or parts[1].lstrip("*") != filename:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_CHECKSUM_INVALID")
    digest = parts[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_CHECKSUM_INVALID")
    return digest


def download_daily(request: DailyRequest, root: Path) -> Path:
    if request.symbol not in SYMBOLS or request.date not in DAILY_DATES:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_SCOPE_INVALID")
    opener = _opener()
    expected = _checksum(_read_official(opener, request.url + ".CHECKSUM"), request.filename)
    destination = root / "futures/um/daily/markPriceKlines" / request.symbol / "1m" / request.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_ARCHIVE_CONFLICT")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{request.filename}.", dir=destination.parent)
    digest = hashlib.sha256()
    try:
        with opener.open(urllib.request.Request(request.url, method="GET"), timeout=60) as response, os.fdopen(descriptor, "wb") as handle:
            resolved = urllib.parse.urlparse(response.geturl())
            if resolved.scheme != "https" or resolved.hostname != ARCHIVE_HOST:
                raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_HOST_INVALID")
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected:
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_ARCHIVE_HASH_MISMATCH")
        os.chmod(temporary_name, 0o400)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _read_mark_rows(path: Path, member: str) -> tuple[list[tuple[datetime, tuple[str, str, str, str]]], int, int, str, bool]:
    invalid = 0
    duplicate_count = 0
    seen: set[datetime] = set()
    rows: list[tuple[datetime, tuple[str, str, str, str]]] = []
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != member:
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_ZIP_MEMBER_INVALID")
        crc = f"{members[0].CRC:08x}"
        crc_ok = archive.testzip() is None
        with archive.open(members[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            if reader.fieldnames is None or not set(MARK_FIELDS).issubset(reader.fieldnames):
                raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_SCHEMA_INVALID")
            previous: datetime | None = None
            for item in reader:
                try:
                    timestamp = _from_epoch(item["open_time"])
                    values = tuple(Decimal(item[name]) for name in ("open", "high", "low", "close"))
                    open_value, high, low, close = values
                    valid = (
                        timestamp.second == 0
                        and timestamp.microsecond == 0
                        and all(value.is_finite() and value > 0 for value in values)
                        and high >= max(open_value, close, low)
                        and low <= min(open_value, close, high)
                    )
                except (InvalidOperation, ValueError, OverflowError):
                    invalid += 1
                    continue
                if not valid:
                    invalid += 1
                if timestamp in seen:
                    duplicate_count += 1
                seen.add(timestamp)
                if previous is not None and timestamp <= previous:
                    invalid += 1
                previous = timestamp
                rows.append((timestamp, (item["open"], item["high"], item["low"], item["close"])))
    return rows, duplicate_count, invalid, crc, crc_ok


def _date_bounds(date: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _monthly_rows_for_date(monthly_row: dict, date: str) -> dict[datetime, tuple[str, str, str, str]]:
    start, end = _date_bounds(date)
    path = Path(monthly_row["file"])
    rows, duplicates, invalid, _, crc_ok = _read_mark_rows(path, monthly_row["zip_members"][0])
    if duplicates or invalid or not crc_ok:
        raise SourceIntegrityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY")
    return {timestamp: values for timestamp, values in rows if start <= timestamp < end}


def apply_precedence(
    monthly: dict[datetime, tuple[str, str, str, str]],
    daily: dict[datetime, tuple[str, str, str, str]],
    expected: Iterable[datetime],
) -> tuple[dict[datetime, tuple[str, str, str, str]], tuple[datetime, ...], tuple[datetime, ...]]:
    merged = dict(monthly)
    conflicts = tuple(sorted(timestamp for timestamp in monthly.keys() & daily.keys() if monthly[timestamp] != daily[timestamp]))
    recovered = []
    for timestamp in expected:
        if timestamp in merged:
            continue
        if timestamp in daily:
            merged[timestamp] = daily[timestamp]
            recovered.append(timestamp)
    missing = tuple(timestamp for timestamp in expected if timestamp not in merged)
    return merged, tuple(recovered), missing


def download_and_audit(repo_root: Path, raw_root: Path) -> list[DailyAudit]:
    verify_r1_immutable(repo_root)
    _, source_records = audit_sources(repo_root)
    audits = []
    for symbol in SYMBOLS:
        for date in DAILY_DATES:
            request = DailyRequest(symbol, date)
            path = download_daily(request, raw_root)
            rows, duplicates, invalid, crc, crc_ok = _read_mark_rows(path, request.member)
            start, end = _date_bounds(date)
            daily = {timestamp: values for timestamp, values in rows}
            coverage_valid = all(start <= timestamp < end for timestamp in daily)
            expected = tuple(start + timedelta(minutes=index) for index in range(1440))
            month = date[:7]
            monthly = _monthly_rows_for_date(source_records[("markPriceKlines", symbol, month)], date)
            merged, recovered, missing = apply_precedence(monthly, daily, expected)
            overlap = monthly.keys() & daily.keys()
            exact = sum(monthly[timestamp] == daily[timestamp] for timestamp in overlap)
            mismatches = len(overlap) - exact
            audit = DailyAudit(
                symbol=symbol,
                date=date,
                daily_exists=True,
                official_path=request.official_path,
                local_file=str(path.resolve()),
                sha256=_sha256_file(path),
                byte_size=path.stat().st_size,
                csv_member=request.member,
                crc=crc,
                crc_ok=crc_ok,
                rows=len(rows),
                minutes=len(daily),
                coverage_start=_iso(rows[0][0]) if rows else None,
                coverage_end=_iso(rows[-1][0]) if rows else None,
                duplicate_count=duplicates,
                invalid_rows=invalid + (0 if coverage_valid else 1),
                overlap_rows_compared=len(overlap),
                exact_matches=exact,
                mismatches=mismatches,
                monthly_missing_minutes=len(expected) - len(monthly),
                daily_recovered_minutes=len(recovered),
                remaining_missing_minutes=len(missing),
            )
            audits.append(audit)
    return audits


def audit_status(audits: list[DailyAudit]) -> str:
    if any(item.mismatches for item in audits):
        return "AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_SOURCE_CONFLICT"
    if any(
        not item.crc_ok
        or item.duplicate_count
        or item.invalid_rows
        for item in audits
    ):
        return "AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_OFFICIAL_SOURCE_GAP"
    critical = [item for item in audits if item.date == "2026-06-29"]
    if len(critical) != len(SYMBOLS) or any(
        item.rows != 1440 or item.minutes != 1440 or item.remaining_missing_minutes
        for item in critical
    ):
        return "AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_OFFICIAL_SOURCE_GAP"
    return "DAILY_VALID_FOR_CONTRACTUAL_GAP_FILL"


def write_audit(path: Path, audits: list[DailyAudit]) -> None:
    payload = {
        "schema_version": "aegis-range-r2-source-gap-daily-audit-v1",
        "status": audit_status(audits),
        "replacement_policy": POLICY,
        "files": [asdict(item) for item in audits],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def source_manifest_payload(repo_root: Path, audits: list[DailyAudit]) -> dict:
    monthly_manifest = repo_root / "data/market_event_economic_path_m1b/archive_manifest.jsonl"
    return {
        "schema_version": "aegis-range-r2-source-gap-manifest-v1",
        "status": "AEGIS_RANGE_R2_SOURCE_GAP_AMENDMENT",
        "monthly_manifest_sha256": _sha256_file(monthly_manifest),
        "daily_gap_files_count": len(audits),
        "replacement_policy": POLICY,
        "files": [
            {
                key: value
                for key, value in asdict(item).items()
                if key in {"symbol", "date", "official_path", "local_file", "sha256", "byte_size", "csv_member", "crc", "crc_ok", "rows", "coverage_start", "coverage_end"}
            }
            for item in audits
        ],
    }


def write_source_manifest(repo_root: Path, audit_path: Path, destination: Path) -> str:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audits = [DailyAudit(**item) for item in audit["files"]]
    if audit.get("status") != "DAILY_VALID_FOR_CONTRACTUAL_GAP_FILL" or audit_status(audits) != audit["status"]:
        raise OfficialSourceGapError("AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_OFFICIAL_SOURCE_GAP")
    payload = source_manifest_payload(repo_root, audits)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return _sha256_file(destination)


def _load_daily_manifest(path: Path) -> tuple[dict[tuple[str, str], dict], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("replacement_policy") != POLICY or payload.get("daily_gap_files_count") != 22:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_MANIFEST_INVALID")
    records = {(item["symbol"], item["date"]): item for item in payload["files"]}
    expected_keys = {(symbol, date) for symbol in SYMBOLS for date in DAILY_DATES}
    if records.keys() != expected_keys:
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_MANIFEST_INVALID")
    for key, item in records.items():
        request = DailyRequest(*key)
        path_value = Path(item["local_file"])
        if (
            item["official_path"] != request.official_path
            or item["csv_member"] != request.member
            or _sha256_file(path_value) != item["sha256"]
            or path_value.stat().st_size != item["byte_size"]
        ):
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_ARCHIVE_CONFLICT")
    return records, _sha256_file(path)


def build_gap_resolved_derived(repo_root: Path, source_manifest_path: Path, output_root: Path) -> dict:
    r1 = verify_r1_immutable(repo_root)
    audits, source_records = audit_sources(repo_root)
    daily_records, source_gap_manifest_sha256 = _load_daily_manifest(source_manifest_path)
    prior_manifest_path = repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_data_readiness/derived_dataset_manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_manifest_sha256 = _sha256_file(prior_manifest_path)
    if prior_manifest_sha256 != "55605c09e3f3de0d3f4d8b335beeac0eab4b0728a0f267512a6429ee8e2186b0":
        raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_DRIFT")

    daily_maps: dict[tuple[str, str], dict[datetime, tuple[str, str, str, str]]] = {}
    for key, item in daily_records.items():
        rows, duplicates, invalid, _, crc_ok = _read_mark_rows(Path(item["local_file"]), item["csv_member"])
        if duplicates or invalid or not crc_ok or len(rows) != item["rows"]:
            raise SourceIntegrityError("AEGIS_RANGE_R2_SOURCE_GAP_ARCHIVE_CONFLICT")
        daily_maps[key] = dict(rows)

    artifacts = []
    symbol_metrics = {}
    total_events = 0
    total_mapped = 0
    total_missing = 0
    total_monthly_missing = 0
    total_recovered = 0
    for symbol in SYMBOLS:
        symbol_events = 0
        symbol_mapped = 0
        symbol_missing = 0
        symbol_monthly_missing = 0
        symbol_recovered = 0
        for month_index, month in enumerate(MONTHS):
            funding_source = source_records[("fundingRate", symbol, month)]
            mark_source = source_records[("markPriceKlines", symbol, month)]
            funding = _funding_events(funding_source)
            needed = {funding_at - timedelta(minutes=1) for _, funding_at, _ in funding}
            marks, mark_rows = _mark_closes(mark_source, needed)
            month_start, month_end = _month_bounds(month)
            monthly_missing = int((month_end - month_start) / timedelta(minutes=1)) - mark_rows
            symbol_monthly_missing += monthly_missing
            source_lineage = [funding_source, mark_source]
            if month_index:
                previous_mark = source_records[("markPriceKlines", symbol, MONTHS[month_index - 1])]
                previous_marks, _ = _mark_closes(previous_mark, needed)
                marks.update(previous_marks)
                source_lineage.append(previous_mark)

            recovered_in_month = 0
            for date in DAILY_DATES:
                if date[:7] != month:
                    continue
                daily_record = daily_records[(symbol, date)]
                daily = daily_maps[(symbol, date)]
                start, end = _date_bounds(date)
                for timestamp, values in daily.items():
                    if start <= timestamp < end and timestamp not in marks and timestamp in needed:
                        marks[timestamp] = values[3]
                monthly_day = _monthly_rows_for_date(mark_source, date)
                recovered_in_month += sum(timestamp not in monthly_day for timestamp in daily)
                source_lineage.append({"actual_sha256": daily_record["sha256"]})
            symbol_recovered += recovered_in_month

            funding_rows = []
            for source_calc_time, funding_at, rate in funding:
                symbol_events += 1
                mark_open = funding_at - timedelta(minutes=1)
                mark_close = marks.get(mark_open)
                if mark_close is None:
                    symbol_missing += 1
                    continue
                symbol_mapped += 1
                funding_rows.append((symbol, _iso(source_calc_time), _iso(funding_at), _iso(funding_at), rate, _iso(mark_open), mark_close))
            relative = Path("funding_mark") / symbol / f"{month}.csv.gz"
            rows, digest = _deterministic_gzip_csv(
                output_root / relative,
                ("symbol", "source_calc_time", "funding_at", "available_at", "funding_rate", "mark_open_time", "mark_close"),
                funding_rows,
            )
            artifacts.append(_artifact(relative, rows, digest, source_lineage))
        remaining = symbol_monthly_missing - symbol_recovered
        symbol_metrics[symbol] = {
            "monthly_missing_minutes": symbol_monthly_missing,
            "daily_recovered_minutes": symbol_recovered,
            "remaining_missing_minutes": remaining,
            "funding_events_total": symbol_events,
            "funding_events_mapped": symbol_mapped,
            "funding_events_missing_mark_price": symbol_missing,
        }
        total_events += symbol_events
        total_mapped += symbol_mapped
        total_missing += symbol_missing
        total_monthly_missing += symbol_monthly_missing
        total_recovered += symbol_recovered

    remaining_missing = total_monthly_missing - total_recovered
    logical = {
        "schema_version": "aegis-range-r2-gap-resolved-derived-v1",
        "status": "AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_READY_FOR_REVIEW",
        "replacement_policy": POLICY,
        "source_interval": {"start_inclusive": _iso(SOURCE_START), "end_exclusive": _iso(SOURCE_END)},
        "source_gap_manifest_sha256": source_gap_manifest_sha256,
        "monthly_manifest_sha256": "1cc559055937f3d2432f0559a6badda6865495fdfd26f52f3f02c0943836f92b",
        "prior_ohlcv_derived": {
            "manifest_sha256": prior_manifest_sha256,
            "logical_sha256": prior_manifest["logical_sha256"],
            "rows": sum(item["ohlcv_5m_rows"] for item in prior_manifest["symbols"].values()),
            "integrity_gap_blocks": sum(item["integrity_gap_blocks"] for item in prior_manifest["symbols"].values()),
        },
        "funding_mark_artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "symbols": symbol_metrics,
        "mark_price_missing_minutes_before_gap_fill": total_monthly_missing,
        "daily_recovered_minutes": total_recovered,
        "mark_price_missing_minutes": remaining_missing,
        "funding_events_total": total_events,
        "funding_events_mapped": total_mapped,
        "funding_events_missing_mark_price": total_missing,
        "partition_access_defaults": SealedPartitionGuard.access_flags({}),
        "r1_manifest_sha256": r1["manifest_sha256"],
        "monthly_source_coverage": {name: asdict(audit) for name, audit in audits.items()},
        "economic_metrics_computed": False,
        "range_engine_executed": False,
    }
    logical_sha256 = hashlib.sha256(_canonical_json(logical).encode("ascii")).hexdigest()
    manifest = {**logical, "logical_sha256": logical_sha256}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "derived_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    if total_events != 31108 or total_mapped != 31108 or total_missing:
        raise OfficialSourceGapError("AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_OFFICIAL_SOURCE_GAP")
    return manifest
