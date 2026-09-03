"""Synthetic-only validation of the Amendment 03 funding contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from .constants import CANONICAL_SYMBOLS, FUNDING_PROVIDER, FUNDING_SCHEMA_VERSION
from .core import canonical_decimal, funding_event_in_interval, validate_canonical_decimal
from .errors import Phase0Error


FUNDING_FIELDS = (
    "schema_version",
    "provider",
    "symbol",
    "funding_time_utc_ms",
    "funding_rate_decimal",
    "source_artifact_sha256",
    "funding_record_id",
)


def funding_record_id(symbol: str, funding_time_utc_ms: int) -> str:
    preimage = f"{FUNDING_SCHEMA_VERSION}|{symbol}|{funding_time_utc_ms}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


@dataclass(frozen=True)
class FundingRecord:
    schema_version: str
    provider: str
    symbol: str
    funding_time_utc_ms: int
    funding_rate_decimal: str
    source_artifact_sha256: str
    funding_record_id: str

    @classmethod
    def build(cls, symbol: str, funding_time_utc_ms: int, rate: str, source_sha256: str) -> "FundingRecord":
        return cls(
            FUNDING_SCHEMA_VERSION,
            FUNDING_PROVIDER,
            symbol,
            funding_time_utc_ms,
            canonical_decimal(rate),
            source_sha256,
            funding_record_id(symbol, funding_time_utc_ms),
        )

    def as_ordered_mapping(self) -> dict[str, str | int]:
        return {field: getattr(self, field) for field in FUNDING_FIELDS}


def validate_record(record: FundingRecord, raw_hashes: set[str]) -> None:
    if record.schema_version != FUNDING_SCHEMA_VERSION:
        raise Phase0Error("FUNDING_SCHEMA_VERSION_MISMATCH", record.schema_version)
    if record.provider != FUNDING_PROVIDER:
        raise Phase0Error("FUNDING_PROVIDER_MISMATCH", record.provider)
    if record.symbol not in CANONICAL_SYMBOLS:
        raise Phase0Error("FUNDING_SYMBOL_UNAUTHORIZED", record.symbol)
    if isinstance(record.funding_time_utc_ms, bool) or not isinstance(record.funding_time_utc_ms, int):
        raise Phase0Error("FUNDING_TIMESTAMP_INVALID", "funding timestamp must be int64 milliseconds")
    validate_canonical_decimal(record.funding_rate_decimal)
    if record.source_artifact_sha256 not in raw_hashes:
        raise Phase0Error("FUNDING_SOURCE_HASH_MISMATCH", record.source_artifact_sha256)
    if record.funding_record_id != funding_record_id(record.symbol, record.funding_time_utc_ms):
        raise Phase0Error("FUNDING_RECORD_ID_MISMATCH", record.funding_record_id)


def normalize_records(records: Iterable[FundingRecord], raw_hashes: set[str]) -> tuple[FundingRecord, ...]:
    accepted: list[FundingRecord] = []
    natural_keys: set[tuple[str, int]] = set()
    record_ids: set[str] = set()
    rates: dict[tuple[str, int], str] = {}
    for record in records:
        validate_record(record, raw_hashes)
        key = (record.symbol, record.funding_time_utc_ms)
        if key in natural_keys:
            code = "FUNDING_CONFLICTING_RATE" if rates[key] != record.funding_rate_decimal else "FUNDING_DUPLICATE_NATURAL_KEY"
            raise Phase0Error(code, f"duplicate funding event {key}")
        if record.funding_record_id in record_ids:
            raise Phase0Error("FUNDING_DUPLICATE_RECORD_ID", record.funding_record_id)
        natural_keys.add(key)
        record_ids.add(record.funding_record_id)
        rates[key] = record.funding_rate_decimal
        accepted.append(record)
    return tuple(sorted(accepted, key=lambda item: (item.symbol.encode("utf-8"), item.funding_time_utc_ms, item.funding_record_id)))


def serialize_jsonl(records: Sequence[FundingRecord]) -> bytes:
    ordered = normalize_records(records, {record.source_artifact_sha256 for record in records})
    lines = [json.dumps(record.as_ordered_mapping(), ensure_ascii=True, allow_nan=False, separators=(",", ":")) for record in ordered]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def parse_jsonl(payload: bytes, raw_hashes: set[str]) -> tuple[FundingRecord, ...]:
    if payload.startswith(b"\xef\xbb\xbf") or (payload and not payload.endswith(b"\n")):
        raise Phase0Error("FUNDING_MANIFEST_INVALID", "funding JSONL encoding or final LF is invalid")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Phase0Error("FUNDING_MANIFEST_INVALID", "funding JSONL is not UTF-8") from exc
    records: list[FundingRecord] = []
    for line in text.splitlines():
        if not line:
            raise Phase0Error("FUNDING_MANIFEST_INVALID", "blank funding JSONL line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase0Error("FUNDING_MANIFEST_INVALID", "invalid funding JSON") from exc
        if tuple(row) != FUNDING_FIELDS or set(row) != set(FUNDING_FIELDS):
            raise Phase0Error("FUNDING_MANIFEST_INVALID", "funding field order or registry mismatch")
        records.append(FundingRecord(**row))
    normalized = normalize_records(records, raw_hashes)
    if serialize_jsonl(normalized) != payload:
        raise Phase0Error("FUNDING_ORDER_INVALID", "funding JSONL is not canonical")
    return normalized


def verify_raw_artifact(raw_bytes: bytes, expected_sha256: str) -> None:
    if hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise Phase0Error("FUNDING_SOURCE_HASH_MISMATCH", "raw source bytes do not match manifest")


def validate_coverage(
    required_start_ms: int,
    required_end_ms: int,
    coverage_by_symbol: Mapping[str, tuple[int, int, bool]],
) -> None:
    if required_end_ms < required_start_ms:
        raise Phase0Error("FUNDING_MANIFEST_INVALID", "coverage interval is inverted")
    for symbol in CANONICAL_SYMBOLS:
        coverage = coverage_by_symbol.get(symbol)
        if coverage is None:
            raise Phase0Error("FUNDING_SOURCE_INCOMPLETE", f"missing coverage for {symbol}")
        start, end, complete = coverage
        if not complete or start > required_start_ms or end < required_end_ms:
            raise Phase0Error("FUNDING_NOT_COMPUTABLE", f"incomplete coverage for {symbol}")


def short_funding_return(
    records: Sequence[FundingRecord],
    symbol: str,
    entry_ms: int,
    termination_ms: int,
    coverage_complete: bool,
) -> Decimal:
    if not coverage_complete:
        raise Phase0Error("FUNDING_NOT_COMPUTABLE", "funding coverage is not proven complete")
    total = Decimal("0")
    for record in records:
        if record.symbol == symbol and funding_event_in_interval(entry_ms, termination_ms, record.funding_time_utc_ms):
            total += validate_canonical_decimal(record.funding_rate_decimal)
    return total


def funding_pnl(total_return: Decimal, frozen_notional: Decimal = Decimal("100.0")) -> Decimal:
    if frozen_notional != Decimal("100.0"):
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "E5 funding notional is frozen at 100.0")
    return frozen_notional * total_return


def funding_manifest(
    records: Sequence[FundingRecord],
    raw_artifacts: Mapping[str, bytes],
    coverage_by_symbol: Mapping[str, tuple[int, int, bool]],
    required_start_ms: int,
    required_end_ms: int,
) -> dict[str, object]:
    raw_hashes = {hashlib.sha256(payload).hexdigest() for payload in raw_artifacts.values()}
    normalized = normalize_records(records, raw_hashes)
    serialized = serialize_jsonl(normalized)
    validate_coverage(required_start_ms, required_end_ms, coverage_by_symbol)
    return {
        "schema_version": FUNDING_SCHEMA_VERSION,
        "provider": FUNDING_PROVIDER,
        "synthetic": True,
        "scientific_use": False,
        "required_coverage_start_utc_ms": required_start_ms,
        "required_coverage_end_utc_ms": required_end_ms,
        "raw_artifact_sha256": sorted(raw_hashes),
        "normalized_artifact_sha256": hashlib.sha256(serialized).hexdigest(),
        "normalized_record_count": len(normalized),
        "coverage_complete": True,
        "duplicate_check": "PASS",
        "reconciliation": "PASS",
    }
