"""Deterministic custodial projection of frozen E3 Fold 1-2 entries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aegis.e5_phase0.core import canonical_json_bytes

from .errors import BlindExportError, BlindExportInterrupted
from .guard import CleanRoomGuard
from .identity import (
    AUTHORITY_CLASSIFICATION,
    IDENTITY_SCHEME,
    CanonicalTradeIdentity,
    canonical_decimal,
    derive_canonical_trade_identity,
)
from .streaming import blind_trade_stream


CONFIG_SCHEMA_VERSION = "e5-blind-fold12-export-config-v1"
SOURCE_SCHEMA_VERSION = "e3-economic-report-static-contract-v1"
MANIFEST_SCHEMA_VERSION = "e5-historical-entry-manifest-v1"
OUTPUT_FIELDS = (
    "trade_id",
    "symbol",
    "fold",
    "signal_timestamp",
    "entry_timestamp",
    "entry_price",
    "side",
    "score",
    "strategy_id",
)
COMPONENTS = ("D3", "RV2", "TRRM", "QMAE", "EQM", "ECON1", "AEGIS")
HISTORICAL_UNAVAILABLE = "NOT_AVAILABLE_HISTORICAL_NOT_PERSISTED"
_TRADE_FIELDS = frozenset({
    "signal", "scenario_id", "entry_timestamp", "exit_timestamp", "entry_price",
    "exit_price", "gross_return_fraction", "cost_fraction", "net_return_fraction",
    "mfe_fraction", "mae_fraction",
})
_SIGNAL_FIELDS = frozenset({"timestamp", "symbol", "side", "score", "strategy_id", "fold", "regime"})


@dataclass(frozen=True)
class ExportConfig:
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    source_path: Path
    source_sha256: str
    output_root: Path
    manifest_name: str


@dataclass(frozen=True)
class ProjectedRow:
    source_ordinal: int
    identity: CanonicalTradeIdentity
    score_decimal: str
    strategy_id: str

    @property
    def trade_id(self) -> str:
        return self.identity.trade_id

    def canonical_bytes(self, fields: Sequence[str] = OUTPUT_FIELDS) -> bytes:
        if tuple(fields) != OUTPUT_FIELDS:
            raise BlindExportError("E5_BLIND_EXPORT_UNAUTHORIZED_FIELD", "output field registry mismatch")
        values = (
            ("trade_id", _json_string(self.identity.trade_id)),
            ("symbol", _json_string(self.identity.symbol)),
            ("fold", _json_string(self.identity.fold)),
            ("signal_timestamp", str(self.identity.signal_timestamp_utc_ms)),
            ("entry_timestamp", str(self.identity.entry_timestamp_utc_ms)),
            ("entry_price", self.identity.entry_price_decimal),
            ("side", _json_string(self.identity.side)),
            ("score", self.score_decimal),
            ("strategy_id", _json_string(self.strategy_id)),
        )
        return ("{" + ",".join(f"{_json_string(key)}:{value}" for key, value in values) + "}").encode("utf-8")


@dataclass(frozen=True)
class ExportResult:
    status: str
    manifest_path: Path
    manifest_sha256: str
    rows_emitted: int
    duplicate_rows: int
    conflicting_rows: int
    invalid_rows: int
    unavailable_nonessential_evidence: int
    unavailable_essential_evidence: int
    prohibited_partitions_emitted: int
    source_sha256: str
    configuration_sha256: str
    code_identity: str


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def package_code_identity() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_config(path: Path, repository_root: Path) -> ExportConfig:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "configuration unreadable") from exc
    required = {"schema_version", "source", "output", "authority", "allowed_output_fields", "components"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema_version") != CONFIG_SCHEMA_VERSION
    ):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "configuration schema")
    source = payload.get("source")
    output = payload.get("output")
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "configuration source/output")
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION or source.get("partition_selector") != "signal.fold":
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "source contract")
    if (
        tuple(payload.get("allowed_output_fields", ())) != OUTPUT_FIELDS
        or tuple(payload.get("components", ())) != COMPONENTS
    ):
        raise BlindExportError("E5_BLIND_EXPORT_UNAUTHORIZED_FIELD", "configuration field registry")
    source_path = repository_root / str(source.get("path", ""))
    output_root = repository_root / str(output.get("root", ""))
    source_hash = source.get("sha256")
    manifest_name = output.get("manifest_name")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or not isinstance(manifest_name, str):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "configuration hashes/names")
    config_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return ExportConfig(path, payload, config_hash, source_path, source_hash, output_root, manifest_name)


def _parse_authorized_row(record_bytes: bytes, source_ordinal: int, selected_fold: str) -> ProjectedRow | None:
    try:
        record = json.loads(record_bytes, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal}") from exc
    if not isinstance(record, dict) or set(record) - _TRADE_FIELDS:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal} field=trade")
    missing_trade = _TRADE_FIELDS - set(record)
    essential_trade = missing_trade & {"signal", "scenario_id", "entry_timestamp", "entry_price"}
    if essential_trade:
        field = sorted(essential_trade)[0]
        raise BlindExportError("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", f"ordinal={source_ordinal} field={field}")
    if missing_trade:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal} field=trade")
    signal = record.get("signal")
    if not isinstance(signal, dict) or set(signal) - _SIGNAL_FIELDS:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal} field=signal")
    missing_signal = _SIGNAL_FIELDS - set(signal)
    if "fold" in missing_signal:
        raise BlindExportError(
            "E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE",
            f"ordinal={source_ordinal} field=signal.fold",
        )
    essential_signal = missing_signal & {"timestamp", "symbol", "side", "score", "strategy_id"}
    if essential_signal:
        field = sorted(essential_signal)[0]
        raise BlindExportError("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", f"ordinal={source_ordinal} field={field}")
    if missing_signal:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal} field=signal")
    if record.get("scenario_id") != "B_BASE" or signal.get("strategy_id") != "full_stack":
        return None
    required = {
        "symbol": signal.get("symbol"),
        "fold": signal.get("fold"),
        "signal_timestamp_utc_ms": signal.get("timestamp"),
        "entry_timestamp_utc_ms": record.get("entry_timestamp"),
        "entry_price_decimal": record.get("entry_price"),
        "side": signal.get("side"),
    }
    try:
        identity = derive_canonical_trade_identity(required)
    except BlindExportError as exc:
        if exc.code == "E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING":
            raise BlindExportError(
                "E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE",
                f"ordinal={source_ordinal} {exc.message}",
            ) from exc
        raise
    if identity.fold != selected_fold:
        raise BlindExportError(
            "E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE",
            f"ordinal={source_ordinal} field=signal.fold",
        )
    if identity.side != "SHORT":
        raise BlindExportError("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", f"ordinal={source_ordinal} field=side")
    if identity.entry_timestamp_utc_ms - identity.signal_timestamp_utc_ms != 300_000:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", f"ordinal={source_ordinal} field=entry_timestamp")
    score = signal.get("score")
    if score is None:
        raise BlindExportError("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", f"ordinal={source_ordinal} field=score")
    score_decimal = canonical_decimal(score, "score")
    return ProjectedRow(source_ordinal, identity, score_decimal, "full_stack")


def validate_unique_rows(rows: Iterable[ProjectedRow]) -> tuple[ProjectedRow, ...]:
    ordered = tuple(sorted(rows, key=lambda row: row.source_ordinal))
    identities: dict[str, tuple[bytes, bytes]] = {}
    ordinals: set[int] = set()
    for row in ordered:
        if row.source_ordinal in ordinals:
            raise BlindExportError("E5_BLIND_EXPORT_DUPLICATE_IDENTITY", "duplicate source ordinal")
        ordinals.add(row.source_ordinal)
        payload = row.canonical_bytes()
        prior = identities.get(row.trade_id)
        if prior is not None:
            prior_preimage, prior_payload = prior
            if prior_preimage != row.identity.preimage:
                raise BlindExportError("E5_CANONICAL_TRADE_ID_HASH_COLLISION", f"trade_id={row.trade_id}")
            if prior_payload != payload:
                raise BlindExportError("E5_CANONICAL_TRADE_ID_CONFLICT", f"trade_id={row.trade_id}")
            raise BlindExportError("E5_BLIND_EXPORT_DUPLICATE_IDENTITY", f"trade_id={row.trade_id}")
        identities[row.trade_id] = (row.identity.preimage, payload)
    return ordered


def manifest_bytes(rows: Iterable[ProjectedRow], fields: Sequence[str] = OUTPUT_FIELDS) -> bytes:
    ordered = validate_unique_rows(rows)
    if tuple(fields) != OUTPUT_FIELDS:
        raise BlindExportError("E5_BLIND_EXPORT_UNAUTHORIZED_FIELD", "manifest field registry")
    return b"".join(row.canonical_bytes(fields) + b"\n" for row in ordered)


def _checkpoint_paths(output_root: Path) -> tuple[Path, Path]:
    root = output_root / "checkpoints"
    return root / "blind_export_checkpoint.json", root / "authorized_fold12_rows.partial.jsonl"


def _write_checkpoint(
    output_root: Path,
    rows: Sequence[ProjectedRow],
    source_ordinal: int,
    source_sha256: str,
    config_sha256: str,
    code_identity: str,
) -> None:
    state_path, rows_path = _checkpoint_paths(output_root)
    partial = manifest_bytes(rows)
    atomic_write(rows_path, partial)
    state = {
        "schema_version": "e5-blind-export-checkpoint-v1",
        "stage": "PARTITION_PROJECTING",
        "identity_scheme": IDENTITY_SCHEME,
        "source_sha256": source_sha256,
        "configuration_sha256": config_sha256,
        "code_identity": code_identity,
        "source_ordinal": source_ordinal,
        "authorized_rows": len(rows),
        "authorized_source_ordinals": [row.source_ordinal for row in rows],
        "authorized_rows_sha256": hashlib.sha256(partial).hexdigest(),
    }
    atomic_write(state_path, canonical_json_bytes(state))


def _row_from_manifest_line(line: bytes, ordinal: int) -> ProjectedRow:
    try:
        value = json.loads(line, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint rows invalid") from exc
    if not isinstance(value, dict) or tuple(value) != OUTPUT_FIELDS:
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint fields invalid")
    identity = derive_canonical_trade_identity({
        "symbol": value["symbol"],
        "fold": value["fold"],
        "signal_timestamp_utc_ms": value["signal_timestamp"],
        "entry_timestamp_utc_ms": value["entry_timestamp"],
        "entry_price_decimal": value["entry_price"],
        "side": value["side"],
    })
    if identity.trade_id != value["trade_id"] or value["strategy_id"] != "full_stack":
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint identity invalid")
    return ProjectedRow(ordinal, identity, canonical_decimal(value["score"], "score"), "full_stack")


def _load_checkpoint(
    output_root: Path,
    source_sha256: str,
    config_sha256: str,
    code_identity: str,
) -> tuple[list[ProjectedRow], int]:
    state_path, rows_path = _checkpoint_paths(output_root)
    try:
        state = json.loads(state_path.read_bytes())
        partial = rows_path.read_bytes()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint unreadable") from exc
    expected = (source_sha256, config_sha256, code_identity, IDENTITY_SCHEME)
    actual = (
        state.get("source_sha256"), state.get("configuration_sha256"),
        state.get("code_identity"), state.get("identity_scheme"),
    )
    if actual != expected or state.get("authorized_rows_sha256") != hashlib.sha256(partial).hexdigest():
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint dependency mismatch")
    lines = partial.splitlines()
    if len(lines) != state.get("authorized_rows"):
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint count mismatch")
    # Source ordinals are retained in deterministic order in checkpoint state by
    # reconstructing them from an internal sidecar list, never in the manifest.
    ordinals = state.get("authorized_source_ordinals")
    if not isinstance(ordinals, list) or len(ordinals) != len(lines):
        raise BlindExportError("E5_BLIND_EXPORT_CHECKPOINT_CONFLICT", "checkpoint ordinals invalid")
    rows = [_row_from_manifest_line(line, int(ordinal)) for line, ordinal in zip(lines, ordinals)]
    return rows, int(state.get("source_ordinal", -1))


def export_fold12(
    *,
    repository_root: Path,
    source: Path,
    expected_source_sha256: str,
    config_path: Path,
    output_root: Path,
    resume: bool = False,
    validation_only: bool = False,
    interrupt_after_authorized_rows: int | None = None,
) -> ExportResult:
    config = load_config(config_path, repository_root)
    if expected_source_sha256 != config.source_sha256:
        raise BlindExportError("E5_COMBINED_SOURCE_HASH_AUTHORITY_UNAVAILABLE", "CLI/config source hash disagreement")
    if output_root.resolve(strict=False) != config.output_root.resolve(strict=False):
        raise BlindExportError("E5_BLIND_EXPORT_OUTPUT_CONFLICT", "output root is not configured authority")
    manifest_path = output_root / "sealed" / config.manifest_name
    guard = CleanRoomGuard(
        config.source_path,
        manifest_path,
        (repository_root / "data/semi_blind", repository_root / "reports/experiments/semi_blind"),
        (repository_root / "data/lockbox", repository_root / "reports/experiments/lockbox"),
    )
    source_path = guard.validate_export_source(source)
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != expected_source_sha256:
        raise BlindExportError("E5_COMBINED_SOURCE_HASH_MISMATCH", "combined source hash")
    code_identity = package_code_identity()
    rows: list[ProjectedRow] = []
    resume_after = -1
    if resume:
        rows, resume_after = _load_checkpoint(output_root, actual_source_hash, config.sha256, code_identity)
    with blind_trade_stream(source_path) as (buffer, records):
        for span in records:
            if span.ordinal <= resume_after:
                continue
            if span.fold in {"F3", "F4"}:
                continue
            row = _parse_authorized_row(buffer[span.start:span.end], span.ordinal, span.fold)
            if row is None:
                continue
            rows.append(row)
            if interrupt_after_authorized_rows is not None and len(rows) >= interrupt_after_authorized_rows:
                _write_checkpoint(output_root, rows, span.ordinal, actual_source_hash, config.sha256, code_identity)
                raise BlindExportInterrupted()
    sealed = manifest_bytes(rows)
    digest = hashlib.sha256(sealed).hexdigest()
    status = "VALIDATED_ONLY" if validation_only else "CREATED"
    if not validation_only:
        if manifest_path.exists():
            existing = manifest_path.read_bytes()
            if existing != sealed:
                raise BlindExportError("E5_BLIND_EXPORT_OUTPUT_CONFLICT", "sealed manifest differs")
            status = "EXISTING_IDENTICAL"
        else:
            atomic_write(manifest_path, sealed)
        if sha256_file(manifest_path) != digest:
            raise BlindExportError("E5_BLIND_EXPORT_NONDETERMINISTIC", "sealed manifest hash mismatch")
    return ExportResult(
        status,
        manifest_path,
        digest,
        len(rows),
        0,
        0,
        0,
        len(rows) * len(COMPONENTS),
        0,
        0,
        actual_source_hash,
        config.sha256,
        code_identity,
    )


def load_sealed_manifest(guard: CleanRoomGuard, path: Path, expected_sha256: str) -> bytes:
    allowed = guard.validate_downstream_path(path)
    payload = allowed.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise BlindExportError("E5_BLIND_EXPORT_OUTPUT_CONFLICT", "sealed manifest hash mismatch")
    return payload
