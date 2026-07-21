"""Independent validation for sealed E5 Phase 1A manifests."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from .errors import BlindExportError
from .exporter import OUTPUT_FIELDS
from .identity import derive_canonical_trade_identity


REQUIRED_TEST_CATEGORIES = (
    "trade_id_repeatability",
    "trade_id_dictionary_order",
    "trade_id_source_order",
    "trade_id_equivalent_utc",
    "trade_id_signal_timestamp_difference",
    "trade_id_entry_timestamp_difference",
    "trade_id_equivalent_decimal",
    "trade_id_entry_price_difference",
    "trade_id_symbol_difference",
    "trade_id_fold_difference",
    "trade_id_side_difference",
    "trade_id_horizon_exclusion",
    "trade_id_source_run_exclusion",
    "trade_id_missing_field",
    "trade_id_invalid_symbol",
    "trade_id_invalid_fold",
    "trade_id_invalid_timestamp",
    "trade_id_invalid_entry_price",
    "trade_id_invalid_side",
    "trade_id_unsupported_version",
    "trade_id_outcome_dependency_denial",
    "trade_id_current_brain_dependency_denial",
    "trade_id_wall_clock_dependency_denial",
    "trade_id_filesystem_dependency_denial",
    "blind_export_fold1",
    "blind_export_fold2",
    "blind_export_fold3_denial",
    "blind_export_fold4_denial",
    "blind_export_unknown_fold",
    "blind_export_missing_fold",
    "blind_export_source_hash_preflight",
    "blind_export_source_schema",
    "blind_export_output_allowlist",
    "blind_export_historical_unavailable",
    "blind_export_essential_field",
    "blind_export_replay_denial",
    "blind_export_imputation_denial",
    "blind_export_duplicate",
    "blind_export_identity_conflict",
    "blind_export_hash_collision",
    "blind_export_canary_stdout",
    "blind_export_canary_stderr",
    "blind_export_canary_logs",
    "blind_export_canary_exceptions",
    "blind_export_canary_checkpoints",
    "blind_export_canary_reports",
    "blind_export_canary_manifest",
    "blind_export_input_order",
    "blind_export_repeatability",
    "blind_export_resume",
    "blind_export_output_conflict",
    "blind_export_existing_identity",
    "clean_room_combined_source_denial",
    "clean_room_sealed_manifest_access",
    "clean_room_semiblind_denial",
    "clean_room_lockbox_denial",
    "cli_discovery_denial",
    "cli_confirmation_denial",
    "cli_shadow_denial",
    "cli_live_denial",
    "offline_execution",
    "exchange_order_isolation",
    "typescript_repository_isolation",
)


def validate_manifest_bytes(payload: bytes) -> int:
    """Validate canonical rows without using scientific outcomes."""

    if payload and not payload.endswith(b"\n"):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "manifest newline")
    seen: set[str] = set()
    count = 0
    for line in payload.splitlines():
        try:
            row: Any = json.loads(line, parse_float=Decimal)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "manifest JSONL") from exc
        if not isinstance(row, dict) or tuple(row) != OUTPUT_FIELDS:
            raise BlindExportError("E5_BLIND_EXPORT_UNAUTHORIZED_FIELD", "manifest fields")
        if row["fold"] not in {"F1", "F2"}:
            raise BlindExportError(
                "E5_BLIND_EXPORT_PROHIBITED_PARTITION_EMISSION",
                "manifest partition",
            )
        identity = derive_canonical_trade_identity({
            "symbol": row["symbol"],
            "fold": row["fold"],
            "signal_timestamp_utc_ms": row["signal_timestamp"],
            "entry_timestamp_utc_ms": row["entry_timestamp"],
            "entry_price_decimal": row["entry_price"],
            "side": row["side"],
        })
        if identity.trade_id != row["trade_id"]:
            raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "manifest trade_id")
        if row["side"] != "SHORT" or row["strategy_id"] != "full_stack":
            raise BlindExportError("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", "manifest authority")
        if row["trade_id"] in seen:
            raise BlindExportError("E5_BLIND_EXPORT_DUPLICATE_IDENTITY", "manifest trade_id")
        seen.add(row["trade_id"])
        count += 1
    return count


if len(REQUIRED_TEST_CATEGORIES) != 63:  # pragma: no cover - import-time invariant
    raise RuntimeError("E5 Phase 1A validation registry must contain 63 categories")
