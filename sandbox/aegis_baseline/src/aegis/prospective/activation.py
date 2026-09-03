"""Atomic activation contract for the first prospective Shadow cohort."""

from __future__ import annotations

import json
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aegis.utils import Sha256HashProvider, canonical_json, sha256_file


ACTIVATION_SCHEMA = "aegis-prospective-shadow-activation-v1"
COHORT_ID = "aegis-prospective-shadow-cohort-1"
PROTOCOL_VERSION = "aegis-prospective-validation-v1"


class ProspectiveActivationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha(value: str, code: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProspectiveActivationError(code)
    return value


def _commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ProspectiveActivationError("PROSPECTIVE_CODE_HASH_INVALID")
    return value


def canonical_activation_payload(payload: Mapping[str, Any]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def create_activation_record(
    *,
    authorization_path: Path,
    activation_timestamp: datetime,
    python_commit: str,
    typescript_commit: str,
    model_identity: str,
    model_bundle_sha256: str,
    trained_artifact_sha256: str,
    feature_contract_sha256: str,
    label_contract_sha256: str,
    signal_schema_sha256: str,
    outcome_schema_sha256: str,
    configuration_sha256: str,
    endpoint_policy_sha256: str,
    symbols: tuple[str, ...],
    intervals: tuple[str, ...],
) -> Mapping[str, Any]:
    if activation_timestamp.tzinfo is None or activation_timestamp.utcoffset() is None:
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_TIME_INVALID")
    timestamp = activation_timestamp.astimezone(timezone.utc)
    if not authorization_path.is_file():
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_AUTHORITY_MISSING")
    hashes = {
        "authorization_document_sha256": sha256_file(authorization_path),
        "model_bundle_sha256": _sha(model_bundle_sha256, "PROSPECTIVE_MODEL_HASH_INVALID"),
        "trained_artifact_sha256": _sha(trained_artifact_sha256, "PROSPECTIVE_MODEL_HASH_INVALID"),
        "feature_contract_sha256": _sha(feature_contract_sha256, "PROSPECTIVE_FEATURE_HASH_INVALID"),
        "label_contract_sha256": _sha(label_contract_sha256, "PROSPECTIVE_LABEL_HASH_INVALID"),
        "signal_schema_sha256": _sha(signal_schema_sha256, "PROSPECTIVE_SCHEMA_HASH_INVALID"),
        "outcome_schema_sha256": _sha(outcome_schema_sha256, "PROSPECTIVE_SCHEMA_HASH_INVALID"),
        "configuration_sha256": _sha(configuration_sha256, "PROSPECTIVE_CONFIG_HASH_INVALID"),
        "public_endpoint_policy_sha256": _sha(endpoint_policy_sha256, "SHADOW_ENDPOINT_POLICY_HASH_INVALID"),
    }
    record: dict[str, Any] = {
        "schema_id": ACTIVATION_SCHEMA,
        "activation_id": f"shadow-cohort-1-{int(timestamp.timestamp() * 1000)}",
        "cohort_id": COHORT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "activation_timestamp_utc": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "activation_timestamp_utc_epoch_ms": int(timestamp.timestamp() * 1000),
        "python_commit": _commit(python_commit),
        "typescript_commit": _commit(typescript_commit),
        "model_identity": model_identity,
        **hashes,
        "symbol_universe": list(symbols),
        "interval_universe": list(intervals),
        "service_mode": "SHADOW",
        "approved_for_live": False,
        "usd16_stage": "INACTIVE",
        "usd100_stage": "INACTIVE",
        "historical_e5_state": "HISTORICAL_E5_NON_EXECUTABLE_MISSING_CONTEMPORANEOUS_ROW_TARGETS",
        "lockbox_state": "NOT_CONSUMED",
        "activation_state": "OPENED",
        "prospective_cohort": "ACTIVE",
        "private_endpoints_enabled": False,
        "credentials_enabled": False,
        "order_operations_enabled": False,
    }
    record["content_sha256"] = Sha256HashProvider().digest_value(record)
    return record


def validate_activation_record(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_RECORD_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_RECORD_INVALID")
    unsigned = dict(payload)
    claimed = str(unsigned.pop("content_sha256", ""))
    if claimed != Sha256HashProvider().digest_value(unsigned):
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_RECORD_CORRUPT")
    required = {
        "schema_id": ACTIVATION_SCHEMA,
        "cohort_id": COHORT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "service_mode": "SHADOW",
        "activation_state": "OPENED",
        "prospective_cohort": "ACTIVE",
        "approved_for_live": False,
        "usd16_stage": "INACTIVE",
        "usd100_stage": "INACTIVE",
        "lockbox_state": "NOT_CONSUMED",
        "private_endpoints_enabled": False,
        "credentials_enabled": False,
        "order_operations_enabled": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_RECORD_INVALID")
    return payload


def persist_activation_record(path: Path, payload: Mapping[str, Any]) -> str:
    """Create once, flush the file and parent directory, and reject rewrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_activation_payload(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_ALREADY_EXISTS") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise ProspectiveActivationError("PROSPECTIVE_ACTIVATION_PERSISTENCE_FAILED") from exc
    validate_activation_record(path)
    return sha256_file(path)


def assert_event_admissible(record: Mapping[str, Any], event_timestamp: datetime) -> None:
    if event_timestamp.tzinfo is None or event_timestamp.utcoffset() is None:
        raise ProspectiveActivationError("PROSPECTIVE_EVENT_TIME_INVALID")
    activated_ms = int(record["activation_timestamp_utc_epoch_ms"])
    if int(event_timestamp.astimezone(timezone.utc).timestamp() * 1000) < activated_ms:
        raise ProspectiveActivationError("PROSPECTIVE_EVENT_BEFORE_ACTIVATION")


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically open prospective Shadow Cohort 1")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-commit", required=True)
    parser.add_argument("--typescript-commit", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--model-bundle-sha256", required=True)
    parser.add_argument("--trained-artifact-sha256", required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--label-contract-sha256", required=True)
    parser.add_argument("--signal-schema-sha256", required=True)
    parser.add_argument("--outcome-schema-sha256", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--endpoint-policy-sha256", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", required=True)
    args = parser.parse_args()
    payload = create_activation_record(
        authorization_path=args.authorization,
        activation_timestamp=datetime.now(timezone.utc),
        python_commit=args.python_commit,
        typescript_commit=args.typescript_commit,
        model_identity=args.model_identity,
        model_bundle_sha256=args.model_bundle_sha256,
        trained_artifact_sha256=args.trained_artifact_sha256,
        feature_contract_sha256=args.feature_contract_sha256,
        label_contract_sha256=args.label_contract_sha256,
        signal_schema_sha256=args.signal_schema_sha256,
        outcome_schema_sha256=args.outcome_schema_sha256,
        configuration_sha256=args.configuration_sha256,
        endpoint_policy_sha256=args.endpoint_policy_sha256,
        symbols=tuple(args.symbols),
        intervals=tuple(args.intervals),
    )
    digest = persist_activation_record(args.output, payload)
    print(canonical_json({"activation_id": payload["activation_id"], "output_sha256": digest, "status": "OPENED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
