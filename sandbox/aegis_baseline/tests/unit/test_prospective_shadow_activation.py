from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.prospective.activation import (
    COHORT_ID,
    ProspectiveActivationError,
    assert_event_admissible,
    create_activation_record,
    persist_activation_record,
    validate_activation_record,
)


def record(tmp_path: Path):
    authority = tmp_path / "authority.md"
    authority.write_text("frozen\n", encoding="utf-8")
    activated = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    return create_activation_record(
        authorization_path=authority,
        activation_timestamp=activated,
        python_commit="a" * 40,
        typescript_commit="b" * 40,
        model_identity="aegis-prospective-shadow-candidate-v1",
        model_bundle_sha256="c" * 64,
        trained_artifact_sha256="d" * 64,
        feature_contract_sha256="e" * 64,
        label_contract_sha256="f" * 64,
        signal_schema_sha256="1" * 64,
        outcome_schema_sha256="2" * 64,
        configuration_sha256="3" * 64,
        endpoint_policy_sha256="4" * 64,
        symbols=("BTCUSDT",),
        intervals=("5m",),
    )


def test_activation_is_atomic_immutable_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "activation.json"
    payload = record(tmp_path)
    digest = persist_activation_record(path, payload)
    assert len(digest) == 64
    assert validate_activation_record(path)["cohort_id"] == COHORT_ID
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ProspectiveActivationError, match="PROSPECTIVE_ACTIVATION_ALREADY_EXISTS"):
        persist_activation_record(path, payload)
    path.chmod(0o644)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["activation_state"] = "NOT_OPENED"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProspectiveActivationError, match="PROSPECTIVE_ACTIVATION_RECORD_CORRUPT"):
        validate_activation_record(path)


def test_preactivation_event_is_rejected_and_boundary_is_inclusive(tmp_path: Path) -> None:
    payload = record(tmp_path)
    boundary = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert_event_admissible(payload, boundary)
    with pytest.raises(ProspectiveActivationError, match="PROSPECTIVE_EVENT_BEFORE_ACTIVATION"):
        assert_event_admissible(payload, boundary - timedelta(milliseconds=1))


def test_live_private_credentials_and_budgets_are_frozen_off(tmp_path: Path) -> None:
    payload = record(tmp_path)
    assert payload["service_mode"] == "SHADOW"
    assert payload["approved_for_live"] is False
    assert payload["private_endpoints_enabled"] is False
    assert payload["credentials_enabled"] is False
    assert payload["order_operations_enabled"] is False
    assert payload["usd16_stage"] == payload["usd100_stage"] == "INACTIVE"
