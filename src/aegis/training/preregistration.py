"""Validation and lockbox controls for an owner-authorized candidate run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..data import CanonicalSeriesSource, DataPurpose
from ..freeze import BundleLifecycleState
from ..utils import Sha256HashProvider, canonical_json


class PreregistrationError(RuntimeError):
    pass


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class PreregistrationAudit:
    experiment_id: str
    content_hash: str
    source_manifest_hash: str
    fold_count: int
    embargo_minutes: int
    lockbox_queries: int
    threshold_pending: bool
    full_run_executed: bool
    initial_lifecycle: BundleLifecycleState


def load_and_validate_preregistration(path: Path, *, audit_source: bool = True) -> tuple[Mapping[str, Any], PreregistrationAudit]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "aegis-candidate-preregistration-v1":
        raise PreregistrationError("unsupported experiment preregistration")
    if payload.get("status") != "PRE_REGISTERED_NOT_EXECUTED" or payload.get("side") != "SHORT":
        raise PreregistrationError("experiment status or side is not eligible")
    if payload.get("allowed_sides") != ["SHORT"] or payload.get("timeframe") != "5m":
        raise PreregistrationError("experiment side/timeframe contract mismatch")
    protocol = payload["protocol"]
    embargo = int(protocol["embargo_minutes"])
    folds = protocol["folds"]
    if len(folds) != int(protocol["fold_count"]) or len(folds) != 4 or embargo != 120:
        raise PreregistrationError("fold/embargo contract mismatch")
    for fold in folds:
        train_end = _utc(fold["train_end"])
        validation_start = _utc(fold["validation_start"])
        if validation_start - train_end < timedelta(minutes=embargo):
            raise PreregistrationError("fold violates temporal embargo")
        if _utc(fold["train_start"]) >= train_end or validation_start >= _utc(fold["validation_end"]):
            raise PreregistrationError("fold chronology is invalid")
    if protocol.get("threshold_value") is not None:
        raise PreregistrationError("productive threshold must remain pending before the full run")
    mandatory = payload["promotion"]["mandatory"]
    required = {
        "minimum_test_signals", "minimum_positive_folds", "minimum_profit_factor_base",
        "minimum_net_expectancy_base", "minimum_worst_fold_expectancy",
        "beat_best_directional_baseline_expectancy", "maximum_symbol_concentration",
        "require_no_known_leakage", "require_side_separated_metrics",
        "qmae_coverage_minimum_each_fold", "qmae_coverage_maximum_each_fold", "maximum_ece_each_fold",
    }
    if not required <= set(mandatory):
        raise PreregistrationError("promotion criteria are incomplete")
    lockbox = payload["lockbox"]
    if int(lockbox["maximum_queries"]) != 1 or _utc(lockbox["start"]) >= _utc(lockbox["end"]):
        raise PreregistrationError("lockbox contract is invalid")
    source = payload["source"]
    if source.get("read_only") is not True or source.get("finality_gate_required") is not True:
        raise PreregistrationError("canonical source is not constrained read-only")
    if audit_source:
        CanonicalSeriesSource(
            Path(source["path"]), DataPurpose.TRAINING,
            expected_manifest_sha256=str(source["manifest_sha256"]),
        ).audit(verify_content=True)
    digest = Sha256HashProvider().digest_value(payload)
    return payload, PreregistrationAudit(
        str(payload["experiment_id"]), digest, str(source["manifest_sha256"]), len(folds), embargo,
        int(lockbox["maximum_queries"]), True, False,
        BundleLifecycleState(str(payload["publication"]["initial_state"])),
    )


@dataclass(frozen=True)
class LockboxBudget:
    path: Path
    maximum_queries: int
    preregistration_hash: str

    def consume(self, *, candidate_hash: str, purpose: str, occurred_at: datetime) -> int:
        if not candidate_hash or not purpose or occurred_at.tzinfo is None:
            raise PreregistrationError("lockbox query metadata is invalid")
        state = {"schema_version": "aegis-lockbox-state-v1", "preregistration_hash": self.preregistration_hash, "queries": []}
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("preregistration_hash") != self.preregistration_hash:
            raise PreregistrationError("lockbox preregistration hash mismatch")
        queries = state.get("queries", [])
        if len(queries) >= self.maximum_queries:
            raise PreregistrationError("lockbox query budget exhausted")
        queries.append({"candidate_hash": candidate_hash, "purpose": purpose, "occurred_at": occurred_at})
        state["queries"] = queries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(canonical_json(state) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        return len(queries)

