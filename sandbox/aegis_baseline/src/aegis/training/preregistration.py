"""Versioned preregistration validation and shared semi-blind lockbox authority."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..data import CanonicalSeriesSource, DataPurpose
from ..freeze import BundleLifecycleState
from ..utils import Sha256HashProvider, canonical_json, sha256_file


E1_PHYSICAL_SHA256 = "2604df3461ca891db05b6d877ab2e6373eac8fc7b7412d08571b2644f351ae39"
E1_CANONICAL_HASH = "ef2c4c236fb09d83886817a296a4aa39d337d7e4fbcbbcbc54e2d3e769c67e78"
E2_PHYSICAL_SHA256 = "759a7c87d2afaec2f6ede7b6451154a287ea5527e00036e040beb26c32732b8e"
E2_CANONICAL_HASH = "dad3d088f41d8aec7fd52f79a0dac8d53c87d6e201c22f7098220991ca8af308"
SEMI_BLIND_BOUNDARY = "2026-04-27T00:00:00Z"


class PreregistrationError(RuntimeError):
    pass


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _required(mapping: Mapping[str, Any], names: set[str], context: str) -> None:
    missing = names - set(mapping)
    if missing:
        raise PreregistrationError(f"{context} missing fields: {sorted(missing)}")


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
    protocol_version: int = 1
    executable: bool = False


def _validate_common(payload: Mapping[str, Any]) -> None:
    if payload.get("status") not in {"PRE_REGISTERED_NOT_EXECUTED", "PRE_REGISTERED_UNEXECUTED"} or payload.get("side") != "SHORT":
        raise PreregistrationError("experiment status or side is not eligible")
    if payload.get("allowed_sides") != ["SHORT"] or payload.get("timeframe") != "5m":
        raise PreregistrationError("experiment side/timeframe contract mismatch")
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
    source = payload["source"]
    if source.get("read_only") is not True or source.get("finality_gate_required") is not True:
        raise PreregistrationError("canonical source is not constrained read-only")


def _validate_v1(payload: Mapping[str, Any]) -> tuple[int, int]:
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
    lockbox = payload["lockbox"]
    if int(lockbox["maximum_queries"]) != 1 or _utc(lockbox["start"]) >= _utc(lockbox["end"]):
        raise PreregistrationError("lockbox contract is invalid")
    return len(folds), int(lockbox["maximum_queries"])


def _validate_inheritance(path: Path, payload: Mapping[str, Any]) -> None:
    parent_path = path.with_name("aegis_short_candidate_e1.yaml")
    if not parent_path.is_file() or sha256_file(parent_path) != E1_PHYSICAL_SHA256:
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: parent physical hash")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if Sha256HashProvider().digest_value(parent) != E1_CANONICAL_HASH:
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: parent canonical hash")
    supersedes = payload["supersedes"]
    expected_supersedes = {
        "experiment_id": "aegis-short-candidate-e1",
        "parent_physical_sha256": E1_PHYSICAL_SHA256,
        "parent_canonical_hash": E1_CANONICAL_HASH,
        "parent_status": "SUPERSEDED_INCOMPLETE_PROTOCOL_NEVER_EXECUTED",
    }
    if any(supersedes.get(key) != value for key, value in expected_supersedes.items()):
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: supersedes")
    if "NOT_CONSUMED" not in str(supersedes.get("no_semi_blind_inspection", "")):
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: semi-blind declaration")
    hasher = Sha256HashProvider()
    for key in ("source", "models", "econ", "promotion", "publication"):
        if hasher.digest_value(payload.get(key)) != hasher.digest_value(parent.get(key)):
            raise PreregistrationError(f"PREREGISTRATION_INHERITANCE_MISMATCH: {key}")
    scalar_keys = (
        "side", "allowed_sides", "timeframe", "universe_id", "symbol_set_hash",
        "feature_schema_version", "feature_hash", "label_schema_version",
    )
    if any(payload.get(key) != parent.get(key) for key in scalar_keys):
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: scientific scalar")
    for key in ("seed", "embargo_minutes", "horizon_bars", "fold_count"):
        if payload["protocol"].get(key) != parent["protocol"].get(key):
            raise PreregistrationError(f"PREREGISTRATION_INHERITANCE_MISMATCH: protocol.{key}")


def _validate_sampling(payload: Mapping[str, Any]) -> None:
    sampling = payload["sampling"]
    _required(sampling, {
        "schema_version", "cadence", "stride_bars", "anchor_rule", "timezone",
        "closed_final_candles_only", "coordinated_symbols_required", "history_bars",
        "horizon_bars", "label_window", "entry_rule", "overlap_policy", "gap_policy",
        "expected_rows",
    }, "sampling")
    if (
        sampling["schema_version"] != "aegis-sampling-v1"
        or sampling["cadence"] != "HOURLY_ANCHOR"
        or sampling["anchor_rule"] != "CLOSE_TIME_ON_THE_HOUR"
        or sampling["timezone"] != "UTC"
        or sampling["closed_final_candles_only"] is not True
        or int(sampling["coordinated_symbols_required"]) != 11
        or int(sampling["history_bars"]) != 288
        or int(sampling["stride_bars"]) != int(sampling["horizon_bars"])
        or int(sampling["horizon_bars"]) != 12
        or sampling["entry_rule"] != "NEXT_BAR_OPEN"
        or sampling["overlap_policy"] != "NONE_BY_CONSTRUCTION"
    ):
        raise PreregistrationError("E2 sampling contract mismatch")
    if sampling["gap_policy"] != {
        "history_gap": "SKIP_CYCLE_AND_COUNT",
        "future_gap": "QUARANTINE_LABEL_AND_COUNT",
        "interpolation": "FORBIDDEN",
    }:
        raise PreregistrationError("E2 gap policy mismatch")
    expected = sampling["expected_rows"]
    if (
        expected["first_anchor_utc"] != "2024-07-12T16:00:00Z"
        or expected["last_dev_anchor_utc"] != "2026-04-26T23:00:00Z"
        or int(expected["approximate_maximum_dev_rows"]) != 172500
        or not math.isclose(float(expected["hard_stop_if_valid_rows_below_fraction"]), 0.90)
    ):
        raise PreregistrationError("E2 expected-row contract mismatch")


def _validate_fold_protocol(payload: Mapping[str, Any]) -> None:
    protocol = payload["fold_protocol"]
    folds = protocol["folds"]
    if protocol.get("schema_version") != "aegis-fold-protocol-v1" or int(protocol.get("embargo_minutes", 0)) != 120:
        raise PreregistrationError("E2 fold protocol mismatch")
    if len(folds) != 4:
        raise PreregistrationError("E2 requires exactly four folds")
    previous_train_end: datetime | None = None
    reserve_start = _utc(payload["refit"]["final_calibration_reserve_start"])
    semi_blind = _utc(payload["lockbox"]["semi_blind_start"])
    for index, fold in enumerate(folds, start=1):
        if int(fold.get("id", 0)) != index:
            raise PreregistrationError("E2 folds are not ordered")
        train_start, train_end = _utc(fold["train_start"]), _utc(fold["train_end"])
        calibration_start, calibration_end = _utc(fold["calibration_start"]), _utc(fold["calibration_end"])
        scoring_start, scoring_end = _utc(fold["scoring_start"]), _utc(fold["scoring_end"])
        half_steps = math.floor(((scoring_end - calibration_start).total_seconds() / 2.0) / 300.0)
        calculated_end = calibration_start + timedelta(seconds=half_steps * 300)
        if calibration_end != calculated_end:
            raise PreregistrationError("E2 fold literal dates do not match internal split formula")
        if scoring_start != calibration_end + timedelta(minutes=120):
            raise PreregistrationError("E2 calibration/scoring embargo mismatch")
        if not (train_start < train_end and train_end + timedelta(minutes=120) <= calibration_start):
            raise PreregistrationError("E2 train/calibration chronology mismatch")
        if not (calibration_start < calibration_end < scoring_start < scoring_end):
            raise PreregistrationError("E2 fold blocks overlap or are empty")
        if previous_train_end is not None and train_end <= previous_train_end:
            raise PreregistrationError("E2 training windows are not expanding")
        if scoring_end >= reserve_start or scoring_end >= semi_blind:
            raise PreregistrationError("E2 scoring touches reserve or semi-blind")
        previous_train_end = train_end


def _validate_v2(path: Path, payload: Mapping[str, Any]) -> tuple[int, int]:
    if int(payload.get("protocol_version", 0)) != 2 or payload.get("experiment_id") != "aegis-short-candidate-e2":
        raise PreregistrationError("PROTOCOL_VERSION_NOT_EXECUTABLE")
    _required(payload, {
        "supersedes", "sampling", "fold_protocol", "calibration", "scoring", "refit",
        "threshold_derivation", "lockbox", "models", "econ", "promotion", "publication",
    }, "E2 preregistration")
    _validate_inheritance(path, payload)
    _validate_sampling(payload)
    local_payload = dict(payload); local_payload["_path"] = str(path)
    _validate_fold_protocol(local_payload)
    calibration = payload["calibration"]
    if calibration != {
        "fit_block": "CALIBRATION",
        "applies_to": ["trrm_tail_probability", "eqm_clean_probability"],
        "qmae_conformal_residuals_block": "CALIBRATION",
        "measurement_block": "SCORING",
        "family_selection_rule": "config/scientific_competition_v1.yaml",
        "forbidden": ["fitting_on_scoring", "model_selection_on_calibration_block", "conformal_residuals_from_scoring"],
    }:
        raise PreregistrationError("E2 calibration contract mismatch")
    if payload["scoring"] != {
        "model_selection_metrics": "RAW_RANKING_METRICS_STABILITY_FIRST",
        "selection_rule": "config/scientific_competition_v1.yaml",
        "econ_per_fold_block": "SCORING", "fitting": "FORBIDDEN",
    }:
        raise PreregistrationError("E2 scoring contract mismatch")
    refit = payload["refit"]
    if (
        refit["final_train_start"] != "2024-07-11T15:20:00Z"
        or refit["final_train_end"] != "2026-02-28T23:59:59Z"
        or _utc(refit["final_train_end"]) + timedelta(minutes=int(refit["embargo_minutes"])) > _utc(refit["final_calibration_reserve_start"])
        or refit["final_calibration_reserve_end"] != "2026-04-26T23:59:59Z"
        or refit["eqm_refit_population"] != "TRRM_VETO_SURVIVORS_OF_FINAL_TRAIN"
    ):
        raise PreregistrationError("E2 refit contract mismatch")
    threshold = payload["threshold_derivation"]
    if (
        threshold["method"] != "ABSOLUTE_QUANTILE_ON_RESERVE_SURVIVORS"
        or float(threshold["quantile"]) != 0.90
        or threshold["executed"] != "PRE_LOCKBOX"
        or threshold["value"] is not None
    ):
        raise PreregistrationError("E2 threshold derivation contract mismatch")
    lockbox = payload["lockbox"]
    if (
        lockbox["window_id"] != "semi-blind-20260427-20260711"
        or lockbox["semi_blind_start"] != SEMI_BLIND_BOUNDARY
        or lockbox["semi_blind_end"] != "2026-07-11T09:20:00Z"
        or int(lockbox["maximum_queries_total_across_preregistrations"]) != 1
        or lockbox["shared_with"] != ["aegis-short-candidate-e1", "aegis-short-candidate-e2"]
        or lockbox["final_test_is_lockbox"] is not True
    ):
        raise PreregistrationError("E2 shared lockbox contract mismatch")
    return len(payload["fold_protocol"]["folds"]), 1


def _validate_v3(path: Path, payload: Mapping[str, Any]) -> tuple[int, int]:
    if int(payload.get("protocol_version", 0)) != 3 or payload.get("experiment_id") != "aegis-short-candidate-e3":
        raise PreregistrationError("PROTOCOL_VERSION_NOT_EXECUTABLE")
    parent_path = path.with_name("aegis_short_candidate_e2.yaml")
    if not parent_path.is_file() or sha256_file(parent_path) != E2_PHYSICAL_SHA256:
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: E2 physical hash")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if Sha256HashProvider().digest_value(parent) != E2_CANONICAL_HASH:
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: E2 canonical hash")
    supersedes = payload.get("supersedes", {})
    expected = {
        "experiment_id": "aegis-short-candidate-e2",
        "parent_physical_sha256": E2_PHYSICAL_SHA256,
        "parent_canonical_hash": E2_CANONICAL_HASH,
        "parent_status": "SUPERSEDED_UNDERSPECIFIED_NEVER_CONSUMED_LOCKBOX",
        "no_semi_blind_inspection": True,
        "shared_lockbox_authority": True,
        "additional_lockbox_budget": 0,
    }
    if any(supersedes.get(key) != value for key, value in expected.items()):
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: E3 supersedes")
    validation_run = supersedes.get("parent_validation_run", {})
    if validation_run != {
        "run_id": "b49f1ec7c71fcf70",
        "classification": "VALID_DIAGNOSTIC_NONCOMPARABLE",
        "threshold_draft_status": "RETAINED_FOR_AUDIT_ONLY",
    }:
        raise PreregistrationError("PREREGISTRATION_INHERITANCE_MISMATCH: E2 classification")
    hasher = Sha256HashProvider()
    for key in ("source", "sampling", "fold_protocol", "refit", "threshold_derivation", "econ", "promotion", "publication"):
        if hasher.digest_value(payload.get(key)) != hasher.digest_value(parent.get(key)):
            raise PreregistrationError(f"PREREGISTRATION_INHERITANCE_MISMATCH: {key}")
    for key in ("side", "allowed_sides", "timeframe", "universe_id", "symbol_set_hash", "feature_schema_version", "feature_hash", "label_schema_version", "protocol"):
        if hasher.digest_value(payload.get(key)) != hasher.digest_value(parent.get(key)):
            raise PreregistrationError(f"PREREGISTRATION_INHERITANCE_MISMATCH: {key}")
    _validate_sampling(payload)
    _validate_fold_protocol(payload)
    competition = payload.get("models", {}).get("competition_protocol", {})
    competition_path = (path.parents[2] / str(competition.get("path", ""))).resolve()
    if not competition_path.is_file() or sha256_file(competition_path) != competition.get("physical_sha256"):
        raise PreregistrationError("E3 competition protocol hash mismatch")
    if payload.get("calibration", {}).get("family_selection_rule") != "config/scientific_competition_v2.yaml":
        raise PreregistrationError("E3 calibration protocol mismatch")
    if payload.get("scoring", {}).get("selection_rule") != "config/scientific_competition_v2.yaml":
        raise PreregistrationError("E3 scoring protocol mismatch")
    if payload.get("eqm_population") != {
        "fold_train": "TRRM_VETO_SURVIVORS_OF_FOLD_TRAIN",
        "fold_scoring": "TRRM_VETO_SURVIVORS_OF_FOLD_SCORING",
        "refit": "TRRM_VETO_SURVIVORS_OF_FINAL_TRAIN",
    }:
        raise PreregistrationError("E3 EQM population contract mismatch")
    veto = payload.get("trrm_veto", {})
    if veto.get("mechanics") != "RANK_BASED_BUDGET" or float(veto.get("veto_budget", -1)) != 0.30:
        raise PreregistrationError("E3 TRRM veto contract mismatch")
    if veto.get("raw_probability_0_70_placeholder") != "FORBIDDEN":
        raise PreregistrationError("E3 raw probability veto is forbidden")
    if payload.get("econ_baselines") != {
        "config_source": "config/scientific_competition_v2.yaml",
        "equal_budget_required": True,
        "equal_costs_required": True,
        "fixed_seed_required": True,
    }:
        raise PreregistrationError("E3 ECON baseline contract mismatch")
    lockbox = payload.get("lockbox", {})
    if lockbox.get("shared_with") != ["aegis-short-candidate-e1", "aegis-short-candidate-e2", "aegis-short-candidate-e3"] or int(lockbox.get("additional_lockbox_budget", -1)) != 0:
        raise PreregistrationError("E3 shared lockbox contract mismatch")
    return len(payload["fold_protocol"]["folds"]), 1


def load_and_validate_preregistration(
    path: Path, *, audit_source: bool = True, require_executable: bool = False,
) -> tuple[Mapping[str, Any], PreregistrationAudit]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PreregistrationError("unsupported experiment preregistration")
    schema = payload.get("schema_version")
    if schema not in {"aegis-candidate-preregistration-v1", "aegis-candidate-preregistration-v2", "aegis-candidate-preregistration-v3"}:
        raise PreregistrationError("unsupported experiment preregistration")
    _validate_common(payload)
    if payload["protocol"].get("threshold_value") is not None:
        raise PreregistrationError("productive threshold must remain pending before the full run")
    if schema.endswith("v1"):
        fold_count, lockbox_queries = _validate_v1(payload)
        protocol_version, executable = 1, False
        if require_executable:
            raise PreregistrationError("PROTOCOL_VERSION_NOT_EXECUTABLE")
    elif schema.endswith("v2"):
        fold_count, lockbox_queries = _validate_v2(path.resolve(), payload)
        protocol_version, executable = 2, False
        if require_executable:
            raise PreregistrationError("PROTOCOL_VERSION_NOT_EXECUTABLE")
    else:
        fold_count, lockbox_queries = _validate_v3(path.resolve(), payload)
        protocol_version, executable = 3, True
    if audit_source:
        CanonicalSeriesSource(
            Path(payload["source"]["path"]), DataPurpose.TRAINING,
            expected_manifest_sha256=str(payload["source"]["manifest_sha256"]),
        ).audit(verify_content=True)
    digest = Sha256HashProvider().digest_value(payload)
    return payload, PreregistrationAudit(
        str(payload["experiment_id"]), digest, str(payload["source"]["manifest_sha256"]),
        fold_count, int(payload["protocol"]["embargo_minutes"]), lockbox_queries,
        True, False, BundleLifecycleState(str(payload["publication"]["initial_state"])),
        protocol_version, executable,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class SharedLockboxAuthority:
    path: Path
    window_id: str
    start: str
    end: str
    maximum_queries: int
    lineage: str = "AEGIS_CLEAN_REBUILD_SHORT"

    def initialize(self, *, e1_hashes: Mapping[str, str], e2_hashes: Mapping[str, str]) -> Mapping[str, Any]:
        expected = {
            "schema_version": "aegis-shared-lockbox-v1", "window_id": self.window_id,
            "semi_blind_start": self.start, "semi_blind_end": self.end,
            "maximum_queries_total": self.maximum_queries, "consumed_queries": [],
            "lineage": self.lineage,
            "preregistrations": {
                "aegis-short-candidate-e1": dict(e1_hashes),
                "aegis-short-candidate-e2": dict(e2_hashes),
            },
            "status": "NOT_CONSUMED",
        }
        if self.path.exists():
            actual = json.loads(self.path.read_text(encoding="utf-8"))
            if actual != expected:
                raise PreregistrationError("LOCKBOX_AUTHORITY_INCOMPATIBLE")
            return actual
        _atomic_json(self.path, expected)
        return expected

    def audit_available(self) -> Mapping[str, Any]:
        if not self.path.is_file():
            raise PreregistrationError("shared lockbox authority is missing")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            payload.get("window_id") != self.window_id
            or payload.get("semi_blind_start") != self.start
            or payload.get("semi_blind_end") != self.end
            or int(payload.get("maximum_queries_total", 0)) != self.maximum_queries
        ):
            raise PreregistrationError("LOCKBOX_AUTHORITY_INCOMPATIBLE")
        if payload.get("consumed_queries") or payload.get("status") != "NOT_CONSUMED":
            raise PreregistrationError("LOCKBOX_ALREADY_CONSUMED")
        return payload


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
        _atomic_json(self.path, state)
        return len(queries)
