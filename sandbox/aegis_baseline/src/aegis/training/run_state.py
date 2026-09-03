"""Persistent Phase-E run state, atomic checkpoints, and exclusive lockbox lease."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn

from ..utils import Sha256HashProvider, canonical_json, sha256_file
from .preregistration import LockboxBudget, PreregistrationError


class PhaseEErrorCode(str, Enum):
    PRECHECK_FAILED = "PRECHECK_FAILED"
    PREREGISTRATION_MISMATCH = "PREREGISTRATION_MISMATCH"
    D3_HASH_MISMATCH = "D3_HASH_MISMATCH"
    LOCKBOX_ALREADY_CONSUMED = "LOCKBOX_ALREADY_CONSUMED"
    LOCKBOX_ACQUISITION_FAILED = "LOCKBOX_ACQUISITION_FAILED"
    DATASET_INVALID = "DATASET_INVALID"
    FOLD_INVALID = "FOLD_INVALID"
    ARTIFACT_WRITE_FAILED = "ARTIFACT_WRITE_FAILED"
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"
    ENVIRONMENT_MISMATCH = "ENVIRONMENT_MISMATCH"
    GIT_COMMIT_MISMATCH = "GIT_COMMIT_MISMATCH"
    TECHNICAL_FAILURE_BEFORE_LOCKBOX = "TECHNICAL_FAILURE_BEFORE_LOCKBOX"
    TECHNICAL_FAILURE_AFTER_LOCKBOX = "TECHNICAL_FAILURE_AFTER_LOCKBOX"
    MODEL_NOT_BEATEN = "MODEL_NOT_BEATEN"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
    QMAE_COVERAGE_FAILED = "QMAE_COVERAGE_FAILED"
    ECON_NOT_POSITIVE = "ECON_NOT_POSITIVE"
    PROMOTION_CRITERIA_FAILED = "PROMOTION_CRITERIA_FAILED"
    FREEZE_VALIDATION_FAILED = "FREEZE_VALIDATION_FAILED"


class PhaseETechnicalError(RuntimeError):
    def __init__(self, code: PhaseEErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class PhaseEScientificRejection(RuntimeError):
    def __init__(self, code: PhaseEErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class RunMode(str, Enum):
    DRY_RUN = "dry-run"
    SMOKE_RUN = "smoke-run"
    VALIDATION_RUN = "validation-run"
    FULL_RUN = "full-run"


class PhaseEState(str, Enum):
    PRE_REGISTERED = "PRE_REGISTERED"
    PREFLIGHT_VALIDATED = "PREFLIGHT_VALIDATED"
    RUN_SNAPSHOT_CREATED = "RUN_SNAPSHOT_CREATED"
    DATASET_BUILT = "DATASET_BUILT"
    FOLDS_READY = "FOLDS_READY"
    TRAINING_IN_PROGRESS = "TRAINING_IN_PROGRESS"
    MODELS_EVALUATED = "MODELS_EVALUATED"
    MODELS_SELECTED = "MODELS_SELECTED"
    CALIBRATION_VALIDATED = "CALIBRATION_VALIDATED"
    QMAE_VALIDATED = "QMAE_VALIDATED"
    REFIT_COMPLETED = "REFIT_COMPLETED"
    THRESHOLD_DERIVED = "THRESHOLD_DERIVED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    LOCKBOX_ACQUIRED = "LOCKBOX_ACQUIRED"
    ECON_EVALUATED = "ECON_EVALUATED"
    CRITERIA_EVALUATED = "CRITERIA_EVALUATED"
    CANDIDATE = "CANDIDATE"
    REJECTED_EXPERIMENT = "REJECTED_EXPERIMENT"
    CANDIDATE_SIMULATED = "CANDIDATE_SIMULATED"
    REJECTED_SIMULATED = "REJECTED_SIMULATED"
    FAILED_SCIENTIFIC = "FAILED_SCIENTIFIC"
    FAILED_TECHNICAL_BEFORE_LOCKBOX = "FAILED_TECHNICAL_BEFORE_LOCKBOX"
    FAILED_TECHNICAL_AFTER_LOCKBOX = "FAILED_TECHNICAL_AFTER_LOCKBOX"


TERMINAL_STATES = frozenset({
    PhaseEState.CANDIDATE, PhaseEState.REJECTED_EXPERIMENT,
    PhaseEState.CANDIDATE_SIMULATED, PhaseEState.REJECTED_SIMULATED,
    PhaseEState.FAILED_SCIENTIFIC, PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX,
})


_LINEAR_TRANSITIONS = (
    PhaseEState.PRE_REGISTERED, PhaseEState.PREFLIGHT_VALIDATED,
    PhaseEState.RUN_SNAPSHOT_CREATED, PhaseEState.DATASET_BUILT, PhaseEState.FOLDS_READY,
    PhaseEState.TRAINING_IN_PROGRESS, PhaseEState.MODELS_EVALUATED,
    PhaseEState.MODELS_SELECTED, PhaseEState.CALIBRATION_VALIDATED,
    PhaseEState.QMAE_VALIDATED, PhaseEState.REFIT_COMPLETED,
    PhaseEState.THRESHOLD_DERIVED, PhaseEState.VALIDATION_COMPLETED,
    PhaseEState.LOCKBOX_ACQUIRED,
    PhaseEState.ECON_EVALUATED, PhaseEState.CRITERIA_EVALUATED,
)
ALLOWED_TRANSITIONS: dict[PhaseEState, frozenset[PhaseEState]] = {
    current: frozenset({following})
    for current, following in zip(_LINEAR_TRANSITIONS, _LINEAR_TRANSITIONS[1:])
}
ALLOWED_TRANSITIONS[PhaseEState.CRITERIA_EVALUATED] = frozenset({
    PhaseEState.CANDIDATE, PhaseEState.REJECTED_EXPERIMENT,
    PhaseEState.CANDIDATE_SIMULATED, PhaseEState.REJECTED_SIMULATED,
})
for state in _LINEAR_TRANSITIONS:
    failure = (
        PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX
        if _LINEAR_TRANSITIONS.index(state) >= _LINEAR_TRANSITIONS.index(PhaseEState.LOCKBOX_ACQUIRED)
        else PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX
    )
    ALLOWED_TRANSITIONS[state] = ALLOWED_TRANSITIONS.get(state, frozenset()) | frozenset({
        failure, PhaseEState.FAILED_SCIENTIFIC,
    })
ALLOWED_TRANSITIONS[PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX] = frozenset({
    PhaseEState.PREFLIGHT_VALIDATED,
})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EnvironmentFingerprint:
    python: str
    numpy: str
    sklearn: str
    platform: str
    content_hash: str

    @classmethod
    def current(cls) -> "EnvironmentFingerprint":
        unsigned = {
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(),
        }
        return cls(**unsigned, content_hash=Sha256HashProvider().digest_value(unsigned))


def deterministic_run_id(
    *, experiment_id: str, preregistration_hash: str, git_commit: str,
    environment_hash: str, mode: RunMode,
) -> str:
    return Sha256HashProvider().digest_value({
        "experiment_id": experiment_id, "preregistration_hash": preregistration_hash,
        "git_commit": git_commit, "environment_fingerprint": environment_hash, "mode": mode.value,
    })[:16]


def atomic_write_json(path: Path, value: Any, *, immutable: bool = True) -> str:
    """Write canonical JSON through an fsynced temporary and atomic replace."""
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    digest = Sha256HashProvider().digest_bytes(encoded)
    if path.exists():
        if path.read_bytes() == encoded:
            return digest
        if immutable:
            raise PhaseETechnicalError(PhaseEErrorCode.ARTIFACT_WRITE_FAILED, f"immutable artifact conflict: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, TypeError) as exc:
        raise PhaseETechnicalError(PhaseEErrorCode.ARTIFACT_WRITE_FAILED, f"atomic write failed: {path.name}") from exc
    return digest


@dataclass(frozen=True)
class StateTransition:
    sequence: int
    state: PhaseEState
    occurred_at: datetime
    git_commit: str
    environment_hash: str
    preregistration_hash: str
    artifact_hashes: Mapping[str, str]
    previous_transition_hash: str | None
    transition_hash: str


class RunStateStore:
    """Hash-chained state store whose recovery verifies every checkpoint."""

    def __init__(
        self, run_dir: Path, *, run_id: str, experiment_id: str,
        preregistration_hash: str, git_commit: str, environment_hash: str,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.state_path = self.run_dir / "state.json"
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.preregistration_hash = preregistration_hash
        self.git_commit = git_commit
        self.environment_hash = environment_hash

    @property
    def state(self) -> PhaseEState | None:
        history = self.recover()
        return history[-1].state if history else None

    def initialize(self, artifact_paths: Mapping[str, Path] | None = None) -> PhaseEState:
        if self.state_path.exists():
            history = self.recover()
            if not history:
                raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "empty run state")
            return history[-1].state
        self._write_transition(PhaseEState.PRE_REGISTERED, artifact_paths or {}, ())
        return PhaseEState.PRE_REGISTERED

    def transition(self, target: PhaseEState, artifact_paths: Mapping[str, Path]) -> PhaseEState:
        history = self.recover()
        if not history:
            raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "run state is not initialized")
        current = history[-1].state
        if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
            raise PhaseETechnicalError(
                PhaseEErrorCode.CHECKPOINT_INVALID, f"invalid Phase-E transition: {current.value}->{target.value}",
            )
        self._write_transition(target, artifact_paths, history)
        return target

    def recover(self) -> tuple[StateTransition, ...]:
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        if not self.state_path.exists():
            return ()
        try:
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
            if any(document.get(key) != expected for key, expected in (
                ("run_id", self.run_id), ("experiment_id", self.experiment_id),
                ("preregistration_hash", self.preregistration_hash), ("git_commit", self.git_commit),
                ("environment_hash", self.environment_hash),
            )):
                raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "run identity mismatch")
            history = []
            previous = None
            for index, item in enumerate(document.get("history", [])):
                unsigned = dict(item)
                claimed = str(unsigned.pop("transition_hash"))
                if unsigned.get("sequence") != index or unsigned.get("previous_transition_hash") != previous:
                    raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "state chain linkage mismatch")
                if Sha256HashProvider().digest_value(unsigned) != claimed:
                    raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "state transition hash mismatch")
                for relative, expected_hash in unsigned.get("artifact_hashes", {}).items():
                    artifact = self.run_dir / relative
                    if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                        raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, f"checkpoint artifact mismatch: {relative}")
                transition = StateTransition(
                    sequence=int(unsigned["sequence"]), state=PhaseEState(unsigned["state"]),
                    occurred_at=datetime.fromisoformat(str(unsigned["occurred_at"]).replace("Z", "+00:00")),
                    git_commit=str(unsigned["git_commit"]), environment_hash=str(unsigned["environment_hash"]),
                    preregistration_hash=str(unsigned["preregistration_hash"]),
                    artifact_hashes=dict(unsigned["artifact_hashes"]),
                    previous_transition_hash=unsigned.get("previous_transition_hash"), transition_hash=claimed,
                )
                history.append(transition)
                previous = claimed
            return tuple(history)
        except PhaseETechnicalError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "unable to recover run state") from exc

    def _write_transition(
        self, target: PhaseEState, artifact_paths: Mapping[str, Path], history: tuple[StateTransition, ...],
    ) -> None:
        hashes: dict[str, str] = {}
        for name, path in sorted(artifact_paths.items()):
            resolved = path.resolve()
            if not resolved.is_file() or self.run_dir not in resolved.parents:
                raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, f"required artifact is invalid: {name}")
            hashes[str(resolved.relative_to(self.run_dir))] = sha256_file(resolved)
        unsigned = {
            "sequence": len(history), "state": target.value, "occurred_at": utc_now(),
            "git_commit": self.git_commit, "environment_hash": self.environment_hash,
            "preregistration_hash": self.preregistration_hash, "artifact_hashes": hashes,
            "previous_transition_hash": history[-1].transition_hash if history else None,
        }
        item = {**unsigned, "transition_hash": Sha256HashProvider().digest_value(unsigned)}
        previous_items = [{
            "sequence": value.sequence, "state": value.state.value, "occurred_at": value.occurred_at,
            "git_commit": value.git_commit, "environment_hash": value.environment_hash,
            "preregistration_hash": value.preregistration_hash, "artifact_hashes": value.artifact_hashes,
            "previous_transition_hash": value.previous_transition_hash, "transition_hash": value.transition_hash,
        } for value in history]
        document = {
            "schema_version": "aegis-phase-e-run-state-v1", "run_id": self.run_id,
            "experiment_id": self.experiment_id, "preregistration_hash": self.preregistration_hash,
            "git_commit": self.git_commit, "environment_hash": self.environment_hash,
            "history": [*previous_items, item],
        }
        atomic_write_json(self.state_path, document, immutable=False)


@dataclass(frozen=True)
class LockboxLeaseRecord:
    run_id: str
    candidate_hash: str
    preregistration_hash: str
    experiment_id: str
    git_commit: str
    environment_fingerprint: str
    acquired_at: datetime
    pid: int
    mode: RunMode
    owner_authorization_hash: str
    physical_preregistration_hash: str = ""


class LockboxLease:
    """Primary O_EXCL exclusion paired with the persistent lockbox query budget."""

    def __init__(self, path: Path, budget: LockboxBudget) -> None:
        self.path = path
        self.budget = budget

    def acquire(self, record: LockboxLeaseRecord) -> LockboxLeaseRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED, "lockbox lease already exists") from exc
        except OSError as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "unable to acquire lockbox lease") from exc
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        try:
            self.budget.consume(
                candidate_hash=record.candidate_hash,
                purpose=f"phase-e:{record.run_id}", occurred_at=record.acquired_at,
            )
            self.validate_coherence(record)
        except (PreregistrationError, OSError, ValueError) as exc:
            raise PhaseETechnicalError(
                PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED,
                "lockbox lease and budget are inconsistent; manual review required",
            ) from exc
        return record

    def validate_coherence(self, expected: LockboxLeaseRecord) -> None:
        lease_exists = self.path.is_file()
        budget_exists = self.budget.path.is_file()
        if lease_exists != budget_exists or not lease_exists:
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "lease/budget presence mismatch")
        lease = json.loads(self.path.read_text(encoding="utf-8"))
        budget = json.loads(self.budget.path.read_text(encoding="utf-8"))
        queries = budget.get("queries", [])
        if (
            lease.get("run_id") != expected.run_id
            or lease.get("candidate_hash") != expected.candidate_hash
            or budget.get("preregistration_hash") != expected.preregistration_hash
            or len(queries) != 1
            or queries[0].get("candidate_hash") != expected.candidate_hash
            or queries[0].get("purpose") != f"phase-e:{expected.run_id}"
        ):
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "lease/budget content mismatch")


class SharedWindowLockboxLease:
    """Exclusive lease and single query record for a semi-blind window lineage."""

    def __init__(self, lease_path: Path, authority_path: Path) -> None:
        self.lease_path = lease_path
        self.authority_path = authority_path

    def acquire(self, record: LockboxLeaseRecord) -> LockboxLeaseRecord:
        if not self.authority_path.is_file():
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "shared authority is missing")
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        try:
            descriptor = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED, "shared lockbox lease exists") from exc
        try:
            os.write(descriptor, encoded); os.fsync(descriptor)
        finally:
            os.close(descriptor)
        authority = json.loads(self.authority_path.read_text(encoding="utf-8"))
        if authority.get("consumed_queries") or authority.get("status") != "NOT_CONSUMED":
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED, "shared window was consumed")
        lease_hash = Sha256HashProvider().digest_bytes(encoded)
        authority["consumed_queries"] = [{
            "experiment_id": record.experiment_id,
            "physical_hash": record.physical_preregistration_hash,
            "canonical_hash": record.preregistration_hash,
            "run_id": record.run_id, "candidate_hash": record.candidate_hash,
            "timestamp": record.acquired_at, "lease_hash": lease_hash,
        }]
        authority["status"] = "CONSUMED"
        atomic_write_json(self.authority_path, authority, immutable=False)
        self.validate_coherence(record)
        return record

    def validate_coherence(self, expected: LockboxLeaseRecord) -> None:
        if not self.lease_path.is_file() or not self.authority_path.is_file():
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "shared lease/authority missing")
        lease = json.loads(self.lease_path.read_text(encoding="utf-8"))
        authority = json.loads(self.authority_path.read_text(encoding="utf-8"))
        queries = authority.get("consumed_queries", [])
        if (
            len(queries) != 1 or authority.get("status") != "CONSUMED"
            or lease.get("run_id") != expected.run_id
            or queries[0].get("run_id") != expected.run_id
            or queries[0].get("candidate_hash") != expected.candidate_hash
            or queries[0].get("canonical_hash") != expected.preregistration_hash
        ):
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ACQUISITION_FAILED, "shared lockbox coherence mismatch")
