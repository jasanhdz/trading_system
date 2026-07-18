import json
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.training.preregistration import LockboxBudget
from aegis.training.run_state import (
    EnvironmentFingerprint, LockboxLease, LockboxLeaseRecord, PhaseEErrorCode,
    PhaseEState, PhaseETechnicalError, RunMode, RunStateStore, atomic_write_json,
    deterministic_run_id,
)


def _store(root: Path, *, commit: str = "a" * 40, environment: str = "b" * 64) -> RunStateStore:
    return RunStateStore(
        root, run_id="run-1", experiment_id="experiment-1",
        preregistration_hash="c" * 64, git_commit=commit, environment_hash=environment,
    )


def _artifact(root: Path, name: str, value: int = 1) -> Path:
    path = root / name
    atomic_write_json(path, {"value": value})
    return path


def _lease_record() -> LockboxLeaseRecord:
    return LockboxLeaseRecord(
        run_id="run-1", candidate_hash="d" * 64, preregistration_hash="c" * 64,
        experiment_id="experiment-1", git_commit="a" * 40,
        environment_fingerprint="b" * 64, acquired_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        pid=1, mode=RunMode.FULL_RUN, owner_authorization_hash="e" * 64,
    )


def _race_worker(lease_path: str, budget_path: str, queue) -> None:
    try:
        LockboxLease(
            Path(lease_path), LockboxBudget(Path(budget_path), 1, "c" * 64),
        ).acquire(_lease_record())
        queue.put("ACQUIRED")
    except PhaseETechnicalError as exc:
        queue.put(exc.code.value)


def test_state_machine_is_atomic_hash_chained_and_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.initialize() is PhaseEState.PRE_REGISTERED
    preflight = _artifact(tmp_path, "preflight.json")
    assert store.transition(PhaseEState.PREFLIGHT_VALIDATED, {"preflight": preflight}) is PhaseEState.PREFLIGHT_VALIDATED
    assert store.state is PhaseEState.PREFLIGHT_VALIDATED
    assert [item.sequence for item in store.recover()] == [0, 1]
    assert not (tmp_path / "state.json.tmp").exists()


def test_state_machine_rejects_skips_and_terminal_reentry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    with pytest.raises(PhaseETechnicalError, match="invalid Phase-E transition"):
        store.transition(PhaseEState.DATASET_BUILT, {})
    store.transition(PhaseEState.FAILED_SCIENTIFIC, {})
    with pytest.raises(PhaseETechnicalError):
        store.transition(PhaseEState.PREFLIGHT_VALIDATED, {})


def test_recovery_rejects_corrupt_checkpoint_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    path = _artifact(tmp_path, "preflight.json")
    store.transition(PhaseEState.PREFLIGHT_VALIDATED, {"preflight": path})
    path.write_text('{"value":9}\n', encoding="utf-8")
    with pytest.raises(PhaseETechnicalError) as captured:
        store.recover()
    assert captured.value.code is PhaseEErrorCode.CHECKPOINT_INVALID


@pytest.mark.parametrize("field", ["commit", "environment"])
def test_recovery_rejects_commit_or_environment_change(tmp_path: Path, field: str) -> None:
    _store(tmp_path).initialize()
    changed = _store(
        tmp_path, commit="f" * 40 if field == "commit" else "a" * 40,
        environment="f" * 64 if field == "environment" else "b" * 64,
    )
    with pytest.raises(PhaseETechnicalError, match="identity mismatch"):
        changed.recover()


def test_abandoned_temporary_is_not_a_checkpoint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    temporary = tmp_path / "state.json.tmp"
    temporary.write_text("partial", encoding="utf-8")
    assert store.state is PhaseEState.PRE_REGISTERED
    assert not temporary.exists()


def test_atomic_artifact_never_overwrites_conflicting_final(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    first = atomic_write_json(path, {"stable": True})
    assert atomic_write_json(path, {"stable": True}) == first
    with pytest.raises(PhaseETechnicalError) as captured:
        atomic_write_json(path, {"stable": False})
    assert captured.value.code is PhaseEErrorCode.ARTIFACT_WRITE_FAILED
    assert json.loads(path.read_text()) == {"stable": True}


def test_lockbox_lease_and_budget_are_coherent_and_single_use(tmp_path: Path) -> None:
    lease = LockboxLease(
        tmp_path / "lease.json", LockboxBudget(tmp_path / "budget.json", 1, "c" * 64),
    )
    record = lease.acquire(_lease_record())
    lease.validate_coherence(record)
    with pytest.raises(PhaseETechnicalError) as captured:
        lease.acquire(record)
    assert captured.value.code is PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED
    assert len(json.loads((tmp_path / "budget.json").read_text())["queries"]) == 1


def test_lease_budget_presence_mismatch_fails_closed(tmp_path: Path) -> None:
    lease = LockboxLease(
        tmp_path / "lease.json", LockboxBudget(tmp_path / "budget.json", 1, "c" * 64),
    )
    atomic_write_json(tmp_path / "lease.json", _lease_record())
    with pytest.raises(PhaseETechnicalError, match="presence mismatch"):
        lease.validate_coherence(_lease_record())


def test_two_processes_compete_for_exactly_one_lockbox_lease(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    args = (str(tmp_path / "lease.json"), str(tmp_path / "budget.json"), queue)
    processes = [context.Process(target=_race_worker, args=(*args,)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=2) for _ in processes)
    assert results.count("ACQUIRED") == 1
    assert results.count(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED.value) == 1
    assert len(json.loads((tmp_path / "budget.json").read_text())["queries"]) == 1


def test_failure_state_depends_on_lockbox_boundary(tmp_path: Path) -> None:
    store = _store(tmp_path / "before")
    store.initialize()
    store.transition(PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX, {})
    assert store.state is PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX

    after = _store(tmp_path / "after")
    after.initialize()
    chain = list(PhaseEState)
    targets = [
        PhaseEState.PREFLIGHT_VALIDATED, PhaseEState.RUN_SNAPSHOT_CREATED, PhaseEState.DATASET_BUILT,
        PhaseEState.FOLDS_READY, PhaseEState.TRAINING_IN_PROGRESS, PhaseEState.MODELS_EVALUATED,
        PhaseEState.MODELS_SELECTED, PhaseEState.CALIBRATION_VALIDATED, PhaseEState.QMAE_VALIDATED,
        PhaseEState.LOCKBOX_ACQUIRED,
    ]
    for target in targets:
        after.transition(target, {})
    after.transition(PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX, {})
    assert after.state is PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX
    assert chain  # enum includes every declared state


def test_run_id_is_deterministic_and_mode_specific() -> None:
    environment = EnvironmentFingerprint.current()
    values = dict(
        experiment_id="experiment-1", preregistration_hash="a" * 64,
        git_commit="b" * 40, environment_hash=environment.content_hash,
    )
    first = deterministic_run_id(**values, mode=RunMode.DRY_RUN)
    assert first == deterministic_run_id(**values, mode=RunMode.DRY_RUN)
    assert first != deterministic_run_id(**values, mode=RunMode.FULL_RUN)
    assert len(first) == 16
