# Aegis Phase-E Orchestrator

## Objective

The Phase-E orchestrator replaces the historical operator-coordinated runbook with a
persistent, deterministic coordinator. It does not implement features, labels, models,
calibration, QMAE, ECON, selection, freeze, or evidence hashing. Those responsibilities
remain in their existing scientific modules.

This implementation did not execute a real full-run, consume the real lockbox, derive a
productive threshold, or publish a real CANDIDATE.

## Architecture

- `training/run_state.py` owns states, legal transitions, atomic JSON writes, checkpoint
  verification, deterministic run IDs, recovery, and the exclusive lockbox lease.
- `training/phase_e.py` owns preflight, orchestration contracts, artifact manifests,
  promotion checks, simulated mechanics, and the structurally complete candidate
  publication path.
- `scripts/run_aegis_candidate_experiment.py` only parses CLI arguments, enforces the
  full-run authorization phrase, selects a backend, invokes the orchestrator, and returns
  a structured exit status.

The orchestrator depends on the existing APIs in `competition.py`, `econ.py`, `models.py`,
`freeze.py`, `preregistration.py`, and `evidence.py`. The smoke backend invokes the real
competition, calibration, QMAE, tree-export, bundle-loader, and ECON APIs with reduced,
deterministic fixtures. It is not an alternative scientific implementation and cannot
publish a real CANDIDATE.

## State machine

The main path is:

```text
PRE_REGISTERED -> PREFLIGHT_VALIDATED -> RUN_SNAPSHOT_CREATED -> DATASET_BUILT
-> FOLDS_READY -> TRAINING_IN_PROGRESS -> MODELS_EVALUATED -> MODELS_SELECTED
-> CALIBRATION_VALIDATED -> QMAE_VALIDATED -> REFIT_COMPLETED
-> THRESHOLD_DERIVED -> VALIDATION_COMPLETED -> LOCKBOX_ACQUIRED
-> ECON_EVALUATED -> CRITERIA_EVALUATED
-> CANDIDATE | REJECTED_EXPERIMENT | CANDIDATE_SIMULATED | REJECTED_SIMULATED
```

`FAILED_TECHNICAL_BEFORE_LOCKBOX` can resume only through a new verified
`PREFLIGHT_VALIDATED` transition. `FAILED_TECHNICAL_AFTER_LOCKBOX` is terminal.
`FAILED_SCIENTIFIC` is terminal. State history is hash chained and every referenced
artifact is re-hashed during recovery.

## Atomic persistence and checkpoints

Relevant JSON artifacts are serialized canonically to `<name>.tmp`, flushed, fsynced,
parsed, and atomically replaced. The containing directory is fsynced. Existing immutable
artifacts may only be reused when byte-identical. An abandoned state temporary is ignored
and removed during recovery; it is never accepted as a checkpoint.

The deterministic `run_id` binds experiment ID, canonical preregistration hash, Python Git
commit, environment fingerprint, and mode. A completed terminal run is returned
idempotently rather than re-executed.

## Lockbox

`LockboxLease` creates the lease with `O_CREAT | O_EXCL | O_WRONLY`. E2 additionally uses
`SharedWindowLockboxLease`, whose authority is keyed by the semi-blind window rather than
the experiment ID. E1 and E2 therefore share one query budget. Lease acquisition is
the instant of consumption. Only after the lease is fsynced does `LockboxBudget.consume`
write the query audit. Both records bind the same run, candidate, preregistration, commit,
environment, timestamp, mode, and authorization hash. Presence or content mismatch fails
closed and is never repaired automatically.

Multiprocess coverage proves exactly one contender acquires a lease and exactly one budget
entry is recorded. Smoke-runs use a lease and budget below their own run directory. They
cannot resolve to the real lockbox path.

## Modes

### Dry-run

Validates pinned config hashes, D3, Git, environment, disk, lifecycle, SHORT-only scope,
TypeScript execution disablement, and lockbox availability. It writes only preflight and
run snapshot artifacts and does not call a scientific backend.

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_candidate_experiment.py --mode dry-run
```

### Smoke-run

Uses reduced deterministic fixtures and the existing scientific APIs. The two mechanical
outcomes are intentionally stored under different report roots because the identity inputs
produce the same deterministic smoke `run_id`.

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_candidate_experiment.py --mode smoke-run \
  --smoke-outcome candidate --reports-root reports/experiments-smoke-candidate

PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_candidate_experiment.py --mode smoke-run \
  --smoke-outcome rejected --reports-root reports/experiments-smoke-rejected
```

Smoke outcomes are mechanics evidence only. They are never evidence of scientific edge.

### Validation-run

E2 supersedes the incomplete, never-executed E1 protocol. E2 freezes exact hourly close
anchors, H12 non-overlap, literal TRAIN/CALIBRATION/SCORING dates, final reserve use,
pre-lockbox threshold derivation, and the shared semi-blind authority. The production
backend uses the existing feature, label, competition, calibrator, QMAE, runtime, and ECON
APIs. It stops at `VALIDATION_COMPLETED`, before any lease.

Two independent runs at commit `2d2518e685d3a1e2c615c943bb3cecea6a24ff97` completed in
841.44 s and 841.42 s. Their dataset, folds, competition, calibration, QMAE, bundle,
threshold, and ECON files were byte-identical. Dataset evidence was 15,680/15,680 hourly
anchors, 172,480 rows, zero skipped cycles, and zero quarantined labels.

### Full-run

The path enforces the exact phrase:

```text
OWNER_AUTHORIZED_PHASE_E_FULL_RUN
```

Without it, the CLI exits before preflight, training, artifacts, or lockbox access. The
path after the lockbox includes ECON, explicit mandatory checks, candidate/rejection,
absolute policy derivation, `SystemFreeze.validate`, and immutable registry publication.
It remains fail-closed behind exact owner authorization and was not executed.

## Promotion and publication

Checks are structured with expected, actual, pass/fail, evidence path, fold, symbol, side,
and severity. They cover minimum signals, positive folds, base profit factor, net and worst
fold expectancy, directional baseline, concentration, ECE, model competition, calibration,
QMAE coverage, robust ECON, leakage status, and SHORT separation.

Only a real full-run with every mandatory check passing can create an absolute
`FrozenSelectionPolicy`, validated `SystemFreeze`, and lifecycle `CANDIDATE` bundle. Smoke
never creates these artifacts. Rejection leaves threshold, policy, freeze, and registry
unpublished.

## Recovery and rollback

- Before the lockbox: retain the run directory, correct only the technical cause, and run
  the same command. Recovery validates preregistration, commit, environment, state chain,
  and artifact hashes before continuing.
- After the lockbox: do not retry. Preserve evidence and request an owner decision.
- Rollback is non-destructive: stop invoking the CLI and retain immutable reports. No
  operational configuration is changed by this implementation.

## E2 diagnostic evidence

- Provisional winners: TRRM random forest, EQM clean HGB, EQM net Extra Trees, QMAE HGB.
- Maximum fold ECE: `0.027179098915491584` (limit `0.08`).
- QMAE fold coverage: `0.9099166`, `0.9095863`, `0.9052509`, `0.8976112`.
- Draft reserve threshold: `4.047730415717134e-05`.
- Experimental bundle: `5ac8d61c489d089c27d874ebf0390022f89923edd7dd599a8b8aa7e9fb992d73`;
  `trained=true`, `approved=false`, lifecycle `EXPERIMENTAL`.
- Dev ECON base: 1,277 trades, expectancy `-0.0014500047`, PF `0.5383978`; every fold
  expectancy was negative. This is diagnostic dev evidence, not a lockbox verdict and not
  a CANDIDATE or REJECTED_EXPERIMENT result.

## Resources and risks

Each production validation consumed about 841 s wall, 1,100 s user CPU, 2.7 s system CPU,
and 46 MiB of persisted reports. Peak RSS was not captured because `/usr/bin/time` is not
installed; this is an observability gap, not inferred data. The open scientific risk is the
negative dev ECON result. The full run remains prohibited until separately authorized.
