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
-> CALIBRATION_VALIDATED -> QMAE_VALIDATED -> LOCKBOX_ACQUIRED
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

`LockboxLease` creates the lease with `O_CREAT | O_EXCL | O_WRONLY`. Lease acquisition is
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

The state path stops at `QMAE_VALIDATED`, before any lease. Its orchestration path is tested
with the same reduced API-backed fixture backend. A production validation backend is not
enabled because the immutable E1 preregistration says only
`temporally_held_out_within_fold` and does not freeze the calibration/scoring subdivision
or dataset sampling cadence. Choosing either value in code would change the scientific
protocol after preregistration. This is the remaining technical blocker.

### Full-run

The path enforces the exact phrase:

```text
OWNER_AUTHORIZED_PHASE_E_FULL_RUN
```

Without it, the CLI exits before preflight, training, artifacts, or lockbox access. The
path after the lockbox includes ECON, explicit mandatory checks, candidate/rejection,
absolute policy derivation, `SystemFreeze.validate`, and immutable registry publication.
It remains fail-closed behind the unresolved production backend and was not executed.

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

## Resources and risks

The preregistration estimates 2-6 CPU hours and 16 GiB peak memory for a future full-run.
That estimate was copied into the dry-run manifest and was not measured here. The open risk
is the missing frozen calibration/scoring split and sample cadence for the production
dataset adapter. Until resolved by a new owner-approved preregistration decision, the CLI
fails closed for production validation/full execution.

