# Aegis Offline Scientific Parity A-F

## Scope and safety

This implementation covers the offline A-D mechanics, freezes the Phase-E experiment
preregistration, and prepares (but does not execute) the Phase-F paired benchmark. It does
not claim scientific parity, does not publish a CANDIDATE, and does not derive a productive
selection threshold from smoke results.

Python remains scientific-only. Canonical market inputs are consumed through the read-only
D3 adapter. The Gen2 directory is only read by the future benchmark. TypeScript remains the
sole operational platform and `execution.enabledByConfig` remains `false`.

## Implemented semantics

- Prediction has one authoritative experiment/runtime/API path.
- `qmae_mean` is explicitly distinct from conformal `qmae_q90`.
- Probability heads require an out-of-fold calibration block.
- Features v2 and SHORT V4 labels are causal and versioned.
- REGIME is diagnostic context. It does not multiply the ranking score.
- TRRM and QMAE are gates. They do not multiply the ranking score.
- Ranking score is `P(clean) * max(0, expected_directional_return)` and has units
  `EXPECTED_CLEAN_RETURN_FRACTION`.
- ECON is an independent next-bar-open to H12-close replay over canonical prices. Cost
  scenarios are A `(4,1,0.5)`, B `(5,2,1)`, and C `(5,5,2)` bps for fee per side,
  slippage per side, and funding per hour respectively.
- Selection Policy is SHORT-specific, absolute, content-hashed, and rejects drift above
  `1e-9`. Its productive threshold remains pending the authorized Phase-E run.
- SYSTEM_FREEZE binds the twelve required artifact/configuration hashes and environment.
- Evidence JSONL is append-only, fsync-backed, chain-verified on recovery, and outcome
  idempotence is keyed by `(decision_id, event_type)`.
- Bundle lifecycle is forward-only: `EXPERIMENTAL -> CANDIDATE -> SHADOW_APPROVED -> LIVE_APPROVED`.

## Phase E preregistration

`config/experiments/aegis_short_candidate_e1.yaml` freezes:

- canonical D3 manifest and read-only finality gate;
- four expanding temporal folds and 120-minute embargo;
- model families and stability-first competition protocol;
- QMAE coverage `[0.87, 0.93]`, per-fold ECE maximum `0.08`;
- ECON methodology, costs, weekly bootstrap, and seed;
- mandatory promotion criteria and side-separated metrics;
- one-query persistent lockbox;
- `threshold_value: null`, because it may only be derived by the authorized full run;
- no automatic promotion and initial lifecycle `EXPERIMENTAL`.

Dry-run command (executed after the orchestrator implementation):

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_candidate_experiment.py --mode dry-run
```

Owner-authorized full-run command (not executed in this phase):

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_candidate_experiment.py \
  --mode full-run --owner-authorization OWNER_AUTHORIZED_PHASE_E_FULL_RUN
```

The persistent state machine, exclusive lease, dry-run, smoke-run, validation stop, ECON,
criteria, policy, freeze, and publication mechanics are implemented and covered by 43
focused tests. A production scientific backend remains fail-closed because E1 does not
freeze the subdivision between calibration and scoring inside each validation fold or the
dataset sampling cadence. Those are scientific protocol inputs and were not invented in
the orchestrator. The real full-run therefore remains unexecuted and Phase E remains open.

Estimated owner-run envelope is 2-6 CPU hours and 16 GiB peak memory. This is a planning
estimate, not measured evidence. Rollback is non-destructive: retain an immutable failed
report and leave active configuration unchanged.

## Phase F preparation

`training/benchmark.py` accepts only a `CANDIDATE`, aligns exact timestamp/symbol SHORT
records, rejects future feature timestamps and duplicates, hashes Gen2 before and after
reading, and refuses output paths under `/home/jasan/Develop/aegis_gen2`.

Future command (not executed):

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python \
  scripts/run_aegis_gen2_paired_benchmark.py \
  --current-decisions <CANDIDATE_DECISIONS_JSONL> \
  --candidate-bundle-id <CANDIDATE_BUNDLE_ID> \
  --candidate-state CANDIDATE \
  --output reports/parity_benchmark/paired_report.json
```

Minimal synthetic report shape exercised by the fixtures:

```json
{"bundle_state":"CANDIDATE","matched_rows":1,"new_no_trade_rate":1.0,"gen2_no_trade_rate":0.0,"source_hash_before":"<sha256>","source_hash_after":"<same-sha256>"}
```

## Traceability of the 16 findings

| # | Finding | Commits / implementation | Test evidence | Status |
|---|---|---|---|---|
| 1 | train/serving skew | `520403e`, authoritative path | `test_authoritative_prediction_path` | CLOSED |
| 2 | q90 was a mean | `4f41908`, quantile/conformal contract; `76ae1f7` tooling | QMAE contract/model competition | IMPLEMENTED; full-fold evidence pending E |
| 3 | uncalibrated probabilities | `4f41908`, `76ae1f7` | calibration fail-closed/export tests | IMPLEMENTED; full OOF report pending E |
| 4 | non-final source / D3 naming | `cf7e029` canonical read-only adapter, REGIME rename | canonical source tests | CLOSED |
| 5 | missing SHORT features | `1d6452c`, 83-column v2 | causal and golden feature tests | CLOSED |
| 6 | no nonlinear competition | `0d061a2`, `76ae1f7` | JSON export and deterministic smoke | IMPLEMENTED; winner pending E |
| 7 | autoreferential ECON | `2a3ecdf`, independent replay | ECON golden/cost/bootstrap tests | CLOSED mechanically; robust result pending E |
| 8 | arbitrary threshold | `2a3ecdf`, frozen policy; reference marked placeholder | drift/hash tests | PRODUCTIVE VALUE PENDING E |
| 9 | terminal-only labels | `1d6452c`, SHORT V4 | path/gap/ambiguity tests | CLOSED |
| 10 | no system freeze | `2a3ecdf` | required component/hash/lifecycle tests | CLOSED |
| 11 | non-persistent evidence | `2a3ecdf`; Phase-E hash-chained run state and isolated evidence | restart, chain tamper, outcome dedupe, 43 orchestrator/state tests | CLOSED offline; G-H N/A |
| 12 | score double counting | `2a3ecdf` | gate invariance and golden parity | CLOSED |
| 13 | approved untrained bundle | `2a3ecdf` | bundle validation | CLOSED |
| 14 | coordinated TS close absent | Phase G | not in offline scope | N/A-OFFLINE |
| 15 | momentum `/4`, duplicate folds | `1d6452c`; single `walk_forward_splits` path | golden/causal/fold tests | CLOSED |
| 16 | hard-coded regime/direction thresholds | versioned REGIME config; one runtime direction threshold | configurable regime golden | CLOSED for offline reference |

## Validation evidence

- Python: 80 tests passed after the final REGIME configuration addition.
- TypeScript: 625 tests passed outside the sandbox restriction; TypeScript build passed.
- Coverage tooling was not installed in the repository environment. No dependency was
  installed merely to generate a percentage.
- Phase-E dry-run and two fixture smoke-runs completed. The real lockbox was not consumed.
- No full Phase-E experiment or Phase-F benchmark was executed.
