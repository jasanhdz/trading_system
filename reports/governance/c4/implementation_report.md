# Frozen E3 C4 implementation

## Scope

C4 was implemented without running Stage 1b, Stage 4b, Stage 5b, or an E3
validation-run. E3 and scientific competition v2 remain byte-identical.

## Productive corrections

- Productive model factories load every hyperparameter from the hash-bound v2
  contract. Smoke overrides are available only to the simulated smoke backend.
- Fold EQM training and scoring use deterministic TRRM rank-survivor populations.
  Final refit uses final-train survivors and freezes the calibrated q0.70 TRRM
  threshold for future runtime use.
- The seven ECON baselines operate on 288 contiguous raw canonical bars, use
  identical H12 entry/exit and costs, and apply independent per-fold budgets.
- The best directional baseline is calculated from the frozen directional list;
  `eqm_only` and `trrm_only` remain diagnostic.

Stage 5 did not identify a final causal population effect. The EQM correction is
therefore justified by historical fidelity and E3 governance only.

No feature, label, sampling, symbol, cost, Selection Policy, Decision Freeze, or
productive selection-threshold formula was changed.

## Governance

Stage 1b and Stage 4b are preregistered but unexecuted. `OWNER_DECISION_O1` is
recorded. Stage 5b remains `OPTIONAL_GOVERNANCE_EVIDENCE` and requires a future
owner decision.

## Technical validation

- Targeted C4/governance suite: 77 passed in 23.62 seconds.
- Full collection: 193 tests. The monolithic invocation was terminated by the
  execution environment after 75% without a pytest failure or summary. The
  exact collected node list was therefore split into two disjoint, exhaustive
  partitions: 97 passed in 7.69 seconds and 96 passed in 26.40 seconds.
- `python -m compileall -q src scripts`: passed.
- `git diff --check`: passed.

No Stage 1b, Stage 4b, Stage 5b, E3 validation-run, full-run, lockbox access, or
semi-blind access was performed by this validation.
