# E5 Protocol Patch 02

## Authority and Scope

This governance-only document supplements, and does not replace or modify,
`reports/governance/e5_signal_edge_protocol/e5_protocol_preregistration.md`.
The original protocol at commit
`b8b86d012c40c4d10f10efb68e5eb9d86d4ac476`, physical SHA-256
`c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770`,
remains authoritative. The failed Patch 01 proposal was never materialized and
has no authority.

This patch completes only procedural definitions omitted from the original
protocol. Every original hypothesis, metric, threshold, gate, failure rule,
statistical method, and safety restriction remains binding.

## 1. Canonical Experiment ID

- Canonical ID: `E5_SIGNAL_EDGE_CONTROL_TEST`.
- Accepted alias: `E5_ENTRY_EDGE_INVESTIGATION`.

Both identifiers name the same frozen experiment and the same protocol. The
difference is administrative and has no scientific effect.

## 2. Discovery and Confirmation

- Discovery folds: Fold 1 and Fold 2.
- Confirmation folds: Fold 3 and Fold 4.
- Confirmation is executed exactly once.

Discovery results may be inspected before confirmation only to validate
implementation behavior, produce already-preregistered descriptive diagnostics,
check data and control coverage, detect technical failures, and verify that
frozen metrics are computable.

Discovery may not add or remove hypotheses; change metrics, thresholds,
controls, horizons, buckets, or gates; select symbols, features, subgroups, or
transformations; or change any final decision rule.

Before confirmation starts, discovery artifacts and the confirmation execution
manifest must be frozen. Once confirmation starts, no discovery artifact or
scientific definition may be changed. Confirmation results may never change a
hypothesis, threshold, control, metric, bucket, horizon, or gate.

## 3. Reconciled Fold Decision Rule

The original requirement that positive results hold in at least three of four
total folds remains unchanged. This patch adds the independent requirement that
both confirmation folds pass the same frozen positive-effect criteria.

`E6_JUSTIFIED` therefore requires both conjunctive fold gates:

1. `CONFIRMATION_CONSISTENCY`: Fold 3 is `PASS` and Fold 4 is `PASS`.
2. `GLOBAL_FOLD_STABILITY`: at least three of all four folds are `PASS`.

Discovery folds contribute to the original global three-of-four stability gate,
but they cannot independently justify E6. Passing both discovery folds without
both confirmation folds can never justify E6. Passing only both confirmation
folds is also insufficient because it supplies only two of four total folds.

The exhaustive authorized examples are:

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold-gate result |
|---|---|---|---|---|
| PASS | PASS | PASS | PASS | PASS |
| PASS | FAIL | PASS | PASS | PASS |
| FAIL | PASS | PASS | PASS | PASS |
| PASS | PASS | PASS | FAIL | FAIL |
| PASS | PASS | FAIL | PASS | FAIL |
| FAIL | FAIL | PASS | PASS | FAIL |

Every other combination fails at least one conjunctive fold gate. No alternate
interpretation is permitted.

## 4. Final Decision Logic

All original mandatory gates and both fold gates in Section 3 are conjunctive.
`E6_JUSTIFIED` requires:

- both confirmation folds pass;
- at least three of four folds total pass;
- statistical significance passes;
- multiplicity correction passes;
- minimum economic magnitude passes;
- net economic utility passes;
- applicable frozen score or label monotonicity passes;
- symbol stability passes;
- matched-control alpha passes;
- every other mandatory gate in the original protocol passes.

Failure of any mandatory gate produces `CLOSE_THIS_SIGNAL_FAMILY`. There is no
weighted score, no majority vote beyond the explicitly frozen three-of-four
gate, no near-pass classification, and no subjective override.

## 5. Frozen Horizons

The complete authorized horizon curve is:

- `H12`: 12 five-minute bars, equal to 1 hour.
- `H48`: 48 five-minute bars, equal to 4 hours.
- `H96`: 96 five-minute bars, equal to 8 hours.

No other horizon is authorized. H12 remains the original primary outcome and
all original H12 gates remain unchanged. H48 and H96 complete the prespecified
horizon curve; they cannot replace a failed H12 gate or independently justify
E6. The curve must be interpreted jointly, and selecting a best horizon after
execution is forbidden.

## 6. Volatility Module

The sole conditioning variable is `ATR(14)` calculated on completed five-minute
bars. For each observation, use the last fully known ATR value immediately
before its entry decision timestamp. Current-bar and future information are
forbidden.

The only buckets are population quintiles `Q1` through `Q5`. Compute bucket
boundaries once from Discovery Folds 1 and 2 before confirmation is accessed.
Freeze those boundaries in the confirmation execution manifest and apply them
unchanged to Folds 3 and 4. Confirmation quantiles may not be recomputed.
Buckets may not be merged, removed, relabeled, or selected after results are
observed.

The volatility module is a frozen stability diagnostic. A favorable isolated
bucket cannot override any original gate or justify E6.

## 7. Economic Barrier Event

The sole path event is symmetric:

- favorable barrier: `+2 * B_BASE round-trip transaction-cost fraction`;
- adverse barrier: `-2 * B_BASE round-trip transaction-cost fraction`;
- maximum evaluation horizon: `H96`.

For SHORT positions, use the canonical frozen SHORT return convention. The
event is true only when the favorable barrier is reached before the adverse
barrier. If both barriers are crossed in the same five-minute bar and order is
unknown, apply the existing conservative ambiguity rule and record the adverse
barrier as first for the primary result. No barrier optimization, alternate
barrier, or optimistic promotion use is allowed.

This event is secondary and cannot replace the original H12 net-return gates.

## 8. Matching Coverage

Minimum valid matched-control coverage is exactly `95.0%` of eligible
experimental observations.

Coverage equals:

`eligible real observations with the required number of valid matched controls`

divided by:

`all eligible real observations`.

Coverage below `0.95` produces the technical stop
`E5_EXECUTION_BLOCKED_BY_CONTROL_COVERAGE`. Unmatched observations may not be
silently discarded or reclassified to increase coverage. Counts and reasons
must be reported by fold and symbol.

## 9. Confirmation Inspection Rule

Before the one-shot confirmation execution, create an immutable manifest that
contains:

- original protocol hash;
- this patch hash;
- implementation commit;
- discovery artifact hashes;
- frozen thresholds;
- frozen ATR bucket boundaries;
- frozen control-generation seed;
- frozen bootstrap seed;
- frozen permutation seed;
- complete confirmatory test family;
- multiplicity-correction family;
- gate implementation hash.

A technical failure may be repaired only when no valid confirmation metric was
exposed, the failure and logs are preserved, and the repair changes no
scientific definition. A valid unfavorable confirmation result may never be
rerun.

## 10. Governance Statement

This patch resolves omitted procedural definitions. It preserves the original
three-of-four fold requirement and adds the stricter requirement that both
confirmation folds pass. It introduces no new hypothesis and changes no
previously frozen scientific metric. It authorizes no E5 execution, consumes no
lockbox query, and grants no operational authority.

## 11. Hashing and Commit Record

| Field | Frozen record |
|---|---|
| Original protocol path | `reports/governance/e5_signal_edge_protocol/e5_protocol_preregistration.md` |
| Original protocol commit | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` |
| Original protocol SHA-256 | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` |
| Patch path | `reports/governance/e5_signal_edge_protocol/e5_protocol_patch_02.md` |
| UTC creation timestamp | `2026-07-20T03:24:02Z` |
| Repository HEAD before patch | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` |
| Working tree before patch | `CLEAN` |
| Lockbox at creation | `NOT_CONSUMED`; `consumed_queries=[]`; `maximum_queries_total=1` |

The patch physical SHA-256 is recorded in the containing commit message under
`E5-Patch-Physical-SHA256`. The repository HEAD after commit is the containing
commit itself and is recorded by Git. These two values cannot be embedded
literally in the file they identify without creating a self-reference that
changes the hash or commit ID. The post-commit working tree must be clean.

## 12. Validation Against the Original Protocol

The original binding fold clauses are preserved at:

- original lines 300–303: top-bottom spread positive in at least three folds;
- original lines 309–315: all four folds included and three-fold stability;
- original lines 527–530: automatic gates 14–17;
- original line 576: fewer than three positive folds closes the family.

The original H12 outcome remains primary at lines 132–145 and 154–155. The
prohibition against replacing H12 with a favorable horizon at line 502 remains
binding. The original protocol contains no prior discovery/confirmation split,
H48 or H96 definition, ATR quintile module, symmetric barrier event, or matching
coverage threshold. This patch therefore leaves every pre-existing definition
and gate intact.
