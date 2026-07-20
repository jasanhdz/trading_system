# E5 Owner-Authorized Amendment 01

## Authority and Prospective Scope

This document is an explicit prospective scientific amendment authorized by
the experiment owner. It is not a clarification and does not represent the
decisions below as having existed in the original protocol.

Authority precedence is:

1. `e5_protocol_preregistration.md` for every definition it already freezes.
2. `e5_protocol_patch_02.md` for its procedural supplements.
3. This amendment only for the previously omitted decisions stated here.

The original protocol remains at commit
`b8b86d012c40c4d10f10efb68e5eb9d86d4ac476`, SHA-256
`c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770`.
Patch 02 remains at commit `92191db1a7c4135252377f64f51b174f180dcd53`,
SHA-256 `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e`.
Neither file is modified by this amendment.

## Owner Acknowledgement

The owner acknowledges that the original protocol did not freeze a separate IC
minimum effect magnitude, subgroup-specific minimum sample sizes, a complete
fold-level monotonicity predicate, or an enumerated confirmation test family.
The decisions below are new preregistered owner decisions made before E5
implementation, dataset construction, discovery, confirmation, or scientific
result inspection.

## Decision 1: IC Has No Separate MREM

No arbitrary minimum magnitude is introduced for Spearman IC. The original
MREM of `0.0005` continues to apply only to the economic quantities frozen in
the original protocol:

- expectancy;
- paired delta against each matched control;
- top-decile minus bottom-decile return spread.

It does not apply to Spearman IC. Confirmatory score IC requires the
preregistered positive orientation and a dependence-aware CI90 with lower bound
strictly greater than zero. IC alone cannot establish economic materiality;
materiality must pass through the existing return-based MREM.

## Decision 2: Symbol and Cycle IC Are Diagnostic

Per-symbol score IC and within-cycle cross-sectional IC are mandatory diagnostic
reports, not independent E6 gates. They may expose concentration, instability,
lack of ordering, symbol dependence, or regime dependence, but cannot rescue a
failed primary economic gate.

Use Spearman rank correlation, the score-association statistic already frozen
by the original protocol, with the frozen score orientation and canonical SHORT
economic outcome. Report all 11 symbols separately and report the complete
distribution of defined cycle-level IC values. Do not create a numerical IC
magnitude threshold or a count of individually significant symbols. Undefined
values are reported under Decision 3.

The original symbol-stability gates based on control deltas, leave-one-symbol-
out results, and positive-PnL concentration remain unchanged and mandatory.

## Decision 3: Sample Size

The only binding numerical minimum is the original rule: at least 100 eligible
trades per fold. No additional numerical minimum is created for symbols, score
buckets, label groups, volatility buckets, individual IC statistics, or
individual cycles.

Every subgroup report must display its exact sample size. A mathematically
undefined subgroup statistic is serialized as `NOT_COMPUTABLE`; it is never
replaced by zero or silently omitted. A limited subgroup result is descriptive
and cannot become an independent confirmatory gate. The fold-level 100-trade
requirement remains mandatory.

## Decision 4: Monotonicity

The original monotonicity gate is preserved exactly:

`MONOTONICITY_GATE_PASS = POOLED_MONOTONICITY_PASS AND
MONOTONICITY_POSITIVE_IN_AT_LEAST_3_OF_4_FOLDS`.

Use the original ten equal-count score buckets and original monotonicity
definition. Do not insert the complete pooled monotonicity gate into each fold
predicate. No per-symbol gate, per-cycle gate, tolerance, bucket merge, or
post-hoc ordering rule is authorized.

## Decision 5: Fold Economic Pass Predicate

For each Fold `k`, evaluate only observations belonging to that fold.
`FOLD_ECONOMIC_PASS(k)` requires every condition below:

1. At least 100 eligible trades exist in Fold `k`.
2. Fold-level paired net-return delta against C1 is strictly positive.
3. Fold-level paired net-return delta against C2 is strictly positive.
4. Each mandatory paired economic delta is at least `0.0005`.
5. Real-entry B_BASE net expectancy is strictly positive.
6. The preregistered top-decile minus bottom-decile B_BASE net-return spread is
   strictly positive and at least `0.0005`.
7. Required matched-control coverage is at least 95%.
8. No frozen concentration or data-validity rule is violated.

Any failed, missing, undefined, or non-computable condition makes that fold
`FAIL`. Per-symbol IC, cycle IC, pooled monotonicity, pooled Holm significance,
and criteria not naturally defined at one-fold scope are excluded from this
predicate. Discovery and confirmation folds use the identical predicate.

The global fold gate remains:

`FOLD_ECONOMIC_PASS(3) AND FOLD_ECONOMIC_PASS(4) AND
COUNT(FOLD_ECONOMIC_PASS(1..4)) >= 3`.

## Decision 6: Exact Confirmatory Test Family

The confirmation population is pooled Folds 3 and 4 only. No Fold 1 or Fold 2
p-value and no four-fold pooled p-value enters this family.

One Holm family contains exactly 12 one-sided confirmation tests:

| IDs | Module | Statistic | Horizons |
|---|---|---|---|
| `A_C1_H12`, `A_C1_H48`, `A_C1_H96` | Matched-control economic alpha | Paired net-return delta, real minus C1 | H12, H48, H96 |
| `A_C2_H12`, `A_C2_H48`, `A_C2_H96` | Matched-control economic alpha | Paired net-return delta, real minus C2 | H12, H48, H96 |
| `B_SPREAD_H12`, `B_SPREAD_H48`, `B_SPREAD_H96` | Score economic ordering | Top-decile minus bottom-decile net-return spread | H12, H48, H96 |
| `C_MONO_H12`, `C_MONO_H48`, `C_MONO_H96` | Monotonic ordering | Spearman rho between score-decile index and decile mean net return | H12, H48, H96 |

The monotonic trend statistic is the Spearman rho explicitly listed in the
original monotonicity metrics and used by its gate. Apply Holm jointly across
these 12 tests using original `alpha = 0.05`, one-sided. The original CI90,
block bootstrap, seeds, and economic MREM remain unchanged. No test may be
added, removed, substituted, or regrouped after confirmation begins.

The original separately Holm-adjusted secondary diagnostic families remain
secondary; they are not members of this primary confirmation family and cannot
override it.

## Decision 7: Confirmatory Success

Confirmatory inference passes only when all conditions below hold:

1. Mandatory C1 effects have the positive direction.
2. Mandatory C2 effects have the positive direction.
3. Every economic effect required by the joint-horizon rule meets `0.0005`.
4. Required confirmation tests survive the frozen Holm correction.
5. Top-minus-bottom net spreads pass economic and statistical requirements.
6. The original monotonicity gate passes.
7. Real-entry net expectancy is positive.
8. Folds 3 and 4 each pass `FOLD_ECONOMIC_PASS`.
9. At least three of four folds pass `FOLD_ECONOMIC_PASS`.
10. Every other unmodified original mandatory gate passes.

Per-symbol and within-cycle IC remain diagnostic and cannot rescue a failed
condition.

## Decision 8: Horizon Interpretation

H12, H48, and H96 remain jointly frozen. No best horizon may be selected.
Because no prior document states how many horizons must pass, the owner now
freezes the conservative rule:

- all three horizons must have the correct economic direction;
- at least two of three horizons must satisfy the complete Holm-corrected
  statistical requirement;
- H96 cannot be the sole passing horizon;
- no weighted horizon average may replace these conditions.

H12 remains the primary original outcome and no longer horizon can rescue a
failed original H12 gate.

## Decision 9: Label-to-Economics Module

This module is a mandatory causal diagnostic. For every already frozen label
family, report without creating or transforming labels:

- label prevalence by fold and symbol;
- gross return by label;
- B_BASE net return by label;
- favorable-first barrier probability by label;
- adverse-first barrier probability by label;
- fold stability;
- score association where mathematically defined.

No independent label-correlation magnitude threshold is introduced. Classify
the bridge mechanically:

- `LABEL_ECONOMICS_DISCONNECTED`: the label lacks the original positive
  association and monotonic ordering with both gross and net outcomes;
- `LABEL_CONNECTED_EFFECT_TOO_SMALL`: association or ordering is present, but
  the highest frozen group fails the original return MREM or net utility;
- `LABEL_CONNECTED_ECONOMICALLY_MATERIAL`: association, monotonicity, return
  MREM, stability, and net utility all pass.

The classifications are diagnostic and do not replace the final conjunctive
gates. A disconnected label prevents E6 when the model's frozen scientific
claim depends on predicting that label.

## Decision 10: Interpretation of Diagnostics

Diagnostics may strengthen closure by showing volatility mirage, label
disconnect, score non-ordering, symbol concentration, cycle instability, or
horizon beta dependence. A diagnostic may not independently justify E6 unless
it is already an explicit mandatory gate. No result may be promoted from
diagnostic to confirmatory after inspection.

## Final Decision Space

The only scientific outcomes remain `CLOSE_THIS_SIGNAL_FAMILY` and
`E6_JUSTIFIED`. There is no inconclusive or partial result, E5B, threshold
revision, second valid confirmation attempt, or post-result amendment.

## Prospective Status at Creation

At amendment creation:

- no E5 implementation exists;
- no E5 scientific dataset exists;
- no discovery or confirmation execution exists;
- no E5 scientific result has been inspected;
- lockbox status is `NOT_CONSUMED`;
- `consumed_queries=[]` and budget remaining is 1;
- the tracked Python working tree was clean before creating this file;
- the TypeScript working tree was clean;
- no Candidate, E6, Selection Policy, or System Freeze exists for E5;
- no shadow, paper, live, Binance, or PM2 action was executed;
- no operational TypeScript change or push occurred.

## Contradiction Validation

No explicit original clause conflicts with these prospective decisions:

- original lines 113-124 limit MREM to economic returns and spreads;
- original lines 286-303 define pooled monotonicity and three-fold stability;
- original lines 309-329 define fold and symbol stability;
- original line 539 freezes 100 completed trades per fold;
- original lines 68-72 and 378-380 preserve secondary inference, while this
  amendment creates a distinct primary confirmation family;
- Patch 02 lines 45-93 preserve the conjunctive confirmation and three-of-four
  fold rules;
- Patch 02 lines 95-107 freeze H12, H48, and H96 jointly.

Missing prior definitions are acknowledged as new owner decisions rather than
misrepresented as historical clauses.

## Governance Record

| Field | Frozen record |
|---|---|
| Original protocol path | `reports/governance/e5_signal_edge_protocol/e5_protocol_preregistration.md` |
| Original protocol SHA-256 | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` |
| Patch 02 path | `reports/governance/e5_signal_edge_protocol/e5_protocol_patch_02.md` |
| Patch 02 SHA-256 | `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e` |
| Phase 0 artifact path | `reports/governance/e5_signal_edge_protocol/e5_phase0_revalidation.json` |
| Phase 0 artifact SHA-256 | `87756f518f1d42109f480b0d4ee7983ac57f70f447dd2616a24349fc25a2f2c1` |
| Amendment path | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_01.md` |
| UTC timestamp | `2026-07-20T05:47:18Z` |
| Repository HEAD before amendment | `92191db1a7c4135252377f64f51b174f180dcd53` |
| Python tracked working tree before amendment | `CLEAN` |
| TypeScript HEAD | `53105ee34c6e29f960c37c3516a58fffd2aa5906` |
| TypeScript working tree | `CLEAN` |
| Lockbox | `NOT_CONSUMED`; `consumed_queries=[]`; budget remaining 1 |

The amendment physical SHA-256 is recorded in the containing commit message as
`E5-Amendment-Physical-SHA256`. The repository HEAD after commit is the commit
containing this file and is recorded by Git. Literal self-identifiers cannot be
embedded in their own hashed content without changing the identifiers.
