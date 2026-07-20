# E5 Owner-Authorized Amendment 02

## Status and Owner Authorization

| Field | Frozen value |
|---|---|
| Document status | `OWNER_AUTHORIZED_PROSPECTIVE_SCIENTIFIC_AMENDMENT` |
| Experiment | `E5_SIGNAL_EDGE_CONTROL_TEST` |
| Accepted alias | `E5_ENTRY_EDGE_INVESTIGATION` |
| Authorization source | Direct instruction from the experiment owner |
| Created UTC | `2026-07-20T08:19:03Z` |
| Repository branch | `feature/aegis-ts-clean-rebuild` |
| Repository HEAD before amendment | `943b98a5091c4d9238f754a1e42e63540a4579a6` |

The owner prospectively authorizes the twelve scientific decisions below. No
E5 implementation, dataset, discovery execution, confirmation execution, or
scientific result existed or was inspected when these decisions were made.
The decisions preserve existing governance, established statistical practice,
temporal and fold dependence, determinism, and auditability. They were not
selected to increase significance, improve performance, or make `E6_JUSTIFIED`
more likely.

## Governance Hierarchy

Authority is ordered as follows:

1. Original preregistration:
   `e5_protocol_preregistration.md`, commit
   `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476`, SHA-256
   `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770`.
2. Procedural Patch 02:
   `e5_protocol_patch_02.md`, commit
   `92191db1a7c4135252377f64f51b174f180dcd53`, SHA-256
   `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e`.
3. Owner Amendment 01:
   `e5_owner_authorized_amendment_01.md`, commit
   `943b98a5091c4d9238f754a1e42e63540a4579a6`, SHA-256
   `c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4`.
4. This amendment, only for the unresolved decisions stated here.

A later document controls only where it explicitly and prospectively resolves
an omitted definition. Every other earlier clause remains frozen.

## Safety State at Creation

- Python tracked working tree: `CLEAN`.
- TypeScript working tree: `CLEAN` at
  `53105ee34c6e29f960c37c3516a58fffd2aa5906`.
- E5 implementation: absent.
- E5 scientific dataset and result artifacts: absent.
- E5 discovery and confirmation: not executed.
- Lockbox: `NOT_CONSUMED`.
- `consumed_queries=[]`.
- `budget_remaining=1`.

## Binding Selection Registry

| Decision | Selected option |
|---|---|
| D1 | `D1-B` |
| D2 | `D2-A` |
| D3 | `D3-B` |
| D4 | `D4-A` |
| D5 | `D5-A` |
| D6 | `D6-C` |
| D7 | `D7-C` |
| D8 | `D8-A` |
| D9 | `D9-D` |
| D10 | `D10-A` |
| D11 | `D11-A` |
| D12 | `D12-B` |

## D1: Horizon-Specific Complete-Case Populations

H12, H48, and H96 use separate complete-case populations. An observation is
eligible at horizon H only when its frozen entry remains valid, every canonical
bar needed for H exists, all mandatory costs and outcomes are computable, and
all existing structural rules pass. Missing H48 or H96 never removes a valid
H12 observation; missing H96 never removes a valid H48 observation.

Forward filling, imputation, synthetic terminal prices, shortened horizons,
and partial-horizon returns are forbidden. Each horizon has an immutable
eligibility manifest listing included and excluded observation IDs, exact
reasons, population, symbol, fold and month counts, and availability and
attrition relative to the frozen entry population.

C1 and C2 are independently constructed for each horizon. A match from one
horizon is not automatically reused at another, although the same algorithm
and seed policy apply. The three estimates are horizon-specific estimands;
differences may reflect both elapsed horizon and eligible-population
composition. This limitation must be prominent in every joint report.

All twelve confirmation p-values remain one Holm family. A horizon is
`NOT_COMPUTABLE` when its mandatory population, coverage, or matching fails.
Later-horizon failure does not invalidate H12 observations, but all existing
joint-horizon and final-verdict rules remain binding.

Prospective rationale: preserve valid H12 evidence while treating later
horizons as honest complete-case estimands.

## D2: Seeded Randomized Augmenting-Path C2 Matching

C2 is run independently per horizon, fold, UTC month, and replicate. Left
nodes are real observations ordered by immutable observation ID. Right nodes
are eligible control cycles ordered by symbol, cycle timestamp, and immutable
cycle ID. Only frozen C2 structural rules determine edges.

For each replicate and stratum, derive a deterministic `PCG64` substream from
the frozen base seed and canonical tuple
`("C2", horizon, fold, month, replicate_index)`. Apply the substream as a
seeded permutation to each left node's eligible adjacency list, then run one
fixed maximum-cardinality augmenting-path matcher using canonical left order,
seeded adjacency order, and deterministic traversal. No cost or outcome
objective is permitted.

A right-side cycle appears at most once within a replicate. It may recur in a
different replicate. Reuse across horizons is allowed because the horizons
are separate analyses, but reuse inside one horizon-stratum-replicate is not.
Perfect cardinality is mandatory; partial matching is forbidden. An impossible
perfect match makes the replicate invalid, records every unmatched node and
eligibility count, and invokes D12 without reducing the required trade count.

Minimum time or volatility distance, outcome or profit similarity,
deterministic minimum-cost matching, and any outcome-based objective are
forbidden. The execution contract must freeze the exact algorithm or library,
version, RNG, seed encoding, stable sorts, traversal, and serialization.

Prospective rationale: realize the preregistered randomized bipartite process
with a practical deterministic and auditable solver.

## D3: Fixed Barrier Geometry and Realized Funding

Symmetric barrier geometry excludes duration-dependent funding. Its distance
contains only the prospectively fixed B_BASE entry fee, exit fee, entry
slippage, and exit slippage. Funding never moves or resizes a barrier.

Canonical funding accrued strictly after entry and at or before the applicable
termination timestamp is included in realized B_BASE net return. Termination
is the earliest applicable frozen favorable barrier, adverse barrier, horizon,
or other already-authorized event. Funding uses canonical timestamps, symbol
rate, SHORT sign, and notional convention; no rate is known before its
canonical timestamp. Same-bar barrier ambiguity retains the frozen adverse-
first rule.

Reports separately expose gross barrier event, gross path return, fixed costs,
realized funding, and final net return. Favorable-first does not imply positive
net utility. Missing mandatory funding makes the net outcome
`NOT_COMPUTABLE`. Estimation, interpolation, optimization, forward filling,
and future-rate inference are forbidden.

Prospective rationale: separate path geometry from realized economics and
eliminate funding look-ahead and circular barriers.

## D4: Wilder ATR(14)

ATR uses completed canonical five-minute bars. For bar t:

`TR_t = max(high_t-low_t, abs(high_t-close_{t-1}), abs(low_t-close_{t-1}))`.

The first ATR is the arithmetic mean of 14 consecutive valid TR values.
Subsequent values use
`ATR_t = ((13 * ATR_{t-1}) + TR_t) / 14`. An entry uses the latest fully
computed ATR from the last completed bar strictly before entry. Current and
future bars are forbidden.

OHLC is never forward filled. A missing canonical bar breaks continuity and
requires a new consecutive 14-TR warm-up. ATR remains `NOT_COMPUTABLE` until
warm-up completes. UTC canonical boundaries apply. The execution contract
must freeze float64 precision and prohibit intermediate discretionary
rounding. Existing code may be reused only after exact conformance is proven.

Prospective rationale: use recognized Wilder semantics consistently with
established project behavior without granting repository precedent retroactive
governance authority.

## D5: Type 7 ATR Quintiles

Learn q20, q40, q60, and q80 once from eligible F1-F2 ATR values using
Hyndman-Fan Type 7 linear quantiles at 0.20, 0.40, 0.60, and 0.80. Freeze and
apply them unchanged to F3-F4.

Assignment is `Q1: x <= q20`, `Q2: q20 < x <= q40`,
`Q3: q40 < x <= q60`, `Q4: q60 < x <= q80`, and `Q5: x > q80`.
Boundary equality enters the lower bucket. Duplicate boundaries and empty
buckets are retained and reported. Identical ATR values are never split using
IDs or jitter. Missing ATR is excluded only from analyses requiring ATR
stratification. Confirmation cannot influence boundaries.

Prospective rationale: use the numerical stack's standard convention without
bespoke tie splitting.

## D6: Fold-Centered Power Simulation

Power evaluates the frozen complete primary absolute-expectancy test, never a
control delta. Use discovery observations only. Within each discovery fold,
compute its trade-weighted B_BASE mean and trade-level residuals. Reconstruct
returns by adding each residual to the fold mean plus MREM `0.0005`.

Resample complete UTC ISO-week blocks with replacement within their original
folds, preserving all trades and within-week dependence. Recombine folds with
the same trade weighting as the primary expectancy statistic. Each simulation
reruns the full frozen primary procedure, including hierarchical week-block
CI and its success predicate. Power is valid successes divided by valid
simulations across exactly 10,000 requested simulations. D12 governs invalid
simulations. Residuals, weighting, MREM, and success criteria cannot be tuned.

Prospective rationale: preserve the primary trade-level estimand, temporal
dependence, and discovery-fold heterogeneity.

## D7: Complete-Week Temporal Permutations

Permutations shift outcome-week blocks. Each repetition selects one nonzero
circular shift per fold, shared by every symbol. Scores, entries, and immutable
observation identities do not shift. Identity shifts are forbidden. Exactly
10,000 repetitions are requested.

An eligible permutation block is a complete UTC ISO calendar week wholly
contained in its fold, Monday 00:00 through the instant before the following
Monday 00:00. Fold-boundary partial weeks are excluded from both observed and
null permutation statistics. Internal observation absence does not redefine a
calendar week; ordinary frozen eligibility still applies. Each fold needs at
least four complete blocks.

For every test name, horizon, fold, and repetition, derive a deterministic
`PCG64` substream from the frozen base seed and canonical tuple
`(test_name, horizon, fold, repetition_index)`. Draw uniformly from integers
1 through W-1. Repeated shifts across repetitions and unequal W across folds
are allowed. D12 governs invalid permutations; reseeding and replacement are
forbidden.

Prospective rationale: strengthen exchangeability by not treating partial
boundary weeks as complete blocks.

## D8: Temporal-Permutation Confirmation P-Values

The spread and monotonicity tests at H12, H48, and H96 use D7's temporal null
and the same eligible complete-week support for observed and null statistics.

Spread is frozen top-score-group mean outcome minus frozen bottom-score-group
mean outcome. Monotonicity is Spearman association between ordered score bucket
and outcome, with average ranks for ties. Every permutation shifts outcomes,
preserves scores and entries, and fully recomputes groups, weights, spread, and
rho. Alternatives are one-sided in the preregistered favorable direction.

For N valid permutations,
`p = (1 + count(null_statistic >= observed_statistic)) / (1 + N)`;
equality enters the null tail. D12 governs invalid permutations. These six
p-values enter the frozen twelve-test Holm family. Switching to bootstrap or
another null after exposure is forbidden.

Prospective rationale: apply one coherent dependence-preserving null and
remove statistic-specific discretion.

## D9: Pooled Concentration Gate

The original pooled positive-PnL symbol-concentration threshold of 30% remains
the sole mandatory scientific concentration gate. Compute it with the original
pooled scope and contribution definition. Fold-level concentration is required
diagnostic reporting but ordinary concentration above 30% does not itself fail
a fold.

A fold may fail this component only for data-integrity defects: duplicated
observation or economic contribution, corrupt symbol identity, inconsistent
aggregation, or impossible totals. Reports distinguish integrity failure from
genuine concentration. Fold diagnostics cannot drop, replace, or rescue the
pooled gate.

When pooled total positive PnL is zero, define the concentration ratio as zero,
emit `NO_POSITIVE_PNL`, and create no additional concentration failure; all
other economic gates operate normally. Three-of-four fold logic is unchanged
except for genuine integrity failures.

Prospective rationale: preserve the original pooled gate without creating four
new small-sample concentration gates.

## D10: H12-Primary Label-to-Economics Classification

H12 is primary. H48 and H96 are mandatory diagnostics and do not independently
change the H12 classification. The canonical mandatory target registry is:

| Family | Canonical field | Static source | Semantics | Favorable ordering |
|---|---|---|---|---|
| TRRM | `tail_event` | `src/aegis/training/dataset.py:23`; `src/aegis/training/labels.py:141`; `src/aegis/training/phase_e.py:691-699` | Indicator that MAE reaches the frozen tail threshold | Lower; `0` is favorable |
| EQM-clean | `clean_quality` | `src/aegis/training/dataset.py:25`; `src/aegis/training/labels.py:120-125`; `src/aegis/training/phase_e.py:705-712` | Indicator that the path satisfies the frozen clean-entry predicate | Higher; `1` is favorable |
| EQM-net | `net_quality_after_costs` | `src/aegis/training/dataset.py:26`; `src/aegis/training/labels.py:115,138`; `src/aegis/training/phase_e.py:705-712` | MFE minus MAE minus frozen round-trip label cost | Higher is favorable |
| QMAE | `qmae` | `src/aegis/training/dataset.py:24`; `src/aegis/training/experiment.py:203-208`; `src/aegis/training/phase_e.py:669-672,887-889` | Canonical MAE fraction regression target | Lower is favorable |

`clean_entry`, `expected_return`, and raw `mae_fraction` are source concepts or
separate fields, not interchangeable target aliases for E5. Schema aliases are
not counted as separate labels. Validity, quarantine, timestamps, path-debug,
barrier-debug, and other non-target metadata are auxiliary and cannot block E6.

For every mandatory target, H12 connection requires pooled association in its
canonical favorable direction, correct target ordering, and favorable
direction in at least three folds. Ordered targets use Spearman with average
ranks; binary targets use favorable-class minus unfavorable-class mean H12
B_BASE return. Materiality uses the corresponding H12 economic difference and
MREM `0.0005`; no correlation threshold exists.

Classification is exact:

- `LABEL_ECONOMICS_DISCONNECTED`: pooled association is non-favorable, target
  ordering is violated, or fewer than three folds are favorable.
- `LABEL_ECONOMICS_CONNECTED_EFFECT_TOO_SMALL`: pooled direction and ordering are
  favorable in at least three folds, but the absolute favorable H12 economic
  difference is below MREM.
- `LABEL_ECONOMICS_CONNECTED_MATERIAL`: pooled direction and ordering are
  favorable in at least three folds and the favorable H12 difference reaches
  MREM.

Only `LABEL_ECONOMICS_DISCONNECTED` for a mandatory target blocks E6. The
effect-too-small class remains diagnostic. Labels cannot be selected, removed,
reordered, or reclassified after exposure.

Prospective rationale: preserve H12 primacy and Amendment 01's explicit
disconnected-label authority without creating a new effect-size gate.

## D11: Complete Diagnostic IC Curve

Spearman IC remains mandatory reporting and never an independent E6 gate.
Report gross and B_BASE net IC at H12, H48, and H96 for each symbol within each
fold, each fold pooled, discovery pooled, confirmation pooled, all folds pooled
descriptively, and each canonical cycle where mathematically computable.

Use average ranks. A scope is computable only with at least two paired
observations and nonconstant score and outcome ranks. No power- or
significance-based minimum is permitted. Undefined scopes are
`NOT_COMPUTABLE` with reasons. Cycle-level IC is summarized descriptively with
unweighted count, mean, median, and fraction positive. D12's hierarchical
week-block bootstrap supplies uncertainty where applicable. No IC magnitude,
p-value, interval, or significant-symbol count can rescue or block E6.

Prospective rationale: provide full horizon transparency without introducing
new confirmation criteria.

## D12: Type 7 Finite-Valid Resampling Policy

Every hierarchical bootstrap, power simulation, or permutation requests
exactly 10,000 replicates unless an earlier clause is stricter. Percentile CI90
uses Hyndman-Fan Type 7 q0.05 and q0.95. A mandatory fold-level bootstrap needs
at least four eligible source week blocks.

A replicate is valid only when mandatory sampled inputs and intermediates
exist, required denominators are valid, and the final statistic is finite. NaN
and positive or negative infinity are invalid. Invalid replicates are excluded
from quantiles or null counts and reported by exact reason. At least 9,500 of
10,000 requested replicates must be valid. No retry, replacement draw,
reseeding, or adaptive extension is allowed.

Below 9,500 valid replicates, a mandatory statistic becomes
`NOT_COMPUTABLE` and its gate fails; a diagnostic is reported
`NOT_COMPUTABLE` without creating a new gate. A zero-loss-denominator PF is
invalid unless prior authority supplies a finite canonical definition. The
same 9,500 threshold applies to permutation validity unless an earlier clause
is stricter.

Prospective rationale: use conventional percentile interpolation while
tolerating a small, transparent number of undefined replicates without
allowing adaptive retries.

## Cross-Decision Consistency

1. D1 and D2 generate controls independently for each horizon population.
2. D3 and D10 use H12 B_BASE net outcomes with realized funding for label
   economics while reporting barrier events separately.
3. D4 and D5 admit only completed Wilder ATR into discovery-frozen Type 7
   quintiles.
4. D6 and D12 share the finite-valid and `NOT_COMPUTABLE` policy.
5. D7 and D8 use complete-week temporal permutations for ordering inference.
6. D8 and D12 use the same 9,500-of-10,000 validity threshold unless prior
   authority is stricter.
7. D9 retains pooled concentration as mandatory and fold concentration as
   diagnostic except for integrity defects.
8. D10 permits only disconnected mandatory targets to block E6.
9. D11 keeps all IC outputs diagnostic regardless of value.

## Unchanged Governance

Nothing here alters the scientific question, hypotheses, frozen E3 entries,
fold boundaries, MREM `0.0005`, alpha `0.05`, frozen costs, model scores, twelve-
test Holm family, one-shot confirmation, lockbox prohibition, or the final
outcomes `CLOSE_THIS_SIGNAL_FAMILY` and `E6_JUSTIFIED`. No E6, Candidate,
Selection Policy, System Freeze, shadow, paper, or live authority is created.

## Prohibited Post-Hoc Changes

After any E5 scientific observation is accessed, no population rule, matcher,
funding treatment, ATR definition, bucket boundary, power model, permutation,
p-value method, concentration scope, label registry, IC scope, resampling rule,
seed, threshold, horizon, exclusion, or missing-data policy may change. A valid
unfavorable confirmation cannot be rerun.

## Determinism Requirements

All implementations must use float64, UTC timestamps, named frozen seeds,
stable total ordering, explicit tie rules, deterministic serialization, and
versioned environment manifests. Filesystem order, hash-map iteration, thread
scheduling, and platform-default randomness cannot affect scientific
artifacts. Every exclusion and invalid replicate is counted by fold, symbol,
horizon, module, and reason.

## Owner Authorization Record

The authorization source is the owner's direct instruction creating this
amendment. No cryptographic owner signature is asserted or fabricated. The
physical SHA-256 is recorded in the containing commit message; the containing
commit records the post-commit repository identity without self-referential
content in this file.

## Final Completeness Declaration

All twelve previously classified Category A decisions are resolved. No
scientific choice among the resolved alternatives is delegated to a future
execution specification. Remaining work is limited to engineering,
serialization, schema validation, and exact implementation conformance. No E5
scientific data or result informed this amendment, and the lockbox remains
unconsumed.
