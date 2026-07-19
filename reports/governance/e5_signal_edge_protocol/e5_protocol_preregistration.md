# E5 Signal Edge Protocol Preregistration

## Governance

| Field | Frozen value |
|---|---|
| Protocol ID | `E5_PROTOCOL_PREREGISTRATION` |
| Experiment ID | `E5_SIGNAL_EDGE_CONTROL_TEST` |
| Classification | `PREREGISTERED_DEV_EXPERIMENT` |
| Status | `PROTOCOL_FROZEN_UNEXECUTED` |
| Scientific family | Frozen E3 SHORT entry signal family |
| Permitted data | Existing development data only |
| Lockbox | Forbidden and unbound |
| Operational use | Forbidden |
| Prior experiments | E3, D1A, Phase O provenance, and E4A remain immutable |

This document freezes the complete scientific protocol before E5 is
implemented or executed. It contains no E5 result. No parameter, gate,
control, metric, or interpretation rule may be changed after an E5 outcome is
observed. A change requires a new experiment identity.

## 1. Primary Scientific Question

Does the frozen E3 SHORT entry signal family produce economically relevant
out-of-sample development-period returns above zero and above comparable
random controls when entry timing, eligible universe, holding period, costs,
and dependence structure are held fixed?

## 2. Hypotheses

### Null hypothesis

`H0` is the union of either condition:

1. Mean B_BASE net expectancy of the experimental entries is less than or
   equal to zero.
2. The experimental entries have no positive association with future returns
   beyond comparable random controls.

For each primary random control `c`, the superiority null is
`H0,c: E[R_exp - R_c] <= 0`.

### Alternative hypothesis

`H1` requires all of the following:

1. Mean B_BASE net expectancy is positive.
2. The experimental entries outperform every primary random control.
3. Both absolute expectancy and control deltas meet the frozen minimum
   economically relevant magnitude.
4. Score ordering, folds, symbols, outlier tests, and cost sensitivity satisfy
   every decision gate in this protocol.

The hypotheses form an intersection-union test: failure of any required
component prevents rejection of the global null.

## 3. Frozen Definitions

### Edge

`edge` means a reproducible conditional advantage of the frozen signal family
such that B_BASE net return is positive, exceeds comparable random controls,
meets the minimum relevant magnitude, and remains directionally stable under
the prespecified robustness checks. Positive gross movement alone is not edge.

### Alpha

`alpha = 0.05`, one-sided. All primary gates must pass. Because the global
alternative is accepted only through an intersection of individually required
tests, no multiplicity credit may be gained by passing only a subset. Secondary
p-values use Holm correction within their declared family and never promote a
failed primary gate.

### Beta

`beta` is the probability of failing the complete primary expectancy test when
the true advantage equals the frozen minimum relevant magnitude. Maximum
permitted `beta = 0.20` (minimum power 80%). Power is measured mechanically by
adding the minimum effect to centered week-block residuals and repeating the
entire primary bootstrap test for 10,000 seeded simulations. Failure to
demonstrate this power closes the signal family; it cannot be reported as
ambiguous evidence.

### Control

A control is a SHORT entry assignment generated without using future outcomes
or the experimental ranking, while preserving the exact comparison contract
defined below. A control may not change costs, horizon, entry rule, fold,
universe, data-quality eligibility, or per-stratum trade budget.

### Experimental group

The experimental group is exactly the frozen E3 `full_stack`, `B_BASE` SHORT
entry set. Entry identity consists of symbol, fold, signal timestamp, entry
timestamp, entry price, and side. Models are not retrained and entries are not
rescored or reselected.

### Control group

The primary control group consists of the two frozen random ensembles `C1`
and `C2` below. Each ensemble contains 10,000 replicates generated with NumPy
`PCG64`, base seed `20260718`, and replicate seed
`SHA256(protocol_id || control_id || replicate_index) mod 2**64`.

### Utility economic

Economic utility is positive B_BASE mean net return per fixed-notional trade,
after the exact E3 fees, slippage, and duration-dependent funding treatment,
with PF at least 1.10, acceptable tail risk, and superiority to controls.
Leverage, compounding, sizing, account equity, and portfolio reinvestment are
outside E5.

### Minimum relevant magnitude

The minimum economically relevant magnitude (`MREM`) is **5 basis points per
trade**, represented as `0.0005` unleveraged net return fraction. It applies
independently to:

1. Experimental B_BASE net expectancy above zero.
2. Experimental B_BASE net expectancy minus the mean expectancy of each
   primary random-control ensemble.
3. Top-score-decile minus bottom-score-decile B_BASE net expectancy.

No smaller effect can justify E6, regardless of nominal significance.

## 4. Frozen Data and Evaluation Contract

E5 uses only the existing canonical development data and the four frozen E3
SCORING folds. The latest permissible candle is the existing development
boundary `2026-04-26T23:59:59Z`. Semi-blind and lockbox data are forbidden.

The entry and outcome contract is:

- side: SHORT only;
- signal cadence: frozen hourly anchors;
- entry: next five-minute bar open;
- exit: close of the twelfth five-minute bar after entry;
- gross label: `(entry_open - h12_close) / entry_open`;
- B_BASE net label: gross label minus the frozen E3 B_BASE cost;
- binary directional label: `gross_label > 0`;
- binary economic label: `net_label_B_BASE > 0`;
- ties at exactly zero are negative for binary labels;
- notional: identical fixed notional for every trade;
- no stop, take profit, trailing, callback, or adaptive exit;
- no interpolation or missing-symbol completion.

The implementation must bind physical and canonical hashes for the E3 entry
set, complete eligible candidate population, score fields, canonical candles,
fold definitions, and cost configuration before reading labels. Missing or
inconsistent bindings are technical failures and leave E5 unexecuted.

## 5. Experimental and Statistical Units

The **experimental unit** is one eligible `(fold, signal_timestamp, symbol)`
SHORT candidate with one frozen score vector and one H12 outcome.

The **selection unit** is one frozen hourly signal cycle. All candidates in a
cycle share market-time exposure and form one cross-sectional cluster.

The **statistical dependence unit** is the UTC ISO calendar week nested within
one fold. Every candidate and selected trade in a sampled week remains together
in bootstrap resampling. No observation may appear in two folds.

## 6. Experimental Design

### Experimental arm

Evaluate the frozen experimental entries without changing their timestamps,
symbols, scores, gates, or selection decisions.

### C1: same-cycle random-symbol control

For every experimental entry, select exactly one symbol uniformly from the
structurally eligible candidate symbols in the same fold and signal cycle.
Structural eligibility includes canonical-data validity, universe membership,
side availability, and complete H12 outcome, but excludes every predictive
model score, veto, threshold, and ranking. Sampling is without replacement
inside a cycle; because the frozen budget is one entry per cycle, this yields
one control entry. C1 exactly preserves experimental timestamps, folds, trade
count, entry rule, side, horizon, and costs.

### C2: matched-time-and-symbol random control

Within each `(fold, UTC calendar month)` stratum, randomly assign the exact
experimental symbol multiset to eligible hourly cycles from that stratum. A
symbol-cycle pair must be structurally eligible. Assignment uses a deterministic
seeded bipartite matching with candidate edges sorted by cycle then symbol.
Cycles are sampled without replacement and no cycle receives more than one
entry. The exact fold-month trade count and symbol counts are preserved. If a
complete matching does not exist for any replicate, that replicate is invalid;
any invalid replicate is a technical protocol failure rather than permission to
relax matching.

### Non-random reference controls

`no_trade` with expectancy zero and the already frozen directional ECON
baselines may be reported as secondary context. They do not replace C1 or C2,
and diagnostic baselines cannot enter the primary random-control maximum.

### Control comparison

For each control, report its 10,000-replicate expectancy distribution, mean,
median, P5, P50, P95, and the experimental-minus-control distribution. The
Monte Carlo p-value is
`(1 + count(control_expectancy >= experimental_expectancy)) / 10001`.
The primary control delta is experimental expectancy minus ensemble mean.
E6 requires both C1 and C2 to pass separately; controls may not be averaged or
selected after results are known.

## 7. Permutation Test

The permutation test targets score-outcome association while preserving
temporal and cross-symbol dependence.

1. Within each fold, order complete hourly cycles chronologically.
2. Treat each UTC ISO week as an indivisible block containing every symbol and
   outcome in that week.
3. Circularly shift outcome-week blocks by a non-zero number of weeks while
   keeping frozen score and selection rows fixed.
4. Use one independently seeded non-zero shift per fold and the same shift for
   all symbols in that fold.
5. Recompute the frozen selected-entry outcome statistic without fitting,
   tuning, or changing selection.
6. Produce 10,000 permutations with the same seed derivation used for controls.

The statistic is pooled B_BASE net expectancy. The p-value uses the same
`(1 + exceedances) / 10001` formula. The unpermuted identity mapping is
forbidden. If a fold has fewer than four complete weekly blocks, E5 cannot
justify E6 and closes the family.

## 8. Bootstrap and Autocorrelation

All confidence intervals are percentile CI90 from 10,000 hierarchical block
bootstrap replicates using `PCG64` seed `20260718`.

1. Resample UTC ISO weeks with replacement independently within each fold.
2. Keep all cycles, symbols, experimental entries, and paired control outcomes
   inside each selected week together.
3. Concatenate the four resampled folds and weight every resulting trade
   equally for pooled metrics.
4. For paired deltas, resample the experimental and control records jointly.
5. Preserve the chronological order of observations within each sampled week
   when computing drawdown.

This week-cluster procedure is the sole autocorrelation correction. IID
bootstrap, row bootstrap, and symbol-independent bootstrap are forbidden for
primary inference.

## 9. Score Evaluation

The score is the final frozen scalar used to rank E3 SHORT candidates, with its
historical direction preserved: larger score means stronger entry preference.
No recalibration, normalization, clipping, refit, or alternate score is allowed.

Evaluate on the complete eligible candidate population in each SCORING fold:

- Spearman correlation between score and continuous B_BASE net label;
- Kendall tau-b between score and continuous B_BASE net label;
- ROC AUC against the binary directional label;
- ROC AUC against the binary economic label;
- average precision for both binary labels;
- equal-count score-decile outcomes;
- top-decile minus bottom-decile net expectancy;
- experimental selected-group lift versus all eligible candidates.

Ties are ordered deterministically by score descending, symbol ascending, then
timestamp ascending. Statistical intervals use the frozen week-block bootstrap.

## 10. Label Evaluation

Labels are evaluated, not redesigned. The implementation must prove that:

- entry uses only the next-bar open;
- H12 uses exactly 12 complete five-minute candles;
- no candle at or before entry is used as a future outcome;
- no candle after the development boundary is accessed;
- every label is recomputable from canonical OHLC and the frozen cost formula;
- recomputed and frozen E3 labels agree within `1e-12`;
- missing H12 paths are counted and never imputed;
- a candidate with an invalid label is excluded from both experimental and
  control populations before randomization, with counts reported.

Any drift in an experimental entry label or any loss of an experimental entry
is a technical failure and leaves E5 unexecuted.

## 11. Monotonicity

Within each fold, assign all eligible candidates to ten equal-count bins using
the deterministic score ordering. Bin 1 is the lowest score and bin 10 the
highest. Report count and B_BASE gross/net expectancy per bin.

Monotonicity metrics are:

- Spearman rho between bin index and bin mean net expectancy;
- Kendall tau-b between bin index and bin mean net expectancy;
- OLS slope of bin mean net expectancy on bin index;
- number and total magnitude of adjacent downward violations;
- bin-10 minus bin-1 net expectancy.

The monotonicity gate requires pooled rho greater than zero, pooled top-bottom
spread at least `0.0005`, the week-block CI90 lower bound for the spread above
zero, and positive top-bottom spread in at least three of four folds. Strictly
monotonic ten-bin means are not required and may not be substituted as a gate.

## 12. Stability

### Fold stability

All four folds are included. The gate requires:

- positive experimental B_BASE net expectancy in at least three folds;
- positive experimental-minus-C1 and experimental-minus-C2 expectancy in at
  least three folds each;
- no fold experimental expectancy below `-0.0005`;
- no fold may be dropped, merged, reweighted, or renamed.

### Symbol stability

All frozen universe symbols are included. Report per-symbol counts, gross/net
expectancy, PF, control deltas, and confidence intervals. The gate requires:

- positive experimental-minus-primary-control delta in at least 7 of 11
  symbols for both C1 and C2;
- leave-one-symbol-out pooled experimental net expectancy remains positive for
  all 11 exclusions;
- leave-one-symbol-out control deltas remain positive for all exclusions;
- no symbol contributes more than 30% of total positive experimental PnL.

Zero-count or unmatchable symbols fail the symbol gate; they may not be removed.

## 13. Economic Utility

The primary scenario is frozen E3 `B_BASE`. Report the identical calculations
under the frozen optimistic and pessimistic scenarios as sensitivity.

Economic utility requires all of:

- B_BASE net expectancy point estimate at least `0.0005`;
- B_BASE net expectancy CI90 lower bound above zero;
- B_BASE PF at least 1.10 and PF CI90 lower bound above 1.00;
- B_BASE delta versus both control means at least `0.0005`;
- B_BASE delta CI90 lower bound above zero for both controls;
- pessimistic-scenario net expectancy not below zero;
- fixed-notional non-compounded maximum drawdown no worse than the P95 drawdown
  of either primary control ensemble;
- CVaR 5% no worse than the lower P5 CVaR of either primary control ensemble.

No gross-only, leverage-adjusted, compounded, or cost-excluded result can pass.

## 14. Complete Metrics

Report pooled, by fold, and by symbol where defined:

- candidate and trade counts;
- gross and net expectancy for all three cost scenarios;
- median return;
- standard deviation and standard error under week clustering;
- P1, P5, P10, P25, P50, P75, P90, P95, and P99 returns;
- win rate, average win, average loss, payoff ratio, and PF;
- CVaR 1% and 5%;
- worst trade;
- fixed-notional non-compounded maximum drawdown;
- turnover and total costs;
- ISO-week expectancy distribution;
- positive folds and positive symbols;
- best-trade, best-1%, best-5%, and best-symbol profit concentration;
- C1 and C2 ensemble distributions and deltas;
- permutation statistic and p-value;
- score correlations, AUC, average precision, decile metrics, and lift;
- monotonicity metrics;
- leave-one-fold-out and leave-one-symbol-out metrics;
- all-trades, excluding-best-1%, excluding-worst-1%, and P1/P99-winsorized
  sensitivity;
- bootstrap CI90 for expectancy, PF, control deltas, decile spread, CVaR, and
  drawdown;
- estimated power and beta at the MREM.

Secondary p-values are grouped into score discrimination, distributional
economics, and stability families and Holm-adjusted separately. Adjusted or
unadjusted secondary significance cannot override a failed primary gate.

## 15. Required Artifacts

An eventual implementation must produce exactly versioned, hash-bound artifacts
covering at least:

1. `preregistration.md` and physical SHA-256 binding.
2. `run_manifest_attempt_1.json` and `run_manifest_attempt_2.json`.
3. `environment_manifest.json`.
4. `input_manifest.json`.
5. `entry_set_manifest.json`.
6. `eligible_population_manifest.json`.
7. `score_manifest.json`.
8. `label_manifest.json`.
9. `control_definition.json`.
10. `control_replicates.parquet`.
11. `permutation_manifest.json`.
12. `permutation_results.parquet`.
13. `experimental_trades.parquet`.
14. `score_metrics.json`.
15. `decile_metrics.json`.
16. `economic_metrics.json`.
17. `fold_metrics.json`.
18. `symbol_metrics.json`.
19. `stability_metrics.json`.
20. `outlier_sensitivity.json`.
21. `power_report.json`.
22. `decision_gates.json`.
23. `scientific_summary.json` and `scientific_summary.md`.
24. `scientific_aggregate.json`.
25. `determinism_report.json`.

Operational timestamps, absolute paths, PIDs, and attempt identifiers must not
enter scientific hashes.

## 16. Required Tests

### Unit and contract tests

- frozen entry identity and source hash;
- SHORT H12 gross and net label formulas;
- next-bar-open and exact twelve-bar boundary;
- cost parity with E3;
- development-boundary enforcement;
- candidate eligibility without predictive gates;
- C1 same-cycle sampling and one-entry budget;
- C2 exact fold-month count and symbol-multiset matching;
- no replacement and deterministic bipartite matching;
- exact `PCG64` seed derivation;
- random-control p-value formula;
- non-zero week circular permutation;
- identical symbol shift within each fold;
- week-block bootstrap and paired resampling;
- no IID primary bootstrap;
- deterministic tie ordering;
- score-decile construction;
- Spearman, Kendall, AUC, average precision, and spread golden cases;
- monotonicity violation accounting;
- fold and symbol stability gates;
- fixed-notional PF, CVaR, and drawdown formulas;
- concentration and outlier removal;
- power and beta calculation at exactly `0.0005`;
- Holm correction confined to secondary metric families;
- every individual decision gate;
- any failed, missing, equal-to-boundary, NaN, or infinite gate closes the
  family;
- no E5 module imports lockbox, Candidate, Selection Policy, Freeze, shadow,
  paper, live, Binance, or PM2 paths;
- E3, D1A, Phase O provenance, and E4A hashes remain unchanged.

### Determinism tests

Execute E5 twice from clean, independent report roots with the same commit,
environment, inputs, and seeds. Require byte-identical scientific artifacts,
control selections, permutation statistics, bootstrap samples, metrics, gates,
and aggregate hash. Numerical comparison tolerance is `1e-12`, but serialized
scientific artifacts must be byte-identical. Any difference is a technical
failure and no scientific disposition is emitted.

## 17. Threats to Validity and Bias Register

### Validity threats

- only four temporal folds;
- development-period regime may not represent future markets;
- cross-symbol and temporal dependence;
- fixed H12 exit may mask or create entry utility;
- random controls may not reproduce every latent operational constraint;
- transaction-cost and funding models may differ from future execution;
- score distributions may drift;
- finite control and bootstrap simulations introduce Monte Carlo error;
- symbol-level sample sizes may differ;
- E5 remains development evidence and is not lockbox confirmation.

### Potential biases

- survivorship in the frozen universe;
- selection bias from the already developed E3 family;
- repeated use of the same development period across prior experiments;
- publication bias toward favorable diagnostics;
- dependence on frozen candidate eligibility definitions;
- concentration in particular weeks, folds, symbols, or extreme trades;
- conditioning C1 on cycles where E3 traded;
- imperfect matching feasibility in C2.

### Potential leakage

- labels or future candles entering features or eligibility;
- fitting, calibration, thresholding, or normalization on SCORING outcomes;
- outcome-aware control matching;
- reuse of H12 candles across a forbidden temporal boundary;
- score reconstruction using post-signal state;
- using future universe membership or data-quality status;
- adapting random seeds, block length, controls, or exclusions after results;
- using semi-blind or lockbox data.

### Potential curve fitting

- choosing the better random control after observation;
- changing MREM, alpha, beta, CI level, or block size;
- selecting a subset of folds, symbols, months, or cost scenarios;
- replacing H12 with a favorable horizon;
- selecting alternate scores, labels, decile counts, or monotonicity tests;
- adding controls or metrics after observing failures;
- optimizing model, veto, threshold, ranking, or entry budget;
- interpreting a nominal secondary p-value as primary evidence;
- proposing E6 from gross-only, subset-only, or outlier-dependent results.

## 18. Automatic Decision Gates

The implementation must serialize every gate with expected value, actual value,
pass/fail status, and evidence path. `E6_JUSTIFIED` requires all gates below:

1. Input hashes and E3 entry identities match exactly.
2. All labels reproduce within `1e-12` and no experimental entry is lost.
3. Both independent executions are byte-identical.
4. Estimated power is at least 80% (`beta <= 0.20`) at MREM.
5. Experimental B_BASE net expectancy is at least `0.0005`.
6. Experimental B_BASE net expectancy CI90 lower bound is above zero.
7. B_BASE PF is at least 1.10 and PF CI90 lower bound is above 1.00.
8. Experimental-minus-C1 mean delta is at least `0.0005`.
9. Experimental-minus-C2 mean delta is at least `0.0005`.
10. CI90 lower bounds for both control deltas are above zero.
11. Both random-control Monte Carlo p-values are at most 0.05.
12. Week-block permutation p-value is at most 0.05.
13. Pooled score monotonicity and top-bottom spread gates pass.
14. Top-bottom spread is positive in at least three folds.
15. Experimental expectancy is positive in at least three folds.
16. Both control deltas are positive in at least three folds.
17. No fold expectancy is below `-0.0005`.
18. All symbol-stability and leave-one-symbol-out gates pass.
19. Positive-PnL symbol concentration is at most 30%.
20. Pessimistic-scenario net expectancy is non-negative.
21. CVaR and drawdown gates pass against both controls.
22. Excluding the best 1% leaves experimental net expectancy positive and both
    control deltas positive.
23. P1/P99 winsorized results retain positive expectancy and both positive
    control deltas.
24. At least 100 completed experimental trades exist in each fold.
25. No forbidden data access, fitting, tuning, or protocol mutation occurred.

## 19. Automatic Closure and E6 Rule

After a technically valid E5 execution there are exactly two scientific
outcomes:

### `E6_JUSTIFIED`

Emit only when every gate 1 through 25 passes. This outcome authorizes only the
separate preregistration of E6. It does not authorize lockbox access, Candidate,
Selection Policy, Freeze, shadow, paper, live, or operational execution.

### `CLOSE_THIS_SIGNAL_FAMILY`

Emit when any gate 1 through 25 fails. This includes insufficient power,
insufficient trades, point estimates below MREM, confidence intervals touching
or crossing zero, p-values above 0.05, unstable folds or symbols, outlier
dependence, failed pessimistic costs, missing metrics, NaN/infinite values, and
exact equality where a gate requires strict positivity.

A technical inability to execute the frozen protocol produces no scientific
outcome and leaves E5 unexecuted. It is not an ambiguity category and cannot be
used to justify E6.

## 20. Prohibition on Positive Reinterpretation

The following can never be described as positive evidence and automatically
produce `CLOSE_THIS_SIGNAL_FAMILY` after a valid execution:

- gross expectancy positive but B_BASE net expectancy non-positive;
- point estimate positive but CI90 touches or crosses zero;
- nominal p-value significant only before correction;
- effect statistically significant but below `0.0005`;
- only one control beaten;
- only a favorable fold, symbol, month, cost scenario, or subset passes;
- fewer than three positive folds;
- result disappears without the best 1% of trades;
- score AUC is favorable but economic gates fail;
- monotonicity is favorable but selected-entry economics fail;
- optimistic costs pass while B_BASE or pessimistic gates fail;
- random controls pass while permutation fails, or vice versa;
- secondary metrics conflict with a failed primary gate;
- an interval is wide enough to contain a favorable effect;
- a result is labelled inconclusive, promising, nearly significant, or
  directionally favorable.

There is no scientific `INCONCLUSIVE`, `PARTIAL`, `PROMISING`, or discretionary
review outcome. No human override may convert a failed gate into E6 under this
experiment identity.

## 21. Safety and Immutability

E5 is development-only and has no lockbox binding or query budget. It may not
read semi-blind data, train or alter prior models, change E3 entries, modify any
prior artifact, create a Candidate, publish a policy or freeze, enable LONG,
enable execution, access Binance, invoke PM2, or start shadow, paper, or live
operation. No implementation may begin until this physical document and its
SHA-256 are reviewed and explicitly approved by the owner.
