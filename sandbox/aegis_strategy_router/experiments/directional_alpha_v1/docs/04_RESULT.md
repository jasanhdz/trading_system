# Directional Alpha V1 Final Report

## Verdict

`DIRECTIONAL_ALPHA_WEAK_OR_UNSTABLE`

The frozen Opportunity radar was successfully reused, but the experiment did
not find robust or economic directional alpha. FINAL_HOLDOUT remains
`SEALED_NOT_OPENED`.

## Ablations

On the primary Opportunity population in VALIDATION:

| Family | AUC | FDR q | Interpretation |
|---|---:|---:|---|
| FLOW_ONLY | 0.4927 | 0.5697 | No signal |
| CROSS_MARKET_ONLY | 0.5845 | 1.89e-10 | Diagnostic predictive signal |
| FLOW_CROSS_MARKET | 0.5647 | 8.13e-07 | Ranking signal, but poor log loss and insufficient support |
| L2_ONLY | Not run | N/A | No clean eligible L2 period |

The combined model's log loss was `0.7235`, worse than the constant baseline
`0.6930`; it therefore failed the frozen combined-signal gate despite AUC being
above the Entry Quality V1 reference of `0.5503`.

## Confidence and economics

The primary top 10% directional-confidence selection contained only 100 rows
across 61 effective blocks:

- favorable-first: 58.00%, versus 49.20% Opportunity-only;
- MFE / MAE: 350.62 / 308.26 bps, ratio 1.137;
- gross expectancy: +3.87 bps;
- net expectancy at 20 bps cost: -16.13 bps;
- block-bootstrap 95% net CI: [-95.92, +9.81] bps.

The risk/coverage curve was not monotonic. Net results for 100%, 50%, 25%,
10% and 5% coverage remained negative. The isolated top 2% bucket was positive
but had only 20 rows/19 blocks and a CI spanning roughly -79 to +181 bps; it is
not treated as evidence.

The preregistered operational abstention selected 138 rows (13.8%) and also
remained negative at -9.77 bps net.

Applying the same frozen model to the preregistered Opportunity populations
did not reveal a hidden economic bucket. At 10% directional coverage, net
expectancy was -18.15 bps on all states, -10.42 bps in Opportunity top 20%,
-16.13 bps in the primary top 10%, and -43.13 bps in top 5%. No population was
selected or redefined after observing these diagnostics.

## Stability

The primary selection was asymmetric:

- LONG: -39.65 bps net;
- SHORT: +10.40 bps net.

Only 55.6% of weeks were positive. Although eight of ten symbols were positive
in the small selected sample, leave-one-symbol-out results were generally
negative and the side/temporal gates failed. No side-specific or symbol-specific
policy is created from this observation.

## Comparison with Entry Quality V1

Directional selection increased favorable-first and MFE/MAE geometry relative
to an Opportunity-only no-direction baseline over the same period, but it did
not create positive conservative-net expectancy. It also did not improve the
frozen Entry Quality V1 top-decile net result materially: -16.13 bps versus
-16.58 bps.

## Decision

Do not open FINAL_HOLDOUT, do not run gradient boosting, and do not begin
prospective collection, Shadow or Live from this result. The limited
cross-market signal may be retained as negative/diagnostic evidence, but a new
hypothesis and genuinely independent data would be required for another
directional experiment.

No production file, collector, exchange connection or financial behavior was
modified.
