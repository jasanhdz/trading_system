# Aegis Joint Path V12 Research Plan

## Objective

V12 tests whether entry quality improves when direction, path cleanliness,
adverse-first risk, unresolved state, barrier and horizon are learned as one
coherent decision rather than intersecting independently trained probabilities.

The economic objective is not to eliminate all losses. It is to reduce adverse
first paths and MAE while retaining positive net utility after the frozen severe
round-trip cost. The existing 40% ROI stop is catastrophe context only and is
not changed or optimized by this experiment.

## Frozen Evidence

- Reuse all V11 episodes, V10 barrier outcomes and V11 clean-entry labels.
- Preserve next-bar-open entry, 5/10/20% ROE barriers, 30/60/120-minute horizons
  and the 20-basis-point severe cost.
- Preserve four purged expanding walk-forward folds and nested calibration.
- Never tune calibration, contract assignment or policy on test data.

## Method

1. Train one joint multiclass estimator per side over all nine contracts.
2. Assign one contract by side and causal regime using only the first half of
   the policy window; use a side-global fallback for unsupported regimes.
3. Derive selection thresholds only from the second half of the policy window.
4. Compare untouched test selections with the unfiltered primary V11 contract.
5. Report MAE, time to positive, maximum favorable excursion, clean rate,
   adverse-first rate and the frozen current-TS stress profile.

## Promotion Gates

At least three of four folds must have a skilled calibrated joint model and
positive economic performance. The worst fold must be non-negative. Selected
entries must improve utility and CVaR over control, preserve payoff ratio at or
above one, contain at least 40% clean paths and no more than 30% adverse-first
paths, and not regress mean MAE. Leave-one-symbol-out validation remains
mandatory after the primary gate passes.

No model is exported and no Shadow or Live runtime is changed by this plan.
