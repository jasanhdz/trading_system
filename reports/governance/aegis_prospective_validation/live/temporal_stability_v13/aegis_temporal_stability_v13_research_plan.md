# Aegis Temporal Stability V13 Research Plan

## Objective

V13 tests whether entry quality becomes stable when a long-history joint-path
model and a recent-window model must agree before selection. It adds causal
regime support, an out-of-distribution gate and a conservative 90th-percentile
MAE estimate. The target remains positive net utility with fewer adverse-first
paths and lower MAE, not elimination of every loss.

## Frozen Authority

V13 reuses the 48,191 independent V11 episodes, all V10 barrier outcomes, V11
clean-entry labels and V12 joint states. It preserves next-bar-open execution,
the nine barrier/horizon contracts, the 20-basis-point severe cost and four
purged expanding walk-forward folds. V12 remains the immutable comparator.

## Method

1. Fit one expanding-history and one 120-day recent joint-state model per side.
2. Fit regime specialists only when their train and calibration support exceed
   preregistered minima.
3. Require matching dominant states and bounded probability divergence.
4. Fit historical and recent q90 MAE models and use the more conservative value.
5. Learn a robust feature-distribution boundary on train/calibration only.
6. Assign barrier/horizon by causal regime in an assignment-only policy window.
7. Choose thresholds in a later policy-only window and evaluate untouched test.

## Fail-Closed Rules

Disagreement, unsupported distribution, excessive predicted MAE, non-positive
utility or insufficient policy candidates results in abstention. No trade quota
is forced and no test threshold is tuned. Manual pyramiding and manual closes
remain a separate future causal-policy study.

## Promotion Gates

Both temporal models must be skilled in at least three folds. At least three
economic folds must pass, the worst fold must be non-negative, selected utility
and CVaR must beat control, payoff ratio must be at least one, clean paths must
be at least 40%, adverse-first paths at most 30%, and mean MAE below unfiltered
control. Leave-one-symbol-out validation is required only after the primary gate.

No model export or Shadow/Live runtime change is authorized by this experiment.
