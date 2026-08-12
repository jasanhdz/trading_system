# Aegis V19 Preregistered Experiment

## Status

V19 is a new, inactive research experiment. It is not a repair of V18 and does
not alter Live or Shadow. `V19_READY_FOR_SHADOW` and `V19_READY_FOR_LIVE` both
start false.

## Evidence Behind The Hypothesis

The independent design dataset contains 96,382 directional episodes. The V11
clean label has 11.92% prevalence. Every clean episode has positive frozen
ROE-10/H12 utility, mean utility 0.4667%, mean MAE 0.4661%, and mean current-TS
protected return 0.3225%. It captures only 34.24% of all positive-utility
episodes, so it is high precision and low recall rather than a complete entry
objective.

V18's raw clean rankers retained measurable discrimination on its validation
window (LONG AUC 0.7281, SHORT AUC 0.7226). Its unconstrained Platt calibrators
learned negative slopes and inverted both rankings. Calibrated AUC fell to
0.2719 and 0.2774. V18 remains failed because it produced no candidates and did
not demonstrate positive economic utility; these diagnostics are not used to
change V18.

## Frozen Hypothesis

V19 tests whether the economically aligned clean label can be used as one
component of a decomposed decision: monotonic clean rank, danger probability,
q90 MAE, probability of a positive current-TS protected path, and q10 economic
utility. Calibration is constrained to preserve ordering. A zero or negative
calibration slope fails closed rather than reversing the model.

Clean ranking alone cannot authorize a candidate. The frozen policy also
requires low danger, bounded predicted MAE, at least 50% protected-success
probability, and positive lower-tail utility. This directly tests whether a
candidate remains economically attractive after uncertainty, not merely
whether it resembles a clean path.

## Evidence Separation

All observations through 2026-08-09 are design evidence only. Fresh validation
runs from 2026-08-13 through 2026-09-30 and may be opened once after maturity.
The final holdout runs from 2026-10-01 through 2026-11-30 and cannot be opened
before 2026-12-01 or before the Shadow gate. Neither future window may be used
to choose features, seeds, thresholds, or calibration.

## Gates

V19 must first demonstrate positive lower-confidence expectancy, profit factor,
MAE, CVaR, temporal stability, and no calibration inversion on fresh
validation. Only then may it enter a separately authorized Shadow forward run.
Live requires an independent final holdout, at least 300 non-overlapping Shadow
opportunities per direction over at least 30 days, durable execution
compatibility, and separate owner authorization.

No model is exported now. No service, configuration, PM2 process, exchange
state, or current trading behavior is changed by this preregistration.
