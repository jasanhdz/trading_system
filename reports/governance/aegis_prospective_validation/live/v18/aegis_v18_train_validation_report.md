# Aegis V18 TRAIN and VALIDATION Report

## Verdict

`V18_READY_FOR_SHADOW = FALSE`

V18 failed its preregistered validation gate. Thresholds, features,
hyperparameters, seed, and calibration were not changed after observing the
result. The final holdout was not opened and remains unavailable until its
preregistered window completes.

## Partitions

- TRAIN source rows: 78,078.
- VALIDATION source rows: 18,304, split evenly between LONG and SHORT before
  independent-episode filtering.
- FINAL HOLDOUT accesses: 0.

The candidate used one fixed seed and no search space. The fit/calibration
split was chronological with a 120-minute purge. VALIDATION was never passed to
model fitting or calibration.

## LONG

No candidate survived in TRAIN or VALIDATION. On VALIDATION, calibrated clean
probability ranged from 0.065895 to 0.067659, below the frozen 0.50 minimum.
Expected utility was negative through the 95th percentile (-0.001124) and only
reached 0.001235 at the maximum. Median predicted q90 MAE was 0.012892.

Clean average precision was 0.059718, danger average precision was 0.358083,
MAE q90 pinball loss was 0.001470, and utility RMSE was 0.004570. With zero
selections, economic confidence intervals and temporal stability could not be
established.

## SHORT

No candidate survived in TRAIN or VALIDATION. On VALIDATION, calibrated clean
probability was approximately 0.0671 for almost the entire population, below
the frozen 0.50 minimum. Expected utility was negative through the 95th
percentile (-0.000792) and reached 0.001859 at the maximum. Median predicted q90
MAE was 0.012212.

Clean average precision was 0.063987, danger average precision was 0.338434,
MAE q90 pinball loss was 0.001353, and utility RMSE was 0.004570. Economic gate
metrics are unavailable because no candidate survived.

## Interpretation

The immediate feasibility failure is the absolute clean-probability threshold,
which is incompatible with the observed calibrated base rate. However, lowering
it on this validation would violate the preregistration and would not address
the deeper result: utility predictions were negative for at least 95% of both
directional populations and clean-entry ranking was weak.

This run therefore does not justify a rescued V18. Any reformulation must be a
new preregistered version with a new untouched holdout, not an amended V18.

## Historical Context

Historical four-fold references also failed their gates:

- V15 LONG mean fold utility: -0.002032; 733 selections; mean MAE 0.003230.
- V15 SHORT mean fold utility: -0.002095; 777 selections; mean MAE 0.003217.
- V17 LONG mean fold utility: -0.002185; 182 selections; mean MAE 0.004733.
- V17 SHORT mean fold utility: -0.001677; 266 selections; mean MAE 0.003202.

These references use prior walk-forward partitions and are contextual, not a
same-window statistical victory or loss for V18.

## Gates and Safety

- Validation gate: FAIL.
- Final holdout: SEALED_NOT_AVAILABLE.
- Model export: false.
- Shadow changed: false.
- Live changed: false.
- Exchange calls: 0.
- Exchange mutations: 0.
- `V18_READY_FOR_LIVE`: false.
