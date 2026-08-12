# Aegis Directional Contract V15 Research Plan

## Objective

V15 tests whether simpler, direction-specific information contracts improve
entry danger, cleanliness, MAE, and realized economic quality. It does not add
indicators, change runtime behavior, or authorize a model.

## Frozen Candidates

The control is the complete 176-position V9 vector. The shared candidate removes
eight exact or mathematically equivalent positions while retaining one canonical
representation. The LONG candidate additionally removes
`LOCAL_MOMENTUM_TREND` and `ROLLING_CONTEXT`, based on the exploratory V14
warning. The SHORT candidate uses only the shared deduplication because V14 did
not support another stable SHORT removal.

The neutral/selective correlation is retained because correlation in the
observed dataset does not prove a permanent semantic identity.

## Validation

Each side and contract receives separate danger, clean-entry, and q90-MAE
models. Thresholds are selected from calibration data using a frozen score that
rewards clean probability and penalizes danger and normalized q90 MAE. Test data
is never used to choose a threshold.

Four purged expanding folds are evaluated. The fourth test partition begins
after V14's evidence ended and is the decisive post-V14 holdout. A candidate
must improve at least three folds and must not regress on that final holdout.
Economic quality includes mean utility after frozen costs, CVaR, adverse-first
rate, mean MAE, and opportunity frequency.

## Interpretation Limits

The first three folds are retrospective and V14 informed the LONG hypothesis.
Only the final partition is new evidence relative to V14. Passing V15 can at
most justify a separately authorized Shadow experiment; it cannot justify Live
promotion.

## Safety

The experiment reads immutable local evidence only. Model export, feature
contract activation, Shadow/Live changes, network calls, exchange calls, and
exchange mutations are prohibited.
