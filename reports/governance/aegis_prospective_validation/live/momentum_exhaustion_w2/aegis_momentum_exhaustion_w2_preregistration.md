# Aegis Momentum Exhaustion W2 Preregistration

## Purpose

W2 asks whether a profitable open position should continue to be held. It does
not use volume waves to choose entries and it does not retune W1. W1 remains
negative and its final holdout remains sealed.

The primary hypothesis is that causal deterioration in favorable-extreme
formation, velocity, taker flow, volume, retracement, structure, 5m/15m
context, and BTC context predicts additional giveback well enough to improve
profit retention over simple trailing and the current mechanical protection.

## Audit Findings Before Registration

The TypeScript execution path currently contains:

- `ProfitGuardian.evaluateGuardianAction`, with break-even and ATR trailing;
- a 20x Aegis profile with 8% ROE break-even, 15% ROE trailing activation,
  1.5 ATR runtime trailing distance, -40% ROE hard stop, and +50% ROE TP;
- profit protection that locks at least 1% ROE and targets peak ROE minus 5%;
- ExitEye signal-aware close/protect behavior that cannot be faithfully
  replayed without historical committee responses;
- durable trade IDs, open/close history, bracket state, and trade events.

The local trade history contains 706 unique IDs with both OPEN and CLOSED
records: 225 LONG and 481 SHORT across all 11 symbols. These outcomes may
include manual intervention and all lie in the new W2 holdout period. They are
therefore an `ACTUAL`, audit-only population and are forbidden for training or
threshold selection.

The primary research population is independently generated and explicitly
`SIMULATED`: M1A independent candidates, next-minute causal entry, an eight
hour maximum path, and an eight-hour symbol/side cooldown. Before labels were
examined this produced 12,874 TRAIN, 8,926 VALIDATION, and 23,033 sealed
HOLDOUT episode identities.

## Episode Contract

The primary unit is `position_episode`, identified by a stable SHA-256 of
symbol, side, and entry timestamp. One episode has one entry and one final exit
per simulated policy. Decision rows are nested observations and may never be
treated as independent trades. Splits, bootstrap, and metrics remain grouped
by episode.

Every result records `outcome_source` as `ACTUAL`, `SIMULATED`, or `SHADOW`.
Actual and simulated entry, exit, gross PnL, and net PnL fields remain distinct.

## Frozen Partitions

- TRAIN: 2024-01-08 through 2024-09-30.
- VALIDATION: 2024-10-01 through 2025-03-31.
- FINAL HOLDOUT: 2025-04-01 through 2026-07-31, `SEALED`.
- Purge: eight hours at boundaries.

This split predates the W1 research interval used for its TRAIN/VALIDATION and
does not read or score the W1 holdout event population.

## Frozen Decision Contract

W2 becomes eligible only after peak MFE first crosses 0.25, 0.50, 0.75, or
1.00 ATR. The primary target is whether another 0.25 ATR is given back before
a new favorable 0.25 ATR extreme within three 5m bars. Secondary targets cover
one/two/three-bar giveback, 25%/40%/60% peak giveback, additional MFE, and
future giveback.

Models are restricted to an interpretable score, L2 logistic regression,
histogram gradient boosting, random forest, and discrete-time logistic hazard.
Calibration and policy threshold selection occur on TRAIN only. Probability
thresholds are frozen at 0.50, 0.60, 0.70, 0.80, and 0.90.

## Baselines

W2 must beat fixed ATR TP, fixed ATR trailing, percentage giveback, time exits,
the current 1.5 ATR trailing, and current break-even/profit protection.
Observed existing exits are available only for ACTUAL episodes. ExitEye cannot
be replayed exactly on simulated history because historical committee outputs
are absent; it must not be fabricated.

## Economic Gate

The selected W2 policy must improve median Profit Capture Ratio by at least 10
percentage points over the best baseline, remain within 2 bps of baseline net
expectancy at worst, survive 20 bps costs, improve at least seven symbols, pass
at least three of four walk-forward folds, and have positive clustered
confidence support. Selection uses TRAIN only. If VALIDATION fails, the final
holdout remains sealed.

No statistically significant but economically trivial result passes.

## Safety

W2 is historical research only. It does not modify TypeScript, PM2, WebSockets,
Shadow, Live, credentials, orders, positions, or exchange state.
