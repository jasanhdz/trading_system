# Aegis Tail-Aware Entry V8 Research Plan

## Finding that motivates V8

V7 showed positive outcomes on more than half of the unfiltered paths, but mean
net returns remained negative. The dominant problem is payoff asymmetry: many
small protected gains are outweighed by fewer large losses. V8 therefore treats
win rate, low MAE, and protection as supporting evidence rather than proof of an
edge.

## Frozen design

V8 keeps LONG and SHORT separate and replaces the dominant hard
`TRANSITION_SELECTIVE` bucket with five soft causal memberships: trend
continuation, breakout, reversal, range reversion, and exhaustion risk. Every
candidate receives a normalized probability-like membership vector; no outcome
field participates in routing.

The forward environment is labeled independently at 30, 60, and 120 minutes
using BTC direction and cross-sectional breadth. A calibrated four-class router
must beat its training prior and majority-class controls outside sample.

The late-entry detector is evaluated as its own scientific claim. It must beat
prevalence baselines in both Brier score and average precision. Its success may
not be inferred from the final selector.

## Entry and tail objectives

The selector estimates clean-entry probability, late-entry probability,
catastrophic-loss probability, positive stress-cost probability, MAE q90, time
to positive, and protected net return. Selection penalizes the negative tail
explicitly and is evaluated on mean net, lower-tail CVaR, payoff ratio, MAE, and
opportunity frequency.

## Protection and cost sensitivity

The current TypeScript profile remains the control. Nine research-only profiles
combine hard stops of -15%, -25%, and -40% ROE with protection armed at 5%, 10%,
and 20% ROE. Each profile is replayed over both admissible OHLC paths and the
worse result is retained.

Expected, stress, and severe round-trip costs are 10, 15, and 20 basis points.
The primary policy is chosen using stress-cost outcomes. Positive results under
expected costs alone cannot pass the gate.

These profiles are counterfactual evidence only. They do not modify production
stops, trailing behavior, TypeScript, PM2, Shadow, or Live.

## Validation

Four purged expanding walk-forward folds use a 120-minute embargo and
independent test snapshots. Thresholds and profile choices are derived only from
training and calibration data. Promotion requires three positive folds, a
non-negative worst fold, stress-cost improvement over the frozen control,
improved CVaR, no worse MAE, payoff ratio of at least one, skilled regime and
late-entry models in three folds, acceptable opportunity frequency, and at least
eight successful leave-one-symbol-out evaluations.

No threshold may be relaxed after test results are observed.

## Safety

- Runtime effect: `NONE`.
- Model export: `PROHIBITED_UNTIL_ALL_GATES_PASS`.
- Shadow activation: `PROHIBITED_UNTIL_SEPARATE_AUTHORIZATION`.
- Live activation: `PROHIBITED`.
- Exchange calls: `0`.
- Exchange mutations: `0`.
