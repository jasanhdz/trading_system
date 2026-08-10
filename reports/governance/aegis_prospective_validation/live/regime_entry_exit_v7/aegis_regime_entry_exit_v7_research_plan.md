# Aegis Regime Entry/Exit V7 Research Plan

## Purpose

V7 tests whether Aegis can select LONG and SHORT entries that become favorable
quickly, suffer less adverse excursion, and convert available favorable movement
into positive net results. It is a research challenger, not a production rule.

Low MAE is necessary evidence about entry timing but is not treated as proof of
profitability. Promotion requires positive protected net expectancy after costs.

## Frozen hypotheses

1. Entry quality and exit capture are distinct and must be attributed separately.
2. A joint target is more useful than direction-only or MAE-only prediction.
3. Trend continuation, breakout, reversal, range reversion, and transition setups
   require separate specialists.
4. Causal multitimeframe, volume, relative-strength, BTC, volatility, and
   support/resistance context can improve selection without future leakage.
5. A late-entry detector can reject correct-direction signals entered after the
   favorable movement is substantially exhausted.
6. Counterfactual protection at 5%, 10%, and 20% ROE must be replayed rather than
   assumed to improve results.
7. Only purged walk-forward evidence, symbol generalization, calibrated
   probabilities, and net results after costs can authorize a later Shadow trial.

## Causal population

The source is the immutable V6 one-year dataset covering all 11 configured
symbols, both sides, next-bar-open entries, and 24 future 5-minute bars. V7 adds
only features available at the signal timestamp. Future candles are used solely
for labels and counterfactual lifecycle outcomes.

## Entry and exit attribution

Every candidate receives one deterministic attribution:

- `CLEAN_REALIZED_WIN`: clean entry and positive protected result.
- `GOOD_ENTRY_POOR_CAPTURE`: favorable opportunity existed but protection did
  not retain a positive result.
- `LATE_OR_ADVERSE_ENTRY`: early reversal, excessive prior extension followed
  by failure, or excessive delay before becoming positive.
- `NO_DIRECTIONAL_EDGE`: no meaningful favorable opportunity.
- `AMBIGUOUS_PATH`: same-bar ordering cannot be known from OHLC.

The audit records MAE, MFE, time underwater, time to positive after costs,
available net opportunity, realized protected net, and capture efficiency.

## Specialists

LONG and SHORT are always fitted separately. Within each side, specialists are
fitted for trend continuation, confirmed breakout, confirmed reversal, range
reversion, and transition/selective conditions. Sparse or unknown conditions
abstain; they never inherit a fabricated decision.

## Protection experiment

Current TypeScript protection is the control. Three research-only profiles arm
protection at 5%, 10%, and 20% ROE. Every profile is replayed over both admissible
OHLC paths, and the worse path is scored. Profile choice is learned only from
training/calibration data and evaluated unchanged on the test fold.

These profiles do not modify TypeScript or any running service.

## Validation

V7 uses four purged expanding walk-forward folds with a 120-minute embargo.
Selection thresholds come only from calibration data. The test population uses
independent non-overlapping snapshots. Results are compared with the current
brain and unfiltered candidates.

Promotion requires at least three positive folds, a non-negative worst fold,
better net expectancy and MAE than the current-brain control, a skilled regime
router in at least three folds, acceptable opportunity frequency, and no
regression in at least eight leave-one-symbol-out evaluations.

No threshold may be relaxed after observing test outcomes.

## Safety

- Runtime selection effect: `NONE`.
- Shadow activation: `PROHIBITED_UNTIL_SEPARATE_AUTHORIZATION`.
- Live activation: `PROHIBITED`.
- Exchange calls: `0`.
- Exchange mutations: `0`.
- Model export: fail-closed until every historical gate passes.

