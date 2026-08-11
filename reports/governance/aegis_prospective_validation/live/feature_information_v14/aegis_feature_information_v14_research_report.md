# Aegis Feature Information V14 Research Report

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V14 does not authorize a model, feature-contract, Shadow, or Live change. The
ten taker-flow candidates failed the preregistered stability gate. Exchange
calls and exchange mutations were zero.

## Evidence

- Source: immutable V11 causal episodes plus the existing OHLCV database opened
  read-only.
- Coverage: 90,442 directional rows / 45,221 episodes (93.8370% of V11 rows).
- Period: 2025-08-09 06:55 UTC through 2026-07-17 18:55 UTC.
- Universe: all eleven symbols at every retained timestamp.
- Causality: only the 24 closed 5-minute bars before next-bar-open entry.
- Baseline: 176 V9 feature positions.
- Candidate: baseline plus ten taker buy/sell imbalance features.
- Validation: four purged expanding walk-forward folds, LONG and SHORT
  independently, with a 120-minute embargo.

Funding and open-interest columns contain zero observations. Historical order
book and liquidation evidence is absent. No values were inferred, downloaded,
or synthesized for those sources.

## Candidate Result

The admission threshold was improvement in at least three of four future folds
for danger log loss, danger average precision, clean-entry log loss, and q90 MAE
pinball loss, on both sides.

| Side | Danger log loss | Danger AP | Clean log loss | q90 MAE | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| LONG | 1/4 | 2/4 | 1/4 | 3/4 | FAIL |
| SHORT | 2/4 | 1/4 | 2/4 | 1/4 | FAIL |

The flow family contains some MAE information for LONG, but it does not
reliably distinguish danger or clean entries. On SHORT it is unstable across
all targets. Adding these ten fields to the decision brain now would increase
complexity without stable incremental evidence.

## Existing Feature Audit

The 176 positions contain one duplicate source name:
`volume_ratio_6_24` at positions 40 and 126. Both positions are perfectly
correlated. V14 preserved both to keep the V13 baseline unchanged and assigned
position-qualified audit names.

Nine near-exact redundancy pairs were found, including:

- equivalent 5m, 15m, and 1h return windows;
- the duplicated volume ratio;
- equivalent close-location fields;
- equivalent neutral/selective encodings;
- alignment and conflict scores that are exact complements.

No feature was constant or near-constant. Two consecutive-candle count fields
reached the preregistered robust-shift warning boundary. These are diagnostics,
not removal authority.

Family removal was not uniformly beneficial. The strongest removal warning was
`LOCAL_MOMENTUM_TREND` for LONG danger (better log loss in 3/4 folds and better
average precision in 4/4). `ROLLING_CONTEXT` also improved both LONG danger
metrics in 3/4 folds. SHORT results did not support the same removals, so a
shared automatic deletion is not justified.

## Interpretation

V13 failed because ranking clean entries remained weak. V14 shows that simply
adding historical taker imbalance does not solve that problem. It also shows
that the current research vector contains duplicated and overlapping
representations, and that some families behave differently for LONG and SHORT.

The next defensible step is a preregistered V15 contract-simplification study:

1. remove exact duplicate positions only in a candidate contract;
2. collapse mathematically equivalent time-window features;
3. test LONG and SHORT family contracts separately;
4. retain V13/V14 as immutable controls;
5. require walk-forward danger, clean-entry, MAE, and economic improvement;
6. do not change Shadow or Live unless a later authorization and validation
   explicitly permit it.

V14 does not claim that more features are needed. Its evidence points first to
reducing redundant information and improving direction-specific contracts.
