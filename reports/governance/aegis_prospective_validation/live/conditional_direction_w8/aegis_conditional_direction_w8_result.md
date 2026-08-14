# Aegis W8 Conditional Direction - Result

## Verdict

`AEGIS_W8_NO_ROBUST_DIRECTIONAL_ALPHA`

- W8_DIRECTION_DATA_VALID: `TRUE`
- W8_LONG_SHORT_ASYMMETRY_EXPLAINED: `TRUE`
- W8_DIRECTION_SIGNAL_FOUND: `FALSE`
- W8_SKIP_CLASS_VALUE_FOUND: `TRUE`
- W8_DIRECTIONAL_ALPHA_FOUND: `FALSE`
- W8_ECONOMIC_EDGE_FOUND: `FALSE`
- W8_READY_FOR_SHADOW: `FALSE`
- W8_READY_FOR_LIVE: `FALSE`
- FINAL_HOLDOUT_W8: `SEALED_NOT_OPENED`

## Why W7 Was 100% SHORT

This was architectural, not a market accident. The qualified artifact used the `aegis-labels-short-v4` contract, its preflight recorded `long_disabled=true`, and `src/aegis/layers.py` unconditionally adds `SIDE_NOT_ENABLED` to every LONG candidate. The stored 90,442 rows contain 2458 SHORT actions, 87984 HOLD actions and vote vector `0:1:0` for every row. Per-candidate rejection reasons were not stored in V14, so they are reported as NOT_PRESENT rather than inferred.

The 45,221 source episodes nevertheless contain a complete LONG/SHORT counterfactual pair with identical pre-entry features. W8 therefore discarded the historical action as a direction label and built symmetric future labels.

## Population

- Broad independent development episodes: 34947
- Frozen W7 Opportunity candidates: 14503
- TRAIN / VALIDATION: 9006 / 5494
- TRAIN labels: {"LONG": 4043, "SHORT": 4636, "SKIP": 327}
- VALIDATION labels: {"LONG": 2520, "SHORT": 2751, "SKIP": 223}

## Selected Candidate

- `H30:PRICE_STRUCTURE:A_MULTICLASS_LOGISTIC`
- Validation trades: 698 (12.70%)
- LONG / SHORT / SKIP: 349 / 349 / 4796
- Taken net expectancy: -16.4990 bps
- Portfolio net expectancy per opportunity: -2.0962 bps
- Portfolio 95% day-block bootstrap CI: [-1.7994, -0.8859]
- Profit factor: 0.3035
- Stress 20 / 30 bps: -22.4990 / -32.4990 bps
- Positive symbols / folds: 0/11, 0/4

## Feature Ablations

All preregistered price, Opportunity, funding/basis, taker and relative-strength variants are retained in `aegis_conditional_direction_w8_verdict.json`. OI, positioning and liquidation ablations were not run because those histories are absent.

## Interpretation

The short-only bias was fully explained and removed from the target construction, but the selected conditional direction policy did not produce robust positive net expectancy after costs and uncertainty. Opportunity remained a magnitude condition, not a reliable sign predictor.

No production, TypeScript, guard, leverage, PM2, Shadow or exchange state was changed.
