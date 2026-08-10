# Aegis Regime-Aware Directional V6 Validation

- Experiment: `aegis-regime-aware-directional-v6-shadow-01`
- Evidence: `2025-08-09T06:55:00+00:00` to `2026-08-09T06:55:00+00:00`
- Validation file SHA-256: `26e3286d7bb39728352a29f82899dbf3638041242cb0b98b3912db1daee48c8e`
- Historical verdict: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Shadow gate: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Runtime effect: `NONE`
- Exchange calls: `0`
- Exchange mutations: `0`

## Directional Results

| Side | Rows | Evaluated folds | Passing folds | Worst fold non-negative | Validation |
|---|---:|---:|---:|---|---|
| LONG | 96371 | 4 | 0 | False | False |
| SHORT | 96371 | 4 | 0 | False | False |

## Fold Outcomes

| Side | Fold | Selected | Protected net | Mean MAE | Protectable | P95 opportunity gap | Router skilled |
|---|---:|---:|---:|---:|---:|---:|---|
| LONG | 1 | 33 | -0.2395% | +0.7725% | +36.3636% | 78.5h | True |
| LONG | 2 | 29 | -0.0780% | +0.5072% | +48.2759% | 87.8h | False |
| LONG | 3 | 38 | -0.2283% | +0.8129% | +34.2105% | 64.39999999999999h | True |
| LONG | 4 | 2 | +0.2000% | +0.0444% | +100.0000% | 332.0h | False |
| SHORT | 1 | 23 | -0.4685% | +0.9765% | +52.1739% | 183.4999999999999h | True |
| SHORT | 2 | 8 | -0.0519% | +0.7825% | +25.0000% | 274.0h | False |
| SHORT | 3 | 13 | -0.0145% | +0.4297% | +38.4615% | 224.29999999999978h | True |
| SHORT | 4 | 11 | -0.7492% | +1.0287% | +27.2727% | 157.59999999999997h | False |

## External Controls

- `LONG_V4`: `RESEARCH_ONLY_NOT_PROMOTABLE` at `data/long_entry_v4_shadow/validation.json` (SHA-256 `52268b6a403d4f7219b6a5dc92d109bcd3d0cf54f134f613ade3286f5edccae5`).
- `LONG_V5`: `RESEARCH_ONLY_NOT_PROMOTABLE` at `data/long_entry_v5_shadow/validation.json` (SHA-256 `7421d739934e02d63e36b2ec2e34552e18d00581282489b8af5531cc2829bfd5`).
- `LONG_V51_ABLATION`: `NO_ROBUST_HEAD_COMBINATION_FOUND` at `data/long_entry_v51_ablation_shadow/validation.json` (SHA-256 `99dc059d47aa1b051f63c6c1869134abfb365c004a93947533bc7536b08e6cb7`).

The external controls use different historical populations and are references,
not direct head-to-head estimates. The current-brain control is embedded in each
fold of the JSON summary.

## Interpretation

- `LONG` best diagnostic ablation: `MAE_AND_SPEED` with 1118 selections, `0/4` positive folds, and -0.1052% weighted protected net.
- `SHORT` best diagnostic ablation: `MAE_AND_SPEED` with 1036 selections, `0/4` positive folds, and -0.0914% weighted protected net.
- The regime router showed skill in `2/4` folds, below the frozen `3/4` requirement.
- Low predicted MAE reduced adverse excursion but did not establish positive direction or net edge.
- V6 therefore remains research-only; thresholds were not relaxed after observing test outcomes.

## Gate Blockers

- `HISTORICAL_VALIDATION_FAILED`
- `VALIDATION_VERDICT_NOT_ELIGIBLE`
- `LONG_VALIDATION_FAILED`
- `LONG_POSITIVE_FOLDS_INSUFFICIENT`
- `LONG_REGIME_ROUTER_SKILL_INSUFFICIENT`
- `LONG_WORST_FOLD_NEGATIVE`
- `LONG_LEAVE_ONE_SYMBOL_OUT_FAILED`
- `SHORT_VALIDATION_FAILED`
- `SHORT_POSITIVE_FOLDS_INSUFFICIENT`
- `SHORT_REGIME_ROUTER_SKILL_INSUFFICIENT`
- `SHORT_WORST_FOLD_NEGATIVE`
- `SHORT_LEAVE_ONE_SYMBOL_OUT_FAILED`

This report does not activate Shadow, export a model, alter Live selection,
or authorize exchange activity.
