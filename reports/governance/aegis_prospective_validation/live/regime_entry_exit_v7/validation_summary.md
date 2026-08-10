# Aegis Regime Entry/Exit V7 Validation

- Experiment: `aegis-regime-entry-exit-v7-research-01`
- Evidence: `2025-08-09T06:55:00+00:00` to `2026-08-09T06:55:00+00:00`
- Verdict: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Gate: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Runtime effect: `NONE`
- Exchange calls: `0`
- Exchange mutations: `0`

## Trajectory attribution

- `AMBIGUOUS_PATH`: 935
- `CLEAN_REALIZED_WIN`: 48194
- `GOOD_ENTRY_POOR_CAPTURE`: 22807
- `LATE_OR_ADVERSE_ENTRY`: 89710
- `NO_DIRECTIONAL_EDGE`: 31096

## Directional folds

| Side | Fold | Selected | Net | MAE | Capture | P95 gap | Passed |
|---|---:|---:|---:|---:|---:|---:|---|
| LONG | 1 | 133 | -0.1561% | 0.6202% | 25.18% | 22.9h | False |
| LONG | 2 | 329 | -0.1237% | 0.5085% | 20.06% | 6.0h | False |
| LONG | 3 | 155 | -0.1474% | 0.5361% | 24.09% | 16.7h | False |
| LONG | 4 | 80 | -0.0985% | 0.4362% | 24.50% | 22.2h | False |
| SHORT | 1 | 102 | -0.1449% | 0.6300% | 22.25% | 26.0h | False |
| SHORT | 2 | 127 | -0.0319% | 0.5291% | 24.28% | 18.0h | False |
| SHORT | 3 | 79 | -0.1187% | 0.4584% | 22.18% | 30.0h | False |
| SHORT | 4 | 73 | -0.1453% | 0.5692% | 27.77% | 38.7h | False |

## Interpretation

- `LONG` improved net loss relative to its frozen control in `4/4` folds, but had `0` positive folds.
- `SHORT` improved net loss relative to its frozen control in `1/4` folds, but had `0` positive folds.
- The regime router was skilled in `2/4` folds for each side, below
  the frozen `3/4` requirement.
- Lower MAE and faster paths reduced some losses but did not establish
  positive expectancy after costs.
- Leave-one-symbol-out was not run because the primary historical gate
  failed; running it could not make this version promotion-eligible.
- Protection profile counts are hindsight diagnostics only and are not
  evidence that one fixed profile should be deployed.

## Hindsight protection profile counts

- `CURRENT_TS`: 103907
- `LOCK_AT_10_ROE`: 25601
- `LOCK_AT_20_ROE`: 34379
- `LOCK_AT_5_ROE`: 28855

## Gate blockers

- `HISTORICAL_VALIDATION_FAILED`
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

V7 remains research-only unless every frozen gate passes. This report does
not activate Shadow, alter Live selection, export a model, or authorize
exchange activity.
