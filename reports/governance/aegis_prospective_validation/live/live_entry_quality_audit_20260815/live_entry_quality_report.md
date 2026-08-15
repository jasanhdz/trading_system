# Aegis Live Entry Quality Audit — 2026-08-15

## Scope

Read-only audit of paired `AEGIS_TURBO_MICRO_LIVE` open/close records. No production rules, services, credentials, or orders were modified.

## Classification

| Split | Class | Episodes | Median MFE bps | Median MAE bps | PnL USDT (secondary) |
|---|---|---:|---:|---:|---:|
| DISCOVERY | BAD_ENTRY | 113 | 25.94 | 195.98 | -1265.31 |
| DISCOVERY | GOOD_CLEAN_ENTRY | 263 | 86.49 | 15.41 | 639.14 |
| DISCOVERY | MIXED_OR_EXIT_DEPENDENT | 129 | 55.81 | 84.01 | 169.51 |
| VALIDATION | BAD_ENTRY | 37 | 14.93 | 258.64 | -633.34 |
| VALIDATION | GOOD_CLEAN_ENTRY | 101 | 106.47 | 22.19 | 447.57 |
| VALIDATION | MIXED_OR_EXIT_DEPENDENT | 75 | 52.31 | 105.20 | 216.73 |

## Existing causal guards

| Split | Guard | Blocked | Bad captured | Good retained | Bad rate if allowed |
|---|---|---:|---:|---:|---:|
| DISCOVERY | entry_quality | 241 | 42.5% | 49.0% | 24.6% |
| DISCOVERY | event_risk | 277 | 49.6% | 41.4% | 25.0% |
| DISCOVERY | decision_brain | 259 | 44.2% | 45.2% | 25.6% |
| DISCOVERY | clean_entry | 277 | 49.6% | 41.4% | 25.0% |
| DISCOVERY | regime | 280 | 51.3% | 41.1% | 24.4% |
| DISCOVERY | probe_mode | 0 | 0.0% | 100.0% | 22.4% |
| VALIDATION | entry_quality | 146 | 75.7% | 37.6% | 13.4% |
| VALIDATION | event_risk | 213 | 100.0% | 0.0% | N/A |
| VALIDATION | decision_brain | 172 | 73.0% | 19.8% | 24.4% |
| VALIDATION | clean_entry | 213 | 100.0% | 0.0% | N/A |
| VALIDATION | regime | 213 | 100.0% | 0.0% | N/A |
| VALIDATION | probe_mode | 213 | 100.0% | 0.0% | N/A |

## Frozen discovery model evaluated on later Live data

- Threshold selected only in discovery: `0.60`.
- Validation allowed/total: `167/213`.
- Validation bad-rate reduction: `-10.3%`.
- Validation clean-good retention: `78.2%`.
- Bootstrap 95% CI for relative bad-rate reduction: `[-23.7%, 4.2%]`.
- `BAD_ENTRIES_CAUSALLY_AVOIDABLE = FALSE`.
- `PRODUCTION_CHANGE_JUSTIFIED = FALSE`.

Largest standardized associations with BAD entries (discovery fit only):

- `missingindicator_close_location`: +0.6543
- `exhaustion_risk`: -0.2840
- `close_location`: +0.2802
- `regime_confidence`: -0.1604
- `atr_pct`: +0.1349
- `choppiness`: +0.1308
- `atr_percentile`: +0.1185
- `missingindicator_regime_confidence`: -0.1126
- `missingindicator_chop_risk`: -0.1126
- `missingindicator_exhaustion_risk`: -0.1126

The strongest coefficient is a missing-value indicator. Feature availability changed materially between discovery and validation, so this is evidence of logging/policy drift rather than a trustworthy market relationship.

Recent predefined reasons `volatility_too_high` and `overextended_short` identified 6 BAD entries among 16 observations, but their discovery behavior was not consistent and the sample is too small for enforcement.

## Interpretation constraints

- Monetary PnL is secondary because some positions were resized manually after entry.
- MFE/MAE describe the full path but do not preserve excursion ordering.
- A retrospective result cannot directly authorize an enforced Live guard.
- Only a frozen prospective observer can establish whether the relationship persists without policy drift.
