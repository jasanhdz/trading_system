# Aegis W11 Entry Safety Gate - Result

## Verdict

`AEGIS_W11_NO_ROBUST_ENTRY_SAFETY_EDGE`

- TRAIN: 437 episodes.
- VALIDATION: 176 episodes.
- FINAL_HOLDOUT_W11: `SEALED_NOT_OPENED`.
- Baseline ENTER_NOW: -21.42 net bps/original signal.
- Combined W11: -4.44 net bps/original signal.
- Improvement: +16.99 bps/original signal.
- 20 bps stress: -5.70 bps/original signal.
- Bootstrap 95% CI improvement: [8.63, 25.49].
- Temporal-block 95% CI: [3.39, 31.40].

## Path order

| Historical class | N | Favorable first | Adverse first | Median MFE | Median MAE |
|---|---:|---:|---:|---:|---:|
| BAD_ENTRY | 128 | 18.8% | 78.1% | 19.0 | 90.8 |
| GOOD_CLEAN_ENTRY | 315 | 59.7% | 34.9% | 58.0 | 18.4 |
| MIXED_OR_EXIT_DEPENDENT | 170 | 16.5% | 74.7% | 24.1 | 51.2 |

## Fixed confirmation delay

| Delay | Net/signal | Improvement | Median MFE | Median MAE | Favorable first |
|---|---:|---:|---:|---:|---:|
| 0m | -21.42 | +0.00 | 31.2 | 32.8 | 37.5% |
| 1m | -20.17 | +1.25 | 30.3 | 32.0 | 38.1% |
| 2m | -17.82 | +3.60 | 30.5 | 34.7 | 39.8% |
| 3m | -19.43 | +2.00 | 33.9 | 37.3 | 35.2% |

## Diagnostic attribution

| Historical class | W11 state | Episodes |
|---|---|---:|
| BAD_ENTRY | CLEAN_NOW | 4 |
| BAD_ENTRY | PREMATURE | 3 |
| BAD_ENTRY | SKIP_EXHAUSTED | 9 |
| BAD_ENTRY | SKIP_NO_SPACE | 2 |
| BAD_ENTRY | SKIP_OPPOSED | 18 |
| GOOD_CLEAN_ENTRY | CLEAN_NOW | 6 |
| GOOD_CLEAN_ENTRY | PREMATURE | 10 |
| GOOD_CLEAN_ENTRY | SKIP_EXHAUSTED | 12 |
| GOOD_CLEAN_ENTRY | SKIP_LOW_EXPECTED_VALUE | 2 |
| GOOD_CLEAN_ENTRY | SKIP_NO_SPACE | 5 |
| GOOD_CLEAN_ENTRY | SKIP_OPPOSED | 42 |
| GOOD_CLEAN_ENTRY | SKIP_VOLATILITY_SHOCK | 5 |
| MIXED_OR_EXIT_DEPENDENT | CLEAN_NOW | 9 |
| MIXED_OR_EXIT_DEPENDENT | PREMATURE | 5 |
| MIXED_OR_EXIT_DEPENDENT | SKIP_EXHAUSTED | 9 |
| MIXED_OR_EXIT_DEPENDENT | SKIP_LOW_EXPECTED_VALUE | 2 |
| MIXED_OR_EXIT_DEPENDENT | SKIP_NO_SPACE | 4 |
| MIXED_OR_EXIT_DEPENDENT | SKIP_OPPOSED | 27 |
| MIXED_OR_EXIT_DEPENDENT | SKIP_VOLATILITY_SHOCK | 2 |

These states are model attributions, not proven causal categories. They are
reported to explain behavior and cannot authorize an entry veto.

## Frozen ablations on VALIDATION

| Policy | Executed | Net/signal | Improvement | GOOD retained | BAD avoided |
|---|---:|---:|---:|---:|---:|
| exhaustion | 97 | -8.48 | +12.94 | 61.0% | 58.3% |
| opposition | 87 | -11.60 | +9.82 | 47.6% | 50.0% |
| space | 111 | -8.86 | +12.56 | 68.3% | 52.8% |
| volatility | 110 | -16.57 | +4.85 | 58.5% | 27.8% |
| confirmation | 120 | -15.71 | +5.71 | 67.1% | 27.8% |
| combined | 37 | -4.44 | +16.99 | 19.5% | 80.6% |
| baseline | 176 | -21.42 | +0.00 | 100.0% | 0.0% |

## Combined behavior

- Actions: `{'ENTER_NOW': 19, 'SKIP_EXHAUSTED': 30, 'SKIP_LOW_EXPECTED_VALUE': 4, 'SKIP_NO_SPACE': 11, 'SKIP_OPPOSED': 87, 'SKIP_VOLATILITY_SHOCK': 7, 'WAIT_1M': 9, 'WAIT_2M': 4, 'WAIT_3M': 5}`.
- Skip reasons: `{'SKIP_EXHAUSTED': 30, 'SKIP_LOW_EXPECTED_VALUE': 4, 'SKIP_NO_SPACE': 11, 'SKIP_OPPOSED': 87, 'SKIP_VOLATILITY_SHOCK': 7}`.
- Mean confirmation delay: 0.86 minutes.
- Symbols executed: 9.
- LONG/SHORT executed: 0/37.

VALIDATION contains only historical SHORT signals, so W11 provides no evidence
that the policy transfers to LONG entries.

## Gates

- `minimum_episodes`: `TRUE`
- `minimum_executed`: `TRUE`
- `coverage`: `TRUE`
- `positive_net_expectancy`: `FALSE`
- `material_improvement`: `TRUE`
- `bad_avoidance`: `TRUE`
- `good_retention`: `FALSE`
- `symbol_breadth`: `TRUE`
- `concentration`: `TRUE`
- `bootstrap_ci`: `TRUE`
- `stress_cost`: `FALSE`

## Flags

- `W11_PATH_ORDER_RECONSTRUCTION_VALID = TRUE`
- `W11_EXHAUSTION_FILTER_VALUE_FOUND = FALSE`
- `W11_OPPOSITION_FILTER_VALUE_FOUND = FALSE`
- `W11_SPACE_FILTER_VALUE_FOUND = FALSE`
- `W11_VOLATILITY_FILTER_VALUE_FOUND = FALSE`
- `W11_CONFIRMATION_TIMING_VALUE_FOUND = FALSE`
- `W11_ENTRY_SAFETY_EDGE_FOUND = FALSE`
- `W11_MODELING_JUSTIFIED = FALSE`
- `W11_READY_FOR_PROSPECTIVE_OBSERVATION = FALSE`
- `W11_READY_FOR_SHADOW = FALSE`
- `W11_READY_FOR_LIVE = FALSE`

No production, TypeScript, Aegis Brain, guards, leverage, PM2, Shadow, Live,
authenticated exchange API, orders, or financial state were modified.
