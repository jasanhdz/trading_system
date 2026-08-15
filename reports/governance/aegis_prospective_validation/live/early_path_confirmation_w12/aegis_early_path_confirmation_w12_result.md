# Aegis W12 Early Path Confirmation - Result

## Verdict

`AEGIS_W12_NO_ROBUST_EARLY_PATH_EDGE`

- TRAIN: 437 signals.
- VALIDATION: 114 signals.
- Resolution: closed `1m`; no causal sub-minute/L2 overlap.
- W12 and W11 holdouts: `SEALED_NOT_OPENED`.
- W12 executed: 26/114 (22.8%).
- Net: -8.96 bps/original signal and -39.29 bps/executed trade.
- Stress 20 bps: -10.33 bps/original signal.
- Median confirmation: 93s.
- Median missed MFE: 2.5 bps.

## Baselines

| Policy | Executed | Net/signal | Net/trade | Median MAE |
|---|---:|---:|---:|---:|
| ENTER_NOW | 114 | -29.71 | -29.71 | 34.1 |
| WAIT_1M_FIXED | 114 | -29.20 | -29.20 | 35.9 |
| WAIT_2M_FIXED | 114 | -27.33 | -27.33 | 38.8 |
| WAIT_3M_FIXED | 114 | -29.42 | -29.42 | 39.2 |
| W11_FROZEN | 23 | -7.86 | -38.94 | 39.2 |

## Ablations

| Features | Executed | Net/signal | Net/trade | GOOD retained | BAD avoided | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| PRICE_PATH_ONLY | 2 | -0.02 | -0.95 | 3.9% | 100.0% | 0.618 |
| PRICE_PATH_PLUS_FLOW | 3 | -0.17 | -6.38 | 2.0% | 96.8% | 0.631 |
| PRICE_PATH_FLOW_CONTEXT | 22 | -10.28 | -53.28 | 17.6% | 67.7% | 0.469 |
| PRICE_PATH_FLOW_CONTEXT_SPACE | 24 | -7.24 | -34.39 | 19.6% | 77.4% | 0.498 |
| FULL_W12 | 26 | -8.96 | -39.29 | 23.5% | 67.7% | 0.492 |

## Primary bootstrap comparisons

| Reference | Improvement bps/signal | Temporal-block 95% CI |
|---|---:|---:|
| ENTER_NOW | +20.74 | [3.62, 36.32] |
| WAIT_2M_FIXED | +18.37 | [0.53, 34.43] |
| W11_FROZEN | -1.11 | [-3.05, 0.72] |

## Early-path diagnostics

| State | Historical class | N | Directional return | Early MFE | Early MAE | Taker alignment | Positive remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | BAD_ENTRY | 31 | -2.3 | 0.0 | 9.1 | -0.098 | 0.0% |
| 1 | GOOD_CLEAN_ENTRY | 51 | 1.7 | 3.6 | 2.8 | +0.035 | 47.1% |
| 1 | MIXED_OR_EXIT_DEPENDENT | 32 | 0.0 | 3.1 | 5.7 | -0.005 | 18.8% |
| 2 | BAD_ENTRY | 31 | -4.2 | 5.3 | 10.5 | -0.086 | 0.0% |
| 2 | GOOD_CLEAN_ENTRY | 51 | 1.3 | 8.0 | 4.0 | +0.018 | 49.0% |
| 2 | MIXED_OR_EXIT_DEPENDENT | 32 | -2.5 | 3.5 | 6.4 | -0.050 | 28.1% |

Historical class is diagnostic only and was not a model target.

## Symbols

| Symbol | Signals | Executed | Net/signal | Net/trade |
|---|---:|---:|---:|---:|
| ADAUSDT | 22 | 6 | -19.82 | -72.67 |
| AVAXUSDT | 16 | 4 | -3.38 | -13.50 |
| BNBUSDT | 1 | 0 | 0.00 | N/A |
| BTCUSDT | 3 | 1 | -72.80 | -218.39 |
| DOGEUSDT | 13 | 1 | 6.63 | 86.13 |
| ETHUSDT | 10 | 2 | -11.11 | -55.55 |
| LINKUSDT | 4 | 1 | -2.62 | -10.49 |
| LTCUSDT | 14 | 4 | -8.78 | -30.72 |
| SOLUSDT | 9 | 2 | 2.83 | 12.73 |
| SUIUSDT | 10 | 1 | -2.69 | -26.91 |
| XRPUSDT | 12 | 4 | -12.78 | -38.35 |

## Primary behavior

- Actions: `{'CANCEL_EARLY_INVALIDATION': 57, 'CANCEL_NO_CONFIRMATION': 31, 'ENTER_AT_STATE_1': 20, 'ENTER_AT_STATE_2': 6}`.
- GOOD retained: 23.5%; BAD avoided: 67.7%; MIXED retained: 12.5%.
- Median MFE/MAE: 31.8/41.3 bps.
- Median entry-price improvement: 0.7 bps.
- LONG/SHORT executed: 0/26.

## Gates

- `minimum_episodes`: `TRUE`
- `minimum_executed`: `TRUE`
- `execution_rate`: `TRUE`
- `positive_per_trade`: `FALSE`
- `positive_per_signal`: `FALSE`
- `beats_enter_now`: `TRUE`
- `beats_wait_2m`: `TRUE`
- `beats_w11`: `FALSE`
- `good_retention`: `FALSE`
- `bad_avoidance`: `TRUE`
- `mae_reduction`: `FALSE`
- `symbol_breadth`: `TRUE`
- `concentration`: `TRUE`
- `bootstrap_enter_now`: `TRUE`
- `bootstrap_wait_2m`: `TRUE`
- `bootstrap_w11`: `FALSE`
- `stress_cost`: `FALSE`

## Flags

- `W12_EARLY_PATH_INFORMATION_FOUND = TRUE`
- `W12_DYNAMIC_CONFIRMATION_VALUE_FOUND = TRUE`
- `W12_REMAINING_EDGE_FOUND = FALSE`
- `W12_EXECUTED_TRADE_EDGE_FOUND = FALSE`
- `W12_PER_SIGNAL_EDGE_FOUND = FALSE`
- `W12_COST_GATE_PASSED = FALSE`
- `W12_SHORT_ONLY_EVIDENCE = TRUE`
- `W12_MODELING_JUSTIFIED = FALSE`
- `W12_READY_FOR_PROSPECTIVE_OBSERVATION = FALSE`
- `W12_READY_FOR_SHADOW = FALSE`
- `W12_READY_FOR_LIVE = FALSE`

No production, TypeScript, Brain, guards, leverage, PM2, exchange, authenticated
API, orders, Shadow, Live, or financial state were modified.
