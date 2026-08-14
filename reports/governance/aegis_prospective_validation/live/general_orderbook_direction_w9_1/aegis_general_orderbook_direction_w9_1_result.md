# Aegis W9.1 General Order-Book Direction - Result

## Status

`AEGIS_W9_1_NO_ROBUST_ORDERBOOK_DIRECTIONAL_EDGE`

## Population

- TRAIN episodes: 12,960.
- VALIDATION episodes: 8,640.
- FINAL_HOLDOUT_W9_1: `SEALED_NOT_OPENED`.
- Unit: non-overlapping two-minute order-book episode.
- Symbols: ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT.
- Reconstructed symbol-days: 30.
- Features: 126 causal order-book/flow features; non-finite values: 0.
- Maximum L2/quote mid p99 difference: 0.000 bps.

## Frozen Candidate

- Family: `MULTINOMIAL_LOGISTIC_L2`.
- Ablation: `STATIC_DYNAMICS`.
- Confidence threshold: 0.60.
- Feature count: 54.

## Validation

- Taken: 1,509 (17.47%).
- LONG: 705; SHORT: 804; SKIP: 7,131.
- Directional accuracy: 8.61%.
- Balanced accuracy: 0.3077.
- Net expectancy after 14 bps: -14.196 bps/episode.
- Profit factor: 0.0762.
- Bootstrap 95% CI: [-14.873, -13.637] bps.
- 20 bps stress: -20.196 bps.
- 250 ms latency: -14.226 bps.

The directional hit rate counts a taken episode whose realized class is
`NEITHER` as a miss: economically, the 25 bps move did not materialize within
60 seconds.

## Feature Ablations

| Features | Model | Taken | Balanced accuracy | Gross bps | Net bps | FDR q |
|---|---|---:|---:|---:|---:|---:|
| PLUS_RESPONSE | MULTINOMIAL_LOGISTIC_L2 | 1,418 | 0.336 | 0.266 | -13.734 | 1.000 |
| FULL | DUAL_BINARY_LOGISTIC_L2 | 1,652 | 0.358 | 0.210 | -13.790 | 1.000 |
| FULL | SHALLOW_TREE_DEPTH3 | 889 | 0.390 | 0.055 | -13.945 | 1.000 |
| FULL | MULTINOMIAL_LOGISTIC_L2 | 1,495 | 0.334 | 0.055 | -13.945 | 1.000 |
| PLUS_ABSORPTION | MULTINOMIAL_LOGISTIC_L2 | 1,322 | 0.319 | -0.129 | -14.129 | 1.000 |
| STATIC_DYNAMICS | MULTINOMIAL_LOGISTIC_L2 | 1,509 | 0.308 | -0.196 | -14.196 | 1.000 |
| PLUS_FLOW | MULTINOMIAL_LOGISTIC_L2 | 1,207 | 0.313 | -0.227 | -14.227 | 1.000 |

No ablation had positive net expectancy or survived FDR. Static state, book
dynamics, trade flow, pressure/response and absorption therefore failed to show
incremental economic direction information under the frozen target.

## Per Symbol

| Symbol | Taken | Coverage | Gross bps | Net bps | Profit factor |
|---|---:|---:|---:|---:|---:|
| ADAUSDT | 462 | 32.08% | 0.233 | -13.767 | 0.070 |
| BNBUSDT | 55 | 3.82% | 1.305 | -12.695 | 0.116 |
| BTCUSDT | 234 | 16.25% | 0.959 | -13.041 | 0.067 |
| ETHUSDT | 453 | 31.46% | -0.379 | -14.379 | 0.072 |
| SOLUSDT | 121 | 8.40% | -2.281 | -16.281 | 0.088 |
| XRPUSDT | 184 | 12.78% | -1.365 | -15.365 | 0.089 |

## Temporal Stability

| Validation month | Taken | Gross bps | Net bps |
|---|---:|---:|---:|
| 2025-12 | 716 | -0.141 | -14.141 |
| 2026-03 | 793 | -0.244 | -14.244 |

## Latency Stress

| Latency ms | Gross bps | Net bps |
|---:|---:|---:|
| 0 | -0.196 | -14.196 |
| 100 | -0.192 | -14.192 |
| 250 | -0.226 | -14.226 |
| 500 | -0.246 | -14.246 |
| 1000 | -0.305 | -14.305 |

## Frozen Horizon Diagnostics

These targets were preregistered. They reuse the primary candidate's model
family, feature ablation and confidence threshold and do not affect promotion.

| Barrier/horizon | Taken | Balanced accuracy | Gross bps | Net bps |
|---|---:|---:|---:|---:|
| b25_h60 | 1,509 | 0.308 | -0.196 | -14.196 |
| b10_h30 | 921 | 0.377 | 0.334 | -13.666 |
| b15_h60 | 1,062 | 0.346 | 0.094 | -13.906 |

## Cross-Symbol Transfer

Each row trains without the named symbol and evaluates only that held-out
symbol.

| Held-out symbol | Taken | Gross bps | Net bps |
|---|---:|---:|---:|
| ADAUSDT | 1,194 | 0.145 | -13.855 |
| BNBUSDT | 54 | -0.030 | -14.030 |
| BTCUSDT | 227 | 1.098 | -12.902 |
| ETHUSDT | 462 | -0.312 | -14.312 |
| SOLUSDT | 185 | -1.268 | -15.268 |
| XRPUSDT | 230 | -0.908 | -14.908 |

## Answers To The Research Questions

1. **L2 directional information:** not robust under the preregistered economic target.
2. **Movie versus snapshot:** adding dynamics did not create positive net expectancy; no defensible superiority was found.
3. **Microprice:** included in STATIC and DYNAMICS, but no validated economic contribution was found.
4. **OBI:** L1/L5/L10/L20 were tested; STATIC did not produce an eligible robust signal.
5. **Depletion/replenishment:** no validated incremental signal.
6. **Trade flow:** `PLUS_FLOW` was negative after costs and failed FDR.
7. **Pressure x response:** it had the least-negative ablation result, but only 0.266 gross bps and remained economically negative.
8. **Absorption proxies:** no validated incremental signal.
9. **Best diagnostic horizon:** 10 bps/30 s had the highest balanced accuracy and gross expectancy, but remained -13.666 net bps; it is not tradable under the cost model.
10. **Cross-symbol transfer:** every held-out symbol remained net negative.
11. **Temporal transfer:** both validation months were net negative.
12. **Latency:** 100-500 ms did not rescue the signal; 250 ms remained negative.
13. **After costs:** no model or ablation was positive at 14 bps; 20 bps stress was worse.
14. **Frozen compass:** not justified.
15. **Combine with W7:** not justified; W9.2 was not executed.

## Gate

- `minimum_validation_taken_episodes`: `TRUE`
- `minimum_net_expectancy`: `FALSE`
- `ci_lower_positive`: `FALSE`
- `profit_factor`: `FALSE`
- `positive_symbols`: `FALSE`
- `positive_months`: `FALSE`
- `stress_20bps`: `FALSE`
- `latency_250ms`: `FALSE`
- `single_symbol_concentration`: `TRUE`


## Decision

- `W9_1_READY_FOR_W9_2 = FALSE`
- W9.2 was not executed.
- `W9_READY_FOR_SHADOW = FALSE`
- `W9_READY_FOR_LIVE = FALSE`

The reconstructed L2 is technically sound, but the tested movie of the book
did not provide a robust compass for a 25 bps first-barrier move over 60
seconds. This is an economic negative result, not a data-quality block.

No production, TypeScript, Aegis Brain, guard, leverage, PM2 or exchange state
was modified.
