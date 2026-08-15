# Aegis W10 Reactive Sequential Momentum - Result

## Verdict

`AEGIS_W10_NO_ROBUST_SEQUENTIAL_MOMENTUM_EDGE`

- TRAIN episodes: 5,166.
- VALIDATION episodes: 3,444.
- States: 215,250; causal features: 72.
- FINAL_HOLDOUT_W10: `SEALED_NOT_OPENED`.

## Frozen Navigator

- Model: `HIST_GRADIENT_BOOSTING` / `FULL`.
- Policy: `E0.65x2-H0.45-X2-C20-M1`.
- Model balanced accuracy: 0.4534.

## Validation

- Trades: 104; LONG/SHORT: 100/4.
- Gross/net: 1.757/-12.243 bps per trade.
- Net per episode: -0.370 bps.
- Profit factor: 0.1918; win rate: 13.46%.
- Median MFE/MAE: 6.048/5.949 bps.
- Median hold: 35.0s; trades/hour: 0.361.
- Reentries/episode: 0.0000; direct flips: 0.
- Cost/positive gross: 2.093.
- Bootstrap episode CI: [-0.705, -0.114] bps.

## Baselines

| Policy | Trades | Gross/trade | Net/trade | Net/episode |
|---|---:|---:|---:|---:|
| NO_TRADE | 0 | 0.000 | 0.000 | 0.000 |
| MOMENTUM_10BPS_FIXED_30S | 1,079 | -0.440 | -14.440 | -4.524 |
| MOMENTUM_10BPS_FIXED_60S | 1,079 | 0.349 | -13.651 | -4.277 |
| MOMENTUM_10BPS_TRAILING_10BPS | 1,079 | -0.154 | -14.154 | -4.435 |
| MOMENTUM_10BPS_GIVEBACK_40PCT | 1,079 | -0.322 | -14.322 | -4.487 |

## Latency

| ms | Trades | Net/trade | Net/episode |
|---:|---:|---:|---:|
| 0 | 104 | -12.243 | -0.370 |
| 100 | 104 | -12.390 | -0.374 |
| 250 | 104 | -12.088 | -0.365 |
| 500 | 104 | -12.075 | -0.365 |
| 1000 | 104 | -12.613 | -0.381 |

## Symbols

| Symbol | Trades | Net/trade | Net/episode |
|---|---:|---:|---:|
| ADAUSDT | 3 | -19.824 | -0.104 |
| BNBUSDT | 1 | -49.791 | -0.087 |
| BTCUSDT | 66 | -12.066 | -1.387 |
| ETHUSDT | 27 | -7.556 | -0.355 |
| SOLUSDT | 3 | -26.904 | -0.141 |
| XRPUSDT | 4 | -20.743 | -0.145 |

## Validation Dates

| Date | Trades | Net/trade | Net/episode |
|---|---:|---:|---:|
| 2025-12-01 | 63 | -11.114 | -0.407 |
| 2026-03-01 | 41 | -13.978 | -0.333 |

## Frozen W7 Diagnostic

- TRAIN overlap: 0 episodes.
- VALIDATION overlap: 0 episodes.
- Testable without refit: `FALSE`.
- Executed: `FALSE`; W7 was not refitted.

## Gates

- `minimum_trades`: `FALSE`
- `minimum_net_per_trade`: `FALSE`
- `minimum_net_per_episode`: `FALSE`
- `profit_factor`: `FALSE`
- `bootstrap_ci`: `FALSE`
- `positive_symbols`: `FALSE`
- `positive_dates`: `FALSE`
- `beats_all_baselines`: `FALSE`
- `cost_20bps`: `FALSE`
- `latency_250ms`: `FALSE`
- `single_symbol_concentration`: `FALSE`
- `anti_churn`: `FALSE`

## Flags

- `W10_MOMENTUM_DETECTION_FOUND = TRUE`
- `W10_MOMENTUM_PERSISTENCE_FOUND = TRUE`
- `W10_DECAY_INFORMATION_FOUND = TRUE`
- `W10_SEQUENTIAL_POLICY_EDGE_FOUND = FALSE`
- `W10_ANTI_CHURN_GATE_PASSED = FALSE`
- `W10_COST_GATE_PASSED = FALSE`
- `W10_LATENCY_GATE_PASSED = FALSE`
- `W10_MODELING_JUSTIFIED = FALSE`
- `W10_READY_FOR_SHADOW = FALSE`
- `W10_READY_FOR_LIVE = FALSE`

W10 did not modify production, TypeScript, Aegis Brain, guards, leverage, PM2,
Shadow, Live or exchange state.
