# W11 Data Audit

Audit date: 2026-08-25. Scope: local, read-only inventory. No network or authenticated
source was used. No sealed holdout outcomes were opened.

## Selected Source

`data/independent_entry_quality_discovery_v1/candles_1m/`

- Eleven Binance USD-M futures symbols: ADA, AVAX, BNB, BTC, DOGE, ETH, LINK,
  LTC, SOL, SUI and XRP against USDT.
- Interval: `[2023-05-01, 2024-01-01)`; SUI starts at
  `2023-05-03T16:00:00Z` because the contract did not exist earlier.
- Timeframe: 1 minute, 352,800 rows per full-history symbol.
- Schema: open/close timestamps, OHLC, base and quote volume, trade count, taker-buy
  base and quote volume.
- Integrity manifest reports zero duplicate timestamps and zero gaps for every
  available symbol.
- Manifest records `sealed_holdout_rows_loaded: 0`.
- Parquet source hashes are recorded in the source manifest and will be copied to
  the W11 artifact manifest.

This source was selected because ephemeral 6-72h experts require continuous market
snapshots. The prior entry-safety W11 panel has only 613 completed signal episodes
and four delayed-action rows per episode, so it cannot represent the continuously
available opportunity set.

## Causal Reconstruction

- One-minute rows are labelled by candle open; `close_time_ms` is the availability
  boundary.
- W11 aggregates to half-open 5-minute bars and labels them by close time.
- A feature snapshot may use only bars whose close time is less than or equal to the
  decision timestamp.
- Decisions occur every 15 minutes after a completed bar.
- Entry is the next 5-minute bar open. Outcomes use later closes at 5/15/30/60m.
- Feature rolling windows use only current and prior completed bars.
- Cross-market features require the same coordinated close timestamp; missing
  symbols are never forward-filled.
- Normalizers, imputation and model fitting use only the recent training interval.
- Validation decisions end at least 60 minutes before model creation, ensuring all
  validation outcomes were knowable when the instance was created.

Future returns are targets only and are never included in `regime_state`.

## Reusable Local Inventory

| Dataset | Coverage / granularity | Causal information | W11 decision |
|---|---|---|---|
| `directional_alpha_v1/candles_1m` | 2022-01 to 2023-03, 10 symbols | OHLCV/taker flow | Valid robustness source; not primary |
| `independent_entry_quality_discovery_v1/candles_1m` | 2023-05 to 2023-12, 11 symbols | OHLCV/taker flow | Primary |
| `aegis_strategy_router_retrospective_v1/candles_1m` | 2023-09 to 2024-09, 10 symbols | OHLCV | Overlaps primary; excluded |
| `market_event_fast_track_m1a/raw` | 2024-01 to 2026-07, 11 symbols | spot/futures klines and aggregate trades | Available but excluded to avoid opening governed later periods |
| `aegis_entry_enhancement_v1/candles_1m` | 2026-01 to 2026-08 | OHLCV | Mixed with sealed August evidence; excluded |
| Entry-safety W11 panel | 613 episodes, May-July 2026 | 284 causal multitimeframe features and 60m paths | Audit reference only |
| W10 sequential states | 215,250 5-second states on 30 symbol-days | L2, quotes, trades, flow | Strong microstructure source but sparse dates; excluded from primary |
| W9.1 order-book episodes | 30 symbol-days | reconstructed L2 and trades | Excluded from primary |
| W1/W2/W3 panels | governed minute/intrabar episodes | flow and path labels | Existing holdouts remain sealed |
| canonical V6-V14 panels | 5-minute derived panels | regimes, flow and future labels | Contents not opened; excluded |
| range R2 source | 2024-2026 intended | OHLC/funding | Blocked by documented source gaps |

## Features Available

The selected source supports returns, realized volatility, true range/ATR, EMA
distance and slope, momentum persistence/efficiency, recent range location,
relative volume, trade count, and taker imbalance. Synchronized symbols support BTC
context, ETH/BTC relative movement, breadth, dispersion, alt-basket return,
relative return, rolling BTC beta and correlation.

VWAP can be approximated from quote/base volume but is omitted to avoid treating an
aggregate ratio as executable VWAP. Funding, open interest, BBO spread and order-book
depth are unavailable in the selected continuous source.

## Costs and Outcomes

The current project standard is 5 bps fee per side plus adverse slippage:

- baseline: 14 bps round trip (5 fee + 2 slippage per side);
- stress: 20 bps (5 + 5 per side);
- severe stress: 30 bps (5 + 10 per side).

Funding is not available in the selected source. This omission is conservative only
for favorable funding and optimistic for adverse funding, so it is an explicit
limitation. Gross close-to-close paths, MFE/MAE and barrier ordering are reconstructible,
but the preregistered W11 target uses next-open to horizon-close returns.

## Gaps and Limitations

- SUI begins 64 hours after the other symbols; coordinated features omit it until
  available.
- Candle bars cannot model intrabar execution priority, queue position or spread.
- Slippage is an assumption, not measured BBO execution.
- Funding is absent.
- The 2023 sample is historical evidence about the procedure, not evidence that the
  same alpha exists in August 2026.
- Repeated symbols at one timestamp share a market shock; inference must use temporal
  blocks, not IID rows.

## Protected Evidence

The experiment will not read W11 August 2026, W12 final evidence, W1-W10 sealed
holdouts, post-August live journals, or compressed canonical dataset rows. Existing
historical verdicts remain unchanged and have no promotion authority for this study.
