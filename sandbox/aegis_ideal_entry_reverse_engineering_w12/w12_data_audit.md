# W12 Data Audit

Audit date: 2026-08-26. This was a local read-only audit. No download, authenticated
API, canonical compressed dataset, production log, or governed holdout was opened.

## Selected Dataset

`data/directional_alpha_v1/candles_1m/`

The source contains Binance USD-M futures 1-minute OHLCV for ADA, AVAX, BNB, BTC,
DOGE, ETH, LINK, LTC, SOL and XRP against USDT. Columns are `open_time_ms`, OHLC,
base/quote volume, `trade_count`, `close_time_ms`, and taker-buy base/quote volume.
Timestamps are UTC epoch milliseconds. Candle information becomes available at the
recorded close boundary.

Seven symbols contain 655,200 contiguous rows from 2022-01-01 through 2023-03-31.
LTC, SOL and XRP have two source gaps before 2022-04-03 and were causally cropped by
the existing dataset builder. After that crop all ten symbols are contiguous through
2023-03-31 with zero gaps and duplicates. W12 therefore uses only the common half-open
interval `[2022-04-03, 2023-04-01)`.

Every Parquet SHA-256 is recorded in the source manifest. W12 verifies hashes, row
counts, monotonic minute spacing and timestamp availability before cache reuse.

## Why This Source

- It is continuous enough for 1-minute path teachers and temporal profiles.
- It predates and does not reuse W11's May-December 2023 prospective evidence.
- It supplies ten synchronized markets for BTC/ETH context, breadth, dispersion,
  relative strength and beta.
- It includes causal taker flow without requiring unavailable exchange access.

Prior research may have used this raw history for different hypotheses. W12 does not
reinterpret those results or inherit their model artifacts; it creates new fixed
chronological partitions and leaves every existing sealed holdout unopened.

## Causal Reconstruction

- Decisions occur every 15 minutes after the current 1-minute candle has closed.
- Features use only candles with `close_time <= decision_time`.
- Entry is the next one-minute open after the decision boundary.
- Future high/low/close paths may only populate teacher, quality and economic target
  columns; all such columns are prohibited from model matrices by an explicit schema.
- Rolling statistics are backward-looking. Cross-market rows require the same closed
  minute and never forward-fill a missing market.
- Imputation, scaling, feature analysis and model fitting are fitted on discovery only.
- Validation selects one frozen formulation. Prospective is evaluated once.
- A 60-minute purge protects every chronological boundary.

## Available Information

The source supports returns and temporal profiles, true range/ATR, realized
volatility, compression/expansion, trend distances/slopes, range position, causal
recent highs/lows and breakout state, relative volume, volume acceleration,
taker imbalance/persistence, BTC and ETH context, alt basket return, breadth,
dispersion, relative rank, and rolling BTC beta/correlation.

Continuous L2, BBO spread, queue position, funding and open interest are not present.
Local W9/W10 L2 evidence exists only for sparse governed symbol-days and is excluded
to avoid mixing populations and protected partitions.

## Economic Contract

The repository's current standard is retained:

- 10 bps round-trip fees;
- 4 bps assumed round-trip baseline slippage, total 14 bps;
- 20 bps stress total;
- 30 bps severe stress total.

Funding is unavailable and documented as a limitation. Candle paths cannot prove a
queue-level executable fill, so conclusions require margin beyond assumed costs.

## Granularity and Partitions

- Teacher paths: 1 minute.
- Model snapshots: 15 minutes.
- Horizons: 15, 30 and 60 minutes.
- Discovery: `[2022-04-03, 2022-09-01)`.
- Validation: `[2022-09-01, 2022-12-01)`.
- Prospective: `[2022-12-01, 2023-04-01)`.

This yields multiple volatility regimes, sides, months and symbols while keeping the
prospective interval untouched during rule definition.

## Limitations

- Slippage is assumed rather than observed from synchronized BBO.
- Funding is omitted.
- Same-minute TP/SL ambiguity is resolved adverse-first.
- Teacher labels describe available path quality, not guaranteed realized execution.
- Shared market shocks make IID row confidence intervals invalid; inference uses
  synchronized UTC-day blocks.
