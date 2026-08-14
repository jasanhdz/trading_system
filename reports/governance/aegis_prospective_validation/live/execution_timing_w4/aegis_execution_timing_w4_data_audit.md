# Aegis W4 Execution Timing Data Audit

## Status

`W4_DATA_QUALITY_INSUFFICIENT`

The current evidence cannot identify execution cost. W4A and W4B must not run
until synchronized top-of-book and intent-time evidence exists.

## Current Aegis Execution

- Entry policy: `MARKET_NOW`.
- Venue call: Binance USD-M `MARKET` with `newOrderRespType=RESULT`.
- The durable lifecycle provides deterministic intent/client-order identity.
- The `ORDER_SUBMITTED` event precedes `marketOpen`, but historical telemetry
  does not contain a synchronized bid/ask snapshot at that timestamp.
- The preregistered primary benchmark is midprice at the intent timestamp.

## Historical Evidence

- Futures aggTrade archives: 132 files, 11
  symbols, 2025-08 through 2026-07.
- Sampled archive rows: 1,320,000; invalid timestamps:
  0; out-of-order rows:
  0; duplicate aggregate-trade IDs:
  0.
- aggTrade provides exchange transaction time, price, quantity and aggressor
  side. It does not provide bid, ask, queue state or local receive time.
- Historical `bookTicker`: 0 rows.
- Sequenced L2 history: 0 rows.
- C2 depth history: 0 rows.
- Legacy depth: 11 isolated
  snapshots across 11 symbols;
  this is not a time series.

## Blocking Conditions

- `HISTORICAL_SYNCHRONIZED_BBO_MISSING`
- `LOCAL_RECEIVE_TIMESTAMP_MISSING`
- `AUTHORITATIVE_FEE_SCHEDULE_MISSING`


Without BBO, `MARKET_NOW` implementation shortfall cannot be separated into
spread and slippage. Without receive timestamps, 100-500 ms latency cannot be
simulated. Without sequenced L2 and queue evidence, passive fills, fill
probability and partial fills cannot be reconstructed honestly.

aggTrade-only future price movement could be modeled, but that would answer a
directional short-horizon question and risk turning W4 into a hidden second
brain. It is therefore not used as a substitute.

## Decision

- `W4_DATA_QUALITY_SUFFICIENT = FALSE`
- W4A not executed.
- W4B not executed.
- FINAL_HOLDOUT_W4 remains `SEALED` and unpopulated.
- No synthetic BBO, spread, queue or fills were created.
- Authenticated requests: 0.
- Exchange mutations: 0.
