# Aegis W4 Execution Timing and Adverse Selection Preregistration

## Scope

W4 treats every Aegis LONG/SHORT intent as external and frozen. It may optimize
only execution cost; it may not predict, reverse or retrospectively judge the
directional signal. The primary unit is one `execution_intent_id`.

The current Aegis baseline is `MARKET_NOW`. The primary benchmark is the
synchronized midprice at the exact intent timestamp. Positive implementation
shortfall and positive total cost mean worse execution.

## Required Evidence

W4A requires a frozen intent timestamp, synchronized bid/ask and sizes,
aggregate trades, exchange timestamps, local receive timestamps and a frozen
fee schedule. Passive execution additionally requires sequenced L2 state,
quantity ahead and traded quantity at price. Candle proxies and optimistic
`low <= limit` fills are prohibited.

## Experimental Contract

- Initial actions: execute now, wait 250 ms, wait 500 ms.
- Limit actions remain disabled unless queue evidence passes its gate.
- Latency scenarios: 100, 250, 500 and 1,000 ms; zero latency is diagnostic.
- Primary metric: total cost per intent, including implementation shortfall,
  fees, delay and missed-opportunity cost.
- Minimum mean saving: 2 bps over `MARKET_NOW`.
- Median improvement must exceed zero; P95/P99 may not deteriorate.
- Improvement must survive fees, 250 ms latency, partial fills and missed
  opportunities, with at least 7 symbols and 3 positive temporal folds.
- 10,000 intent/day-block bootstraps and Benjamini-Hochberg FDR are required.

## Partitions

A future untouched collection will be divided chronologically 50% TRAIN, 25%
VALIDATION and 25% FINAL HOLDOUT, with a ten-second purge. At least 60 calendar
days and 1,000/500/500 intents are required. FINAL_HOLDOUT_W4 remains `SEALED`
until data, features, models, fees, latency assumptions and policy are frozen
and W4A/W4B pass all prior gates.

## Fail-Closed Rule

If synchronized BBO or timestamp integrity is absent, W4 stops before W4A.
aggTrade-only future movement must not be relabeled as execution cost because
that would create a hidden directional model. No production or exchange action
is authorized by this preregistration.
