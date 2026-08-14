# Aegis W4 Execution Timing and Adverse Selection Result

## Status

`AEGIS_EXECUTION_TIMING_W4_BLOCKED_DATA_QUALITY`

```text
W4_DATA_QUALITY_SUFFICIENT = FALSE
W4_EXECUTION_COST_PREDICTABLE = FALSE
W4_EXECUTION_EDGE_FOUND = FALSE
W4_MODELING_JUSTIFIED = FALSE
W4_READY_FOR_SHADOW = FALSE
W4_READY_FOR_LIVE = FALSE
```

This is not evidence that execution timing has no value. It is evidence that
the available historical data cannot answer W4 without inventing spreads,
latency or fills. W4 stopped before modeling, VALIDATION and HOLDOUT.

## Baseline Execution

Aegis currently submits a Binance USD-M `MARKET` order immediately after the
`ORDER_SUBMITTED` telemetry event. The adapter requests a `RESULT` response.
The durable lifecycle has deterministic intent and client-order IDs, but it is
not yet the historical microstructure dataset W4 needs.

The frozen W4 benchmark is synchronized midprice at the intent timestamp. An
implementation shortfall greater than zero means worse execution for both BUY
and SELL.

## Data Audit

- 132 monthly Futures aggTrade archives, 12 months for each of 11 symbols.
- Coverage: August 2025 through July 2026; about 28.47 GB compressed.
- 1,320,000 sampled rows: zero invalid timestamps, ordering errors or duplicate
  aggregate-trade IDs in the audited prefixes.
- A C2 database contains 2,574,816 aggTrades for one symbol over 31 days.
- Historical synchronized `bookTicker`: absent.
- Sequenced L2 depth: absent.
- C2 depth rows: zero.
- Legacy depth: 11 isolated snapshots, not a usable time series.
- Local receive timestamps: absent from archives.
- Decision-time BBO joined to real fills: absent.
- Authoritative historical account-tier maker/taker schedule: not frozen.

## Why W4A Was Not Run

aggTrade records traded price, quantity, aggressor side and exchange time. It
does not reveal the bid/ask that existed when Aegis decided, so it cannot
separate spread, slippage or implementation shortfall. It also cannot identify
microprice, queue position, partial fills or realistic passive fill
probability.

Modeling the post-intent aggTrade path alone would predict short-term price
movement. That would violate W4's frozen-direction contract by creating a
hidden directional filter, not an execution model.

## Preregistered Future Experiment

The W4 contract is frozen for a future untouched collection:

- 50% TRAIN, 25% VALIDATION, 25% SEALED HOLDOUT;
- at least 60 days and 1,000/500/500 independent intents;
- execute now, wait 250 ms and wait 500 ms in the initial action space;
- 100/250/500/1,000 ms latency scenarios;
- primary metric includes implementation shortfall, fees, delay and missed
  opportunity per intent;
- minimum mean saving of 2 bps, median improvement above zero, no P95/P99 tail
  degradation, 7 symbols and 3 temporal folds;
- passive limits remain disabled unless sequenced L2 and queue evidence exist.

No fee value was guessed. It must be frozen from the applicable account tier
before the future dataset is opened.

## Required Questions

1. Real current execution cost: `NOT_IDENTIFIABLE` from existing history.
2. Fee component: `NOT_FROZEN`.
3. Spread component: `NOT_IDENTIFIABLE_WITHOUT_BBO`.
4. Slippage component: `NOT_IDENTIFIABLE_WITHOUT_BBO_AND_FILL_JOIN`.
5. Adverse selection after a real fill: `NOT_IDENTIFIABLE_WITHOUT_FILL_JOIN`.
6. Symbol differences: not tested.
7. BUY/SELL differences: not tested.
8. Microstructure prediction: not tested because required state is absent.
9. Microprice value: not testable.
10. Order-book imbalance value: not testable.
11. aggTrade incremental value: available but insufficient alone.
12. Wait 100-500 ms: not economically testable without intent BBO/latency.
13. Opportunity decay: definable, not measurable for frozen Aegis intents.
14. Passive savings: not testable without queue evidence.
15. Realistic fill probability: not identifiable.
16. Partial fills: not identifiable.
17. Size impact: not identifiable without depth.
18. Latency survival: not testable without receive timestamps.
19. Fee/slippage survival: not tested.
20. Stability: not tested.
21. Adaptive policy vs MARKET_NOW: W4B not authorized because W4A did not run.
22. Minimum effect: preregistered at 2 bps; no result exists.
23. P95/P99: not measurable.
24. Shadow: insufficient evidence.

## Safety and Integrity

- FINAL_HOLDOUT_W4: `SEALED` and unpopulated.
- Synthetic BBO, depth, queues and fills: none.
- Authenticated requests: 0.
- Network requests: 0.
- Exchange mutations: 0.
- TypeScript changes: none.
- Production, WebSocket and PM2 changes: none.
- W1, W2 and W3 artifacts: unchanged.

Artifact identities at evaluation time:

- Preregistration config SHA-256: `f98b0a9fea9ee92deca420823f129d33680e96b40f06e3ea72a857ca60f08328`.
- Private data-audit JSON SHA-256: `e45dd1593817bdbff79656298a428c0f7e5e31b75f7bcc970def38e313c2ba18`.
- Research module SHA-256: `8e90054f370d3dae4b230eef70480d16673cde16534022cd3306a347f9f755c5`.
- Audit script SHA-256: `f90ae4bb2e78d1ece92bdd77380e524c9899e37ad64f73a31f4f29f12583a18b`.

## Next Evidence Required

A separate, explicitly authorized public-data collector would need to record,
for every future Aegis intent, the local monotonic/UTC decision time,
exchange-time `bookTicker`, bid/ask quantities, aggTrades and subsequent BBO.
Sequenced L2 is additionally required before passive-limit evaluation. This
collector can remain observational and unauthenticated, but it was not started
by W4 because the task prohibited production WebSocket changes without a
separate approval.
