# Shared Market Data Audit and Reuse Report

Recorded: `2026-08-18` UTC

## Verdict

- `EXISTING_MARKET_DATA_AUDITED = TRUE`
- `SHARED_MARKET_DATA_REUSE_FEASIBLE = TRUE`
- `STRATEGY_ROUTER_AEGIS_INDEPENDENCE_PRESERVED = TRUE`
- `DUPLICATE_BINANCE_COLLECTION_REQUIRED = TRUE`
- `GENERAL_CANDIDATE_COLLECTION_ACTIVE = FALSE`
- `PRODUCTION_EVENT_EXPORT_ADAPTER_REQUIRED = TRUE`
- `FRESH_DATA_SUFFICIENCY = NOT_YET_MET`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `EDGE_VALIDATION_PERFORMED = FALSE`

`DUPLICATE_BINANCE_COLLECTION_REQUIRED` describes the current operational
state, not the desired architecture. The Router still needs its separate
public candle refresh because no continuous, neutral 1m spool exists. After
the proposed W13-P export extension is independently reviewed and activated,
this flag is expected to become `FALSE`.

No production source, PM2 configuration, active process, exchange connection,
order path, or Aegis decision was changed in this work.

## Governing boundary

The implementation enforces:

```text
SHARED MARKET DATA != SHARED DECISION LOGIC
```

Market observations may be shared. Aegis signal side, confidence, scores,
committee output, entry quality, reason codes, `ENTER/WAIT/SKIP`, labels, and
outcomes are prohibited at the Router adapter boundary. Initial discovery
continues to evaluate LONG and SHORT independently from side-neutral snapshots.

## Existing data audit

### W13-P public collector

The active PM2 process `w13p-passive-microstructure-collector` uses only public
Binance USD-M endpoints. Its frozen universe contains 11 symbols:

`ADA, AVAX, BNB, BTC, DOGE, ETH, LINK, LTC, SOL, SUI, XRP` against USDT.

Per symbol, the existing combined websocket receives:

- `@depth@100ms` incremental L2;
- `@bookTicker` BBO quote;
- `@trade` raw trades.

The local book is initialized with the public `/fapi/v1/depth` snapshot. It
tracks update IDs, gaps, duplicates, out-of-order events, crossed books,
invalid intervals, and resynchronization. Events retain exchange event/trade
timestamps, local wall and monotonic receive timestamps, collector write time,
book generation, and book validity.

At audit time the process was online with all 11 books valid, zero collector
or disk drops, and about 46,649 events in bounded in-memory rings. Lifetime
health counters reported 322 sequence gaps, 211 duplicates, zero out-of-order
events, zero crossed books, 660 resyncs, and 452 reconnects. These counters are
not hidden; affected intervals are invalidated and rebuilt.

The configured rings retain 90 seconds with a hard cap of 100,000 events per
symbol. Market and disk queues are bounded at 50,000 and 100,000 records,
respectively. Parquet flushes at 50,000 rows or 30 seconds. Collection stops
itself, not trading, below 100 GB free disk or above 100 GB collector usage.
There is no silent gap filling or configured outcome-based retention policy.

The important limitation is selection: raw events are continuously observed
in memory, but persisted Parquet parts cover only `-30s/+180s` windows around
Aegis signals. Current persisted data contains 786,845 market events over seven
symbols and 116 signal quality records, of which 103 are W13 eligible. The
event payload is neutral, but the persisted population is signal-conditioned.
It is therefore `UNSAFE_OR_DECISION_DERIVED` for general-market candidate
discovery and cannot be used as a substitute for a continuous market spool.

### TypeScript market inputs

The production TypeScript bot already consumes public futures candles,
aggregate trades, and partial depth. Its Binance adapter also obtains mark
price, funding, and basis information. Candle/taker state and most auxiliary
data are held in process caches rather than persisted as a continuous neutral
dataset.

The TypeScript Binance adapter also contains authenticated financial methods.
Importing or invoking it from the sandbox would violate the isolation contract,
even if a caller intended to use only public methods. No Router adapter imports
production TypeScript code.

### Persisted local datasets

| Source | Coverage found | Assessment |
|---|---|---|
| `data/binance_candles.db` | 6,789,034 rows, 11 symbols, 5m only, ending 2026-07-17 | Historical warmup/discovery only; stale and no 1m |
| `data/long_entry_v3_shadow/public_microstructure.db` | Historical kline microstructure, funding, OI and taker ratio; sparse depth; ending before fresh freeze | Historical fixtures/debugging only |
| `data/market_microstructure_events_c2/c2_archive.db` | 2,574,816 aggregate trades ending 2026-07-31 | Historical replay only; no current L2 continuity |
| `data/live_entry_quality_audit_20260815/candles_1m` | Immutable 1m candle warmup | `REUSABLE_DIRECTLY` for causal warmup |
| `data/aegis_strategy_router_fresh/candles_incremental` | Public 1m increment through 2026-08-18 04:43 UTC | `REUSABLE_DIRECTLY`, but not continuously refreshed |
| W13-P event Parquet | High-resolution BOOK/QUOTE/TRADE only near Aegis signals | `UNSAFE_OR_DECISION_DERIVED` for general population |

No existing persistent source provides continuous neutral 1m candles from the
fresh freeze onward. Continuous funding/OI history is also absent. Funding/OI
is not required by the frozen Phase 1/2 generators and remains explicitly
missing rather than synthesized.

### Relevant process state

The read-only PM2 audit found `01-Trading-Bot`, `02-Aegis-API`,
`aegis-prospective-shadow-cohort-1`, and
`w13p-passive-microstructure-collector` online. The Router has no PM2 process.
No process was restarted, reconfigured, or signaled during this audit.

## Compatibility matrix

| Requirement | Existing source | Classification | Notes |
|---|---|---|---|
| 1m OHLCV | Warmup plus Router increment | `REUSABLE_DIRECTLY` | Current batch only; no continuous shared spool |
| 5m/15m/1h/4h/1d | Deterministic aggregation from closed 1m | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Phase 1 availability contract applies |
| BOOK/L2 | W13-P in-memory sequenced book | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Not continuously persisted for general market |
| Persisted W13-P BOOK | Signal-window Parquet | `UNSAFE_OR_DECISION_DERIVED` | Selection depends on Aegis signal occurrence |
| QUOTE/BBO | W13-P public bookTicker | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Same persistence limitation |
| TRADE/taker flow | W13-P public trade; 1m taker-buy volume | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Closed 1m is enough for current Phase 1/2 |
| BTC/ETH context | Shared neutral 1m candles | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Backward-as-of only |
| Structural levels | Phase 1 confirmed pivots | `REUSABLE_DIRECTLY` | Derived causally inside sandbox |
| Feature availability timestamps | Closed-candle timestamp | `REUSABLE_WITH_READ_ONLY_ADAPTER` | Local receive time absent from old candle files |
| L2 sequence/data-quality flags | W13-P health and event metadata | `REUSABLE_DIRECTLY` | Only for intervals actually exported |
| Funding/OI continuous history | In-memory/stale historical stores | `MISSING` | Not required for Phase 1/2 |
| Aegis signals/scores/actions | Journals and signal snapshots | `UNSAFE_OR_DECISION_DERIVED` | Explicitly prohibited |
| Production Binance adapter | Mixed public and authenticated capabilities | `UNSAFE_OR_DECISION_DERIVED` | Must not be imported by sandbox |

## Implemented sandbox architecture

```text
existing immutable neutral Parquet
                |
                v
SharedNeutralMinuteCandleSource (read-only schema firewall)
                |
                v
Phase 1 causal snapshots
                |
                v
Phase 2 independent LONG/SHORT generators
                |
                v
general-market candidate dataset
```

The adapter:

- opens only local Parquet files;
- has no network client, writer, signal reader, or exchange dependency;
- rejects decision, signal, score, outcome, target, and future-derived columns;
- merges immutable partitions deterministically;
- deduplicates exact overlaps and rejects conflicting historical duplicates;
- fails closed on gaps and stale coverage;
- exposes source schema/hash and coverage audit metadata.

Snapshot source hashes now include only candles causally available at the
snapshot boundary. Appending future candles therefore cannot change an
existing historical snapshot ID or candidate record.

The general-market CLI now consumes this shared read-only boundary. It remains
a batch/incremental replay command; no persistent supervisor was created or
activated.

## Selected shared-data architecture

The target architecture is:

```text
                         +--------------------> Aegis
                         |
Binance public market data
                         |
                         +--> neutral append-only market spool
                                      |
                                      +--> Strategy Router read-only adapter
                                               |
                                               +--> Phase 1 --> Phase 2
```

There is no Aegis signal or decision edge into the Router.

## Minimal Event Export Adapter proposal

The current sources are insufficient for unattended independent collection.
The recommended correction is not another Binance client. It is a minimal
extension of the already-running W13-P public combined websocket:

1. add `<symbol>@kline_1m` to the existing public stream subscription;
2. persist only closed 1m bars to a neutral append-only spool;
3. never attach `signal_id`, side, confidence, score, action, reason code, or
   any other Aegis interpretation;
4. make writes bounded, asynchronous, fail-open, atomic, and idempotent;
5. expose per-symbol checkpoints, gaps, duplicates, stale status, and health.

Proposed schema:

```text
schema_id
symbol
open_time_ms
close_time_ms
open high low close volume taker_buy_volume
exchange_event_timestamp_us
local_receive_wall_timestamp_us
local_receive_monotonic_ns
collector_write_timestamp_us
available_at_us
complete
quality_status
source
```

The unique key is `(symbol, open_time_ms)`. Exact duplicates are ignored;
conflicting duplicates and gaps invalidate the affected interval. Partitions
should be Parquet/ZSTD by date and symbol, with atomic manifest/checkpoint
replacement.

An independent implementation review would cover only:

- `src/aegis/research/prospective_microstructure_w13p.py`;
- `config/experiments/aegis_w13p_prospective_collection.yaml`;
- `tests/unit/test_prospective_microstructure_w13p.py`;
- a new neutral data root such as `data/shared_market_data/candle_1m/`.

No TypeScript source or PM2 definition needs to change if this extension is
accepted. It was not implemented or activated in this round.

## Duplicated acquisition

All sandbox replay and tests now use local files only; they make zero Binance
requests. The existing direct Router candle downloader remains necessary only
to refresh the dataset until a neutral spool is approved. Once the exporter is
active, that duplicate refresh can be retired. Historical warmup remains a
local immutable source and does not require redownload.

## Current label-free collection state

The latest completed independent batch remains current through
`2026-08-18T04:43:00Z`:

- 11 symbols;
- 330 valid snapshots;
- 3,300 candidate evaluations;
- 23 raw candidate population events;
- 18 independent episodes after overlap control;
- 5 suppressed duplicate/overlap events;
- zero candle gaps and zero rejected anchors.

Independent episodes remain: Breakout/Retest 17 and Regime
Transition/Reversal 1; the other three strategies have zero. This is event-rate
only. No outcome or economic metric was loaded.

`GENERAL_CANDIDATE_COLLECTION_ACTIVE = FALSE` because neither a neutral
continuous spool nor an authorized persistent sandbox consumer exists. The
active W13-P process is signal-conditioned persistence and cannot satisfy this
flag. Fresh support remains below the frozen minimum of 2,000 independent
TRAIN candidates, six symbols, and four weekly blocks per specialist.

## Safety and validation

Tests cover operation without Aegis, decision-column rejection, read-only file
behavior, deterministic replay, exact-duplicate idempotence, gap/stale failure,
and no-retroactivity after future candle append. Existing sandbox isolation
tests continue to prohibit financial/exchange capabilities.

No PnL, win rate, future MFE/MAE, return label, strategy ranking, or holdout was
opened or evaluated.
