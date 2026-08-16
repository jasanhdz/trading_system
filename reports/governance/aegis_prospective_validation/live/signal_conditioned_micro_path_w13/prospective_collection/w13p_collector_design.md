# AEGIS W13-P Collector Design

## Scope

W13-P is acquisition only. It consumes public Binance USD-M market data and tails a
contemporary, append-only Aegis evidence journal from a separate process. It cannot
block, alter, route, place, or cancel a financial action.

## Data path

1. Public combined streams continuously populate bounded 90-second rings for the 11
   Aegis symbols: `depth@100ms`, `bookTicker`, and raw `trade`. The longer physical
   retention absorbs journal publication delay; the persisted logical window remains
   exactly T0-30s/T0+180s.
2. The sidecar tails `signal_evidence_v1.jsonl` from a durable byte offset. Only a
   frozen `ENTER_NOW` envelope opens a logical capture window.
3. At T0 the complete source envelope, model/configuration hashes, commits, reason
   codes, scores, collector wall/monotonic clocks, and nearest causal BBO are hashed
   into an immutable signal snapshot.
4. The ring contributes T0-30s and the live stream contributes through T0+180s.
5. Overlapping signals share persisted market events; each signal retains its own
   logical time range and quality record.
6. Bounded queues decouple ingestion and Parquet/ZSTD batches. Overflow invalidates
   research data and never applies backpressure to Aegis.

## Isolation

- No import or callback from the trading process.
- No TypeScript modification.
- No account or order state query. Those fields are explicitly
  `NOT_COLLECTED_PUBLIC_ONLY`.
- First start begins at journal EOF; restarts resume a byte checkpoint. Missed or
  incomplete windows remain ineligible.
- Disk safety stops only W13-P.

## Recovery

WebSocket reconnects, sequence gaps, queue drops, crossed books, and process restarts
are never concealed. L2 is invalidated and rebuilt from a fresh public snapshot.
Signals whose window intersects an invalid interval fail the quality gate.

The future W13 TRAIN/VALIDATION split is not assigned by the collector. It remains an
outcome-blind temporal preregistration after at least 1,500 eligible signals exist.
