# W13-P Schema

All absolute timestamps use integer UTC microseconds unless the field ends in `_ns`.
Monotonic timestamps are process-local nanoseconds and are not UTC-comparable.

## SIGNAL

Primary key: `signal_id` (the contemporary `prospective_signal_id`). Includes canonical
T0, collector observation clocks/delay, symbol, side, causal BBO/mid, source envelope
JSON, model/config hashes, commits, reasons, scores, schema version and snapshot hash.

Quality gate v2 also embeds the latest valid full L2 checkpoint at or before T0-30s:
checkpoint exchange timestamp, update ID, generation, and sorted bid/ask levels. Raw
diffs from that checkpoint onward are persisted so L1/L5/L10/L20 and depth can be
reconstructed without a future snapshot.

Account-derived state is never queried; descriptive fields contain
`NOT_COLLECTED_PUBLIC_ONLY`.

## EVENT

Primary event identity is exchange-derived:

- BOOK: symbol + U + u;
- QUOTE: symbol + update/event ID;
- TRADE: symbol + aggregate trade ID.

Columns include event type, symbol, exchange event/trade time, local receive wall and
monotonic clocks, collector write time, L2 validity/generation, capture segment, and
lossless normalized public payload JSON.

## QUALITY

One terminal row per completed signal: logical start/T0/end, pre/post completeness,
L2 base snapshot and continuity, quote/trade coverage, maximum observed gap, event
counts, reconnect/drop state, quality version, and `W13_ELIGIBLE`.

Physical layout is Parquet/ZSTD partitioned by record kind, UTC date and symbol.
Overlapping signals reference a shared event stream by time and segment rather than
duplicating every update.
