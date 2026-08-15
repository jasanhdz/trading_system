# W13-P Schema

All absolute timestamps use integer UTC microseconds unless the field ends in `_ns`.
Monotonic timestamps are process-local nanoseconds and are not UTC-comparable.

## SIGNAL

Primary key: `signal_id` (the contemporary `prospective_signal_id`). Includes canonical
T0, collector observation clocks/delay, symbol, side, causal BBO/mid, source envelope
JSON, model/config hashes, commits, reasons, scores, schema version and snapshot hash.

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
L2 continuity, quote/trade coverage, maximum observed gap, event counts, reconnect and
drop state, quality version, and `W13_ELIGIBLE`.

Physical layout is Parquet/ZSTD partitioned by record kind, UTC date and symbol.
Overlapping signals reference a shared event stream by time and segment rather than
duplicating every update.
