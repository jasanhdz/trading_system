# W13-P Safety Audit

## Existing prototype

The W13 prototype was an inert in-memory sink. It accepted caller-fed BOOK/QUOTE/TRADE
events and produced -30/+180 windows, but had no persistence, sequence reconstruction,
bounded queue, disk gate, reconnect handling, contemporary model bundle, or runtime.

## Contemporary signal source

The existing prospective journal is fsync-appended and provides an immutable signal
ID, canonical signal/information cutoff, model identity/hash, configuration hash,
Python and TypeScript commits, probabilities, component evidence, reasons and risk
intent. W13-P reads this file; it does not inject an execution hook.

Current evidence schema exposes `SHORT`/`NO_TRADE`; W13-P accepts and stores both LONG
and SHORT when future evidence schemas emit them. No side is discarded by collection.

## Threat review

Static and behavioral tests confirm:

- exact public host/path allowlists;
- no API key, signing secret, account client, private endpoint, order, cancel, position,
  balance, leverage, or money movement capability;
- bounded market and disk queues;
- fail-closed data eligibility after loss or reconnect;
- disk pressure stops only the sidecar;
- sidecar crash does not share process state with trading;
- configuration is separate from trading configuration.

## Public validation

On 2026-08-15 a full-universe 15-second public dry run reconstructed all 11 books with:

- valid books: 11/11;
- generations/resyncs: one per symbol;
- sequence gaps: 0;
- crossed books: 0;
- out-of-order events: 0;
- collector/disk drops: 0;
- WebSocket reconnects: 0.

Initial dry runs exposed and fixed two sidecar-only defects: event-loop fairness and
serial snapshot initialization. Neither run consumed signal events or affected Aegis.

A later runtime audit corrected the snapshot-to-diff bridge to accept a first update
starting at `lastUpdateId + 1`. It also added one-at-a-time public snapshots and a
global backoff for HTTP `418`/`429`. Rate limiting therefore invalidates collection
quality instead of producing a retry storm or blocking a trading process.

## Verdict

`W13P_SAFE_TO_COLLECT = TRUE` applies only to passive public acquisition. It is not a
W13 model, guard, Shadow decision, or Live trading authorization.
