# W13-P Activation Report

## Phase A: offline/synthetic

PASS. Tests cover immutable signal snapshots, ring/window boundaries, L2 snapshot and
diff continuity, duplicate/gap/cross handling, reconnect invalidation, overlap
deduplication, bounded queues, disk stop, checkpoint tails, Parquet output, crash
isolation, public allowlists and absence of financial capabilities.

## Phase B: public market data

PASS. Public-only dry run on all 11 symbols produced 11 valid local books and no gaps,
crosses, out-of-order events, drops or reconnects. No credentials were read and no
signal journal event was consumed during this phase.

## Phase C: prospective signals

The approved architecture tails only newly appended `ENTER_NOW` evidence from the
existing fsync journal. It does not modify its producer. Activation status and runtime
health must be checked separately from W13 scientific readiness.

PASS. A read-only dry run preserved all 11 valid books with zero gaps, crosses, drops
or reconnects. The dedicated PM2 sidecar was then authorized for passive acquisition;
it is not a dependency of either trading process.

An explicit collector restart changed only the W13-P PID. Trading Bot PID 1754 and
Aegis API PID 1767 remained unchanged. The restarted collector rebuilt all 11 books
validly with zero drops, and the PM2 process list was saved for host-reboot recovery.

One real SUIUSDT SHORT signal arrived during the deliberate restart test. Its immutable
snapshot and logical window were preserved, but it was correctly marked ineligible
(`pre_window_complete=false`, L2 invalid interval, no trade coverage). No attempt was
made to repair or relabel it.

Collection does not authorize W13 research before sample minima and does not authorize
Shadow or Live decisions.
