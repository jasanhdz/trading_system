# W13-P Data Quality Specification

A signal is eligible only when all preregistered acquisition conditions pass:

- pre-window reaches T0-30s within 1.5s tolerance;
- post-window reaches T0+180s within 1.5s tolerance;
- no L2 sequence invalidity, crossed book, queue/disk drop or reconnect in the window;
- a valid L2 base checkpoint exists at or before T0-30s;
- at least one quote and trade event;
- inter-event gaps are recorded descriptively but are not a rejection condition:
  Binance streams are event-driven, so market inactivity is not evidence of data loss.
  Sequence continuity, socket reconnects and queue drops are the causal loss tests.

Every discontinuity is explicit. No gap interpolation, synthetic depth, retrospective
signal recreation or outcome label is permitted.

The first six prospective signals remain permanently ineligible. They exposed a
measurement defect: a 30-second physical ring could not cover T0-30s after the journal
publication delay, and `aggTrade` emitted no events on the active public endpoint.
They are not rewritten after switching to a 90-second ring and raw `trade` stream.

Ten subsequent bundles passed the legacy core-path gate and are valid for BBO/price and
trade-flow pilot analysis. They lack a causal full-book base snapshot, so they are not
counted toward the v2 W13 minimum and cannot support absolute L2 depth/OBI research.

Thirty-one quality-v2 bundles include a causal L2 base snapshot, but a later audit found
that journal publication occurred approximately 7-18 seconds after T0 and the ring was
persisted only through T0. The missing T0-to-observation interval makes those bundles
invalid for 100ms-10s early-path claims. They remain available for delayed endpoint
description but do not count toward the current W13 minimum.

Quality-v3 backfills all already-observed ring events through collector activation
before continuing live capture. Only quality-v3 bundles count toward prospective W13
micro-path TRAIN/VALIDATION.

The collector records quality only; it does not assign GOOD/BAD, ENTER/CANCEL,
TRAIN/VALIDATION or W13 thresholds. Minimum future samples remain TRAIN >= 1,000 and
VALIDATION >= 500. `W13_FINAL_HOLDOUT = SEALED_NOT_OPENED`.
