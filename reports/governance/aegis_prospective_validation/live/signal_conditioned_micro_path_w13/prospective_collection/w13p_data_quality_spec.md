# W13-P Data Quality Specification

A signal is eligible only when all preregistered acquisition conditions pass:

- pre-window reaches T0-30s within 1.5s tolerance;
- post-window reaches T0+180s within 1.5s tolerance;
- no L2 sequence invalidity, crossed book, queue/disk drop or reconnect in the window;
- at least one quote and trade event;
- maximum combined-stream gap no greater than 1,000ms.

Every discontinuity is explicit. No gap interpolation, synthetic depth, retrospective
signal recreation or outcome label is permitted.

The collector records quality only; it does not assign GOOD/BAD, ENTER/CANCEL,
TRAIN/VALIDATION or W13 thresholds. Minimum future samples remain TRAIN >= 1,000 and
VALIDATION >= 500. `W13_FINAL_HOLDOUT = SEALED_NOT_OPENED`.
