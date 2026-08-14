# Aegis W9 Historical Order Book Direction - Data Audit

## Status

`AEGIS_W9_BLOCKED_DATA_QUALITY`

W9 is measurable at the normalized-book level, but the free samples overlap
too few frozen W7 Opportunity episodes to support the preregistered economic
validation. Modeling stopped before any directional hypothesis was fitted.

## Frozen Opportunity Coverage

- Total W7 Opportunity episodes in source: 14,503.
- Episodes on Tardis free-sample days: 459.
- TRAIN: 228 episodes, 11 symbols,
  4 months.
- VALIDATION: 231 episodes,
  11 symbols, 4 months.
- FINAL_HOLDOUT_W9: `SEALED_NOT_OPENED`; future evidence is not yet collected.

## L2 Reconstruction Pilot

- Provider: Tardis public first-day sample.
- Instrument/date: `ADAUSDT`, `2025-09-01`.
- Compressed size: 121,458,818 bytes.
- Rows: 17,477,817.
- Message groups: 1,472,930.
- Snapshot groups: 2.
- Crossed/locked messages: 0.
- Invalid messages: 0.
- Local timestamp reorderings: 0.
- Gaps over five seconds: 0.
- Normalized reconstruction valid: `TRUE`.
- Quote rows: 1,073,789; quote audit:
  `TRUE`.
- Trade rows: 1,546,516; trade audit:
  `TRUE`.

The normalized CSV preserves provider capture order and absolute L2 updates.
It does not expose Binance native `U/u/pu` sequence IDs. Tardis documents that
its raw collector validates `pu`/`u`, restarts on gaps, and validates snapshot
overlap. W9 records this limitation rather than claiming independent native
sequence verification from the normalized file.

## Blocking Conditions

- `INSUFFICIENT_TRAIN_EPISODES:228<1000`
- `INSUFFICIENT_VALIDATION_EPISODES:231<500`


Downloading every free L2 day would add gigabytes but would not create more
than the 459 overlapping frozen Opportunity episodes. Bootstrap resampling
cannot replace independent episodes. Fitting the requested ablations on this
population would invite symbol/month selection and produce an unreliable
economic verdict.

## Verdict

- `W9_DATA_QUALITY_SUFFICIENT = FALSE`
- `W9_ORDERBOOK_RECONSTRUCTION_VALID = TRUE`
- W9 directional modeling: not executed.
- Economic edge: not tested, not disproved.
- `W9_READY_FOR_PROSPECTIVE_COLLECTION = FALSE`
- `W9_READY_FOR_SHADOW = FALSE`
- `W9_READY_FOR_LIVE = FALSE`

This result means **insufficient overlapping historical evidence**, not
`AEGIS_W9_NO_ORDERBOOK_DIRECTIONAL_EDGE`.

## Safety

- Public unauthenticated data only.
- Exchange mutations: 0.
- Production, TypeScript, brain, guards, leverage and PM2 changes: 0.
- Orders and production WebSockets: 0.
