# Aegis W9 Historical Order Book Direction - Result

## Verdict

`AEGIS_W9_BLOCKED_DATA_QUALITY`

This is a sample-coverage block, not a negative economic result.

## What Was Proven

The public Tardis pilot for `ADAUSDT` on 2025-09-01 is internally
reconstructable:

- 17,477,817 normalized L2 rows.
- 1,472,930 ordered message groups.
- 2 full snapshots, including a recovery reset.
- 0 crossed/locked messages.
- 0 invalid messages.
- 0 local timestamp reorderings.
- 0 gaps longer than five seconds; maximum observed gap 1.268361 seconds.
- 1,073,789 valid quotes.
- 1,546,516 valid trades.
- 0 adjacent duplicate trade IDs.

The normalized CSV omits Binance `U/u/pu` identifiers, so W9 does not claim
independent native sequence verification from that file. Tardis documents that
its raw collector checks `pu`/`u`, restarts after missed updates and validates
REST snapshot overlap. That provider assertion is recorded separately from the
checks W9 performed itself.

## Why Modeling Stopped

The frozen W7 source contains 14,503 Opportunity episodes, but only 459 occur
on the first days of months available without a Tardis subscription:

| Partition | Episodes | Symbols | Months | Required episodes |
|---|---:|---:|---:|---:|
| TRAIN | 228 | 11 | 4 | 1,000 |
| VALIDATION | 231 | 11 | 4 | 500 |

Monthly counts were 89, 33, 20, 86, 14, 107, 76 and 34. Downloading all 11
symbols for all eight days would add gigabytes of updates but no independent
Opportunity episodes. It therefore cannot repair the statistical deficiency.

No classifier, threshold, ablation, latency policy or directional rule was
fitted. Bootstrap was not used to manufacture independence. FINAL_HOLDOUT_W9
remains `SEALED_NOT_OPENED`.

## Flags

- `W9_DATA_QUALITY_SUFFICIENT = FALSE`
- `W9_ORDERBOOK_RECONSTRUCTION_VALID = TRUE`
- `W9_LIQUIDITY_INFORMATION_FOUND = FALSE` (not tested)
- `W9_FLOW_INFORMATION_FOUND = FALSE` (not tested)
- `W9_ABSORPTION_INFORMATION_FOUND = FALSE` (not tested)
- `W9_DIRECTIONAL_SIGNAL_FOUND = FALSE` (not tested)
- `W9_ECONOMIC_EDGE_FOUND = FALSE` (not tested)
- `W9_READY_FOR_PROSPECTIVE_COLLECTION = FALSE`
- `W9_READY_FOR_SHADOW = FALSE`
- `W9_READY_FOR_LIVE = FALSE`

## Interpretation

Historical L2 can be reconstructed honestly from this provider, but the free
overlap is insufficient to answer whether order-book dynamics form a useful
directional compass. The correct status is blocked data quality, not
`AEGIS_W9_NO_ORDERBOOK_DIRECTIONAL_EDGE`.

No production code, runtime, PM2 service, exchange account or order was
modified.
