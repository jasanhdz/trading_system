# Phase 2 Independent-Mode Transition Report

Recorded: `2026-08-18` UTC

## Verdict

- `INITIAL_EXPERIMENT_MODE = INDEPENDENT_STRATEGY_DISCOVERY`
- `INDEPENDENT_MARKET_PIPELINE_WORKING = TRUE`
- `GENERAL_CANDIDATE_COLLECTION_ACTIVE = FALSE`
- `GENERAL_CANDIDATE_BATCH_CURRENT_THROUGH = 2026-08-18T04:43:00Z`
- `PHASE_2_INDEPENDENT_MODE_COMPLETE = TRUE`
- `FRESH_DATA_SUFFICIENCY = NOT_YET_MET`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `EDGE_VALIDATION_PERFORMED = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

`GENERAL_CANDIDATE_COLLECTION_ACTIVE` is deliberately `FALSE`: the public
acquisition and replay commands work and the persisted dataset is current, but
no permitted persistent supervisor is running them. A one-shot catch-up is not
reported as continuous collection. This operational limitation does not alter
the completed independent-mode implementation.

## Scope transition

The governing amendment is
`17_INDEPENDENT_STRATEGY_DISCOVERY_AMENDMENT.md`. Initial discovery no longer
loads Aegis signals or uses Aegis to choose anchors, symbols, sides, candidates,
training rows, validation rows, or routing decisions. The same immutable
snapshot is evaluated independently for LONG and SHORT. Rejection of one side
does not support the other side, and no candidate on either side is a valid
`NONE` state.

No Phase 1 or prior Phase 2 infrastructure was removed. The five frozen
generators and all causal snapshot contracts remain unchanged.

## Fresh timeline and source recovery

The Phase 0 checkpoint `dcd445c` has commit timestamp
`2026-08-17T20:59:31Z`. The first previously recorded post-freeze collection
timestamp remains the effective fresh start; this replay used
`2026-08-17T21:14:26.093000Z` as its lower boundary.

After the Binance public-IP restriction was corrected, the public USD-M 1m
candle acquisition completed for all 11 frozen symbols:

- requested increment: `2026-08-15T07:00:00Z` through
  `2026-08-18T04:44:00Z`;
- final closed candle: `2026-08-18T04:43:00Z`;
- rows in the increment: 4,184 per symbol;
- duplicate open times: 0;
- timestamp gaps: 0.

The snapshot replay merged this increment with the immutable warmup source.
Each symbol had 157,244 continuous one-minute rows available to construct the
required 4h/1d warmup. No anchor was rejected.

## Label-free population audit

The replay generated:

- 330 valid side-neutral snapshots, 30 per symbol;
- 3,300 deterministic evaluations: five strategies times two sides;
- 23 raw `CANDIDATE`/`ENTERABLE` events;
- 18 independent episodes after setup identity and 60-minute overlap control;
- 5 overlap/setup suppressions retained in the audit;
- 307 snapshots with `NONE` across the candidate population.

Independent support by strategy:

| Strategy | Episodes | Symbols | Weekly blocks | Frozen minimum met |
|---|---:|---:|---:|---|
| Trend Continuation | 0 | 0 | 0 | No |
| Pullback Continuation | 0 | 0 | 0 | No |
| Breakout/Retest | 17 | 7 | 1 | No |
| Range Mean Reversion | 0 | 0 | 0 | No |
| Regime Transition/Reversal | 1 | 1 | 1 | No |

The frozen support gate requires 2,000 independent TRAIN candidates, at least
six symbols, and at least four weekly blocks for any fitted specialist.
Breakout/Retest already spans seven symbols but is far below the episode and
temporal minimums. Every strategy therefore has insufficient population for
specialist implementation or validation.

The nonzero independent counts were:

- Breakout/Retest: AVAX LONG 1, AVAX SHORT 1, BNB LONG 2, BNB SHORT 2,
  DOGE SHORT 1, ETH LONG 1, LINK SHORT 2, LTC LONG 3, LTC SHORT 1, SOL LONG 3;
- Regime Transition/Reversal: XRP SHORT 1.

These are population counts only. They do not state that any event, side,
symbol, or strategy was profitable or predictive.

## Persistence and deterministic execution

The independent dataset is persisted under
`data/aegis_strategy_router_fresh/general_market_phase2/` as:

- `snapshots.jsonl`;
- `candidate_evaluations.jsonl`;
- `independent_candidate_episodes.jsonl`;
- `suppressed_candidates.jsonl`;
- `manifest.json`.

The CLI supports deterministic symbol partitioning with multiple workers.
Tests compare merged partitions with serial replay at snapshot bytes, candidate
records, and manifest level. Parallelism changes runtime only.

## Prohibited information

No PnL, win rate, future MFE/MAE, barrier outcome, label, holdout, or edge
metric was loaded or inspected. The persisted manifest records
`aegis_signals_loaded = false`, `outcomes_loaded = false`, and
`edge_validation_performed = false`. W1-W14 holdouts remain sealed.

## Remaining blockers

1. Fresh support is much smaller than the frozen minimums.
2. Only one weekly block exists.
3. Several strategies have zero independent candidates in this short window.
4. Public acquisition is operational, but continuous unattended refresh is
   not active because PM2/production integration is prohibited and no separate
   sandbox supervisor has been authorized.

No threshold or rule may be changed to increase event rate. Phase 2 remains a
label-free collection exercise until the frozen support gate is met.
