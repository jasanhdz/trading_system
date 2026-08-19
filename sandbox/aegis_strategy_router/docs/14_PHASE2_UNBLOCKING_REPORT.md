# Phase 2 Snapshot Pipeline and Decision-Gap Review

Status: `FRESH_SNAPSHOT_PIPELINE_WORKING_RULE_FREEZE_PENDING`

Evaluation date: `2026-08-17 UTC`

This report is label-free. No PnL, win rate, future return, MFE/MAE, barrier
outcome, specialist comparison, or holdout was read.

## 1. Cause of zero fresh snapshots

The prior report counted W13-P market events and signal bundles but no Phase 1
snapshot producer existed. The causes were technical:

1. W13-P stores `BOOK/QUOTE/TRADE` only around signals. It does not persist a
   continuous 1m candle series.
2. `DeterministicSnapshotBuilder` requires continuous 1m candles with enough
   preroll to form 99 fully closed 1d bars.
3. The available preroll ended on `2026-08-15`; fresh signals occurred on
   `2026-08-17`, leaving an invalid 62-hour gap.
4. No adapter joined W13-P signal metadata to public candles, invoked the Phase
   1 builder, invoked the five generators, or persisted the resulting audit.
5. The Phase 1 candle adapter rejected duplicates but did not explicitly reject
   missing 1m timestamps.

This was not a failure of the 4h/1d warmup thresholds.

## 2. Technical correction

Implemented inside the sandbox:

- strict continuous-1m gap validation; gaps become `INVALID` and are never
  filled or interpolated;
- `latest_candle` in each immutable timeframe snapshot, with availability and
  future-leakage checks;
- schema version `aegis-strategy-router-snapshot-v2-latest-candle`;
- a read-only multi-partition Parquet candle source;
- deterministic merge, conflict, gap, and boundary-revision handling;
- exact last-closed-minute freshness check at every signal; stale candle sources
  reject the signal instead of emitting a stale snapshot;
- source-content hash in snapshot provenance;
- W13-P eligible-signal adapter;
- deterministic `signal -> Phase 1 snapshot -> five Phase 2 evaluations`
  pipeline;
- atomic snapshot, candidate, and manifest persistence;
- label-free coverage/event audit consuming the persisted manifest.

The candle source remains fail closed. Conflicting duplicate candles are
rejected except when the only conflict is the final candle of an older segment;
in that case the later complete public segment replaces that boundary candle.
An interior conflict remains fatal.

Public 1m candles were used without authentication to bridge the observed
preroll gap. Pre-freeze candles serve only as causal warmup. Candidate snapshots
begin after the governance checkpoint and no outcome was loaded.

Reproducible pipeline command:

```text
.venv_rocm62/bin/python \
  sandbox/aegis_strategy_router/tools/build_fresh_phase2_pipeline.py \
  --signal-root data/w13p_prospective_collection \
  --candle-root data/live_entry_quality_audit_20260815/candles_1m \
  --candle-root data/aegis_strategy_router_fresh/candles_incremental \
  --output data/aegis_strategy_router_fresh/phase2_audit
```

## 3. Current fresh coverage

```text
fresh public events:       9,434
fresh eligible signals:        2
symbols:                       2 (SUIUSDT, ADAUSDT)
sides:                         SHORT only
continuous candles/symbol: 156,900
candle gaps:                   0
complete Phase 1 snapshots:    2
generator evaluations:        10
rejected fresh signals:        0
```

Each strategy was evaluated once per snapshot. All ten evaluations currently
return `BLOCKED_FROZEN_DECISION_GAP`. This is evaluation coverage, not candidate
event rate. Eligible candidate event rate remains unavailable until rules are
frozen.

## 4. Rules recovered from existing frozen decisions

The following no longer require a methodological decision:

1. Structural levels use confirmed deterministic complete-linkage levels only.
2. Candidate data must be fully available; missing timeframe data returns
   `UNKNOWN` before rules run.
3. Favorable structural space is measured against the nearest causal level in
   every available structural timeframe. The exact common-target requirement
   is at least `0.50 ATR`; the minimum available space is binding.
4. Breakout penetration is exactly `0.10 ATR` beyond a prior causal level.
5. Retest proximity is exactly `0.20 ATR`, followed by a close on the breakout
   side.
6. Breakout too-late is remaining structural space below `0.50 ATR`.
7. LONG/SHORT use one direction-normalized formula.
8. Shock and volatility-shock candidate vetoes are removed. Phase 0 section 8
   explicitly makes market critics diagnostic-only initially; using shock as a
   Phase 2 eligibility veto would contradict the later controlling decision.
9. `BREAKOUT_CONFIRMED` specialist eligibility is not a Phase 2 rule. Phase 2
   can create `BREAKOUT_CANDIDATE`; later specialist work remains unauthorized.
10. All pending/enterable/invalidated substate dispositions and episode
    identity behavior are frozen and implemented.

On the two fresh snapshots, the exact breakout observations and structural
space facts were recorded, but were not interpreted as evidence of performance.

## 5. Remaining decisions and recommended minimal definitions

These recommendations are proposals only. They are not active rules and cannot
be implemented until explicitly frozen.

### Trend continuation

| Gap | Recommended minimal causal definition |
|---|---|
| `TREND_ALIGNMENT_UNSPECIFIED` | A timeframe aligns when direction-normalized EMA25 slope is strictly positive and its latest two confirmed HIGH pivots and latest two confirmed LOW pivots are non-contradictory: both rise for LONG, both fall for SHORT. Require this on 1h and 4h. Missing pivot pairs produce `UNKNOWN`. |
| `TREND_15M_INVALIDATION_UNSPECIFIED` | LONG invalidates when the latest closed 15m candle closes below the most recently available confirmed 15m LOW pivot; SHORT is symmetric above the latest HIGH pivot. |
| `ISOLATED_CANDLE_UNSPECIFIED` | Require direction-normalized `return_3_bps > 0` on 5m and 15m and direction-normalized `path_efficiency_6 > 0` on 5m. This uses only sign boundaries and prevents one isolated candle from defining continuation. |

### Pullback continuation

| Gap | Recommended minimal causal definition |
|---|---|
| `PULLBACK_HTF_ALIGNMENT_UNSPECIFIED` | Reuse the exact frozen Trend alignment definition; do not create a second trend definition. |
| `PULLBACK_OPPOSITION_UNSPECIFIED` | `PULLBACK_FORMING` when direction-normalized 1-bar return is strictly negative on both 1m and 5m while HTF alignment remains valid. |
| `PULLBACK_INVALIDATION_LEVEL_UNSPECIFIED` | Use the most recently available confirmed 1h LOW pivot for LONG and HIGH pivot for SHORT. A closed 5m candle beyond it invalidates. Missing level produces `UNKNOWN`. |
| `PULLBACK_REALIGNMENT_UNSPECIFIED` | At a later snapshot, require direction-normalized 1-bar return strictly positive on 1m and 5m, direction-normalized 1m taker imbalance strictly positive, and no invalidation. The later snapshot creates a new candidate timestamp and price. |

### Breakout/retest

| Gap | Recommended minimal causal definition |
|---|---|
| `BREAKOUT_TIMEFRAME_UNSPECIFIED` | Use 15m only for initial Phase 2 breakout/retest generation. 1h/4h/1d remain location context, preventing five correlated breakout populations. |
| `BREAKOUT_AGE_UNSPECIFIED` | Do not add another numeric age threshold. A level is sufficiently prior when it already satisfies the frozen two-touch construction and `level.available_at <= breakout_candle.open_at`. |
| `RETEST_SEQUENCE_IDENTITY_UNSPECIFIED` | Episode key hashes symbol, side, `15m`, level ID, and breakout-candle close time. Evaluate subsequent closed 15m candles for at most the frozen common 60-minute horizon. First close back inside before a valid retest is `FALSE_BREAKOUT`; otherwise use the frozen retest predicate. |

### Range mean reversion

| Gap | Recommended minimal causal definition |
|---|---|
| `LOW_EFFICIENCY_UNSPECIFIED` | On 15m require `abs(path_efficiency_6) <= 0.35`. This threshold is a proposed geometric definition, not data-selected, and requires explicit approval. |
| `FLAT_HTF_SLOPE_UNSPECIFIED` | Require `abs(ema25_slope_atr) <= 0.05` on both 1h and 4h. This proposed small-slope boundary requires explicit approval. |
| `STABLE_RANGE_UNSPECIFIED` | Use nearest causal 15m LOW support and HIGH resistance, both available before the current candle and already satisfying two-touch clustering. Current close must remain inside and range width must be at least `1.00 ATR`, exactly two common half-ATR barriers. |
| `RANGE_EDGE_UNSPECIFIED` | LONG candidate within `0.20 ATR` of support; SHORT within `0.20 ATR` of resistance, reusing the frozen structural tolerance. Target direction points toward the range midpoint. |

### Regime transition/reversal

| Gap | Recommended minimal causal definition |
|---|---|
| `PRIOR_REGIME_UNSPECIFIED` | Prior regime is `TREND` only under the frozen Trend alignment rule, `RANGE` only under the frozen stable-range rule, otherwise `UNKNOWN`. Do not add a third classifier. |
| `REGIME_DETERIORATION_UNSPECIFIED` | For a prior trend, require direction-normalized 15m EMA25 slope non-positive and a 15m close through the prior regime's 15m invalidation pivot. Both conditions are required. |
| `NEW_STRUCTURE_UNSPECIFIED` | Require two fully confirmed 15m HIGH pivots and two LOW pivots aligned with the proposed new side. This cannot be satisfied by one opposing candle. |
| `TRANSITION_CONFIRMATION_UNSPECIFIED` | `NEW_REGIME_CONFIRMED` requires the new 15m structure plus direction-normalized EMA25 slope strictly positive on 15m and 1h. Before that it remains a candidate/terminal WAIT according to the frozen substate contract. |

## 6. Verification

Verification:

```text
Sandbox Phase 1/2 suite: 42 passed
Existing causal feature regressions: 7 passed
Python compileall: passed
git diff --check: passed
Nested TypeScript production repository: clean
```

Tests cover source gaps, stale coverage, conflicting duplicates, safe boundary
revision, complete warmup, deterministic persistence, latest-candle causality,
LONG/SHORT symmetry, episode overlap, exact gap registry, isolation, and
future-data rejection.

## 7. Verdict

- `FRESH_SNAPSHOT_PIPELINE_WORKING = TRUE`
- `FRESH_CANDLE_CONTINUITY_VALID = TRUE`
- `FRESH_SNAPSHOT_PERSISTENCE_VALID = TRUE`
- `FRESH_PHASE1_SNAPSHOTS = 2`
- `PHASE2_GENERATOR_EVALUATIONS = 10`
- `FROZEN_DECISION_GAPS_REDUCED_FROM = 22`
- `FROZEN_DECISION_GAPS_REMAINING = 18`
- `ELIGIBLE_CANDIDATE_EVENT_RATE_AVAILABLE = FALSE`
- `PHASE_2_RULE_EXECUTION_COMPLETE = FALSE`
- `PHASE_2_TECHNICAL_ACCEPTANCE = PARTIAL_RULE_FREEZE_REQUIRED`
- `FRESH_DATA_SUFFICIENCY = NOT_YET_MET`
- `READY_TO_VALIDATE_PHASE_2 = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`
- `EDGE_VALIDATION_PERFORMED = FALSE`

The snapshot pipeline is unblocked. Phase 2 candidate activation remains
blocked only by the 18 explicitly listed decisions. Freezing them is a separate
governance action; they must not be selected from future performance.
