# Phase 2 Deterministic Candidate Rule Freeze

Freeze date: `2026-08-18` UTC

This document closes the 18 methodological gaps listed in the removed build-stage
report. The rules were selected from minimal, causal, LONG/SHORT-symmetric
proposals. No PnL, win-rate, MFE/MAE, outcome, label, validation result, or
prior holdout was inspected or used to choose them.

This freeze authorizes deterministic candidate generation only. It does not
authorize specialists, training, calibration, critics, routing, sequential
trading decisions, shadow, or live use.

## Common semantics

- Every input must be available at or before the snapshot `decision_at`.
- All candles are fully closed. Every pivot and clustered level is already
  causally confirmed.
- LONG uses direction `+1`; SHORT uses direction `-1`. Signed predicates use
  one direction-normalized formula.
- A missing required feature, candle, pivot pair, ATR, or invalidation level
  produces `UNKNOWN`. It is never imputed.
- The existing complete-linkage structural-level tolerance (`0.20 ATR`),
  breakout penetration (`0.10 ATR`), retest touch (`0.20 ATR`), and common
  target-space requirement (`0.50 ATR`) remain unchanged.
- `path_efficiency_6` is an unsigned magnitude in the frozen source adapter.
  It is never multiplied by direction. Direction is supplied by signed returns
  and structure.
- A candidate evaluation remains snapshot-specific. A persistent
  `setup_episode_id` links causally related substates without reusing the old
  snapshot price or timestamp.
- Phase 2 replay may reconstruct deterministic candidate substates. It does
  not implement router `WAIT`, execution, or a trading state machine.

## The 18 frozen decisions

### Trend continuation

1. A timeframe is aligned when direction-normalized `ema25_slope_atr > 0`, the
   latest two confirmed HIGH pivots rise for LONG/fall for SHORT, and the latest
   two confirmed LOW pivots rise for LONG/fall for SHORT. Both 1h and 4h must
   align. Equal pivot prices are not aligned. Missing pivot pairs produce
   `UNKNOWN`.
2. LONG invalidates when the latest closed 15m close is strictly below the most
   recently available confirmed 15m LOW pivot. SHORT invalidates symmetrically
   above the latest confirmed HIGH pivot. Equality does not invalidate.
3. Direction-normalized `return_3_bps` must be strictly positive on 5m and 15m,
   and unsigned 5m `path_efficiency_6` must be strictly positive. This is a
   sign-only temporal persistence check; no new numeric threshold is introduced.

### Pullback continuation

4. Pullback HTF alignment reuses decision 1 exactly.
5. `PULLBACK_FORMING` requires direction-normalized `return_1_bps < 0` on both
   1m and 5m while decision 1 remains valid.
6. Use the most recently available confirmed 1h LOW pivot for LONG and HIGH
   pivot for SHORT. A latest closed 5m close strictly beyond that pivot
   invalidates.
7. A later snapshot can emit `PULLBACK_CONFIRMED` only after the same
   symbol/side had a `PULLBACK_FORMING` setup and now has direction-normalized
   `return_1_bps > 0` on 1m and 5m, direction-normalized 1m `taker_imbalance >
   0`, continued HTF alignment, and no invalidation. Confirmation uses the
   later snapshot's timestamp and reference price.

### Breakout/retest

8. Breakouts are evaluated at 15m only. 1h/4h/1d remain structural context and
   do not generate additional breakout populations.
9. A level is prior only if it already satisfies frozen two-touch construction
   and `level.available_at <= breakout_candle.open_at`; no age threshold is
   added.
10. `setup_episode_id` hashes symbol, side, `15m`, level ID, and breakout candle
    close time. Only subsequent closed 15m candles up to 60 minutes after the
    breakout are evaluated. A frozen `0.20 ATR` touch followed by a close on
    the breakout side emits `RETEST_CONFIRMED`. The first prior close back
    inside emits `FALSE_BREAKOUT`. An unconfirmed setup remains
    `RETEST_PENDING` until expiry. Every later evaluation has its own snapshot
    candidate ID.

### Range mean reversion

11. Low efficiency is unsigned 15m `path_efficiency_6 <= 0.35`.
12. Flat HTF slope is `abs(ema25_slope_atr) <= 0.05` on both 1h and 4h.
13. Use the nearest causal 15m LOW support at/below price and HIGH resistance
    at/above price. Both levels must be available no later than the current
    candle open and already satisfy frozen two-touch clustering. The latest
    close must remain inside, and width must be at least `1.00 ATR`.
14. LONG is at the support edge when distance is at most `0.20 ATR`; SHORT is
    symmetric at resistance. Target direction points toward the midpoint. A
    same-candle rejection is confirmed only when the candle touches within that
    same band and closes strictly inward from its edge.

### Regime transition/reversal

15. Prior regime is `TREND` only under decision 1 and `RANGE` only under
    decision 13; otherwise it is `UNKNOWN`. For a reversal candidate, prior
    TREND must align with the side opposite the proposed new side. RANGE is
    directionless.
16. A prior TREND deteriorates only when old-direction-normalized 15m
    `ema25_slope_atr <= 0` and the latest 15m close crosses the old trend's
    decision-2 invalidation pivot. A prior RANGE deteriorates only when the
    latest 15m close is strictly beyond its frozen boundary in the proposed new
    direction.
17. The latest two fully confirmed 15m HIGH pivots and latest two LOW pivots
    must both align with the proposed new side under decision 1's strict
    structure comparison.
18. `NEW_REGIME_CONFIRMED` requires decision 17 plus direction-normalized
    `ema25_slope_atr > 0` on both 15m and 1h. Deterioration without new
    structure is `OLD_REGIME_DETERIORATING`; new structure without slope
    confirmation is `TRANSITION_CANDIDATE`.

## Deterministic precedence

Rules are evaluated in this order: missing causal input -> `UNKNOWN`;
structural invalidation/break -> invalidated substate; too-late structural
space -> too-late substate; confirmation -> confirmed substate; forming or
pending setup -> waiting/candidate substate; otherwise `INELIGIBLE`.

When multiple prior levels could break on one 15m candle, choose the level with
the smallest directional penetration, then `level_id` lexicographically. This
uses only current causal geometry. Sequence state is keyed by strategy, symbol,
and side and is replayed in ascending `(decision_at, snapshot_id)` order.

## Governance flags

- `FROZEN_DECISION_GAPS_REMAINING = 0`
- `PHASE_2_RULES_FROZEN = TRUE`
- `EDGE_VALIDATION_PERFORMED = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`
