# Phase 2 Deterministic Candidate Rule Freeze

Freeze date: `2026-08-18` UTC

This document closes the 18 methodological gaps listed in
`14_PHASE2_UNBLOCKING_REPORT.md`. The rules were selected from the minimal,
causal, LONG/SHORT-symmetric proposals already documented there. No PnL,
win-rate, MFE/MAE, outcome, label, validation result, or prior holdout was
inspected or used to choose them.

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
  It is therefore never multiplied by direction. Direction is supplied by
  signed returns and structure.
- A candidate evaluation remains snapshot-specific. A persistent
  `setup_episode_id` links causally related substates without reusing the old
  snapshot price or timestamp.
- Phase 2 replay may reconstruct deterministic candidate substates. It does
  not implement router `WAIT`, execution, or a trading state machine.

## The 18 frozen decisions

### Trend continuation

1. `TREND_ALIGNMENT_UNSPECIFIED` is closed as follows. A timeframe is aligned
   when direction-normalized `ema25_slope_atr > 0`, the latest two confirmed
   HIGH pivots rise for LONG/fall for SHORT, and the latest two confirmed LOW
   pivots rise for LONG/fall for SHORT. Both 1h and 4h must align. Equal pivot
   prices are not aligned. Missing pivot pairs produce `UNKNOWN`.
2. `TREND_15M_INVALIDATION_UNSPECIFIED` is closed as follows. LONG invalidates
   when the latest closed 15m close is strictly below the most recently
   available confirmed 15m LOW pivot. SHORT invalidates symmetrically above
   the latest confirmed HIGH pivot. Equality does not invalidate.
3. `ISOLATED_CANDLE_UNSPECIFIED` is closed as follows. Direction-normalized
   `return_3_bps` must be strictly positive on 5m and 15m, and unsigned 5m
   `path_efficiency_6` must be strictly positive. This is a sign-only temporal
   persistence check; no new numeric threshold is introduced.

### Pullback continuation

4. `PULLBACK_HTF_ALIGNMENT_UNSPECIFIED` reuses decision 1 exactly.
5. `PULLBACK_OPPOSITION_UNSPECIFIED` is closed as follows.
   `PULLBACK_FORMING` requires direction-normalized `return_1_bps < 0` on both
   1m and 5m while decision 1 remains valid.
6. `PULLBACK_INVALIDATION_LEVEL_UNSPECIFIED` is closed as follows. Use the
   most recently available confirmed 1h LOW pivot for LONG and HIGH pivot for
   SHORT. A latest closed 5m close strictly beyond that pivot invalidates.
7. `PULLBACK_REALIGNMENT_UNSPECIFIED` is closed as follows. A later snapshot
   can emit `PULLBACK_CONFIRMED` only after the same symbol/side had a
   `PULLBACK_FORMING` setup and now has direction-normalized `return_1_bps > 0`
   on 1m and 5m, direction-normalized 1m `taker_imbalance > 0`, continued HTF
   alignment, and no invalidation. Confirmation uses the later snapshot's
   timestamp and reference price.

### Breakout/retest

8. `BREAKOUT_TIMEFRAME_UNSPECIFIED` is closed at 15m only. 1h/4h/1d remain
   structural context and do not generate additional breakout populations.
9. `BREAKOUT_AGE_UNSPECIFIED` adds no age threshold. A level is prior only if
   it already satisfies frozen two-touch construction and
   `level.available_at <= breakout_candle.open_at`.
10. `RETEST_SEQUENCE_IDENTITY_UNSPECIFIED` is closed as follows. The
    `setup_episode_id` hashes symbol, side, `15m`, level ID, and breakout
    candle close time. Only subsequent closed 15m candles up to 60 minutes
    after the breakout are evaluated. A frozen `0.20 ATR` touch followed by a
    close on the breakout side emits `RETEST_CONFIRMED`. The first prior close
    back inside emits `FALSE_BREAKOUT`. An unconfirmed setup remains
    `RETEST_PENDING` until expiry. Every later evaluation has its own snapshot
    candidate ID.

### Range mean reversion

11. `LOW_EFFICIENCY_UNSPECIFIED` is closed at unsigned 15m
    `path_efficiency_6 <= 0.35`.
12. `FLAT_HTF_SLOPE_UNSPECIFIED` is closed at
    `abs(ema25_slope_atr) <= 0.05` on both 1h and 4h.
13. `STABLE_RANGE_UNSPECIFIED` is closed as follows. Use the nearest causal
    15m LOW support at/below price and HIGH resistance at/above price. Both
    levels must be available no later than the current candle open and already
    satisfy frozen two-touch clustering. The latest close must remain inside,
    and width must be at least `1.00 ATR`.
14. `RANGE_EDGE_UNSPECIFIED` is closed as follows. LONG is at the support edge
    when distance is at most `0.20 ATR`; SHORT is symmetric at resistance.
    Target direction points toward the midpoint. A same-candle rejection is
    confirmed only when the candle touches within that same `0.20 ATR` band
    and closes strictly inward from its edge; this introduces no extra
    threshold.

### Regime transition/reversal

15. `PRIOR_REGIME_UNSPECIFIED` is closed as follows. Prior regime is `TREND`
    only under decision 1 and `RANGE` only under decision 13; otherwise it is
    `UNKNOWN`. For a reversal candidate, prior TREND must align with the side
    opposite the proposed new side. RANGE is directionless.
16. `REGIME_DETERIORATION_UNSPECIFIED` is closed as follows. A prior TREND
    deteriorates only when old-direction-normalized 15m `ema25_slope_atr <= 0`
    and the latest 15m close crosses the old trend's decision-2 invalidation
    pivot. A prior RANGE deteriorates only when the latest 15m close is
    strictly beyond its frozen boundary in the proposed new direction. No
    volatility or outcome threshold is added.
17. `NEW_STRUCTURE_UNSPECIFIED` is closed as follows. The latest two fully
    confirmed 15m HIGH pivots and latest two LOW pivots must both align with
    the proposed new side under decision 1's strict structure comparison.
18. `TRANSITION_CONFIRMATION_UNSPECIFIED` is closed as follows.
    `NEW_REGIME_CONFIRMED` requires decision 17 plus direction-normalized
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
uses only current causal geometry. Sequence state is keyed by strategy,
symbol, and side and is replayed in ascending `(decision_at, snapshot_id)`
order.

## Governance flags

- `FROZEN_DECISION_GAPS_REMAINING = 0`
- `PHASE_2_RULES_FROZEN = TRUE`
- `EDGE_VALIDATION_PERFORMED = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

