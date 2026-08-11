# Aegis Decomposed Entry V9 Research Plan

## Motivation

V8 reduced adverse excursion and lower-tail loss in several folds, but every
LONG and SHORT fold retained negative stress-cost expectancy. Its single broad
late-entry label also failed to beat prevalence baselines in every fold. V9
therefore separates three scientific questions that V8 mixed together:
direction, entry timing, and the likely path after entry.

## Frozen architecture

The direction model sees only side-neutral market features plus causal rolling
4-hour context and predicts `LONG`, `SHORT`, or `ABSTAIN`. Direction labels use
the unprotected 24-bar path under stress costs and require an advantage over the
opposite side. Protection outcomes cannot define direction.

Timing remains side-specific. Seven independent heads estimate exhausted move,
counter-trend failure, false breakout, weak-volume impulse, overextension
failure, transition failure, and adverse continuation. Their labels may use
future outcomes, but their model inputs may contain only information available
before entry. Each head is evaluated independently against prevalence Brier and
average-precision controls.

The trajectory component estimates positive and catastrophic protected return,
stress-cost net return, MAE q90, MFE q50, and time to positive. Its purpose is
to distinguish attractive direction from an unattractive entry path.

## Timeframes

V9 retains the existing 5-minute, 15-minute, and 1-hour causal features. It adds
rolling 4-hour and 12-hour context computed exclusively from closed 5-minute
candles preceding the candidate. The current partial bar and every future bar
are excluded.

## Entry versus protection

Primary validation always uses the frozen `CURRENT_TS` protection profile.
Alternative V8 protection profiles are retained only for diagnostics after a
candidate has been selected. They cannot participate in model fitting, policy
selection, ranking, or the promotion gate.

## Validation and success

Four expanding walk-forward folds use a 120-minute purge and calibration-only
thresholds. A successful side needs at least three positive folds, a
non-negative worst fold, positive stress-cost expectancy above the current
brain control, improved CVaR, no worse MAE, payoff ratio at least one, skilled
direction, timing, and trajectory components, acceptable opportunity frequency,
and at least eight successful leave-one-symbol-out evaluations.

Failure of any independent scientific claim blocks promotion. Thresholds may
not be changed after test results are inspected, and no trade quota may force a
selection.

## Safety

- Runtime effect: `NONE`.
- Model export: `PROHIBITED_UNTIL_ALL_GATES_PASS`.
- Shadow activation: `PROHIBITED_UNTIL_SEPARATE_AUTHORIZATION`.
- Live activation: `PROHIBITED`.
- Exchange calls: `0`.
- Exchange mutations: `0`.
