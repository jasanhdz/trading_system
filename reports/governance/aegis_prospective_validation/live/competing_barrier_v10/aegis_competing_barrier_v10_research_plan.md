# Aegis Competing Barrier V10 Research Plan

## Motivation

V9 demonstrated out-of-sample skill in direction and seven timing labels, but
its exact stress-return regressor failed in seven of eight side/fold tests and
the final selector remained negative. Several timing labels also included the
causal setup they were intended to diagnose. V10 removes both weaknesses.

## Outcome-only labels

Every V10 label is determined exclusively by OHLC observations after entry.
For each side, the system asks whether a favorable or equally sized adverse
price barrier is reached first. Causal indicators, regime, volume,
overextension, and archetype never participate in label construction; they are
model inputs only.

The frozen contracts combine 5%, 10%, and 20% ROE barriers at 15x reporting
leverage with horizons of 30, 60, and 120 minutes. Each outcome is exactly one
of favorable first, adverse first, same-bar ambiguous, or neither reached.
Same-bar ambiguity is retained and valued as the adverse outcome when realized
utility is audited.

## Non-overlapping episodes

Candidates for the same symbol are reduced to deterministic 120-minute
episodes. The first eligible timestamp is retained and the next timestamp must
be at least 120 minutes later. The rule is side-neutral and cannot inspect an
outcome. This prevents one market move from appearing as several independent
successes or failures.

## Models and utility

A side-neutral direction model predicts LONG, SHORT, or abstention from one
frozen 10%-ROE/60-minute contract. Nine side-specific competing-risk models
predict the four barrier outcomes. No model predicts an exact return.

For each candidate, conservative utility equals predicted favorable value less
predicted adverse value, severe round-trip costs, and a penalty for ambiguous
or unresolved outcomes. The system may abstain and may select at most one
symbol per timestamp. Policies are chosen using calibration only.

## Separation from exits

V10 evaluates entry-path evidence, not a production exit strategy. The current
TypeScript protection result, MAE, and time to positive are reported only as
diagnostics. No alternative stop, take-profit, trailing, or protection profile
may influence model fitting, ranking, selection, or promotion.

## Validation

Four purged expanding folds and a 120-minute embargo are frozen. Direction and
at least six of nine competing-risk contracts must beat prior/majority controls
independently. Promotion additionally requires three positive realized-utility
folds, a non-negative worst fold, improved lower-tail CVaR, payoff ratio at
least one, acceptable opportunity frequency, and eight successful
leave-one-symbol-out evaluations.

No threshold may be changed after test outcomes are observed.

## Safety

- Runtime effect: `NONE`.
- Model export: `PROHIBITED_UNTIL_ALL_GATES_PASS`.
- Shadow activation: `PROHIBITED_UNTIL_SEPARATE_AUTHORIZATION`.
- Live activation: `PROHIBITED`.
- Exchange calls: `0`.
- Exchange mutations: `0`.
