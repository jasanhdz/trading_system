# Aegis W12 Early Path Confirmation - Preregistration

## Frozen question

Given a frozen historical Aegis direction, can genuinely new path information
observed after the signal support one dynamic `PENDING -> ENTER/CANCEL` decision
with positive remaining economic value?

## Resolution audit

Live signals cover May-July 2026. Validated Tardis L2 samples cover isolated
first days from September 2024 through March 2026 and do not overlap the Live
signals. W12 therefore uses closed 1m candles and their taker-buy volume only.
It does not fabricate 15s, 30s, or 90s observations.

The two causal states are the close of the first and second complete 1m bars
that begin after the signal. Depending on the signal's position inside its
minute, these occur approximately 60-120s and 120-180s after T0.

## Splits

- TRAIN: before 2026-07-01 UTC, 437 source signals.
- VALIDATION: 2026-07-01 through 2026-07-26 UTC, 114 source signals.
- W12 FINAL_HOLDOUT: 2026-07-27 through 2026-07-31 UTC, 62 source signals,
  initially `SEALED_NOT_OPENED`.
- W11's August holdout remains excluded and sealed.

## Frozen policy

At each state, a regularized logistic model estimates the probability that an
entry at that instant has positive 60m net value and reaches the favorable
30-bps barrier before the adverse 20-bps barrier. A Ridge model estimates the
remaining 60m net bps. Enter when probability is at least 0.65 and predicted
remaining net value is at least 2 bps. Cancel at the first state below 0.30;
otherwise remain pending until state two, then cancel if entry criteria fail.

Features are grouped into price path, taker flow and price response,
multi-timeframe context, structural space/extension, and the full frozen set.
No historical GOOD/BAD class is used as a model target.

## Economics and baselines

Primary utility is side-oriented 60m return from the candidate fill minus 14
bps. Stress cost is 20 bps. W12 is compared with ENTER_NOW, fixed waits of one,
two and three minutes, frozen W11 decisions, and NO_TRADE. Cancelled signals are
zero in net bps per original signal.

## Gate

W12 must be positive per executed trade and original signal, beat ENTER_NOW,
WAIT_2M and frozen W11 by at least 2 bps/signal, execute at least 25 signals and
20% of validation, retain at least 40% of historical GOOD, avoid at least 20%
of historical BAD, reduce median MAE by at least 20%, remain positive at 20-bps
cost, and pass 10,000 episode and temporal-block bootstrap comparisons.

## Restrictions

Python offline research only. No production, TypeScript, Brain, guards,
leverage, PM2, exchange, authenticated API, orders, Shadow, or Live changes.
`W12_READY_FOR_LIVE` remains `FALSE` regardless of the offline result.
