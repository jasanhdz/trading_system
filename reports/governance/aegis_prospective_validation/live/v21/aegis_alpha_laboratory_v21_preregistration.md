# Aegis Alpha Laboratory V21 Preregistration

## Purpose

V21 asks a narrower question than V15 through V20: does any simple, causal,
economically explicit opportunity rule have stable net value before machine
learning is introduced? It is a research-only experiment. It cannot alter
Live, Shadow, exchange state, capital, guards, or model authority.

V20 showed why this ordering matters. Several families had attractive win
rates while remaining unprofitable because their uncommon losses were larger
than their common gains. V21 therefore treats net expectancy, profit factor,
MAE, drawdown, temporal stability, and cost sensitivity as primary evidence.
Accuracy and win rate are diagnostics only.

## Frozen Hypotheses

1. `CROSS_SECTIONAL_MOMENTUM` selects the two strongest side-adjusted 4-hour
   relative movers at each timestamp, requiring short-horizon continuation,
   volume, and taker-flow agreement. It uses the current TypeScript protection
   replay.
2. `EXTREME_REVERSAL` selects the two worst side-adjusted 4-hour extremes and
   requires a causal turn in return, acceleration, wick structure, and taker
   flow. It uses the already computed `LOCK_AT_5_ROE` profile.
3. `BREAKOUT_FLOW_FUNDING` requires breakout membership, range and volume
   expansion, taker direction and acceleration, and non-crowded funding. It
   uses the already computed `LOCK_AT_10_ROE` profile.
4. `FUNDING_BASIS_CARRY` requires historical futures mark, spot, executable
   basis, funding, and costs for both legs. No directional proxy may replace a
   missing delta-neutral leg. If those data are absent, the hypothesis is
   classified `DATA_GAP` rather than profitable or unprofitable.

The rule thresholds and exit assignments are frozen in
`config/experiments/aegis_alpha_laboratory_v21.yaml`. They may not be changed
after results are observed and still be called V21.

## Temporal Evidence

The evidence is split chronologically into discovery, validation, and a final
one-shot holdout. Events are spaced by at least 60 minutes for each
symbol/side/family to reduce overlapping-path inflation. The holdout may be
opened once because the complete rules and gates are frozen before execution.
It may not be reused to tune a V21 rule.

Passing requires sufficient samples in every period, positive validation and
holdout expectancy, positive holdout utility, holdout profit factor of at
least 1.10, mean holdout MAE no greater than 0.60%, temporal stability, and
outperformance of both no-trade and a period/side-matched random control.

## Model Boundary

No V21 model will be trained unless a simple rule first passes every gate.
Any future model is limited to abstention, path-risk estimation, and economic
ranking. It may not manufacture an edge that the underlying family does not
show. The first permitted baselines are logistic regression, histogram
gradient boosting, and quantile regression; deep learning and reinforcement
learning are outside V21.

## Data Limits

The hash-bound V14 source supplies causal price, volume, multi-timeframe,
regime, taker-flow and precomputed protected outcomes. Funding history is
available separately and is hash-bound for this run. Long historical open
interest, order-book imbalance, liquidation flow, and an executable spot-basis
leg are not available. These gaps are reported and never filled with zeros or
synthetic values.

## Safety

V21 performs local, read-only research. It has no exchange authority, makes no
network calls, exports no runtime model, starts no service, and cannot promote
itself. Live and Shadow remain unchanged.
