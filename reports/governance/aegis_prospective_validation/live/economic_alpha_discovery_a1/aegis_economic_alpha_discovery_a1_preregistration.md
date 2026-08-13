# Aegis Economic Alpha Discovery A1 - Preregistration

## Purpose

A1 changes the research order. It does not ask machine learning to rescue a
negative candidate family. It first tests whether three economically distinct
mechanisms contain enough stable gross information to survive realistic costs.

The mechanisms are trend acceptance, extreme reversal and carry convergence.
Each is evaluated independently for LONG and SHORT at fixed horizons of 1, 4,
8, 12 and 24 hours.

## Causal State

The state grid is 15 minutes and is constructed only from completed public
one-minute Spot, USD-M Futures, mark-price and funding archives. Entry is the
next 15-minute open after the state closes. Future paths retain one-minute
resolution for MAE, MFE and funding.

No order-book, liquidation, news or whale feature is imputed because causal
historical sources with sufficient provenance are not currently available.

## Economic Mechanisms

`TREND_ACCEPTANCE` requires sustained breakout acceptance, persistent volume,
side-aligned taker flow and relative strength. It is not a one-candle breakout.

`EXTREME_REVERSAL` requires a training-defined extreme extension, exhaustion of
the prior flow and an actual reclaim. It is not RSI oversold or overbought by
itself.

`CARRY_CONVERGENCE` requires extreme basis, side-favorable funding and observed
basis convergence. It tests a cross-market mechanism rather than candle shape.

All scores and eligibility rules are frozen before validation. At each
timestamp, mechanism and side, only the highest-ranked eligible symbol is
selected. A daily-spaced view is primary to control dependence and turnover.

## Economic Gate

The primary round-trip cost is 14 bps. A mechanism must show at least 42 bps
gross expectancy, positive primary and 20-bps stress expectancy, positive
day-block uncertainty lower bounds, stable temporal thirds, broad symbol
participation and superiority to random and simple momentum/reversal controls.

Accuracy and win rate cannot pass the gate. Zero-cost results are attribution
only.

## Contamination

History through July 2026 has already influenced earlier Aegis experiments.
This validation can reject mechanisms or justify a separately preregistered
modeling study, but cannot authorize Shadow or Live. Any modeling, threshold
change or deployment requires a new experiment and new evidence.

No runtime, service, order, capital, model or TypeScript behavior is changed.
