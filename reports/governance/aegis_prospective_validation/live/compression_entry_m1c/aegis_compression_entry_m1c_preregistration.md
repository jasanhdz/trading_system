# Aegis Compression Entry M1C - Preregistration

## Purpose

M1C tests one disclosed hypothesis from M1B: Compression LONG contained gross
information and lower-MAE subsets, but did not remain profitable after costs.
M1C asks whether a causal pullback-and-reclaim entry and multi-timeframe context
can improve entry quality without mining the already opened validation period.

## Primary Hypothesis

The primary challenger is fixed as `PULLBACK_RECLAIM__CURRENT_TS_240`. After a
frozen M1A Compression LONG event, it observes at most five completed 1-minute
candles. It confirms only when a candle pulls back to the original event close,
closes at or above the original prior high and closes above its own open. Entry
is the next 1-minute open. No confirmation means no trade.

Immediate entry remains a baseline. Two alternative exits are diagnostics and
cannot replace the primary exit after validation results are known.

## Feature and Model Boundary

M1C retains the 23 M1B causal features and adds 15 fixed multi-timeframe
features spanning 5 minutes, 15 minutes, 1 hour and 4 hours. No feature may use
data after its entry-confirmation cutoff. Missing, reordered, extra or
non-finite features fail closed.

Three intentionally simple estimators answer separate questions:

1. calibrated probability of positive net outcome;
2. q90 adverse-excursion risk;
3. expected net utility.

Training, probability calibration and retrospective validation remain
temporally separated. Seeds, features and policy quantiles are frozen.

## Economic Boundary

The primary cost is 14 bps round trip. Eight bps is optimistic attribution,
20 bps is stress and zero cost is explanatory only. Funding is included.
Profit without costs cannot pass the gate.

The gate requires positive net expectancy, positive day-block uncertainty
lower bounds, profit-factor lower bound above one, mean MAE at most 0.5%, at
least two positive temporal thirds, controlled symbol concentration, positive
stress expectancy and stability across predefined direction and volatility
regimes.

## Contamination and Deployment

M1B already exposed data through July 2026. M1C retrospective validation has no
promotion authority regardless of its result. It may only justify passive
fresh-forward collection beginning 2026-08-13. Shadow or Live promotion
requires new evidence and separate authorization.

This experiment changes no runtime, TypeScript behavior, Shadow process,
capital rule, order path or exchange state. Exchange calls and mutations are
zero.
