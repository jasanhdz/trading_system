# Aegis Specialized Committee V2.1 Preregistration

## Authority

- Experiment: `aegis-specialized-committee-v21-shadow-01`
- Mode: `SHADOW`
- Runtime authority: `OBSERVATIONAL_ONLY`
- Live or exchange authority: `NONE`
- Automatic training or promotion: `PROHIBITED`
- Parent control: current canonical SHORT selection
- Parent observer: Committee V2 remains unchanged in `SHADOW`

This document freezes the V2.1 hypothesis before fitting or evaluating it.
V2.1 cannot change TypeScript decisions, guards, capital, leverage, sizing,
orders, positions, or the current Python decision.

## Problem

Committee V2 treated every nonzero reversal proxy as an equal Boolean veto.
Replay showed that this assumption was false: the meaning of a proxy depended
on its magnitude, the other proxies present, symbol, and market regime. Several
flags named as risks were associated with favorable outcomes in part of the
observed sample.

V2.1 therefore tests a narrower claim:

> A calibrated model of causal magnitudes and preregistered interactions can
> rank the risk of a net-negative 12-bar SHORT outcome better than the V2
> Boolean OR rule and the unfiltered canonical control.

This is a testable risk-ranking hypothesis, not a profitability claim.

## Outcome

The primary label is:

`adverse = net_return_after_0.10_percent_round_trip_cost <= 0`

The entry price is the signal close. The outcome price is the close after 12
fully closed 5-minute bars. MAE, MFE, worst-decile return, and coverage are
secondary diagnostics. The model cannot use future data, fill outcomes,
execution results, or post-entry fields.

## Model

The preregistered model is an L2-regularized logistic regression over:

- causal continuous feature magnitudes;
- the current control score, QMAE q90, tail probability, and TRRM
  compatibility;
- one-hot symbol and factorized direction, volatility, and structure regimes;
- the exact pairwise and context interactions listed in
  `config/experiments/aegis_committee_v21_preregistered_v1.yaml`.

The base model is fitted only on the training interval. A separate Platt
logistic calibrator is fitted only on the calibration interval. Model
coefficients, standardization values, calibrator coefficients, feature order,
and the 70th-percentile calibrated-risk threshold are exported to a
hash-addressed JSON artifact. Runtime inference is deterministic and cannot
train.

The calibrated value means:

`estimated probability of a net-negative 12-bar SHORT outcome`

It is not a directional vote, model consensus, certainty of loss, or
authorization to trade.

## Frozen Time Splits

- Training: `2026-05-01T00:00:00Z` through `2026-06-20T23:55:00Z`
- Calibration: `2026-06-21T00:00:00Z` through `2026-07-04T23:55:00Z`
- Diagnostic only: `2026-07-05T00:00:00Z` through
  `2026-07-11T09:20:00Z`
- Prospective evidence: first complete cycle after the exact V2.1 artifact and
  runtime configuration are activated

The diagnostic interval may reveal implementation or gross generalization
failure, but it cannot authorize promotion because its outcomes predate this
preregistration. Prospective records cannot be reused for refitting this
artifact.

## Counterfactual Policy

For a canonical SHORT selected by the existing control:

- calibrated risk at or below the frozen calibration threshold:
  `ENTER_NOW`;
- calibrated risk above the threshold: `WAIT_CONFIRMATION`.

For a symbol not selected by the control: `DO_NOT_ENTER`.

`WAIT_CONFIRMATION` means abstention only. V2.1 does not define delayed entry
and may not create one. The threshold is the 70th percentile of calibrated
risk on the calibration split, preserving a comparable coverage target
without manually choosing a profitable threshold after looking at outcomes.

## Calibration And Incremental Evidence

Reports must include:

- Brier score versus the constant calibration base-rate predictor;
- expected calibration error over 10 fixed-width probability bins;
- observed adverse rate by predicted-risk band;
- calibration slope and intercept;
- retained coverage;
- paired net-return delta versus canonical control;
- MAE, worst-decile return, temporal-block stability, and confidence interval;
- V2 versus V2.1 paired comparison.

No promotion claim is permitted unless prospective Shadow evidence has at
least 300 globally non-overlapping episodes, the 95% lower confidence bound of
paired incremental value is positive, calibration and risk ordering pass, MAE
does not increase, worst-decile performance does not degrade, and at least
three of four chronological blocks are positive. Symbol-specific conclusions
require at least 50 non-overlapping episodes for that symbol.

## Prohibitions

V2.1 must not:

- replace or modify Committee V2;
- affect Live execution;
- fabricate directional votes;
- interpret one estimator as a committee majority;
- tune thresholds from diagnostic or prospective outcomes;
- train online;
- auto-promote;
- alter current model, features, committee, guards, capital, sizing, leverage,
  or exchange state.

Any future V2.1 change requires a new versioned preregistration and new
prospective evidence.
