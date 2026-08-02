# Aegis Adverse-Trend Learning Shadow V1

## Purpose

This document freezes a causal Shadow experiment for learning from weak entries,
large adverse excursion, and persistent directional moves. It does not change
the canonical Python decision, TypeScript guards, execution, capital, sizing,
leverage, brackets, trailing, callback, or position management.

- mode: `SHADOW`;
- runtime authority: `OBSERVATIONAL_ONLY`;
- exchange authority: `NONE`;
- automatic online learning: `PROHIBITED`;
- automatic averaging or pyramiding: `PROHIBITED`;
- automatic Live promotion: `PROHIBITED`.

## ADA Case Reconstruction

The original bot opened an ADAUSDT short on 2026-08-01 around 04:30 UTC near
0.1704. The canonical estimator selected SHORT, and TypeScript allowed the
entry because the analytical entry-quality, event-risk, probe, and regime
controls were observational at that time.

The same decision cycle contained materially adverse Shadow evidence:

- setup grade `WEAK`;
- entry-quality score approximately `0.00006155`;
- tail-risk score approximately `0.53742`;
- event-risk result that would have denied a weak setup;
- probe result that would have denied excessive tail risk;
- high-volatility regime warning;
- regime-avoid counterfactual;
- no confirmed timing setup.

Public closed-candle reconstruction shows:

- ADA fell about 0.53% over the 30 minutes preceding the entry;
- ADA was nevertheless stronger than BTC over the preceding 2-4 hours;
- ADA rose about 1.00% after 6 hours, 1.88% after 12 hours, and 7.81% after
  24 hours while BTC remained comparatively stable;
- maximum adverse price excursion for the short reached about 9.57% before
  leverage;
- the new causal observer classifies the entry candle as
  `CONFLICTING_TRANSITION` and would have produced
  `WAIT_CONFIRMATION_COUNTERFACTUAL`;
- it first detects `UPWARD_PRESSURE` at 04:50 UTC and full
  `UPWARD_ACCELERATION` at 07:10 UTC.

These results distinguish a poor entry context from a claim that the model was
random. The canonical estimator selected a valid artifact-defined candidate,
but the promoted production policy did not enforce several contrary Shadow
signals. The timing and adverse-regime evidence therefore warrant prospective
study.

## Manual Pyramiding Boundary

Adding to a losing short lowers the average entry price but increases notional
exposure precisely while the market is contradicting the original thesis. A
high historical recovery rate cannot bound the loss from a persistent squeeze.
The ADA episode is an example of that tail risk.

The local bot journal contains the original automated ADA quantity, but not the
complete sequence of owner-initiated additions. Consequently:

- the final local lifecycle is not a clean label for the automated policy;
- manual additions must be marked as intervention-contaminated evidence;
- the automated entry result and the manual management result must be evaluated
  separately;
- no model may infer that averaging was beneficial without exact fill, quantity,
  timestamp, fee, funding, and counterfactual no-add evidence;
- the Shadow observer never emits an `ADD` authorization.

## Causal Observer

`directional_acceleration_shadow` uses only values available at the latest
closed five-minute candle. It records separate upward and downward evidence:

1. multi-horizon return alignment;
2. directional candle persistence;
3. EMA trend-stack alignment;
4. EMA slope alignment;
5. ATR-scaled impulse;
6. relative divergence from BTC;
7. volume confirmation;
8. range and trend expansion;
9. causal rolling-level break.

It also records short-chase and reversal flags already present in
`aegis-features-v2`. Its states are:

- `UPWARD_ACCELERATION`;
- `UPWARD_PRESSURE`;
- `DOWNWARD_ACCELERATION`;
- `DOWNWARD_PRESSURE`;
- `CONFLICTING_TRANSITION`;
- `BALANCED`.

For a short candidate, upward pressure or acceleration produces observational
`DO_NOT_ENTER_COUNTERFACTUAL` and `DO_NOT_ADD_COUNTERFACTUAL`. A transition or
short-chase condition produces `WAIT_CONFIRMATION_COUNTERFACTUAL` and
`DO_NOT_ADD_COUNTERFACTUAL`. Other states never authorize adding; they report
`INSUFFICIENT_EVIDENCE_TO_ADD`.

## Preliminary Replay

The thresholds were frozen before replaying the local read-only candle store.
The first replay covered 94,908 symbol evaluations across all eleven symbols
from 2026-06-17 through 2026-07-17.

The observer detected the ADA event, but its standalone persistent-move
precision was not sufficient for Live authority:

- upward precision: approximately 14.1%;
- upward recall: approximately 18.5%;
- downward precision: approximately 15.9%;
- downward recall: approximately 20.6%.

For ADA, upward acceleration was associated with a larger future maximum
upward excursion than balanced observations, but that is not enough to prove a
profitable blocker. The replay is diagnostic evidence, not validation.

## Learning Design

The observer now journals a future outcome for every symbol/candle after the
frozen horizon, including terminal return, maximum upward/downward excursion,
time-side persistence, and persistent-move labels. This enables a later hazard
model to answer narrower questions than the current entry model:

- probability of a persistent upward squeeze against a short;
- probability of a persistent downward cascade against a long;
- expected adverse excursion conditional on current regime and timing;
- whether waiting for confirmation reduces MAE without eliminating too many
  winners.

No online weight update is allowed. Training must use immutable snapshots,
purged walk-forward folds, embargoed outcomes, symbol-level reporting, and a
held-out final period. Manual-intervention lifecycles must be excluded or
explicitly modeled as contaminated.

## Promotion Criteria

Any future Live proposal requires, at minimum:

- comparison against unchanged canonical selection;
- at least 300 non-overlapping selected episodes overall;
- at least 50 episodes per included symbol;
- seven temporal blocks;
- positive incremental expectancy with a positive 95% lower confidence bound;
- measurable MAE and time-underwater improvement;
- explicit accounting for missed winners;
- stable performance in both temporal halves;
- bounded symbol concentration;
- zero unexplained interaction with TypeScript execution and protection;
- exact artifact hashes and separate owner authorization.

The experiment must remain Shadow when these requirements are not met.

## Immediate Interpretation

The system should learn two separate lessons from ADA:

1. weak short entries with conflicting timing, high tail risk, and adverse
   regime evidence require prospective counterfactual evaluation;
2. averaging a losing position is a management intervention with nonlinear
   tail risk, not proof that the original signal was good.

The current implementation provides evidence collection for both questions
without changing production decisions.
