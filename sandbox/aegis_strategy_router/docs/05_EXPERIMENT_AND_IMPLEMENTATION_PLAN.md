# Experiment and Implementation Plan

## Program status

Phase 0 is approved under the frozen decision record. Phase 1 technical
acceptance is met. The governance amendment in
`12_PHASE2_GOVERNANCE_AMENDMENT.md` authorizes Phase 2 implementation while
fresh collection continues, but prohibits Phase 2 validation and specialist
implementation until their separate gates pass.

## Phase 0: design freeze and data audit

Deliverables:

- approve specialist catalog and critic catalog;
- approve signal-conditioned versus independent scope;
- inventory causal 1m/5m/15m/1h/4h/1d coverage;
- inventory taker, L2, funding/OI, BTC/ETH, and structural-level support;
- quantify candidate event rates using rules only;
- verify collection feasibility for the frozen fresh split;
- verify the frozen effective-sample gates per specialist;
- freeze snapshot, assessment, and decision contracts;
- threat model all imports and prove no financial capability.

Stop conditions:

- no fresh validation window;
- insufficient causal warmup/coverage;
- candidate rules depend on hindsight levels;
- strategy definitions cannot be made mutually interpretable;
- sandbox cannot remain isolated.

## Phase 1: shared feature and snapshot adapters

Implementation allowed only after Phase 0 approval.

Tasks:

- create immutable snapshot domain objects;
- adapt existing causal multi-timeframe features;
- add 4h/1d coverage and warmup flags;
- create causal level/structure adapter;
- prohibit undeclared feature access;
- hash feature schema and source versions;
- add leakage and timestamp contract tests;
- build deterministic replay fixtures.

No specialist model is trained in this phase.

Acceptance:

- same input produces byte-equivalent snapshot;
- every feature has availability timestamp and owner;
- future-column injection tests fail closed;
- missing higher timeframe remains unknown, never false confirmation.

## Phase 2: deterministic candidate generators

Implement one package per strategy with rules only.

Order:

1. trend continuation;
2. pullback continuation;
3. breakout/retest;
4. range mean reversion;
5. regime transition/reversal.

For each generator:

- produce candidate and ineligibility reasons;
- produce structural invalidation;
- record all continuous features before thresholding;
- run general-market event-rate audit;
- run signal-conditioned coverage audit;
- compare LONG/SHORT and symbols without selecting winners;
- verify no future levels or outcome fields.

Stop any generator with tiny or unstable populations. Do not loosen rules after
validation to manufacture sample size.

## Phase 3: rules-only strategy baselines

Before models, evaluate each rule-defined candidate population.

Metrics:

- directional correctness versus empirical baseline;
- favorable/adverse first;
- gross path return;
- MFE, MAE, MFE > MAE;
- time-to-event;
- tail MAE and expected shortfall;
- candidate coverage;
- per-symbol/side/period stability;
- cost-adjusted results as secondary diagnostics.

Purpose:

- determine whether candidate definitions isolate distinct behavior;
- establish a baseline every model must beat;
- reject strategies whose rules do not define a coherent population.

## Phase 4: specialist training

Train specialists independently. Each specialist has its own preregistration,
feature allowlist, label family, split, and verdict.

Model sequence:

1. constant/empirical baseline;
2. regularized logistic barrier model;
3. regularized MFE/MAE regressions;
4. shallow tree;
5. boosting only if incremental validation value is justified.

Required ablations:

- structure/location only;
- trend/extension only;
- volatility/volume;
- flow/response;
- cross-market context;
- full specialist.

Promotion requirements per specialist:

- beats its own rule-only baseline outside sample;
- calibrated probability and acceptable ECE/Brier score;
- directional/path improvement is economically and statistically material;
- no dependency on one symbol, side, or fold unless preregistered;
- stable feature contribution and no leakage evidence;
- sufficient sample support and OOD coverage;
- survives temporal/block bootstrap and multiple-comparison control.

Failed specialists remain documented but are excluded from routing.

## Phase 5: critics

Implement critics first as deterministic diagnostics. A critic becomes a hard
veto only if an independent experiment proves that veto improves the specified
primary objective without unacceptable opportunity destruction.

Required outputs:

- severity: INFO/CAUTION/HIGH/CRITICAL;
- evidence and missing data;
- affected specialists;
- hard-veto eligibility;
- calibration/false-positive audit.

Data quality and critical OOD may fail closed by design. Market-risk critics
require evidence before enforcement.

## Phase 6: deterministic static router replay

Prerequisites:

- at least two validated specialists, otherwise routing is not meaningful;
- all included specialists calibrated;
- compatibility and evidence-correlation matrices frozen;
- critics frozen;
- router utility and dominance margins preregistered.

Baselines:

- `NO_TRADE`;
- original Aegis decision;
- best validated single specialist;
- simple highest calibrated probability;
- simple compatible-consensus rule;
- deterministic full router.

Primary router requirement:

The router must outperform the best included single specialist, not merely the
weak original baseline. It must also demonstrate that abstention and conflict
handling add value.

Router diagnostics:

- chosen/rejected hypothesis counts;
- winner-runner-up margin;
- disagreement rate;
- abstention rate;
- strategy concentration;
- specialist regret;
- false abstention and harmful selection;
- MFE/MAE and tail risk;
- turnover/churn assumptions;
- per-symbol/side/time stability.

Phase 6 `WAIT` is terminal. It does not create a pending episode, inspect a
future snapshot, or simulate delayed entry.

## Phase 7: pending confirmation experiment

Only after static routing passes.

Evaluate:

- `ENTER_NOW` versus causal `WAIT`;
- remaining MFE/MAE from actual confirmation time;
- consumed movement;
- expiry and cancellation;
- confirmation latency;
- no reentry/no flip;
- event-level dependence.

This phase must beat static routing, not repair a static router that failed.

## Phase 8: learned meta-router consideration

Only after sufficient out-of-fold specialist predictions exist.

Compare a small regularized meta-model against the deterministic router. Its
inputs are specialist outputs, critic outputs, uncertainty, and broad context;
it does not receive raw unrestricted features initially.

Reject the meta-router if:

- gains are calibration artifacts;
- feature importance is unstable;
- it selects one specialist almost exclusively without added value;
- it fails temporal transfer;
- it increases tail MAE or removes abstention.

## Phase 9: prospective observation

Only after fresh validation and sealed holdout pass.

For each real signal, log without influencing trading:

- immutable snapshot;
- candidate strategies;
- specialist outputs;
- critics;
- router action and reasons;
- baseline action;
- future counterfactual outcome.

No blocking, entry, exit, sizing, or financial mutation.

## Phase 10: shadow and live

Not authorized by this plan. Shadow requires a separate approval and safety
review. Live requires a later independent decision after prospective evidence.

## Testing strategy

Required suites:

- unit tests for every candidate rule and critic;
- property tests for LONG/SHORT symmetry;
- timestamp/leakage tests;
- snapshot determinism tests;
- model-feature allowlist tests;
- calibration tests;
- overlapping episode/split tests;
- router conflict and abstention tests;
- no-direct-flip lifecycle tests;
- sandbox import safety tests;
- deletion/integration-boundary test.

## Documentation generated by every run

- preregistration and config hash;
- data audit and split manifest;
- feature dictionary;
- candidate registry;
- hypothesis registry including failures;
- specialist model cards;
- calibration and ablation reports;
- router decision ledger;
- per-symbol/side/fold results;
- bootstrap/FDR results;
- cost/latency stress results;
- verdict JSON with explicit flags;
- sealed-holdout status.
