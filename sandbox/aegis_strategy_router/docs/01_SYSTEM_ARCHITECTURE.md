# System Architecture

## 1. Design objective

The system is a hypothesis arbiter, not a monolithic market predictor. It must
answer four separate questions:

1. What market setups are causally eligible now?
2. How strong is the evidence for each eligible setup?
3. What independent risks contradict or weaken each setup?
4. Is one setup sufficiently dominant to justify `ENTER`, or should the system
   `WAIT` or `SKIP`?

The architecture must make `NONE` a first-class outcome. Failure of hypothesis
A is not evidence for hypothesis B.

## 2. Architectural layers

### Layer 0: immutable market snapshot

One timestamped snapshot is shared by all components. It contains only values
known at the decision boundary. Every specialist receives the exact same
snapshot identity and feature-version hash.

Required metadata:

- `snapshot_id` and `signal_id` when signal-conditioned;
- exchange and local timestamps;
- symbol and proposed side;
- last closed bar for every timeframe;
- source-data completeness and gap indicators;
- feature schema/version/hash;
- model/config/git identifiers;
- reference price and intended decision horizon.

No component may query future bars, independently refresh data, or use a
different timestamp interpretation.

### Layer 1: context and location map

This layer describes the market without deciding an action:

- 1d/4h broad structure and location;
- 1h active regime and structural transition;
- 15m operational setup context;
- 5m/1m timing and immediate flow;
- recent support/resistance levels computed from prior data only;
- distance to the next level in both directions;
- trend age, extension, volatility, and range efficiency;
- BTC/market alignment and asset-specific residual movement.

It produces facts and uncertainty, not labels such as `GOOD_TRADE`.

### Layer 2: deterministic candidate generators

Each strategy owns explicit eligibility rules. A generator answers only:

> Is there enough structural evidence to ask this specialist for an opinion?

Generators reduce nonsensical evaluations. They do not estimate outcome
probabilities and must not use learned thresholds initially.

### Layer 3: specialist models

Each specialist operates only on its candidate population and estimates a
strategy-specific future path distribution. Specialists share an output
contract but may use different feature subsets and horizons.

Initial model family:

- regularized logistic for barrier ordering;
- regularized regression for MFE/MAE magnitude;
- shallow tree as an interpretability comparison;
- gradient boosting only after simple baselines show stable information.

No deep learning, reinforcement learning, or online retraining in the initial
program.

### Layer 4: independent critics

Critics identify conditions capable of invalidating multiple strategies:

- incomplete or stale data;
- volatility/liquidity shock;
- late entry or exhaustion;
- insufficient structural space;
- model uncertainty or out-of-distribution state;
- conflict among timeframes or specialists;
- correlated portfolio exposure, in a later phase.

A critic emits risk severity and evidence. It does not propose a direction.

### Layer 5: calibration

Raw probabilities from different specialists are not directly comparable.
Every model must be calibrated on a temporally separate calibration fold.

Calibration outputs:

- calibrated favorable/adverse probabilities;
- expected calibration error;
- reliability bins;
- prediction interval or uncertainty band;
- sample support near the current prediction;
- out-of-distribution score.

### Layer 6: deterministic router

The first router is deliberately not learned. It applies frozen eligibility,
calibration, dominance, risk, and abstention rules. This prevents a meta-model
from overfitting the errors of immature specialists.

The router may output:

- `ENTER(strategy, side, horizon)`;
- `WAIT(candidate_set, expiry)`;
- `SKIP(reason)`.

It must record all accepted and rejected hypotheses.

### Layer 7: lifecycle controller

The initial research scope ends at entry selection. A later, independent
sequential experiment may manage:

- pending confirmation;
- current thesis persistence;
- invalidation;
- exit;
- cooldown;
- new independent entry.

Direct `LONG -> SHORT` and `SHORT -> LONG` transitions are prohibited.

## 3. Dependency policy

The sandbox may import stable existing research utilities through adapters:

- causal candle aggregation and indicators from
  `src/aegis/research/live_entry_multitimeframe.py`;
- path reconstruction concepts from
  `src/aegis/research/entry_safety_gate_w11.py`;
- immutable signal and prospective collection schemas where compatible;
- existing bootstrap/report utilities only after contract verification.

The sandbox must not import:

- authenticated exchange adapters;
- TypeScript execution services;
- position mutation interfaces;
- PM2/runtime control;
- production configuration with financial authority.

## 4. Compatibility matrix

The router needs a frozen compatibility matrix rather than informal reasoning.

| Hypothesis A | Hypothesis B | Relationship |
|---|---|---|
| Trend continuation LONG | Pullback continuation LONG | Compatible; may reinforce |
| Trend continuation LONG | Breakout/retest LONG | Compatible if horizons overlap |
| Trend continuation LONG | Mean reversion SHORT | Opposed; require abstention or dominance |
| Breakout LONG | Transition reversal LONG | Potentially compatible but avoid double counting |
| Breakout LONG | Breakout SHORT | Mutually exclusive |
| Range mean reversion LONG | Range mean reversion SHORT | Mutually exclusive at same timestamp |
| Any strategy | Shock critic CRITICAL | Diagnostic initially; enforcement requires independent validation |
| Any strategy | Data quality invalid | Hard veto |
| Any strategy | OOD critical | Hard abstention |

Compatible specialists do not count as independent votes when they rely on the
same underlying evidence. The router groups correlated evidence families before
calculating confidence.

## 5. Non-functional requirements

- Deterministic replay from snapshot and version identifiers.
- No component reads outcome fields during inference.
- Complete explanation for every router action.
- Bounded runtime and memory in offline replay.
- No network dependency during model evaluation.
- Strategy packages can be removed independently.
- Router behavior remains unchanged when a disabled specialist is absent.
- All uncertainty or data failure degrades to `WAIT/SKIP`, never forced entry.
