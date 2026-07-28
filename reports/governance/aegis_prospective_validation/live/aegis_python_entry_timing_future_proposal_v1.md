# Aegis Python Entry Timing Future Proposal V1

## Status and Scope

This document freezes a future research proposal. It does not modify the
current Python decision, TypeScript execution semantics, trading guards,
capital management, position sizing, leverage, exchange behavior, or runtime
configuration.

Current status:

- proposal state: `FUTURE_RESEARCH_PROPOSAL`;
- runtime authority: `NONE`;
- exchange authority: `NONE`;
- automatic promotion: `PROHIBITED`;
- current Live behavior changed: `NO`;
- current Shadow behavior changed by this document: `NO`.

The objective is to separate two questions that the current system partially
combines:

1. Is there a potentially favorable opportunity over the configured horizon?
2. Is the current closed candle a favorable time to enter that opportunity?

The proposal does not assume that adding more rules improves trading. It
requires prospective evidence against the unchanged current decision as the
control.

## Current Decision Path

The current production path is:

```text
aligned closed 5m candles for all 11 symbols
  -> 83 causal features
  -> current SHORT-only multi-output estimator
  -> scientific layers
  -> candidate construction
  -> global ranking and selection
  -> Python/TypeScript canonical decision contract
  -> TypeScript operational and exchange-safety checks
  -> original TypeScript sizing, execution, brackets, and position management
```

### Market and Feature Input

The Python service evaluates the canonical eleven-symbol universe using aligned
closed 5-minute candles. The `aegis-features-v2` contract contains 83 features,
including:

- returns over multiple horizons;
- candle body, range, and wick structure;
- volume changes, z-scores, and volume trend;
- ATR, realized volatility, and range expansion;
- EMA gaps, slopes, and trend stacks;
- persistence, momentum acceleration, trend strength, and choppiness;
- cross-sectional rank, breadth, dispersion, and concentration;
- BTC and ETH trend, volatility, and divergence context;
- breakdown and failed-breakdown proxies;
- distance to rolling highs and lows;
- room-to-fall and extension proxies;
- exhaustion, rebound, squeeze, reclaim, and immediate-reversal risks;
- consecutive red and green candle counts.

The same causal feature implementation is used by training and inference.

### Current Model Outputs

The current artifact contains one estimator with several predictive heads:

- LONG probability;
- SHORT probability;
- NEUTRAL probability;
- expected return;
- tail-risk probability;
- entry-quality probability;
- mean adverse excursion;
- q50 adverse excursion;
- q90 adverse excursion.

The artifact is structurally SHORT-only. Its directional heads use fixed
biases and no feature-dependent directional weights:

- LONG bias: `-8`;
- NEUTRAL bias: `-8`;
- SHORT bias: `+8`.

Consequently, the reported SHORT probability is approximately one for normal
successful inference. It is a side-authority mechanism, not a calibrated
probability that a short trade will be profitable.

The response currently contains one directional vote because the bundle
contains one estimator. This is not independent multi-model consensus.

### Current Scientific Layers

The layers use the model outputs as follows:

- direction: selects the model side above the frozen direction threshold;
- RV2: observes predicted tail-risk probability;
- TRRM: vetoes tail risk above the frozen maximum;
- QMAE: vetoes missing or excessive q90 adverse excursion;
- EQM: multiplies clean-entry probability by positive directional expected
  return;
- candidate construction: packages score, risk intent, regime, and reasons;
- global selection: ranks all eligible symbols and selects at most one per
  decision cycle.

For a SHORT candidate:

```text
expected_short_return = -model_expected_return
eqm_score = clean_probability * max(0, expected_short_return)
```

The current runtime regime label is descriptive context. It is not an
independent veto in the canonical layer result.

D3 is historical data and experimental discipline rather than a per-request
runtime gate. ECON1 is an offline economic evaluation program rather than a
separate online layer.

### TypeScript Responsibility

TypeScript validates the exact canonical Python contract and then applies
operational and exchange-safety controls. These include:

- Live mode and service authorization;
- exact model, bundle, configuration, and feature identities;
- no fallback inference;
- Python canonical `selected=true`;
- same-symbol position protection;
- trade-count, cooldown, loss, and liquidity controls;
- symbol execution state;
- exchange position duplication checks;
- wallet-based sizing under the original behavior;
- exchange precision and minimum-notional checks;
- bracket creation and confirmation;
- trailing, callback, recovery, and position management.

For current Phase O SHORT entries, secondary analytical TypeScript guards are
recorded in Shadow while hard operational safety remains enforced.

## Seven Current Conceptual Problems

### 1. No Independent Directional Committee

The system vocabulary can suggest that multiple independent models vote on
direction. The current artifact contains only one estimator and therefore one
directional vote. The pipeline is multi-output and layered, but it is not a
directional voting ensemble.

Risk:

- operators may overestimate consensus;
- reports may treat one vote as independent confirmation;
- adding vote-count thresholds would create false assurance.

Required future correction:

- describe the current object as a SHORT-only multi-output estimator with
  layered selection;
- reserve the word `committee` for genuinely independent members or define it
  explicitly as a pipeline committee;
- never fabricate additional votes.

### 2. SHORT Probability Is Not Profitability Confidence

Because the directional heads are fixed, `short_prob` is nearly constant and
does not express the probability that a short will be profitable.

The economically relevant outputs are:

- expected short return;
- entry-quality probability;
- tail risk;
- QMAE;
- final EQM score;
- global rank.

Risk:

- dashboards or guards may use `short_prob` incorrectly;
- a high number may be interpreted as high trade quality;
- apparent signal confidence may remain high even when expected edge is weak.

Required future correction:

- label the field as SHORT side authority or preserve it only for transport
  compatibility;
- expose a distinct, well-defined opportunity score;
- document calibration and outcome meaning for every probability.

### 3. Directional Vote Count Adds No Independent Evidence

`votes.short=1` is a serialization of one estimator result. It is not agreement
between independent models, horizons, seeds, or model families.

Risk:

- a two-vote rule cannot be satisfied honestly by the current artifact;
- duplicating the same decision to create two votes would manufacture
  evidence;
- legacy vote logic can conflict with the canonical Python selection.

Required future correction:

- TypeScript must use canonical selection rather than reinterpreting legacy
  vote counts;
- future independent votes require separately trained and validated members;
- correlation between members must be measured before treating them as
  independent evidence.

### 4. Candidate Confidence Is Mechanically Overstated

Candidate confidence is calculated from model disagreement. With one estimator,
disagreement is always zero, so confidence becomes one and uncertainty becomes
zero.

This value currently does not appear to determine canonical eligibility, but
it is semantically misleading.

Risk:

- telemetry can claim perfect confidence without supporting evidence;
- future consumers may accidentally use the field as a guard or sizing input;
- monitoring can conceal the absence of ensemble diversity.

Required future correction:

- report disagreement as `NOT_APPLICABLE_SINGLE_ESTIMATOR`;
- derive uncertainty only from validated quantities such as calibration,
  predictive dispersion, conformal intervals, or an actual ensemble;
- do not change selection until the replacement uncertainty measure has
  independent evidence.

### 5. Runtime Regime Is Partial

The current canonical regime classifier uses market-wide direction together
with local volatility and structure features. Its result is contextual and
does not veto canonical candidates.

The stateful factorized Regime V2 is stronger structurally because it separates
direction, volatility, and market structure and adds hysteresis. However, its
direction input still relies substantially on the common market direction.

Risk:

- a symbol-specific deterioration can be hidden by broader market direction;
- one categorical label can combine materially different contexts;
- regime telemetry may be mistaken for proven entry authorization.

Required future correction:

- retain separate global-market and symbol-specific direction axes;
- retain volatility and structure as separate axes;
- validate transition frequency, stability, and outcome separation;
- keep regime observational until it demonstrates incremental value over the
  current control.

### 6. Opportunity Quality and Entry Timing Are Not Explicitly Separated

The current horizon is twelve 5-minute candles. A candidate can correctly
identify a potential move within that horizon while entering on a locally poor
candle.

Examples:

- chasing an already extended decline;
- shorting close to support with little room left;
- entering before a rebound or reclaim;
- entering before a pullback that creates a materially better price;
- accepting excessive time underwater even when the final direction is right.

Risk:

- unnecessary MAE;
- avoidable stop-outs;
- poor reward-to-risk at the exact fill;
- correct opportunity forecasts producing poor realized trades.

Required future correction:

- add an explicit entry-timing research layer;
- compare immediate entry with delayed confirmation and invalidation;
- measure opportunity cost as well as avoided losses.

### 7. Python Selection and TypeScript Operational State Are Split

Python globally ranks the eleven symbols and normally selects one candidate per
cycle. TypeScript subsequently knows the authoritative wallet, positions,
cooldowns, exchange filters, and execution state.

This separation is intentional, but the Python snapshot does not represent all
of TypeScript's live operational state.

Risk:

- Python can select a symbol that TypeScript cannot execute;
- the second-best eligible candidate is not necessarily reconsidered;
- research selection counts may differ from executable opportunity counts.

Required future correction:

- preserve TypeScript as operational authority;
- journal both Python rank and TypeScript disposition under one correlation
  identity;
- study whether an operationally unavailable winner should allow the next
  Python-ranked candidate, without implementing that behavior before evidence
  and explicit authorization.

## Proposed Python Entry Timing Policy

The preferred future architecture places entry timing in Python:

```text
current model and scientific layers
  -> eligible opportunity candidate
  -> Entry Timing Policy
  -> confirmed, waiting, invalidated, or expired
  -> global selection
  -> canonical TypeScript contract
```

TypeScript should not independently recalculate this strategy logic. It should
continue to validate the contract, apply operational safety, execute, and
manage positions.

### Proposed Setup State

Each candidate timing lifecycle should have one of:

- `CANDIDATE_SEEN`;
- `WAITING_FOR_RETEST`;
- `TIMING_CONFIRMED`;
- `INVALIDATED`;
- `EXPIRED`.

A stateless guard is insufficient for retest timing. If the current candidate
is rejected, the model selection may disappear on the next candle even though
the original setup is still waiting for confirmation. A bounded, durable,
causal setup identity is therefore required.

The identity must derive from:

- symbol;
- side;
- model and configuration identities;
- closed market timestamp;
- feature-vector hash;
- original candidate hash.

It must not depend on future returns, fills, process identity, or randomness.

### SHORT Continuation Retest Hypothesis

A higher-quality continuation may require:

- bearish global context;
- bearish symbol-specific trend;
- short EMA alignment;
- a prior breakdown or established downward structure;
- a controlled pullback toward an EMA or broken level;
- bearish rejection from the retest;
- sufficient remaining room to the next support;
- compatible volume;
- acceptable QMAE and tail risk;
- no strong reclaim, squeeze, or reversal evidence.

### SHORT Exhaustion Avoidance Hypothesis

An immediate short may be delayed or invalidated when the closed-candle
evidence indicates:

- excessive downside extension;
- too many consecutive red candles;
- little room to the next support;
- strong lower-wick reclaim;
- failed breakdown;
- squeeze plus reclaim;
- rebound risk;
- extreme volatility expansion;
- QMAE or tail risk inconsistent with a clean entry.

This is a hypothesis, not an authorized Live rule.

### Confirmed Breakdown Hypothesis

A breakdown candidate may require:

- a closed break below a causal reference level;
- compatible volume and body structure;
- no immediate reclaim;
- either continuation confirmation or a failed retest;
- continued positive expected short return after confirmation;
- enough remaining room after costs.

### LONG Research Boundary

The current canonical artifact is SHORT-only. LONG timing must not be derived
from its fixed directional heads.

Future LONG research requires:

- an independently trained LONG opportunity model;
- directionally valid expected return;
- LONG-specific MAE and tail-risk validation;
- bullish symbol-specific context;
- pullback and reclaim hypotheses;
- separate offline and prospective Shadow evidence.

The current experimental LONG observer remains observational and is not Live
authority.

## Shadow Experiment Design

Every canonical cycle should produce parallel, non-executing alternatives:

1. `CONTROL_IMMEDIATE`: unchanged current selection and entry timestamp;
2. `CONTEXT_FILTERED`: current candidate subject to factorized context;
3. `TIMING_RANKED`: timing score changes ranking but not the model outputs;
4. `WAIT_RETEST`: bounded delayed entry after a causal retest confirmation;
5. `EXHAUSTION_AVOID`: counterfactual rejection of extended entries.

Each alternative must preserve:

- original candidate identity;
- original feature vector;
- original model outputs;
- original layer outputs;
- proposed timing state and reasons;
- hypothetical entry timestamp and price;
- expiration or invalidation reason;
- outcome horizon and cost assumptions.

### Required Metrics

Evaluate:

- net return after frozen costs;
- MAE and MFE;
- time underwater;
- stop-loss incidence;
- profit factor;
- expectancy and confidence interval;
- entry delay;
- avoided losses;
- missed winning opportunities;
- signal and symbol concentration;
- performance by symbol;
- performance by regime axes;
- first-half and second-half stability;
- operational executability rate.

### Evidence Discipline

At minimum:

- discovery and validation evidence must be separated;
- outcomes must be non-overlapping or embargoed;
- purged walk-forward validation must be used for trained ranking;
- the unchanged current selection must remain the control;
- at least 300 independent selected episodes are required overall;
- at least 50 independent episodes per included symbol are required;
- evidence must span at least seven temporal blocks;
- symbol concentration must remain bounded;
- mean MAE must satisfy the frozen prospective threshold;
- profit factor must exceed one;
- the 95% expectancy interval must have a positive lower bound;
- both temporal halves must remain positive;
- no automatic training or promotion is permitted;
- an owner-authorized exact artifact promotion is required.

These numbers are research minimums, not guarantees of profitability.

## Integration Boundary

### During Shadow

Python should:

- calculate setup and timing observations from the canonical closed-candle
  batch;
- persist append-only signal and outcome evidence;
- expose optional, non-authoritative metadata such as
  `entry_timing_shadow`;
- leave canonical `selected` unchanged;
- never load exchange mutation credentials;
- never submit an exchange request.

TypeScript may:

- persist the opaque timing metadata with the canonical correlation identity;
- compare the counterfactual timing result with actual execution and lifecycle
  outcomes;
- continue ignoring the timing result for entry authorization.

TypeScript should not:

- recalculate Python indicators from a separate candle stream;
- create strategy votes from the timing labels;
- enforce the timing result while its mode is Shadow;
- alter brackets, trailing, callback, sizing, or capital behavior.

### Future Live Promotion

If and only if the evidence contract passes, the timing policy should become
part of Python's canonical selection before TypeScript receives
`selected=true`.

The promotion should:

- use an exact versioned configuration and artifact hash;
- preserve the current model outputs unchanged;
- update the canonical decision contract explicitly;
- require parity and fail-closed tests;
- retain TypeScript operational and exchange-safety authority;
- avoid a second strategy implementation in TypeScript;
- require explicit owner authorization.

A YAML mode may support `SHADOW` and `LIVE`, but changing the word alone must
not authorize production. `LIVE` must additionally require an exact promotion
record and validated hashes.

## Why Python Is the Preferred Location

Python already owns:

- aligned closed market data;
- the canonical 83-feature vector;
- model inference;
- tail risk and QMAE;
- entry-quality scoring;
- cross-symbol ranking;
- prospective research journals.

Implementing the same logic in TypeScript would create:

- duplicate indicator implementations;
- timestamp and candle-alignment drift;
- competing strategy authorities;
- harder parity validation;
- increased risk of one side changing without the other.

TypeScript should remain responsible for:

- contract validation;
- runtime and exchange safety;
- wallet-based sizing;
- precision and minimum-notional checks;
- order submission;
- brackets;
- trailing and callback;
- reconciliation and recovery;
- position management.

## Implementation Phases

### Phase 1: Terminology and Telemetry Correction

- document that the current artifact is one SHORT-only estimator;
- mark vote consensus as not applicable;
- mark disagreement confidence as not applicable for one estimator;
- distinguish side authority, opportunity quality, and entry timing.

No trading behavior changes.

### Phase 2: Entry Timing Shadow Observer

- implement the bounded setup state machine in Python research code;
- use existing causal features first;
- add no new indicator unless a missing concept is demonstrated;
- journal all control and counterfactual alternatives;
- test restart recovery, deduplication, and causal timestamps.

No trading behavior changes.

### Phase 3: Historical Replay

- replay the hypotheses using frozen data and costs;
- use purged walk-forward evaluation;
- reject hypotheses that only improve discovery data;
- analyze each symbol and regime separately.

No trading behavior changes.

### Phase 4: Prospective Shadow

- collect the frozen minimum evidence;
- monitor drift, data quality, feature variability, and regime collapse;
- compare actual fills with hypothetical delayed entries;
- publish owner-review evidence.

No automatic promotion.

### Phase 5: Explicit Promotion Decision

- either reject the proposal;
- continue collecting evidence;
- promote only ranking;
- promote only exhaustion avoidance;
- or promote the complete timing policy.

Any promotion is a separate, owner-authorized strategy change.

## Non-Goals

This proposal does not authorize:

- forced trades;
- fabricated votes;
- a new capital cap;
- changed wallet fractions;
- changed leverage;
- changed stops, take profits, trailing, or callback;
- changes to TypeScript execution semantics;
- automatic model retraining;
- automatic threshold adaptation;
- automatic Live promotion;
- USD100 activation.

## Recommended Decision

Proceed with the terminology and telemetry correction and build the timing
policy as a Python Shadow observer. Do not create a new TypeScript strategy
guard and do not change the canonical Live selection until prospective
evidence demonstrates incremental benefit over the unchanged control.

The central hypothesis is:

> The current system may identify a valid SHORT opportunity over twelve bars
> but still enter on a poor candle. Separating opportunity quality from entry
> timing may reduce MAE and avoid exhausted entries without changing the
> underlying model.

This hypothesis must be tested rather than assumed.
