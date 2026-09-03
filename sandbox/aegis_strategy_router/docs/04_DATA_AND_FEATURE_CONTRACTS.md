# Data, Labels, and Feature Contracts

## 1. Statistical units

The primary unit is a strategy candidate episode, not a candle evaluation.

Identifiers:

- `market_snapshot_id`: one causal state at timestamp t;
- `signal_episode_id`: one original Aegis signal when conditioned;
- `candidate_episode_id`: one strategy candidate derived from a snapshot;
- `pending_episode_id`: all repeated WAIT evaluations for one candidate;
- `position_episode_id`: reserved for later position-management research.

Multiple specialists may evaluate the same snapshot. They are correlated and
must remain grouped in bootstrap and split logic.

## 2. Snapshot timing contract

For every timeframe, only the last fully closed candle may be used. A signal at
10:07 cannot use the 10:05-10:10 5m close or the still-open 1h candle.

Required timestamps:

- event/open/close time from exchange data;
- signal timestamp;
- feature cutoff timestamp;
- local receive timestamp when prospectively available;
- dataset build timestamp;
- timezone/unit declarations.

All joins use backward-as-of semantics with explicit exact-match rules.

## 3. Feature families

### Price and path

- returns over causal windows;
- normalized velocity and acceleration;
- path efficiency;
- favorable/adverse excursion observed so far;
- prior impulse and pullback geometry;
- consumed versus remaining movement.

### Structure and location

- causal swing highs/lows;
- HH/HL/LH/LL state;
- break/reclaim/retest state;
- distance to recent and higher-timeframe levels;
- range edge, midpoint, width, age, and touch count;
- invalidation distance and available favorable space.

### Trend and extension

- EMA7/25/99 slopes and directional extensions;
- trend age;
- cumulative move in ATR;
- directional RSI extension/remaining room;
- slope and momentum decay.

### Volatility and volume

- ATR and percentile;
- realized volatility and expansion/contraction;
- range shock;
- volume ratio and log-volume z-score;
- contextual 15m/1h volume state.

### Flow and response

- taker buy/sell imbalance;
- directional delta and persistence;
- flow velocity/acceleration;
- price impact per favorable/adverse flow;
- flow without price response;
- absorption/replenishment proxies when data supports them.

### Cross-market

- BTC/ETH returns and regime;
- beta-adjusted residual return;
- relative strength and cross-sectional rank;
- breadth, dispersion, and correlation regime.

### Data quality and uncertainty

- missing/warmup flags;
- maximum gap;
- source coverage;
- stale-data age;
- L2 reconstruction validity;
- model support/OOD features.

## 4. Feature ownership

Shared features are computed once by a versioned feature engine. Specialists
declare an allowlist and cannot silently read undeclared columns.

Forbidden as features:

- future MFE/MAE;
- future barriers;
- realized exit/PnL;
- post-decision classifications;
- labels from another specialist;
- validation/holdout identifiers that encode time outcome;
- manually corrected hindsight levels.

## 5. Strategy-specific labels

No specialist is trained to reproduce its candidate rules. The target measures
what happened after eligibility.

Shared label components:

- first favorable/adverse barrier from actual candidate timestamp;
- fixed-horizon directional return;
- MFE and MAE from now;
- MFE/MAE geometry;
- time to favorable/adverse event;
- structural invalidation;
- path efficiency;
- `NEITHER/NONE`.

Strategy-specific labels are defined in the specialist catalog. LONG and SHORT
use exactly symmetric formulas. The initial experiment prohibits side-specific
models; any later exception requires a separately preregistered experiment.

Same-bar barrier ambiguity resolves adverse-first unless higher-resolution data
can establish ordering.

## 6. Horizon policy

Every specialist uses the frozen common router target: symmetric `0.50 ATR14`
barriers and a 60-minute horizon from its actual decision timestamp.

Frozen secondary diagnostics are limited to:

- trend continuation: structural survival through 60 and 120 minutes;
- pullback continuation: common 60-minute target from confirmation;
- breakout/retest: common 60-minute target from breakout or retest
  confirmation, never from the earlier pending timestamp;
- range reversion: midpoint before range break within 60 minutes;
- regime transition: new-regime structural persistence through 120 minutes.

These diagnostics cannot replace the common target, select a different entry
price, or create an additional horizon search on validation.

## 7. Dataset populations

Two datasets answer different questions:

### General candidate dataset

Run deterministic generators over the market timeline with spacing/overlap
controls. This provides enough examples to train each specialist without being
limited to historical Aegis entries.

### Aegis signal-conditioned dataset

Evaluate specialists/router only at immutable Aegis signal timestamps. This
answers whether the architecture improves the current system.

Training on general candidates and evaluating only selected Live trades without
checking population shift is prohibited. The OOD critic and transfer analysis
must quantify the difference.

## 8. Split and leakage rules

- Use the frozen `FRESH_TRAIN`, `FRESH_CALIBRATION`,
  `SPECIALIST_VALIDATION`, `ROUTER_VALIDATION`, and `FINAL_SYSTEM_HOLDOUT`
  windows in the Phase 0 decision record.
- Existing W1-W14 holdouts remain sealed, and all data that influenced W1-W14
  or this architecture is discovery-only.
- Purge candidate horizons around split boundaries.
- Embargo overlapping episodes.
- Keep all specialists for one snapshot in the same split.
- Fit scalers, feature selection, models, calibration, and thresholds on their
  authorized folds only.
- Router training uses out-of-fold specialist predictions.
- Bootstrap at candidate/signal episode and temporal block levels.

## 9. Sample sufficiency

Before model training, produce an event-rate audit per specialist, side,
symbol, regime, and time window.

Frozen minimums are defined in `10_PHASE0_FROZEN_DECISIONS.md`: 2,000 TRAIN,
500 calibration, and 500 specialist-validation candidate episodes per fitted
specialist; 300 signal episodes in router validation and final holdout; at least
150 validation episodes per side for side-specific claims; at least four weekly
blocks and six symbols, with no symbol above 35%.

Model degrees of freedom must additionally retain at least 20 observed outcomes
of the least frequent common-target class per effective fitted degree of
freedom. This constraint may simplify or block a model, never weaken the frozen
episode minima.

If a specialist lacks support, retain its rules as descriptive and do not fit a
model merely to complete the architecture.
