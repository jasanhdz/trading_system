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
use exactly symmetric formulas before any side-specific model is considered.

Same-bar barrier ambiguity resolves adverse-first unless higher-resolution data
can establish ordering.

## 6. Horizon policy

Each specialist owns a small preregistered horizon family. It cannot search
hundreds of barrier/horizon combinations.

Provisional research horizons:

- timing/pullback confirmation: 1-15 minutes;
- breakout/retest: 5-60 minutes;
- trend continuation: 15-120 minutes;
- range reversion: 5-60 minutes;
- regime transition: 30-240 minutes.

These are planning ranges, not frozen experiment values. A movement-scale audit
on TRAIN must select the final limited families before validation.

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

- New TRAIN/VALIDATION/FINAL_HOLDOUT windows only.
- Existing W1-W14 holdouts remain sealed.
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

Provisional minimums requiring final preregistration:

- enough TRAIN episodes for at least 10-20 outcome events per effective model
  degree of freedom;
- at least 500 independent validation candidates per promoted specialist where
  event frequency permits;
- minimum temporal and symbol breadth;
- sufficient LONG and SHORT evidence before claiming symmetry.

If a specialist lacks support, retain its rules as descriptive and do not fit a
model merely to complete the architecture.

