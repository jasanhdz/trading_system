# W11 Ephemeral Regime Alpha Preregistration

Status: **FROZEN BEFORE RESULT EXECUTION**

Seed: `20260825`

Scope: offline historical research only.

## Hypothesis and Falsification

The hypothesis is that a model trained only on the prior 6-72 hours can sometimes
identify net-positive opportunities for a short prospective lifetime. The null is
that apparent local predictability does not survive forward validation and realistic
costs. The experiment is designed to reject the idea unless a causal procedure, not
one retrospectively selected day/symbol/side, survives.

## Frozen Partitions

- DISCOVERY: `[2023-05-01, 2023-09-01)`
- VALIDATION: `[2023-09-01, 2023-11-01)`
- PROSPECTIVE: `[2023-11-01, 2024-01-01)`

The methodology and thresholds are fixed before any pipeline result is computed.
Prospective results may not change thresholds. External governed holdouts remain
sealed. The first 72h of usable data are warmup, not decisions.

## Decision Clock and Labels

Completed 1m candles are aggregated into completed 5m bars. Feature snapshots occur
every 15m. At decision time `t`, entry is the next 5m bar open and exit is the close
after 5/15/30/60m. Training and validation samples whose outcome is not fully known
at model creation are excluded.

Opportunity is `abs(gross_return_bps) >= 19`, five bps beyond baseline cost. Direction
is LONG/SHORT only among opportunities; all other cases are SKIP. Every economic
result reports gross, 10 bps fees, assumed slippage, total cost and net bps at 14,
20 and 30 bps.

## Regime State

The frozen 20-feature vector contains recent returns, volatility, ATR, EMA state,
efficiency, range location, relative volume, taker flow, BTC context, beta/correlation,
breadth, dispersion, alt basket return and ETH/BTC relative movement. No future path,
target, trade result, or noncausal forward fill is allowed.

## Experts and Models

Candidate families are `ERE_6H`, `ERE_12H`, `ERE_24H`, `ERE_48H`, `ERE_72H`, each
evaluated at all four horizons. Every creation timestamp is six hours apart.

Each candidate uses:

1. median imputation fitted on its recent training rows;
2. standardization fitted on those rows;
3. regularized logistic opportunity classification;
4. regularized logistic direction classification on opportunity rows.

Fixed thresholds are 0.55 opportunity and 0.55 directional confidence. A class with
fewer than ten observations fails closed. Deep learning, retrospective symbol
whitelists and random splits are forbidden.

At creation time, the six hours immediately preceding the 60m outcome embargo are
forward validation. A candidate activates only with at least 12 validation trades,
three symbols, no symbol above 50%, positive 14-bps and 20-bps expectancy, and at
least 80% probability of positive mean from a deterministic one-hour temporal-block
bootstrap. At most one new instance is selected per creation timestamp, ranked by
validation 20-bps expectancy, then shorter window and horizon. Failure means SKIP.

Baselines are always skip, always long, always short, 15m momentum and 15m mean
reversion. Baselines use the same prospective timestamps, horizons and costs.

## Regime Similarity

Three causal diagnostics are compared inside forward validation:

- standardized Euclidean similarity to the final quarter of the training regime;
- cosine similarity in training-standardized space;
- diagonal covariance distance.

The method with the strongest positive validation rank relationship between
similarity and realized selected-trade net edge is frozen per instance; ties prefer
standardized Euclidean. Its drift threshold is the 10th percentile of training
similarity. This selection uses no prospective outcomes.

## Expiration Guardian

Primary TTL is fixed by training window: 6h, 6h, 12h, 24h and 24h respectively.
TTL alternatives 6/12/24/36/48h are descriptive sensitivity checks and cannot choose
the primary result.

An instance expires irreversibly on the first condition in this priority order:

1. `EDGE_DECAY`: after 12 resolved post-creation trades, the trailing 12-trade mean
   baseline net edge is below -2 bps;
2. `REGIME_DRIFT`: similarity is below its frozen threshold for three consecutive
   snapshots;
3. `TTL`: the fixed expiry timestamp is reached.

Only post-creation outcomes known at the current simulation timestamp may update edge
decay. An expired ID is appended to an immutable registry and never reactivated.

Instance IDs use `ERE_<CREATED_UTC>_<WINDOW>H_<HORIZON>M_<SEQUENCE>`. Decision records
freeze family, version, ID, training/validation windows, creation/expiry, similarity,
expected edge, direction, horizon, decision and reason. Trade attribution never
changes when an exit mechanism changes.

## Edge Half-Life

For each instance, realized net edge is measured in age buckets beginning at 0-1h,
then through 3h, 6h, 12h, 24h and 48h where available. Initial edge is the first
nonempty age bucket. Half-life is the first later bucket whose absolute positive edge
is at most 50% of initial edge. Economic lifetime ends at the first non-positive
cumulative age bucket. Unobserved half-life is reported as censored, never imputed.

## Inference and Multiple Testing

- Selection is causal at every creation timestamp.
- The system-level prospective stream, not the best retrospective expert, is primary.
- Confidence intervals resample synchronized complete UTC-day blocks across symbols.
- Fixed seed and stable ordering are mandatory.
- Window/horizon results are descriptive family diagnostics; they cannot replace the
  primary causal selection stream.
- Subgroups discovered after execution are marked exploratory.

## Success Gates

`EPHEMERAL_ALPHA_CONFIRMED` requires all of:

- at least 100 prospective trades from at least five instances and four symbols;
- no symbol above 50% of trades;
- positive mean net at both 14 and 20 bps;
- 95% temporal-bootstrap lower bound above zero at 14 bps;
- Guardian improves net expectancy or drawdown versus TTL-only;
- similarity has a positive prospective relationship with later edge;
- results are not concentrated in one day, symbol, side or family.

Signal without economic robustness is
`EPHEMERAL_SIGNAL_DETECTED_NOT_YET_ECONOMIC`; no useful evidence is
`NO_EPHEMERAL_EDGE_FOUND`; inability to run a valid test is `INSUFFICIENT_DATA`.
No result authorizes production, Shadow, E4 or TypeScript changes.
