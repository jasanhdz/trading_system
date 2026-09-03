# Directional Alpha V1 Preregistration

## Question

Inside states ranked highly by the frozen Entry Quality V1 Opportunity model,
can causal flow effectiveness and cross-market propagation identify which side
has materially better future path utility?

This experiment does not retrain Opportunity, use Aegis, or condition on the
five Strategy Router V1 generators.

## Frozen population

The primary population is defined by the TRAIN 90th-percentile Opportunity
score threshold, reused unchanged in CALIBRATION, VALIDATION and the sealed
holdout. TRAIN 80th- and 95th-percentile thresholds define frozen diagnostics.
The all-state population is the control. Thresholds use only TRAIN scores,
never outcomes; their realized coverage may therefore differ in later splits.

The Opportunity component and its ordered feature schema are loaded from the
immutable Entry Quality V1 model artifact. Any hash or schema mismatch stops
the experiment.

## Data and splits

The main panel uses ten Binance USD-M symbols from 2022-01 through 2023-03.
SUIUSDT is excluded because it was not listed. LTCUSDT and XRPUSDT each contain
two pre-TRAIN source gaps; their inputs begin after the last gap and remain
`UNKNOWN` until the full 99-bar daily warmup rebuilds. No row is filled. The period precedes both Entry
Quality V1 development and the explicitly contaminated Strategy Router V1
period.

TRAIN, CALIBRATION, VALIDATION and FINAL_HOLDOUT are temporally separated by
one-day embargoes. FINAL_HOLDOUT receives features only; labels are not built.
W1-W14 holdouts are not read.

## Primary target and model

For every state, LONG and SHORT receive the same symmetric 0.50 ATR/60-minute
path label used by Entry Quality V1. Same-minute dual barrier touches resolve
adverse-first. Utility is common barrier-or-terminal payoff minus 20 bps.

`directional_advantage = utility_long - utility_short`.

The primary model is dual-side Ridge regression of net utility. At inference,
both sides are scored independently; the higher predicted utility supplies the
proposed side and their difference supplies confidence. No losing LONG is
converted mechanically into SHORT evidence.

The operational abstention is frozen before outcomes:

- predicted best net utility must be positive; and
- predicted directional advantage must be at least 20 bps.

Coverage curves are reported independently of that operational gate so a
zero-trade abstention cannot hide model weakness.

## Feature families

The main comparable panel evaluates `FLOW_ONLY`, `CROSS_MARKET_ONLY`, and
`FLOW_CROSS_MARKET`. Sequence features use frozen 1/3/5/15/30/60-minute
lookbacks ending before the decision timestamp.

The available Tardis L2 days were already used by W9.1; later days also overlap
sealed W3/W2 holdout periods. They are therefore not eligible for clean V1
evaluation. `L2_ONLY` is audited but not fitted.

OI and liquidations are not proxied. A positioning subexperiment is only
permitted if authentic, sufficiently covered history is found.

## Promotion boundary

The primary result is top 10% directional confidence inside the top 10%
Opportunity population. Promotion requires positive conservative-net CI,
improved MFE/MAE geometry versus Opportunity-only, monotonic selectivity,
multi-symbol and temporal stability, and sufficient effective blocks.

No result opens FINAL_HOLDOUT automatically or authorizes collection, Shadow,
Live, production changes, or a more complex model.
