# Independent Entry Quality Discovery V1

## Status

- `EXPERIMENT = INDEPENDENT_ENTRY_QUALITY_DISCOVERY_V1`
- `AEGIS_DEPENDENCY = NONE`
- `STRATEGY_ROUTER_V1_CANDIDATE_DEPENDENCY = NONE`
- `FINAL_HOLDOUT = SEALED_NOT_OPENED`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

## Question

Can a supervised model rank causal market states, evaluated symmetrically as
LONG and SHORT hypotheses, by future path quality and conservative net value?
This is entry-quality discovery, not strategy-family classification.

## Sampling and independence

The market is sampled once per UTC hour. Each market state creates two rows,
LONG and SHORT, sharing one market-state group. Same-symbol anchors therefore
do not overlap under the frozen 60-minute horizon. Bootstrap and effective N
use the UTC-hour group across symbols, not individual rows.

## Frozen targets

The primary directional target reuses the compatible frozen target from Phase
0: ±0.50 ATR14 of the last closed 15m bar, 60-minute horizon and adverse-first
when both barriers occur in the same 1m candle. Continuous labels include MFE,
MAE, MFE-MAE, terminal directional return, event times and common payoff.
Conservative net subtracts 20 bps.

Direction-independent Opportunity is true when the maximum absolute excursion
within 60 minutes reaches at least the larger of 0.50 ATR and 20 bps. No other
horizon or barrier is searched in V1.

## Splits

All intervals are `[start, end)` with one-day embargoes:

| Partition | Interval |
|---|---|
| TRAIN | 2023-09-01 through 2023-10-15 |
| CALIBRATION | 2023-10-17 through 2023-11-05 |
| VALIDATION | 2023-11-07 through 2023-12-05 |
| FINAL_HOLDOUT | 2023-12-07 through 2023-12-31 |

January-September 2024 is discovery-contaminated and excluded. W1-W14
holdouts remain sealed. The 2023 source may have appeared in unrelated prior
research, so this study is temporal OOS discovery, not pristine prospective
confirmation.

FINAL_HOLDOUT features may be frozen and hashed, but its future labels and
performance cannot be built or read in this round.

## Models and selection

Models are fixed to empirical constants, L2 logistic regression, Ridge
regression and a depth-3 tree with minimum leaf size 200. Logistic probabilities
are sigmoid-calibrated only on CALIBRATION. Gradient boosting is prohibited
unless simple models first demonstrate validation signal and a new freeze is
recorded.

Risk/coverage is fixed to 100%, 50%, 25%, 10%, 5% and 2%. No threshold is
selected from validation. The primary view is top 10%, frozen before fitting.

## Success boundary

Promising requires useful calibration/ranking, monotonic improvement through
100/50/25/10% coverage, positive conservative-net CI at top 10%, at least 500
effective groups, six positive symbols and two-thirds positive weekly blocks.
Net mean above 20 bps is reported separately as economic materiality.

No result from this experiment authorizes production, Shadow or opening the
FINAL_HOLDOUT.
