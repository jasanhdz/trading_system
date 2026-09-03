# Independent Entry Quality Discovery V1 Result

## Verdict

`PREDICTIVE_SIGNAL_NOT_YET_ECONOMIC`

The experiment found strong information about whether a large movement will
occur and weak but statistically detectable information about which proposed
side has the better path. It did not find a positive economic entry-quality
edge. The final holdout remains `SEALED_NOT_OPENED`.

## Data support

- TRAIN: 23,700 side-state rows, 1,079 UTC-hour blocks, 11 symbols.
- CALIBRATION: 10,558 rows, 480 blocks.
- VALIDATION: 15,266 rows, 694 blocks.
- Primary top-10% selection: 1,527 rows and 540 effective UTC-hour blocks.

Both frozen support gates passed.

## Models

| Model | AUC | Log loss | Constant log loss | ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| Opportunity logistic | 0.8016 | 0.1619 | 0.1845 | 0.0089 | Clear magnitude/opportunity information |
| Direction logistic | 0.5503 | 0.6917 | 0.6931 | 0.0256 | Weak; barely above the frozen AUC gate |

The direction result is statistically detectable, but the improvement is too
small to establish economic usefulness. Gradient boosting was not run because
the frozen protocol requires simple models first and no economic candidate
survived validation.

## Risk and coverage

| Coverage | Favorable first | Gross bps | Net bps at 20 bps cost | Effective blocks |
|---:|---:|---:|---:|---:|
| 100% | 49.30% | -0.00 | -20.00 | 694 |
| 50% | 51.70% | +1.35 | -18.65 | 694 |
| 25% | 53.21% | +2.42 | -17.58 | 689 |
| 10% | 53.77% | +3.42 | -16.58 | 540 |
| 5% | 53.53% | +4.51 | -15.49 | 371 |
| 2% | 53.59% | +5.19 | -14.81 | 196 |

Quality ranking is monotonic through the preregistered 10% level and has
Spearman 1.0 across the required coverage levels. This is useful predictive
structure, but it is not tradable structure: even the most selective frozen
buckets remain materially negative after costs.

At the primary 10% coverage:

- MFE: 106.35 bps.
- MAE: 101.48 bps.
- Net block-bootstrap 95% CI: `[-20.93, -14.21]` bps.
- Expected shortfall net: -115.74 bps.
- Tail MAE: 457.88 bps.

The selected states have more movement, but their favorable/adverse geometry
is not asymmetric enough to pay for entry.

## Cost stress

Primary top-10% mean expectancy:

| Assumed cost | Net bps |
|---:|---:|
| 0 bps | +3.42 |
| 14 bps | -10.58 |
| 20 bps | -16.58 |
| 30 bps | -26.58 |

The result fails well before the frozen 20 bps materiality hurdle.

## Stability

- Positive symbols after 20 bps costs: 0 of 11.
- Positive validation weeks: 0%.
- LONG top-10% net: -16.61 bps.
- SHORT top-10% net: -16.57 bps.
- Leave-one-symbol-out results remain negative.

There is no hidden winning side, symbol, or week in the frozen primary
selection.

## Feature ablations

Direction AUC by frozen family:

| Family | AUC | Symbols > 0.5 | Weeks > 0.5 | FDR result |
|---|---:|---:|---:|---|
| Price/path | 0.5294 | 10/11 | 5/5 | Pass |
| Structure/location | 0.5228 | 11/11 | 4/5 | Pass |
| Trend/extension | 0.5335 | 11/11 | 5/5 | Pass |
| Volatility/volume | 0.5001 | 2/11 | 1/5 | Fail |
| Flow/response | 0.5254 | 9/11 | 5/5 | Pass |
| Cross-market | 0.4952 | 5/11 | 1/5 | Fail |
| Full | 0.5503 | 10/11 | 4/5 | Pass |

Several families carry weak incremental directional information. None was
promoted independently and no family selection was made from validation.

## Baselines

- Unconditional validation: -0.00 gross / -20.00 net bps.
- Simple directional persistence: -1.95 gross / -21.95 net bps.
- Top-decile volatility: 0.00 gross / -20.00 net bps.
- Frozen random 10%: -1.84 gross / -21.84 net bps.

The model ranks better than these simple baselines, but not by enough to make
an entry economically defensible.

## Mandatory gates

- `ENTRY_QUALITY_DATASET_BUILT = TRUE`
- `LEAKAGE_CHECK_PASSED = TRUE`
- `TRAIN_SUPPORT_SUFFICIENT = TRUE`
- `VALIDATION_SUPPORT_SUFFICIENT = TRUE`
- `OPPORTUNITY_MODEL_HAS_SIGNAL = TRUE`
- `DIRECTION_MODEL_HAS_SIGNAL = TRUE`
- `QUALITY_RANKING_MONOTONIC = TRUE`
- `OUT_OF_SAMPLE_EDGE_POSITIVE = FALSE`
- `NET_EDGE_POSITIVE = FALSE`
- `NET_EDGE_ABOVE_20BPS = FALSE`
- `MULTI_SYMBOL_STABLE = FALSE`
- `TEMPORALLY_STABLE = FALSE`
- `LONG_SHORT_STABLE = FALSE`
- `FINAL_HOLDOUT_OPENED = FALSE`
- `FINAL_HOLDOUT_PASSED = FALSE`
- `INDEPENDENT_ENTRY_QUALITY_PROMISING = FALSE`
- `READY_FOR_PROSPECTIVE_COLLECTION = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

## Decision

Choose outcome 2: continue only as a newly versioned research hypothesis if
there is a genuinely different target, data source, or execution formulation.
The present V1 demonstrates predictive magnitude/ranking information but no
economic entry-quality edge. It does not justify prospective collection,
shadow, live use, opening the holdout, or adding model complexity.
