# AEGIS E4 Final Result

## Verdict

`E4_NO_MEANINGFUL_IMPROVEMENT`

E4 successfully constructed a more realistic training population and exposed
the hourly-to-5m population shift. It also learned tail-risk and late-entry
diagnostics. It did not learn enough directional entry quality to separate MFE
from MAE or produce positive expectancy.

At top 10% quality coverage, MFE was 22.69 bps and MAE 22.65 bps. Favorable-first
was 49.41%. Net expectancy was -13.95 bps at baseline costs. The ranking was not
monotonic, LONG was unstable, and the holdout remained sealed.

## Flags

- `E4_DATASET_BUILT = TRUE`
- `E4_5M_TRAIN_LIVE_ALIGNMENT_COMPLETE = TRUE`
- `EFFECTIVE_EPISODE_ACCOUNTING_COMPLETE = TRUE`
- `MULTITIMEFRAME_CONTEXT_COMPLETE = TRUE`
- `FLOW_FEATURES_COMPLETE = TRUE`
- `FLOW_EFFECTIVENESS_HAS_SIGNAL = FALSE`
- `REMAINING_MOVE_HAS_SIGNAL = FALSE`
- `CROSS_MARKET_HAS_INCREMENTAL_SIGNAL = FALSE`
- `L2_HAS_INCREMENTAL_SIGNAL = FALSE`
- `OI_POSITIONING_HAS_INCREMENTAL_SIGNAL = FALSE`
- `REALISTIC_EXECUTION_TESTED = TRUE`
- `LEAKAGE_CHECK_PASSED = TRUE`
- `CALIBRATION_IMPROVED_VS_E3 = TRUE`
- `MFE_MAE_PREDICTION_IMPROVED_VS_E3 = FALSE`
- `TRAIN_LIVE_POPULATION_SHIFT_REDUCED = TRUE`
- `TEMPORALLY_STABLE = TRUE`
- `MULTI_SYMBOL_STABLE = TRUE`
- `SHORT_STABLE = TRUE`
- `LONG_STABLE = FALSE`
- `NET_EXPECTANCY_POSITIVE = FALSE`
- `ECONOMIC_EDGE_FOUND = FALSE`
- `ROBUSTNESS_SUCCESS = FALSE`
- `FINAL_HOLDOUT_OPENED = FALSE`
- `FINAL_HOLDOUT_PASSED = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

## Decision

Do not replace E3 and do not advance E4 to shadow. Preserve E4 as evidence that
cadence alignment matters and as a reusable risk/late-entry research dataset.
Any attempt to turn its tail/late diagnostics into a gate must be a separately
preregistered experiment on new data; it cannot be inferred from this run.
