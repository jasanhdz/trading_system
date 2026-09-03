# AEGIS E4 Robust Training Preregistration

## Status

- Classification: `RETROSPECTIVE_DISCOVERY_WITH_TEMPORAL_OOS_VALIDATION`
- E3 remains frozen and unchanged.
- Production, shadow, execution, leverage and FINAL_HOLDOUT access are prohibited.
- Primary cadence: every fully closed 5-minute bar.
- Primary horizon: 60 minutes.
- Both LONG and SHORT are evaluated independently from the same market state.

The frozen machine-readable protocol is `config/preregistration_v1.json`. No
threshold, horizon, symbol, side or period was selected from E4 outcomes.

## Questions

E4 tests whether matching LIVE cadence, causal multi-timeframe context, real
candle-level taker flow, flow effectiveness, cross-market state and
consumed/remaining-move diagnostics improve calibration or economics relative
to an E3-like hourly population.

Two success levels are separate:

- `ROBUSTNESS_SUCCESS`: reliable causal and distributional alignment with useful
  out-of-sample prediction.
- `ECONOMIC_SUCCESS`: positive, stable cost-adjusted expectancy in addition to
  robustness.

## Evidence limitation

The 2023 source period has been exposed to prior research. Temporal VALIDATION
is honest within this run, but it is not prospective-clean confirmation. The
FINAL_HOLDOUT remains feature-only and sealed.
