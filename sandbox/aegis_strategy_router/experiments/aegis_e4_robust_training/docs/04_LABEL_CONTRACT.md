# E4 Label Contract

Labels use only the 60 minutes after a decision and are never inputs.

- Entry baseline: next available 1-minute open after the closed 5-minute bar.
- Barrier: symmetric `0.5 * ATR14` from the last closed 15-minute context.
- Same-minute ambiguity: `ADVERSE_FIRST`.
- Heads: favorable-first, adverse-first, neither, severe tail risk, entry
  quality, late-entry risk, MFE, MAE, fixed return and event timing.
- Continuous outputs remain continuous; no 39/41 bps discontinuity is imposed
  on MFE, MAE or return regressions.
- Baseline/stress costs: 14/20 bps.

FINAL_HOLDOUT outcomes were not constructed. The sealed parquet contains only
metadata and causal features.
