# Retrospective Rules-Only Falsification Report

Date: `2026-08-18` UTC

Classification: `RETROSPECTIVE_DISCOVERY_ONLY`

## Integrity

- Rules were verified by frozen SHA-256 before replay.
- Candidate rules, thresholds, horizons, substates and episode logic were not changed.
- W1-W14 sealed holdouts were not loaded or opened.
- Candidate anchors were restricted to `2024-01-01` through `2024-09-30 23:00 UTC`.
- Source data ended before `2024-10-01`, the earliest populated sealed holdout boundary.
- No model, specialist, critic or router was fitted.
- No production, collector, TypeScript, order, position, PM2 or exchange behavior changed.

## Historical data audit

The authoritative source is checksum-verified Binance USD-M public monthly
1-minute kline archives. Warmup begins `2023-09-01`. Each of the 11 symbols
contains 570,240 rows through `2024-09-30 23:59 UTC`. The preparation manifest
records every archive SHA-256 and confirms zero sealed-holdout rows.

The replay used the production-independent Phase 1 snapshot contract and the
unchanged Phase 2 generators. `PrecomputedSnapshotBuilder` is a replay-speed
implementation whose canonical output was tested byte-for-byte against
`DeterministicSnapshotBuilder` at multiple historical boundaries.

## Population

- valid causal snapshots: 289,290;
- generator evaluations: 2,892,900;
- raw population candidates: 60,182;
- independent episodes after frozen overlap control: 29,741;
- overlap-suppressed candidates: 30,441 (50.58%);
- fail-closed anchors: 21, all due to incomplete 1m feature state;
- symbols: 11;
- temporal blocks: nine calendar months.

## Primary results

| Strategy | Independent N | Favorable first | Gross bps | Net bps | 95% net CI | Matched-baseline improvement | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Trend Continuation | 2,490 | 48.11% | -0.27 | -20.27 | [-21.59, -18.93] | -0.37 | NEGATIVE |
| Pullback Continuation | 1,007 | 46.77% | -1.31 | -21.31 | [-23.43, -19.20] | -1.39 | NEGATIVE |
| Breakout/Retest | 24,702 | 49.11% | -0.26 | -20.26 | [-20.68, -19.83] | -0.27 | NEGATIVE |
| Range Mean Reversion | 195 | 44.62% | -1.61 | -21.61 | [-25.28, -17.88] | -1.54 | INSUFFICIENT_HISTORICAL_SUPPORT |
| Regime Transition/Reversal | 1,347 | 48.11% | -0.83 | -20.83 | [-22.62, -19.08] | -0.75 | NEGATIVE |

The empirical unconditional favorable-first prevalence was 49.25% and gross
expectancy was -0.02 bps. No family improved materially over its comparable
symbol/side/month population. Benjamini-Hochberg q-values were approximately
0.91. The frozen 20 bps hurdle was not approached even before costs.

All 11 symbols had negative mean net payoff for each supported strategy. Some
months and directional branches had small positive gross means, but those
descriptive values were unstable, below cost and not converted into new policy.

## Catalog viability

- Breakout/Retest contributed 83.06% of independent episodes.
- Candidate-bearing snapshots with multiple strategies: 4.57%.
- Snapshots with opposing LONG/SHORT hypotheses: 0.94%.
- Four strategies met the historical support gate; Range did not.

Although multiple families have sample support, the largest-family share
exceeds the frozen 80% diversity ceiling. More importantly, no supported family
contains rules-only predictive or economic evidence for a future router to
route between.

## Decision

`4_STOP_PROGRAM_NO_RETROSPECTIVE_VIABILITY`

Do not implement specialists and do not continue 24/7 prospective collection
for the current frozen catalog. Any investigation of timing, horizon, side
asymmetry or rule changes is a new discovery hypothesis with a new version and
cannot use this same period as clean confirmation.

## Flags

- `RETROSPECTIVE_BACKTEST_COMPLETE = TRUE`
- `SEALED_HOLDOUTS_PRESERVED = TRUE`
- `RULES_CHANGED_DURING_BACKTEST = FALSE`
- `LEAKAGE_CHECK_PASSED = TRUE`
- `SAMPLE_SUPPORT_TREND = TRUE`
- `SAMPLE_SUPPORT_PULLBACK = TRUE`
- `SAMPLE_SUPPORT_BREAKOUT = TRUE`
- `SAMPLE_SUPPORT_RANGE = FALSE`
- `SAMPLE_SUPPORT_TRANSITION = TRUE`
- `RETROSPECTIVE_EDGE_TREND = FALSE`
- `RETROSPECTIVE_EDGE_PULLBACK = FALSE`
- `RETROSPECTIVE_EDGE_BREAKOUT = FALSE`
- `RETROSPECTIVE_EDGE_RANGE = FALSE`
- `RETROSPECTIVE_EDGE_TRANSITION = FALSE`
- `ANY_STRATEGY_RETROSPECTIVELY_PROMISING = FALSE`
- `ANY_STRATEGY_NET_ABOVE_20BPS = FALSE`
- `CATALOG_DIVERSITY_SUFFICIENT_FOR_FUTURE_ROUTER = FALSE`
- `CONTINUE_PROSPECTIVE_COLLECTION_RECOMMENDED = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `EDGE_VALIDATION_STATUS = RETROSPECTIVE_DISCOVERY_ONLY`

## Reproduction

```bash
PYTHONPATH=sandbox/aegis_strategy_router/src:src \
  .venv/bin/python sandbox/aegis_strategy_router/tools/run_retrospective_falsification.py \
  --workers 3 --overwrite

PYTHONPATH=sandbox/aegis_strategy_router/src:src \
  .venv/bin/python sandbox/aegis_strategy_router/tools/analyze_retrospective_falsification.py
```

Machine-readable results, per-symbol/month/side/substate metrics, frequency,
status rates, raw independent episodes and source manifests are under
`sandbox/aegis_strategy_router/artifacts/retrospective_falsification_v1/`.
