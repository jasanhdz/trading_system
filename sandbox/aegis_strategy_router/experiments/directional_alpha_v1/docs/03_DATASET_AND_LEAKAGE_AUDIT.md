# Directional Alpha V1 Dataset And Leakage Audit

## Construction

The experiment uses the same `PrecomputedSnapshotBuilder`, closed-candle
availability contracts, structural-level implementation and symmetric target
builder as Entry Quality V1. Anchors occur hourly and each state has exactly
two rows: one LONG hypothesis and one SHORT hypothesis. Both share a
`market_state_group_id`; the 60-minute horizon and cadence prevent same-symbol
outcome overlap.

The development dataset contains 48,786 TRAIN, 28,748 CALIBRATION and 43,654
VALIDATION directional rows. These represent 24,393, 14,374 and 21,827 market
states respectively. The feature-only FINAL_HOLDOUT contains 27,840 rows and
has no `target__*` columns.

## Frozen Opportunity gate

The Opportunity component was loaded from its immutable artifact with SHA256
`6ba81bd4d1e1ccd80f2566de8c07168e6b03a69658426c28caac0720304a611f`.
It was not retrained. Thresholds were computed from TRAIN scores only:

- TRAIN p80: `0.9994616961688733`
- TRAIN p90, primary: `0.9999066987326859`
- TRAIN p95: `0.9999866180286152`

The primary threshold yielded 4,880 TRAIN rows, 512 CALIBRATION rows and 1,998
VALIDATION rows. This score-distribution shift is retained rather than
re-quantiling later splits.

## Causality and isolation

- all `max_feature_available_at <= decision_at`;
- all features pass an explicit allowlist and contain no future/outcome fields;
- no Aegis, Phase 2 candidate, committee or production-decision field is read;
- same-bar dual barrier ambiguity is adverse-first for both sides;
- missing source periods fail closed; no candle is filled or interpolated;
- directional feature hashes are deterministic;
- FINAL_HOLDOUT labels were not constructed or opened.

`LEAKAGE_CHECK_PASSED = TRUE`

## Support gate

The frozen primary minimum was 2,500 VALIDATION directional rows and 1,500
effective blocks. Actual support is 1,998 rows across 490 blocks. Therefore:

`VALIDATION_SUPPORT_SUFFICIENT = FALSE`

All model results are diagnostic temporal-OOS discovery. They cannot promote
the experiment, trigger a more complex model or open FINAL_HOLDOUT.
