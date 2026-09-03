# Dataset and Leakage Audit

## Scope

`INDEPENDENT_ENTRY_QUALITY_DISCOVERY_V1` uses the existing causal snapshot and
replay infrastructure without Aegis outputs and without the five Phase 2
candidate generators. Each market state is evaluated as two symmetric,
independent hypotheses: `LONG` and `SHORT`.

The experiment is classified as
`RETROSPECTIVE_DISCOVERY_WITH_TEMPORAL_OOS_VALIDATION`. It is not final
validation and it is not prospective evidence.

## Source

- Venue: Binance USD-M public monthly 1-minute klines.
- Coverage loaded: 2023-05-01 through 2024-01-01 exclusive.
- Universe: 11 frozen symbols.
- Source continuity: 352,800 rows per symbol, except SUIUSDT with 348,960 rows
  because its coverage starts on 2023-05-03.
- Source manifest SHA-256:
  `c5ef99316be5d20167985dce7d8168ecd3c9f9ab291f0910fd04ff648d059631`.
- January-September 2024 is excluded as `DISCOVERY_CONTAMINATED` by the prior
  Strategy Router V1 falsification.
- W1-W14 sealed holdouts were not read.

Historical periods before 2024 may have influenced unrelated prior research.
The temporal validation here is therefore useful discovery evidence but is not
a pristine never-observed holdout.

## Sampling and identity

- One anchor per symbol per UTC hour.
- Two rows per state: one `LONG`, one `SHORT`.
- The two sides share `market_state_group_id` and `temporal_block_id`.
- Same-symbol anchors are separated by the full 60-minute outcome horizon.
- Bootstrap resamples UTC-hour blocks rather than treating both sides or all
  symbols at the same hour as independent.

## Frozen splits

| Split | Start | End | Rows | UTC-hour blocks |
|---|---|---|---:|---:|
| TRAIN | 2023-09-01 | 2023-10-16 | 23,700 | 1,079 |
| CALIBRATION | 2023-10-17 | 2023-11-06 | 10,558 | 480 |
| VALIDATION | 2023-11-07 | 2023-12-06 | 15,266 | 694 |
| FINAL_HOLDOUT | 2023-12-07 | 2024-01-01 | 13,200 | not labeled |

Each adjacent split has a one-day purge/embargo, exceeding the 60-minute
outcome horizon.

## Feature contract

The explicit allowlist contains 276 causal features from the frozen families:

- price/path;
- structure/location;
- trend/extension;
- volatility/volume;
- flow/price response derivable from klines and taker-buy volume;
- BTC/ETH and cross-sectional context;
- warmup/data-quality support fields.

All 5m/15m/1h/4h/1d values are built from closed 1-minute observations. Every
row stores `max_feature_available_at`, and the audit requires it to be no later
than `decision_at`.

## Fail-closed behavior

Forty cross-market rows were rejected because complete contemporaneous BTC/ETH
context was unavailable. Missing context was not imputed across timestamps.
Incomplete snapshots are recorded in per-symbol audit files rather than
silently accepted.

## Leakage result

- Feature availability after decision: 0 rows.
- Target columns in model allowlist: none.
- Aegis/committee/decision-derived columns: none.
- Phase 2 candidate fields: none.
- FINAL_HOLDOUT target columns: none.
- FINAL_HOLDOUT labels built: false.

`LEAKAGE_CHECK_PASSED = TRUE`

The machine-readable evidence is in `artifacts/run_01/leakage_audit.json`,
`split_manifest.json`, and `artifacts/dataset_v1/dataset_manifest.json`.
