# Aegis W9.1 Data Audit

## Verdict

`W9_1_DATA_QUALITY_SUFFICIENT = TRUE`

- Rows: 21,600.
- Independent episode IDs: 21,600.
- Symbol-days: 30.
- Episodes per symbol-day: 720.
- Symbols: ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT.
- Dates: 2024-09-01, 2025-03-01, 2025-09-01, 2025-12-01, 2026-03-01.
- Causal features: 126.
- Non-finite feature values: 0.
- Maximum reconstructed-L2/quote mid p99 difference: 0.000 bps.

## Integrity Checks

- `manifest_complete`: `TRUE`
- `episode_ids_unique`: `TRUE`
- `features_finite`: `TRUE`
- `only_preregistered_months_present`: `TRUE`
- `final_holdout_absent`: `TRUE`
- `fixed_anchor_spacing`: `TRUE`
- `expected_episodes_per_symbol_day`: `TRUE`

The provider CSV omits native sequence identifiers, so reconstruction quality
was established through mandatory snapshots, monotonic capture order, zero
crossed/invalid books, and agreement with the independent quote stream. The
sealed `2026-06` holdout was not downloaded or opened.
