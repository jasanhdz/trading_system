# Aegis Scientific Validation

## Scope

This report covers the clean-rebuild scientific pipeline on 2026-07-17. It is offline-only. It does not authorize execution, access Binance, or promote an artifact to live use.

## Audit matrix

| Component | State | Direct evidence | Remaining risk | Action |
|---|---|---|---|---|
| Domain and contracts | Implemented | Contract, invalid numeric and serialization tests | No static type checker installed | Keep runtime validation authoritative |
| Configuration | Implemented | Eleven-symbol, order and stable-hash tests | TS loader is intentionally not wired | Require manifest handshake before any integration |
| Snapshot validation | Implemented | Stale, future, gap, OHLC, duplicate and depth tests | Source quality remains a TS responsibility | Fail closed |
| Features | 39 causal features | parity, order independence, label isolation, finite-value tests | No production drift history yet | Collect only after an approved shadow bundle |
| Model runtime/registry | Implemented | corrupt/missing/hash/schema/universe/timeframe tests | Current configured bundle is reference-only | Do not promote it |
| D3/RV2/TRRM/QMAE/EQM/ECON1 | Implemented | direct normal/boundary/monotonicity tests | Mathematical validation needs more independent datasets | Keep thresholds frozen per bundle |
| Candidate/selection/freeze | Implemented | LONG, SHORT, NO_TRADE, tie, blocking and idempotency tests | Portfolio context is contractual, not operational sizing | TS remains operational authority |
| Evidence/outcome | Implemented | hash-chain, duplicate-outcome and shadow maturity tests | File growth/rotation is not yet operationalized | Add retention before activation |
| API | Implemented | health/readiness/manifest/evaluate/outcome tests | No deployed service in this phase | Offline only |
| Training/evaluation | Implemented | temporal split, embargo, train-only normalization and reproducibility tests | One local data source | Candidate rejected |
| TS contract/gate | Implemented | 28 focused contract/gate/shadow tests | Coordinator is not registered in a running process | Leave disabled |

No functional `NotImplementedError` remains. The only source `TODO` check rejects placeholder bundle IDs; the other TODO text is historical architecture documentation.

## Test strategy

Python has 43 behavioral tests after Phase 2, up from 14. They cover the critical modules `domain`, `config`, `features`, `models`, `layers`, `decision`, `evidence`, `runtime`, `api`, `training`, `registry`, candidate evaluation and shadow replay. Fixtures include deterministic LONG, SHORT, NO_TRADE, competitive ranking, portfolio blocking, stale/invalid data and corrupt artifacts.

`coverage.py` and `pytest-cov` are not installed in the project environment. No package was installed and no global environment was changed. Therefore no percentage is claimed. Direct module/branch coverage is recorded by the audit matrix and test names instead of inventing a number.

## Feature catalogue

All features are evaluated at the coordinated final candle `t`. Training and inference call the same `DeterministicFeaturePipeline`. Normalization is `(x - train_mean) / train_scale`, clipped to `[-12, 12]`, using bundle-frozen statistics only. Degenerate train scales become `1.0`; invalid published scales fail closed.

| Feature | Definition | Window/input | Causal role/consumer |
|---|---|---|---|
| `ret_1` | `close[t]/close[t-1]-1` | 1 bar | local return; models/layers |
| `ret_3` | `close[t]/close[t-3]-1` | 3 bars | short momentum; models |
| `ret_6` | `close[t]/close[t-6]-1` | 6 bars | momentum/regime/cross-section |
| `ret_12` | `close[t]/close[t-12]-1` | 12 bars | H12 trend; models/D3 |
| `ret_24` | `close[t]/close[t-24]-1` | 24 bars | medium trend; models |
| `log_ret_1` | `ln(close[t]/close[t-1])` | 1 bar | stable local return; models |
| `close_to_open_return` | `close[t]/open[t]-1` | current final bar | candle direction; models |
| `candle_range_fraction` | `(high-low)/close` | current final bar | intrabar volatility; RV2/QMAE |
| `candle_body_fraction` | `abs(close-open)/close` | current final bar | directional impulse; D3 |
| `upper_wick_fraction` | `(high-max(open,close))/(high-low)` | current final bar | rejection; models |
| `lower_wick_fraction` | `(min(open,close)-low)/(high-low)` | current final bar | rejection; models |
| `body_to_range` | `abs(close-open)/(high-low)` | current final bar | candle efficiency; EQM |
| `close_position_in_range` | `(close-low)/(high-low)` | current final bar | closing pressure; models |
| `volume_return_1` | `volume[t]/volume[t-1]-1` | 1 bar | volume impulse; models |
| `volume_zscore_24` | `(volume[t]-mean24)/std24` | 24 bars | abnormal activity; D3 |
| `volume_ratio_6_24` | `mean(volume,6)/mean(volume,24)` | 6/24 bars | volume acceleration; models |
| `range_mean_6` | mean `(high-low)/close` | 6 bars | short volatility; RV2 |
| `range_mean_24` | mean `(high-low)/close` | 24 bars | baseline volatility; RV2 |
| `atr_12` | mean normalized true range | 12 bars | adverse range; RV2/QMAE |
| `atr_24` | mean normalized true range | 24 bars | regime volatility; D3/RV2 |
| `volatility_ratio_6_24` | `std(ret,6)/std(ret,24)` | 6/24 bars | volatility transition; D3 |
| `ema_gap_6_12` | `EMA6/EMA12-1` | max 48 bars | short trend; D3/models |
| `ema_gap_12_24` | `EMA12/EMA24-1` | max 48 bars | medium trend; D3/models |
| `ema_slope_12` | `EMA12[t]/EMA12[t-1]-1` | 25 bars | trend persistence; D3 |
| `momentum_acceleration_3_12` | `ret3-ret12/4` | 12 bars | momentum acceleration; models |
| `return_zscore_24` | `(ret1-mean(ret,24))/std(ret,24)` | 24 returns | return anomaly; models |
| `persistence_6` | mean sign of last six returns | 6 returns | directional persistence; D3 |
| `chop_12` | `1-min(1,abs(close[t]-close[t-12])/sum(abs(ret),12))` | 12 bars | range/trend context; D3 |
| `trend_strength_12` | `abs(ret12)/mean(true_range,12)` | 12 bars | normalized trend; D3 |
| `range_expansion` | `mean(range,6)/mean(range,24)-1` | 6/24 bars | volatility change; RV2 |
| `relative_return_6` | symbol `ret6` minus universe mean | same cut, 11 symbols | relative strength; selection/models |
| `relative_return_12` | symbol `ret12` minus universe mean | same cut, 11 symbols | relative trend; selection/models |
| `cross_rank_return_6` | average tie rank of `ret6` in universe | same cut, 11 symbols | stable global rank; selection |
| `cross_dispersion_return_6` | population std of universe `ret6` | same cut, 11 symbols | cross-sectional regime; D3 |
| `market_breadth_6` | fraction of positive universe `ret6` | same cut, 11 symbols | market breadth; D3 |
| `market_direction_6` | mean universe `ret6` | same cut, 11 symbols | dominant direction; D3 |
| `market_concentration_6` | max absolute `ret6` / sum absolute `ret6` | same cut, 11 symbols | concentration; selection context |
| `btc_divergence_6` | symbol `ret6` minus BTC `ret6` | same cut | benchmark divergence; models |
| `eth_divergence_6` | symbol `ret6` minus ETH `ret6` | same cut | benchmark divergence; models |

Safe division returns zero only for a near-zero denominator. Every raw and normalized value must be finite. The feature-name hash is `9a6e74720e14ce52800033e14979c4309c2a7f3d3c49b0218e727f29fe64248d`.

## Causality and leakage

- Snapshot validation admits only final, ordered 5m candles ending at the coordinated cut.
- Cross-sectional features use all eleven symbols at that same cut, never the next cut.
- Labels are built after feature transformation and cannot alter a feature batch; a regression test changes labels while asserting identical features.
- The local SQLite source is opened with `mode=ro&immutable=1`.
- The temporal partition is train, 120-minute embargo, validation, 120-minute embargo, then test.
- Normalizers fit only train indices. Calibration/threshold configuration does not inspect final test outcomes.
- No random split is used. Seeded randomness exists only in the random baseline.
- Conflicting duplicate source rows are excluded; gaps produce skipped cycles, not interpolation.

No known leakage was found by these invariants. This is evidence against obvious leakage, not a proof that the dataset is free of all provenance errors.

## Numerical and layer properties

Constant prices, zero volume, zero range, extreme values, insufficient history, non-finite values, invalid normalizers and invalid probability domains are covered. Invalid states fail closed or yield finite neutral features; none produces an executable proposal.

The direct layer tests establish: TRRM compatibility does not improve as tail probability rises; QMAE quality does not improve as adverse excursion rises; EQM does not improve with model disagreement; ECON1 edge declines as friction rises and rejects negative net edge. D3/RV2 boundaries, reason codes, ordering and determinism are also exercised.

## Offline performance measurement

On 200 complete unique evaluations of the eleven-symbol/60-bar fixture:

- p50: 119.77 ms
- p95: 129.97 ms
- p99: 139.93 ms
- mean: 121.32 ms
- peak traced memory: 13,665,001 bytes
- request: 156,396 bytes
- response: 3,250 bytes
- evidence event: 56,294 bytes

Mean stage times were validation 3.57 ms, features 11.67 ms, models 1.04 ms, layers 1.82 ms, candidates 2.47 ms and selection 0.29 ms. The remaining measured total is dominated by canonical hashing/evidence serialization under `tracemalloc`. These are measurements, not production SLAs.
