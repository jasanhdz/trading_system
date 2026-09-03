# Aegis Range Strategy V1 - R1 Implementation Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R1_READY_FOR_REVIEW`

**Rama:** `work/entry-quality-evidence-20260726`

**HEAD base:** `56e7c977bd7ac988939ab24cb13f47df15ac926e`

## 1. Resultado

R1 implementa el motor puro, causal y determinista de Aegis Range Strategy V1 en
`sandbox/aegis_range_strategy_v1/`. La implementacion usa exclusivamente fixtures
sinteticos y no contiene loaders de datasets economicos, backtester, optimizador,
grid runner, integracion runtime o llamada E4.

```text
tests_total: 62
tests_passed: 62
tests_failed: 0
spec_divergence: none
```

Comando ejecutado:

```text
/tmp/opencode/aegis-range-venv/bin/python -m pytest
```

## 2. Archivos creados

### Implementacion

- `src/aegis_range_v1/models.py`
- `src/aegis_range_v1/numeric.py`
- `src/aegis_range_v1/data_adapter.py`
- `src/aegis_range_v1/atr.py`
- `src/aegis_range_v1/regime.py`
- `src/aegis_range_v1/candidates.py`
- `src/aegis_range_v1/levels.py`
- `src/aegis_range_v1/detector.py`
- `src/aegis_range_v1/safety.py`
- `src/aegis_range_v1/signal.py`
- `src/aegis_range_v1/breakout.py`
- `src/aegis_range_v1/costs.py`
- `src/aegis_range_v1/thesis.py`
- `src/aegis_range_v1/lifecycle.py`
- `src/aegis_range_v1/engine.py`
- `src/aegis_range_v1/__init__.py`

### Tests, config y evidencia sintetica

- `tests/conftest.py`
- `tests/test_data_regime_atr.py`
- `tests/test_numeric_thesis_candidates.py`
- `tests/test_levels_signal_safety.py`
- `tests/test_breakout_episode.py`
- `tests/test_lifecycle_fills.py`
- `tests/test_engine_master.py`
- `tests/test_costs.py`
- `fixtures/thesis_golden_v1.json`
- `config/candidate_grid.json`
- `artifacts/README.md`
- `pyproject.toml`
- `README.md`

## 3. Arquitectura

```text
RangeDataAdapter
        |
RangeRegimeAdapter + RangeAtr14V1
        |
RangeLevelsV1
        |
RangeDetectorV1
        |
RangeSafetyV1
        |
RangeSignalV1
        |
RangeBreakoutV1
        |
RangeLifecycleV1
        |
RangeEngineV1 (orquestacion por vela)
```

Todos los estados son locales por instancia/simbolo. No se usa wall clock,
randomness, timezone local, iteracion de sets no ordenada para decisiones ni
estado global mutable.

## 4. Mapping R0 a modulos

| Contrato R0 | Modulo |
|---|---|
| Validacion 1m, agregacion causal 5m y gaps | `data_adapter.py` |
| Snapshot exacto de 160 candles | `regime.py`, `data_adapter.py` |
| ATR14 Wilder raw | `atr.py` |
| Candidate de ocho parametros y grid 384 | `candidates.py` |
| Pivots, clusters, touches y pairs | `levels.py` |
| Episodios, IDs y replacement | `detector.py`, `numeric.py`, `engine.py` |
| Hard blockers y score descriptivo | `safety.py` |
| Counted-touch rejection LONG/SHORT | `signal.py` |
| Breakout de episodio y trade | `breakout.py` |
| Pending entry, TP/SL, fills, max hold y cooldown | `lifecycle.py` |
| Costos y funding sintetico puro | `costs.py` |
| Thesis freeze y SHA-256 | `thesis.py` |
| Orden completo por candle | `engine.py` |

## 5. Mapping Technical Clarification a codigo

| Aclaracion | Implementacion |
|---|---|
| `RangeAtr14V1`, 160 bars, binary64 raw | `atr.py` |
| Sin cuantizacion intermedia | operaciones `float` en modulos de decision |
| HALF_EVEN 12dp solo IDs/hashes | `numeric.canonical_decimal_12dp` |
| `market` ausente y output limitado | `regime.RangeRegimeAdapter` |
| `PAIR_REPLACED` mismo close y reset | `detector.py`, `engine.py` |
| `PENDING_ENTRY` flat y orden exacto | `lifecycle.py`, `engine.py` |
| Rejection exige counted touch nuevo | `signal.py` |
| Cooldown elegible en close 12 | `lifecycle.py` |
| `pair_recency_at=min(...)` | `levels.py` |
| Schema y bindings exactos de thesis | `thesis.py` |

## 6. Evidencia obligatoria

```text
atr_wilder_parity: PASS
exact_160_bar_window: PASS
market_absent: PASS
future_pivot_invisible: PASS
no_lookahead_master: PASS
determinism_master: PASS
cluster_assignment_expiration_no_merge: PASS
touch_rearm_six_bar_separation: PASS
counted_touch_rejection: PASS
pair_ranking_and_recency_min: PASS
pair_replacement_same_close: PASS
no_same_close_rebirth: PASS
range_ids_deterministic: PASS
episode_48h_expiration: PASS
episode_breakout_previous_levels: PASS
pending_entry_invariant_and_next_bar: PASS
entry_42bps_and_reward_risk_gates: PASS
thesis_freeze: PASS
tp_sl_freeze_and_adverse_first: PASS
gap_stop_and_target_no_improvement: PASS
trade_breakout_next_open: PASS
max_hold_next_open: PASS
cooldown_close_12: PASS
one_long_one_short_two_trade_quota: PASS
e4_decision_independence: PASS
hard_blockers_override_score: PASS
candidate_cartesian_product_384: PASS
```

Golden thesis fixture sintetico:

```text
fixture: fixtures/thesis_golden_v1.json
sha256: dbc38ff08459e9673c96cfa72d675c4c6a817871a999604f13177e8ba2a745cb
```

El master no-lookahead compara estructuralmente todos los outputs hasta T,
incluidos pivots/clusters/touches, episodio, IDs, senal, pending entry y thesis,
contra la misma historia extendida con candles futuras arbitrarias. El master de
determinismo ejecuta repetidamente el mismo fixture y exige igualdad estructural.

## 7. Frontera experimental preservada

```text
datasets_read: none
train_read: false
calibration_read: false
validation_opened: false
holdout_opened: false
r2_executed: false
pre_validation_spec_frozen: false
production_modified: false
e4_modified: false
regime_engine_v2_modified: false
child_repo_modified: false
fundingRate_materialized: 237/341
markPriceKlines_materialized: 237/341
r2_policy: BLOCK_R2_UNTIL_DOWNLOADED_AND_VERIFIED
```

El adapter de regimen usa una dependencia read-only inyectada y omite el
argumento `market`; no carga BTC, ETH, market confirmation, E4 ni campos no
autorizados para decisiones Range.

## 8. Verificaciones finales requeridas

El manifest `r1_implementation_manifest.json` fija SHA-256 de todos los archivos
R1 y de la autoridad R0. Su hash propio se excluye para evitar autorreferencia y
se informa en la verificacion final.

```text
manifest_json_valid: true
manifest_recorded_hashes_match: true
python_compileall: PASS
git_diff_check: PASS
untracked_r1_diff_check: PASS
git_status_short:
 M binance-futures-bot-ts
?? docs/aegis-range-v1/AEGIS_RANGE_R1_IMPLEMENTATION_REPORT.md
?? docs/aegis-range-v1/r1_implementation_manifest.json
?? sandbox/aegis_range_strategy_v1/
```

No se hizo commit. El gitlink preexistente `binance-futures-bot-ts` permanece
fuera del alcance y conserva `GITLINK_MISMATCH_RECORDED_NOT_RESOLVED`.

`AEGIS_RANGE_R1_READY_FOR_REVIEW`
