# AEGIS Range V2 Discovery Diagnostic Report

## 1. Estado y conclusion

**Estado:** `AEGIS_RANGE_V2_DISCOVERY_READY_FOR_REVIEW`

**Etiquetas:** `DISCOVERY_ONLY`, `NO_WHITELIST_AUTHORITY`.

El diagnostico TRAIN explica el rechazo economico de V1, pero no identifica una
especificacion Range V2 con autoridad de seleccion. Las 22,016 filas de
candidatos representan solo 382 oportunidades canonicas. En la vista primaria
por oportunidad, las dos confirmaciones investigadas empeoran la expectativa
neta V1 y permanecen con profit factor inferior a 1.

Las confirmaciones evitan entre 66.1% y 68.7% de los stops originales mediante
abstencion, no transformando el outcome de entradas retenidas. Al mismo tiempo
pierden entre 58.8% y 64.3% de los targets originales. Ningun simbolo alcanza
el floor causal de 30 oportunidades unicas en la ventana prior de 60 dias, por
lo que no existe autoridad para una whitelist o tercil de suitability.

Este trabajo es generacion post hoc de hipotesis sobre TRAIN 2024. No reabre ni
revierte el rechazo V1, no selecciona candidato y no autoriza `CALIBRATION`,
`VALIDATION`, `HOLDOUT`, produccion o live trading.

## 2. Autoridad y lineage

- Commit base: `5ecfa7009f6cfe3f700cdc1607aacb15ce245614`.
- HEAD de la autoridad R2: `55c3bb7665790572d83db014509c2978bf3effaf`.
- R1 immutable: `42/42 MATCH`.
- R1 manifest SHA-256:
  `2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c`.
- Dataset logical SHA-256:
  `587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8`.
- Post-R1 revalidation manifest SHA-256:
  `9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1`.
- Lineage reconciliation report SHA-256:
  `b21fc0e85286c09e5ac53a12a30965858aaa15858464847ec9486509a94274c5`.
- Politica funding/mark: `MONTHLY_PRIMARY_DAILY_GAP_FILL_V1`.
- R2 permanece `AEGIS_RANGE_R2_TRAIN_BACKTEST_COMPLETE_REJECT_NO_EDGE`.

La unica particion abierta fue TRAIN. Los flags registrados fueron
`TRAIN=true`, `CALIBRATION=false`, `VALIDATION=false` y `HOLDOUT=false`.

## 3. Alcance y unidad estadistica

- Ventana: `[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`.
- Timeframe: 5 minutos.
- Simbolos: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT,
  ADAUSDT, AVAXUSDT, LINKUSDT, SUIUSDT y LTCUSDT.
- Filas V1 aceptadas: 22,016.
- Oportunidades canonicas: 382.
- Multiplicidad por oportunidad: 8 a 256 filas.
- Escenarios: `BASELINE`, `STRESS_20` y `STRESS_30`.
- Confirmacion A: `NEXT_CLOSE_PROGRESS`.
- Confirmacion B: `REJECTION_EXTREME_RECLAIM`.

La vista primaria asigna a cada fila peso `1 / group_multiplicity`, de modo que
cada `canonical_opportunity_id` suma exactamente 1. La vista candidate-weighted
se publica solo para mostrar la sensibilidad a la duplicacion contractual.

El ID canonico usa simbolo, lado, `decision_at`, `entry_at`, soporte,
resistencia y midpoint normalizados a 12 decimales. Excluye candidato, stop,
target, costos y outcomes.

## 4. Contrato metodologico

- MFE/MAE incluyen el fill de entrada como extrema cero y nunca son negativos.
- La barra terminal usa el modelo conservador oficial; el OHLC terminal completo
  se publica aparte para revelar ambiguedad intrabar.
- El post-stop comienza en la primera vela completa estrictamente posterior a
  la barra de salida STOP.
- Recovery se mide a 15, 30, 60 y 120 minutos; las excursiones desde el fill del
  stop tambien tienen piso cero.
- Los exits contrafactuales conservan stop, target, adverse-first, breakout,
  max hold, funding y costos V1.
- Suitability usa maturity lag fijo de 12 horas, ventana prior de 60 dias y floor
  de 30 oportunidades unicas maduras antes de la decision.
- La abstencion de confirmacion es incremental respecto de las entradas V1 ya
  aceptadas; la abstencion agregada oficial V1 sigue siendo 98.49%.

## 5. Evidencia materializada

Fuente oficial discovery:

`sandbox/aegis_range_strategy_v1/artifacts/range_v2_discovery/`

| Artefacto | Filas | SHA-256 |
|---|---:|---|
| `diagnostics_manifest.json` | 1 | `77b27abf29efa2cffae853ea991da4af2183dad9b1feff150b7d527bba4346e3` |
| `diagnostic_summary.json` | 1 | `38ff8f28bc4b6a5dc204c14fe1aa6b6bd3d57fe56df3873cf467fd6303fd56fb` |
| `opportunity_paths.jsonl.gz` | 22,016 | `796af43415a7398f04704a4ca22d60d7c7f5b9e7d54b9589ea1700c3aaaf9b00` |
| `stop_recovery.jsonl.gz` | 16,280 | `796fec88152c7cd08161691f2198b3cdd1486f7eeec8d6719fafa8bdedce86f7` |
| `confirmation_counterfactuals.jsonl.gz` | 44,032 | `aaca07c8055d96753fc56a5b83e0d57a0be69c90f67938c3f8794aab2c40044c` |
| `symbol_suitability.jsonl.gz` | 382 | `6061ff4b5102f1227ce85578674144e2c9be1b9d5eddff80312df1cff7b9ac34` |

Los inputs congelados `run_a` conservaron sus hashes:

- `run_manifest.json`: `5f62022f35fb38de174e6f7c573397d1c1ceebc75d76f7d848260c35456012b8`.
- `candidate_metrics.json`: `12f72be45420099d7ab0a56524ca934e791dfbaa9da0c0add87277d7939b656f`.
- `episodes.jsonl.gz`: `82989a83a68935ed44866afb2f5904e703c81e27b50acde5fe1c2fabd6af5270`.
- `trades.jsonl.gz`: `125f31dcb1bf27e6f183bbbb02da901a5133847e577196a9bd6a59be42cd4537`.
- `regime_cache_manifest.json`: `a9699e874537bcdf14042e3d811594448e886b6811f48741b9b6ce5ad7e9c22b`.

## 6. Auditoria integral

- Hashes y conteos discovery: `6/6 MATCH`.
- Gzip determinista con `mtime=0`: `4/4 MATCH`.
- Hashes `run_a`: `5/5 MATCH`.
- Regime caches: `11/11 MATCH`.
- IDs canonicos reconstruidos: `382/382`; mismatches: 0.
- Suma total de peso unico: 382.0; errores por ID: 0.
- Joins de episodios y contexto de regimen faltantes: 0.
- MFE/MAE negativos en paths, full-terminal y suitability: 0.
- Excursiones post-stop negativas en todos los horizontes: 0.
- Errores de reconstruccion del exit oficial: 0.
- Errores de categoria/horizonte recovery: 0.
- Contrafactuales censurados o purgados: 0.
- Errores de cronologia o cardinalidad de confirmacion: 0.
- Violaciones de ventana prior, maturity o sample floor: 0.
- Observaciones fuera de TRAIN: 0.
- Tests Python finales: `106/106 PASS`.
- Tests TypeScript frozen: `17/17 PASS`.
- R1 al cierre: `42/42 MATCH`.

## 7. Q1 - Es material la duplicacion contractual

Si. Las 22,016 filas colapsan a 382 oportunidades. La multiplicidad mediana es
64 y el maximo 256. La conclusion candidate-weighted para Confirmacion A parece
mejor que V1, mientras la vista primaria por oportunidad muestra lo contrario.
Elegir retrospectivamente un contrato favorable seria invalido; por eso todos
los contratos se promedian dentro de cada oportunidad.

## 8. Q2 - Donde falla V1 durante el trade

| Metrica | Candidate-weighted | Unique-opportunity-weighted |
|---|---:|---:|
| MFE medio | 0.4644% | 0.4226% |
| MAE medio | 0.3367% | 0.3302% |
| Midpoint alcanzado mientras abierto | 11.54% | 11.39% |
| Stop first | 73.95% | 76.09% |
| Target first | 22.86% | 21.42% |

La entrada V1 suele sufrir una excursion adversa suficiente para tocar stop
antes de materializar el retorno al midpoint. El problema no es solo costo de
ejecucion: la expectativa gross agregada ya es negativa en ambas vistas.

## 9. Q3 - Que ocurre despues de un STOP

Los 16,280 STOP tienen los cuatro horizontes completos.

| Categoria | Filas / tasa | Peso unico / tasa |
|---|---:|---:|
| `STOP_THEN_MIDPOINT_RECOVERY` | 3,056 / 18.77% | 55.67 / 19.15% |
| `STOP_THEN_ENTRY_RECOVERY` | 7,024 / 43.14% | 130.00 / 44.72% |
| `STOP_TRUE_FAILURE` | 3,784 / 23.24% | 68.00 / 23.39% |
| `STOP_AMBIGUOUS` | 2,416 / 14.84% | 37.00 / 12.73% |

Un 63.88% del peso unico recupera entry dentro de 60 minutos o midpoint dentro
de 120. Sin embargo, solo 19.15% recupera midpoint; la mayoria de las
recuperaciones no demuestra que mantener el riesgo V1 hasta target sea viable.

## 10. Q4 - Con que rapidez ocurre la recuperacion

| Horizonte | Entry candidate / unica | Midpoint candidate / unica |
|---:|---:|---:|
| 15m | 34.00% / 36.01% | 2.06% / 1.26% |
| 30m | 47.03% / 50.40% | 4.18% / 3.15% |
| 60m | 58.97% / 61.98% | 9.39% / 8.66% |
| 120m | 76.76% / 76.61% | 18.77% / 19.15% |

La recuperacion de entry crece rapido, pero la recuperacion estructural al
midpoint permanece baja. Esto sugiere investigar timing/entrada como hipotesis
nueva, no ampliar retrospectivamente el stop V1.

## 11. Q5 - Cuanto filtran las confirmaciones

| Regla | Filas retenidas | Abstencion candidate | Peso unico retenido | Abstencion unica |
|---|---:|---:|---:|---:|
| A `NEXT_CLOSE_PROGRESS` | 7,848 | 64.35% | 125.17 | 67.23% |
| B `REJECTION_EXTREME_RECLAIM` | 8,272 | 62.43% | 137.25 | 64.07% |

No hubo censura. Esta abstencion es adicional al 98.49% de abstencion agregada
V1, por lo que no puede presentarse como mejora gratuita.

## 12. Q6 - Evitan stops sin perder targets

| Regla/vista | Stop original evitado | Target original perdido |
|---|---:|---:|
| A candidate | 66.93% | 59.94% |
| A unica | 68.72% | 64.26% |
| B candidate | 65.36% | 56.60% |
| B unica | 66.11% | 58.76% |

Todos los stops evitados y targets perdidos se deben a `NO_TRADE`. Entre las
entradas retenidas, `original_stop_no_longer_stop=0` para ambas reglas: las
confirmaciones no cambian el outcome de los stops que aceptan.

## 13. Q7 - Mejoran la economia BASELINE

| Sistema/vista | Gross | Net | PF | Win rate |
|---|---:|---:|---:|---:|
| V1 candidate | -0.0574% | -0.1572% | 0.5791 | 25.40% |
| V1 unica | -0.0954% | -0.1951% | 0.4803 | 23.25% |
| A candidate | -0.0475% | -0.1472% | 0.6359 | 30.58% |
| A unica | -0.1048% | -0.2044% | 0.5073 | 26.56% |
| B candidate | -0.0604% | -0.1598% | 0.6118 | 31.04% |
| B unica | -0.1089% | -0.2082% | 0.5101 | 27.50% |

La mejora aparente de A bajo candidate weighting desaparece al corregir la
duplicacion. En la unidad primaria, A y B son peores que V1 y todos los PF son
inferiores a 1.

## 14. Q8 - Sobreviven costos y stress

| Sistema/vista | BASELINE net | STRESS_20 net | STRESS_30 net |
|---|---:|---:|---:|
| V1 candidate | -0.1572% | -0.2172% | -0.3172% |
| V1 unica | -0.1951% | -0.2551% | -0.3552% |
| A candidate | -0.1472% | -0.2071% | -0.3071% |
| A unica | -0.2044% | -0.2643% | -0.3643% |
| B candidate | -0.1598% | -0.2198% | -0.3197% |
| B unica | -0.2082% | -0.2682% | -0.3682% |

La expectativa gross y el break-even fee calculado son negativos para todos
los sistemas en la vista primaria. Reducir fees o slippage a cero no rescata la
senal agregada; los stresses la degradan monotonicamente.

## 15. Q9 - Que describe suitability por simbolo

| Simbolo | N unico | Midpoint hit | False range | MFE estructural | MAE estructural |
|---|---:|---:|---:|---:|---:|
| ADAUSDT | 37 | 27.03% | 21.62% | 0.8574% | 0.5237% |
| AVAXUSDT | 48 | 14.58% | 39.58% | 0.4925% | 0.5536% |
| BNBUSDT | 26 | 23.08% | 15.38% | 0.7603% | 0.3222% |
| BTCUSDT | 25 | 36.00% | 12.00% | 0.8068% | 0.3608% |
| DOGEUSDT | 37 | 21.62% | 21.62% | 0.6426% | 0.6799% |
| ETHUSDT | 24 | 29.17% | 25.00% | 0.6619% | 0.5704% |
| LINKUSDT | 42 | 28.57% | 33.33% | 1.0490% | 0.5367% |
| LTCUSDT | 34 | 26.47% | 17.65% | 0.7698% | 0.6460% |
| SOLUSDT | 46 | 26.09% | 34.78% | 0.9484% | 0.7184% |
| SUIUSDT | 30 | 43.33% | 21.67% | 1.0936% | 0.7579% |
| XRPUSDT | 33 | 32.32% | 36.36% | 1.0895% | 0.4776% |

SUI tiene el midpoint hit descriptivo mayor y AVAX el menor. Las diferencias
son post hoc, tienen muestras pequenas y no autorizan inclusion o exclusion.

## 16. Q10 - Existe una whitelist causal defendible

No. Las 382 observaciones son maduras a 12 horas, pero el maximo sample prior
por simbolo dentro de 60 dias es 17, por debajo del floor 30. El resultado es:

- `SUITABILITY_STATUS=INSUFFICIENT_HISTORY`: 382/382.
- `prior_tercile=INSUFFICIENT_HISTORY`: 382/382.
- Simbolos elegibles: 0/11.
- `future_by_prior_tercile={}`.

No puede estimarse poder predictivo causal ni publicarse whitelist con esta
evidencia.

## 17. Hipotesis permitidas y limitaciones

La evidencia permite preregistrar, no adoptar, preguntas futuras sobre timing
de entrada, calidad de reclaim y recuperacion parcial a entry. No justifica:

- ampliar stops V1 para capturar recuperaciones post-stop;
- elegir retrospectivamente contratos, lados, meses o simbolos;
- usar A o B como gate final;
- investigar solo costos de ejecucion cuando gross ya es negativo;
- abrir una particion OOS sin una especificacion nueva congelada.

Los paths usan una convencion conservadora para la barra terminal; el OHLC
completo revela la ambiguedad, pero no determina orden intrabar. Las metricas no
son un backtest de portfolio con sizing, concurrencia o leverage. TRAIN permite
diagnostico, no confirmacion independiente.

## 18. Frontera de fase

- `candidate_selected=false`.
- `promotion_authorized=false`.
- `calibration_opened=false`.
- `pre_validation_spec_frozen=false`.
- `validation_opened=false`.
- `holdout_opened=false`.
- `production_modified=false`.
- No hubo cambios live ni ejecucion de ordenes.

**Decision final:** cerrar este diagnostico como evidencia para formular una
hipotesis Range V2 futura. No existe todavia una especificacion seleccionada ni
autoridad para promocion. Cualquier formula, filtro, lifecycle o familia nueva
requiere preregistro y evidencia independiente.
