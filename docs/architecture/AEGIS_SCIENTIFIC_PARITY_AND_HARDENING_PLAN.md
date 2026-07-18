# AEGIS — Plan de Paridad Científica y Endurecimiento (Clean Rebuild)

Documento maestro. Fuente de verdad para la siguiente etapa de implementación.

- Estado: `PROPOSED` — pendiente de ejecución por fases.
- Rama objetivo (ambos repos): `feature/aegis-ts-clean-rebuild`.
- Rama histórica de referencia (solo repo Python): `feature/wraith-phantom-v8` (`1841dbf`).
- Artefactos científicos congelados y evidencia forward: `/home/jasan/Develop/aegis_gen2/` (fuera de ambos repos; READ-ONLY absoluto).
- Base: auditoría comparativa científica/arquitectónica de 2026-07-17/18 (veredicto **C**: la nueva implementación va por buen camino, pero todavía no alcanza paridad científica; confianza alta).

Decisiones ratificadas que este plan NO reabre:

1. La arquitectura nueva se conserva. No se vuelve a `aegis_alpha/`.
2. Python (`src/aegis/`) es exclusivamente cerebro científico. TypeScript (`binance-futures-bot-ts`) es exclusivamente plataforma operacional.
3. Se porta **conocimiento**, no arquitectura legada.
4. Alcance científico actual: **SHORT + NO_TRADE**. LONG queda deshabilitado (`RESEARCH_ONLY`).
5. Los cuatro P0 se corrigen antes de entrenar cualquier bundle candidato nuevo.
6. Ningún bundle puede aprobarse para live desde este plan. El techo de este plan es shadow no ejecutable con evidencia forward.

---

## 0. Principio de migración

**PORTAR CONOCIMIENTO, NO PORTAR ARQUITECTURA LEGADA.**

Se recupera: fórmulas, features útiles, labels, metodología de entrenamiento y selección, calibración, validación, criterios económicos, freezes, invariantes, evidencia y benchmarks.

No se recupera bajo ninguna circunstancia:

- Binance/red en Python (el histórico `gen2_decision_loop.py:default_symbol_filters` llamaba a exchange info desde Python; eso queda prohibido).
- Adaptadores operacionales, bridge, Telegram, status, maintenance, arm tokens ni ejecución en Python.
- Scripts transitorios por fase (auditorías TRRM A–F, forensics Phase O) como código activo.
- Copia de carpetas completas desde `feature/wraith-phantom-v8`.
- Duplicación training/inference.
- Manifests mutables.
- Acceso directo no controlado al SQLite vivo como fuente de entrenamiento.
- Pickles como formato de artefacto publicado (ver §7.3: los modelos nuevos deben exportarse a un formato inspeccionable y verificable por hash).
- El allowlist `premium_symbols=(ADA,SUI,SOL,AVAX)` (construido sobre un subset sesgado por bugs de conteo del auditor retrospectivo).

---

## 1. Estado de partida (verificado en código, 2026-07-17)

### 1.1 Arquitectura actual (fortalezas que se conservan)

Python (`src/aegis/`, 2.853 líneas):

- Dominio inmutable y serialización canónica (`domain.py`, `utils/hashing.py`).
- Universo coordinado de 11 símbolos, timeframe 5m, hash de universo (`config.py`, `config/universe.yaml`).
- Validador de snapshot coordinado, fail-closed (`features.py:MarketSnapshotValidator`).
- Pipeline de features única para training e inferencia: 39 features (`features.py:FEATURE_NAMES`, líneas 24–36), normalizador congelado en el bundle.
- Bundle JSON con content-hash y validación estricta (`models.py:load_model_bundle`); registry inmutable (`training/registry.py:FileArtifactRegistry`).
- Modelo actual: ridge lineal multi-cabeza determinista (`training/train.py:DeterministicLinearTrainer`).
- Capas D3→RV2→TRRM→QMAE→EQM→ECON1 (`layers.py:OrderedScientificLayers`).
- Candidatos + selección global + freeze de decisión (`decision.py`).
- Evidencia append-only con cadena SHA-256 (`evidence.py`).
- Shadow determinista no ejecutable con replay (`shadow.py`).
- API científica (`api.py`, `runtime.py:BrainRuntime`).

TypeScript (`binance-futures-bot-ts/src/brain/`):

- Parser estricto del contrato (`contract.ts`), cliente HTTP fail-closed (`client.ts`), manifest handshake (`manifest.ts`).
- Gate estricto (`decision-gate.ts:StrictDecisionGate`): cualquier razón ⇒ `DENY`; `SHADOW_MODE_NON_EXECUTING` incondicional en shadow; `execution.enabledByConfig=false`; `OperationalContext.allowedSides` ya existe (`decision-gate.ts:7`).
- Shadow coordinado (`shadow.ts`).

### 1.2 Validación actual

- 43 tests Python (verificados en verde), 28 tests TS focalizados de `src/brain` (verificados en verde); la suite completa TS del bot reporta ~624.
- Tests de causalidad directa (`tests/unit/test_feature_causality.py`), embargo 120 min, folds expansivos, paridad de features, fail-closed de bundle corrupto.
- Experimento pre-registrado (`config/candidate_experiment.yaml`): 574.200 filas, 4.231 ciclos coordinados (ene–jun 2026), full-layer 34 señales, expectancy 0.003421, PF 2.043 — **RECHAZADO** por criterios propios: solo 1/4 folds positivo, 2 folds con 0 señales, SUIUSDT 61,8% > 30% de concentración. Artefacto `REJECTED_EXPERIMENT`, no publicado.
- El bundle del registry (`config/bundles/aegis-offline-reference-v1.json`) tiene `"trained": false` y pesos escritos a mano; es referencia, no ciencia.

### 1.3 Lo que estas fortalezas NO prueban

Nada de lo anterior demuestra paridad científica. El runtime hoy no contiene ningún modelo entrenado aprobado; las capas son composiciones heurísticas; el umbral de selección (0.50) no tiene derivación económica; el experimento y el runtime no comparten transformaciones (P0.1); la fuente de datos no tiene gate de finalidad (P0.4); no existe evidencia forward propia (`config/brain.yaml`: `persistence_enabled: false`).

### 1.4 El benchmark histórico vivo

`/home/jasan/Develop/aegis_gen2/` contiene el sistema Gen2 congelado y **en producción** (candidato `gen2-20260711T202935Z`):

- `GEN2_SYSTEM_FREEZE.json` — ata dataset (`d3-v1-build_a`, sha `86be5a15…`), TRRM (`69c03e12…`), EQM (`77887b7c…`), ECON1 (`ea0b6c6f…`), feature_hash (`0fadf4b2…`).
- `GEN2_SELECTION_POLICY.json` — umbral absoluto congelado `0.014322404529216534` = q0.90(s_eqm | 32.405 supervivientes del veto TRRM sobre dev H12).
- `GEN2_LOCKBOX_MANIFEST.json` — lockbox semi-blind con presupuesto de queries.
- `d3/datasets/d3-v1-build_a/` — dataset canónico: 111 features causales, 165.000 filas dense, doble build bit-idéntico, replay parity G7 = 99,984%.
- `rv2/20260711T171832Z/` — TRRM RandomForest (worst-fold PR-AUC 0.317) + calibrador isotónico (ECE 0.198→0.016) + QMAE cuantílico q50/q90 con ajuste conformal (+0.0226; cobertura 0.889–0.904 dentro de banda 0.87–0.93 en todos los folds).
- `eqm1/20260711T201456Z/` — EQM: ExtraTrees regresión sobre `net_quality_after_costs` + HGB clasificación sobre `clean_entry_v4`; score final `reg_component` (el compuesto clf×reg fue **descartado empíricamente**).
- `econ1/` — backtest con precios reales, 3 escenarios de coste, bootstrap por bloques semanales: `eqm_plus_trrm` +$0.140/trade por $100 notional, PF 1.741, CI [0.084, 0.198], única estrategia positiva en los 4 folds.
- `forward/` — **8.096 decisiones forward + 365 selection outcomes, acumulándose en vivo** (última verificada: 2026-07-17 23:32 UTC).

Este directorio es benchmark y fuente de conocimiento. Es READ-ONLY para todo el plan: prohibido escribir, mover, renombrar o "reorganizar" nada dentro.

---

## 2. Definición de paridad científica

Paridad NO significa: mismo número de archivos, mismas clases, mismo código, mismo número de features, ni igualar métricas aisladas.

Paridad significa cumplir **todo** lo siguiente (checklist de cierre del plan; cada punto con evidencia):

1. **Datos finales y canónicos**: todo entrenamiento/backtest consume velas finales verificadas (gate de finalidad) desde snapshots inmutables con hash.
2. **Features causales** con información SHORT equivalente o superior a la histórica (familias de setup SHORT y estado de riesgo presentes).
3. **Labels path-aware** equivalentes o superiores (MFE/MAE, net quality tras costes, ambigüedad hit/stop), no solo retorno terminal.
4. **Modelos competitivos y calibrados**: competición pre-registrada stability-first, no un único ridge por defecto.
5. **QMAE cuantílico válido**: q90 real (pinball + conformal) con cobertura verificada, o renombrado y prohibido como q90.
6. **Tail risk calibrado**: probabilidad con ECE/Brier medidos out-of-fold y calibrador dentro del bundle.
7. **ECON independiente**: replay económico con precios reales, no reutilización del label de entrenamiento.
8. **Threshold económico congelado**: umbral absoluto derivado de la economía validada, con hash y modo validate.
9. **Freeze integral**: un manifest que ata dataset+features+labels+folds+modelos+calibradores+thresholds+policy+universo+entorno+commit+criterios.
10. **Evidencia persistente**: cadena SHA-256 en disco, con recuperación tras reinicio y sin duplicados.
11. **Reproducibilidad**: doble ejecución bit-idéntica (o con tolerancia declarada) del dataset, del entrenamiento y del experimento.
12. **Robustez por fold**: criterios pre-registrados sobre el peor fold, no solo el agregado.
13. **Concentración controlada**: cap por símbolo obligatorio (se conserva el ≤30% actual).
14. **Comparación pareada contra el benchmark Gen2** (mismos timestamps/símbolos/side) documentada.
15. **SHORT + NO_TRADE**: ninguna métrica mezclada LONG/SHORT puede aprobar un bundle SHORT.
16. **Runtime idéntico al evaluado**: el experimento y la API producen bit-a-bit (o con tolerancia explícita ≤1e-12) la misma decisión para el mismo bundle y FeatureBatch.

---

## 3. P0 — Bloqueadores de validez científica

Ningún experimento candidato nuevo (Fase E) puede ejecutarse hasta cerrar P0.1–P0.4.

### P0.1 — Training/serving skew (transformaciones divergentes)

**Problema (verificado):** el experimento evalúa salidas lineales crudas con clipping y un gate propio (`training/experiment.py:311–318`: `tail = clip(raw)`, `|direction| ≥ direction_threshold=0.10`, score experimental de 4 factores), mientras el runtime aplica `softmax` a dirección y `sigmoid` a tail/quality (`models.py:210–222`) y compone un score de 6 factores con otros umbrales (`layers.py:96–103`; `config/models.yaml`: direction 0.50, selection 0.50). El mismo bundle exportado por `build_candidate_bundle` (`experiment.py:361–386`) se comporta distinto en `api.py` que en el experimento que lo "midió".

**Consecuencia:** las métricas de promoción no describen el sistema que decidiría en shadow. Es el mecanismo exacto por el que un candidato podría aprobarse midiendo otra cosa.

**Solución obligatoria:**

- UNA única función autoritativa de predicción y UNA única transformación de cabezas: la del runtime (`DeterministicModelRuntime.predict` + `OrderedScientificLayers.apply` + `ScientificCandidateBuilder` + `GlobalSelectionPolicy`). El experimento debe **importar y ejecutar exactamente esos objetos**, construyendo `MarketSnapshot`/`FeatureBatch`/`PortfolioContext` sintéticos por ciclo, en lugar de `_prediction()` + `evaluate_strategies` con su score paralelo.
- Eliminar los caminos alternos: `experiment.py:_prediction` y el bloque de score/gate propio (líneas 305–323) desaparecen o quedan reducidos a adaptadores triviales sobre el camino único.
- Golden fixtures comunes: un conjunto versionado de FeatureBatch + bundle con salidas esperadas (dirección, probabilidades, tail, qmae, eqm, econ, score, ranking, decisión) usado por tests de training, runtime, shadow y API.
- Test de paridad experimento→bundle→runtime→API: mismo bundle + mismo FeatureBatch ⇒ misma decisión bit-a-bit; si hay float no determinista, tolerancia explícita documentada (≤1e-12) — nunca implícita.

**Criterio de cierre:** existe un test que falla si CUALQUIER transformación difiere entre experimento y runtime (dirección, probabilidades, tail risk, QMAE, EQM, ECON, score, ranking, decisión). Los criterios de promoción de Fase E se computan exclusivamente sobre decisiones del camino único.

### P0.2 — `qmae_q90` no es un q90 real

**Problema (verificado):** la cabeza `qmae_q90` se entrena por regresión ridge de media condicional sobre la excursión adversa realizada (`experiment.py:207` construye el target `qmae = max(0, adverse)`; `train.py:41–57` ajusta por mínimos cuadrados), pero se consume como cuantil 90 en la capa QMAE (`layers.py:68–69`) y como `stop_distance_fraction` del `RiskIntent` (`decision.py:57`). Referencia histórica correcta: `gen2_rv2_train.py:232–284` — HistGradientBoosting con `loss="quantile"` q50/q90, split-conformal one-sided sobre residuos de calibración, banda de cobertura pre-registrada 0.87–0.93 verificada por fold, por horizonte y por símbolo.

**Consecuencia:** subestimación sistemática de la excursión adversa; una distancia de stop científica basada en una media, etiquetada como cuantil.

**Decisión recomendada (explícita): opción A — implementar QMAE cuantílico real.** La opción B (renombrar la salida a `qmae_mean` y prohibir su uso como q90 y como stop distance) solo es aceptable como paso intermedio si la Fase C se retrasa; en ese caso el renombrado es obligatorio e inmediato y `RiskIntent.stop_distance_fraction` debe dejar de derivarse de esa cabeza.

**Especificación del QMAE objetivo:**

- Target: MAE del lado de la hipótesis dentro del horizonte H12, definido igual que el histórico (para SHORT: `max(0, (max(high_futuro) − entry)/entry)`; para LONG quedará definido pero NO entrenado/promovido en este plan — ver §6).
- Unidades: fracción de precio (la conversión a ROE es responsabilidad de reporting, no del modelo). Documentar la equivalencia con el histórico (ROE = fracción × leverage).
- Horizonte: 12 barras de 5m, ventana `[t+1, t+12]` sobre velas finales.
- Modelos: cuantílicos con pinball loss para q50/q90 (candidato principal: HistGradientBoosting quantile; baseline: cuantil incondicional del train).
- Conformal: split-conformal one-sided sobre residuos de un bloque de calibración temporal embargado (metodología de `gen2_rv2_train.py:261–268`).
- Aceptación: cobertura conformal q90 dentro de [0.87, 0.93] en TODOS los folds; pinball ≤ 0.9× baseline incondicional en todos los folds; cobertura por símbolo dentro de una banda declarada; cobertura por régimen reportada.
- Fail-closed: si el bundle carece de bloque QMAE cuantílico válido (o cobertura fuera de banda en validación), la capa QMAE debe degradar a veto (`qmae_quality=0`, reason code explícito), nunca a un valor optimista.

**Criterio de cierre:** ninguna salida llamada `q90` en el sistema proviene de una regresión de media; hay tests de cobertura conformal y de fail-closed.

### P0.3 — Calibración de probabilidades

**Problema (verificado):** `models.py:220–221` aplica `sigmoid` a cabezas ridge entrenadas sobre targets binarios (`tail_event`, `clean_quality`); la escala resultante (~[0.5, 0.73] para entradas en [0,1]) no es una probabilidad. Los umbrales (`trrm_max_tail_probability: 0.70` en `config/models.yaml`) operan sobre una escala sin significado demostrado. El Brier del experimento usa la pseudo-probabilidad `(direction+1)/2` (`experiment.py:346`). Referencia histórica: `gen2_rv2_train.py:198–229` — calibración por fold (raw vs Platt vs isotónica), selección por ECE medio, calibrador congelado dentro del artefacto.

**Solución obligatoria:**

- Etapa de calibración out-of-fold para toda cabeza interpretada como probabilidad (tail risk, clean/quality, dirección si se usa como probabilidad): candidatos raw / Platt (baseline) / isotónica; selección pre-registrada por ECE con Brier como desempate.
- Métricas reales: Brier sobre el target verdadero, ECE con binning declarado, reliability diagrams como artefacto del reporte (datos JSON; el render es opcional).
- El calibrador seleccionado se almacena DENTRO del bundle (parámetros serializados en JSON: para isotónica, los puntos de la función escalonada; para Platt, coeficientes) y el runtime lo aplica de forma idéntica (mismo código compartido).
- Fallback: bundle sin bloque de calibración ⇒ la cabeza no puede usarse como probabilidad; la capa correspondiente entra en modo veto fail-closed con reason code.
- El umbral de tail risk NO se fija hasta tener escala calibrada; el 0.70 actual queda marcado como placeholder inválido y se re-deriva en Fase D.

**Criterio de cierre:** tests que verifican (a) aplicación idéntica del calibrador en training y runtime sobre golden fixtures; (b) ECE out-of-fold reportado por candidato; (c) fail-closed ante calibrador ausente.

### P0.4 — Finalidad y canonicidad de datos (la función real de D3)

**Problema (verificado):** el experimento de Fase 2 leyó `data/binance_candles.db` directamente (`config/candidate_experiment.yaml: source`). Esa fuente tiene riesgo documentado de velas capturadas mid-bar (auditoría F0.2, `audit_candle_finality_f03.py` en la rama histórica: 18–25% de velas de jun–jul 2026 eran subconjuntos estrictos de las klines finales de Binance). El D3 histórico era una disciplina de integridad de datos (snapshots inmutables, gates G1 esquema / G2 duplicados / G3 finalidad por double-fetch / G4 gaps / G5 cobertura, doble build bit-idéntico, G7 replay parity ≥99.9% — `gen2_d3_build.py`), NO un clasificador de régimen. El "D3" actual (`layers.py:classify_regime`, líneas 130–147) es ciencia nueva sin validar que usurpó el nombre.

**Solución obligatoria (dos opciones, decisión en Fase A):**

- **Opción 1 (recomendada como punto de partida):** consumir READ-ONLY los snapshots canónicos existentes de `/home/jasan/Develop/aegis_gen2/d3/` (verificando sus manifests y hashes antes de cada uso) como fuente de entrenamiento para el periodo que cubren, y definir el mecanismo nuevo solo para extender el rango temporal.
- **Opción 2:** reconstruir la disciplina D3 dentro de `src/aegis/` (módulo nuevo, p. ej. `src/aegis/data/canonical.py`): snapshot inmutable con manifest+sha256, gate de finalidad (double-fetch de klines finales o comparación contra una segunda captura diferida), detección de mutaciones vs snapshot previo, esquema/duplicados/gaps/cobertura, doble build bit-idéntico, replay parity, alineación de timestamps, cuarentena de datos ambiguos (excluidos con registro, nunca "reparados" silenciosamente).

Nota de pureza: la captura de datos (si requiere red) NO vive en `src/aegis/` runtime; es tooling offline separado (p. ej. `scripts/`) con salida inmutable. El cerebro solo consume snapshots ya verificados. Esto mantiene "cero Binance en Python científico" en el sentido que importa: el runtime de decisión jamás toca red.

- Renombrar/redefinir la capa actual "D3" de `layers.py`: el clasificador de régimen pasa a llamarse por lo que es (p. ej. `REGIME`) o se documenta que D3-integridad vive en la ruta de datos y el clasificador es un componente distinto (ver P1.4).

**Criterio de cierre:** prohibido entrenar el bundle candidato de Fase E sobre datos que no hayan pasado el gate de finalidad; test que rechaza una fuente con velas no finales sembradas; doble build del dataset bit-idéntico verificado.

---

## 4. P1 — Recuperación de paridad científica

### P1.1 — Feature schema v2 (`aegis-features-v2`)

Nuevo esquema versionado que conserva las 39 actuales (especialmente las cross-sectional nuevas: `relative_return_*`, `cross_rank_return_6`, `cross_dispersion_return_6`, `market_breadth_6`, `market_direction_6`, `market_concentration_6`, `btc/eth_divergence_6`) y recupera SOLO familias históricas con valor demostrado. Fuente histórica: `aegis_alpha/tools/build_trrm_causal_feature_dataset_d.py:compute_causal_features` (111 features en el dataset congelado).

Criterios de selección (no se portan las 111 automáticamente): causalidad estricta, estabilidad entre folds, utilidad para SHORT, no redundancia con las 39 existentes, importancia en los modelos históricos (los artefactos de `aegis_gen2/rv2` y `eqm1` permiten extraer importancias reales), mantenibilidad.

Matriz de decisión por familia (la matriz completa feature-a-feature es entregable de Fase B; toda feature portada debe llevar fórmula, ventana, test de causalidad y golden value):

| Familia histórica | Fórmula/fuente histórica | Consumidor histórico | Equivalente actual | Decisión | Justificación |
|---|---|---|---|---|---|
| SHORT setup proxies (15: `breakdown_proxy_12/24`, `close_below_rolling_low_12/24`, `distance_to_rolling_low/high_12/24`, `failed_breakdown_proxy`, `fake_breakdown_risk_proxy`, `room_to_fall_proxy_24`, `extension_down_proxy`, `exhaustion_down_proxy`, `rebound_risk_proxy`, `squeeze_risk_proxy_causal`) | `compute_causal_features` líneas 316–336 | TRRM/QMAE/EQM | ninguno | **PORTAR** (prioridad 1) | señal específica de SHORT; familia completa ausente |
| Risk state proxies (5: `immediate_reversal_risk_proxy`, `overextended_down_risk_proxy`, `low_room_to_fall_risk_proxy`, `high_wick_reclaim_risk_proxy`, `squeeze_plus_reclaim_risk_proxy`) | líneas 347–351 | TRRM | ninguno | **PORTAR** (prioridad 1) | alimentaba el modelo de tail |
| EMA/trend (ema_48, `close_vs_ema_6/12/24/48`, `ema_slope_6/24`, `trend_stack_short/long`, `trend_compression`) | líneas 301–309 | todos | solo `ema_gap_6_12/12_24`, `ema_slope_12` | **PORTAR** | estructura de tendencia perdida |
| Momentum asimétrico (`downside_momentum_6`, `upside_momentum_6`) | líneas 313–314 | todos | ninguno | **PORTAR** | asimetría relevante para SHORT |
| Volatilidad (`rolling_range_std_12/24`, `volatility_compression_12_24`, `volatility_expansion_6_24`) | líneas 288–292 | todos | parcial (`volatility_ratio_6_24`, `range_expansion`) | **PORTAR** las std y compresión | dispersión de rango perdida |
| Volumen (`volume_spike_12`, `volume_trend_12`, `volume_zscore_12`) | líneas 294–299 | todos | parcial | **PORTAR** spike y trend | eventos de volumen perdidos |
| Régimen relativo (`high_vol_regime_proxy`, `low_vol_regime_proxy` vs cuantiles rolling 96) | líneas 345–346 | todos | ninguno | **PORTAR** | régimen relativo a la propia historia |
| Contexto BTC/ETH (`btc/eth_volatility_12`, `btc/eth_trend_proxy`, `btc/eth_ret_3/12`) | `add_market_context` | todos | solo divergencias ret_6 | **PORTAR** vol y trend | contexto direccional del mercado líder |
| Velas consecutivas (`consecutive_red/green_count`) | líneas 280–281 | todos | ninguno | **PORTAR** | secuencia de presión |
| Retornos extra (ret_2, log_ret_3/6) | líneas 260–263 | todos | parcial | **RECHAZAR** salvo evidencia de importancia | redundancia probable |
| `is_red/green_candle` | líneas 278–279 | todos | derivable de `close_to_open_return` | **RECHAZAR** | redundante |
| EMAs absolutas (`ema_6/12/24/48` en precio) | línea 302 | (interno) | gaps normalizados actuales | **RECHAZAR** como feature directa | no comparable entre símbolos; usar solo derivadas normalizadas |

Investigación obligatoria: `momentum_acceleration_3_12 = ret_3 − ret_12/4` (`features.py:259`) difiere del histórico `momentum_accel_3_12 = momentum_3 − momentum_12` (línea 310). El `/4` no tiene justificación documentada. Resolver: o se documenta como normalización por barra intencional Y se añade golden test, o se restaura la semántica histórica bajo un nombre inequívoco. No puede quedar como está.

Reglas de implementación: `FEATURE_SCHEMA_VERSION = "aegis-features-v2"` con hash nuevo; los bundles v1 siguen siendo válidos contra v1 (compatibilidad por verificación de hash, no por conversión); tests de causalidad y golden values numéricos por cada feature nueva; presupuesto razonable de dimensionalidad (objetivo orientativo: 60–80 features, no 111).

### P1.2 — Labels SHORT V4 path-aware

Fuente histórica: `aegis_alpha/turbo/short_quality_v4_labels.py` (`compute_short_path_metrics_v4`, `ShortV4Config`, `_hit_before_stop`).

Los labels actuales (`experiment.py:203–207`) son retorno terminal + MAE + umbral fijo 3%. Se sustituyen por labels path-aware versionados (`aegis-labels-short-v4`):

- **Entrada hipotética**: definir explícitamente. El histórico ECON1 usó next-bar-open; los labels V4 usaron close de la barra de señal. El plan exige elegir UNA convención por label, documentarla, y que ECON (P1.5) use next-bar-open siempre.
- **MFE / MAE** (fracción de precio, lado SHORT): `mfe = max(0, (entry − min(low[t+1..t+12]))/entry)`; `mae = max(0, (max(high[t+1..t+12]) − entry)/entry)`.
- **net_quality_after_costs** = `mfe − mae − coste_round_trip` (coste versionado: fees + slippage por lado; funding si el horizonte lo justifica; documentar unidades — el histórico lo expresaba en ROE a 20x: registrar la equivalencia y elegir fracción de precio como unidad canónica nueva).
- **clean_entry / bad_entry / tail_event**: umbrales versionados equivalentes a los históricos (`ShortV4Config`: clean_mfe/mae/ratio/tiempo; bad_mae/low_mfe; tail = MAE ≥ umbral), convertidos de ROE a fracción de precio de forma documentada.
- **hit-before-stop con ambigüedad**: portar `_hit_before_stop` — si target y stop se tocan en la misma barra, el resultado es `ambiguous=true` y se trata conservadoramente (para SHORT: como stop), nunca como hit.
- **Gaps**: si la ventana futura contiene un gap temporal, el label se marca inválido (cuarentena), no se interpola.
- **time_to_mfe / time_to_mae / mfe_before_mae**: portar (alimentan diagnósticos y el criterio clean).
- Sin `premium_symbols` ni ningún allowlist por símbolo.

Tests: golden values contra casos construidos a mano; propiedad de causalidad (labels solo de `[t+1, t+horizonte]`); test de ambigüedad; test de gaps.

### P1.3 — Modelos y competición stability-first

Se porta la **metodología** de `gen2_rv2_train.py` y `gen2_eqm1_train.py`, no los pickles.

- Candidatos mínimos — TRRM (clasificación tail): RandomForest, HistGradientBoosting, y baseline logístico/ridge (el ridge actual queda como baseline obligatorio, nunca como único). QMAE: cuantílicos (P0.2). EQM: ExtraTrees/HGB regresión sobre net_quality + HGB/RF clasificación sobre clean_entry + baselines lineales; **evaluación separada de clean y net_quality, nunca mezclados en un target**.
- Selección pre-registrada ANTES de correr: stability-first — peor fold primero (`worst_fold` como clave primaria de ranking, como `select_winner` histórico), luego media, luego varianza, luego coste computacional; hipótesis H1/H2/H3 declaradas (lift mínimo vs prevalencia por fold, ratio de estabilidad, incrementalidad del stack).
- Además: calibración (P0.3) por candidato, concentración por símbolo del top-decil, robustez por símbolo y por régimen reportadas.
- No asumir que el ganador histórico (RF) gana con las features v2; no asumir que el ridge basta.
- **Formato de artefacto:** prohibidos los pickles como artefacto publicado. Los modelos de árboles deben exportarse a una representación JSON inspeccionable (estructura de árboles serializada) con evaluador determinista propio en `src/aegis`, o el candidato no es publicable. Este requisito es innegociable: preserva la clase de garantías que eliminó el bug de los dos venvs (`ENVIRONMENT_MISMATCH_VS_FREEZE`). El coste de implementar el export/eval de árboles se paga una vez y protege para siempre. Si un candidato no puede exportarse de forma verificable, no compite.
- El `ModelBundle` v2 (`models.py`) gana bloques opcionales tipados: `calibration`, `qmae_quantile`, `tree_ensembles` — todos bajo el mismo content-hash.

### P1.4 — Estructura semántica correcta de las capas

Mapa objetivo (sustituye la cadena lineal `D3→RV2→TRRM→QMAE→EQM→ECON1` actual, que codifica una jerarquía que nunca existió):

```
DATOS    D3        = disciplina de integridad/canonicidad (gates, snapshots, replay)  [ruta de datos, no capa de inferencia]
MODELOS  RV2       = programa/dominio de riesgo que contiene:
                       TRRM  = probabilidad CALIBRADA de tail risk (modelo)
                       QMAE  = cuantil q90 conformal de excursión adversa (modelo)
         EQM       = calidad de entrada (clean) y net_quality (modelos separados)
GATES    veto TRRM = corte por presupuesto/umbral congelado (gate)
         gate QMAE = q90 ≤ máximo tolerado (gate)
INFER.   REGIME    = clasificador de régimen (contexto/diagnóstico; NO multiplica el score hasta validarse)
SCORE    score     = magnitud ÚNICA con unidades declaradas para rankear candidatos
ECON     ECON      = programa de validación económica independiente (no una capa de runtime)
POLICY   Selection = umbral absoluto congelado derivado de ECON
EVID.    evidencia = registro encadenado de todo lo anterior
```

Definiciones que el código y los docs deben respetar: **modelo** (aprende de datos), **capa/gate** (regla determinista sobre salidas de modelos), **programa de validación** (proceso offline que produce evidencia y umbrales), **métrica** (medida de evaluación), **score** (cantidad de ranking con unidades declaradas), **evidencia** (registro inmutable).

Correcciones concretas en `layers.py`:

- Eliminar el doble conteo `rv2_tail`/`trrm_compatibility` (`layers.py:66–67, 96–103`): la misma cantidad no puede aparecer como factor del score Y como veto. El veto es gate; el score no la re-multiplica.
- `d3_confidence` deja de multiplicar el score (`layers.py:102`) hasta que exista validación del clasificador de régimen; el régimen se conserva como contexto/reporting.
- El score deja de ser un producto heurístico de 6 factores: la definición nueva del score (unidades, composición) se pre-registra en Fase D y se deriva del EQM validado (como el `s_eqm = reg_component` histórico), no de una multiplicación ad hoc.
- Reevaluar `ReasonCode` para que cada gate tenga razón propia y estable.

### P1.5 — ECON real (replay económico independiente)

Sustituye `econ_edge = expected_edge − 0.0014` (`layers.py:80`) y la evaluación autorreferencial del experimento (`experiment.py:337` reutiliza `target.expected_return`). Metodología de referencia: `gen2_econ1_backtest.py`.

Especificación del motor ECON nuevo (módulo offline, p. ej. `src/aegis/training/econ.py`):

- Entrada **next-bar-open** tras la señal; salida por horizonte H12 close; sin gestión intermedia (o con reglas predefinidas y pre-registradas si se añaden stops — nunca improvisadas).
- Precios reales del snapshot canónico (no labels, no features).
- MFE/MAE del trade simulado; registro de invalidación/expiración.
- Costes: 3 escenarios pre-registrados (optimista/base/pesimista) con fee+slippage por lado y funding por hora (valores de partida: los históricos A/B/C de `gen2_econ1_backtest.py:COST_SCENARIOS`).
- Igualdad de presupuesto entre estrategias comparadas (mismo nº de trades por fold).
- Baselines mínimos: no-trade, aleatorio con veto, momentum, reversión, señal sin gates, señal con gates.
- Bootstrap por bloques temporales (semanales) con CI 90% de la expectancy.
- Métricas: expectancy, PF, win rate, max drawdown, CVaR 5%, worst trade, turnover, concentración (por símbolo/mes/trade), desglose por fold, por símbolo, por régimen y **por side**.
- Separación estricta de vocabulario y unidades: score predictivo (adimensional/unidades del EQM) ≠ probabilidad ≠ quality (fracción de precio neta) ≠ edge económico (fracción sobre notional en el replay) ≠ PnL simulado (moneda por notional). Ningún número cruza de categoría sin conversión explícita.

### P1.6 — Selection Policy con umbral absoluto congelado

Metodología de referencia: `gen2_selection_policy.py`. El 0.50 actual (`config/models.yaml: selection_score`) queda invalidado como criterio científico.

- Procedimiento: aplicar vetos (TRRM calibrado, QMAE) → población superviviente del dev set → derivar umbral absoluto = cuantil del score correspondiente al presupuesto que ECON validó (p. ej. q0.90 si ECON validó top-decil) → congelar valor + hashes (dataset, modelos, calibradores, feature schema, bundle).
- Modo `validate`: recompute-from-scratch con hard-stop ante cualquier drift (> 1e-9) o mismatch de hash.
- Regla de decisión: si ningún candidato supera el umbral ⇒ `NO_TRADE` con reason code (`BELOW_FROZEN_THRESHOLD`). Prohibido best-of-cycle-regardless.
- El umbral es específico de: versión de policy, bundle, y **side** (solo SHORT en este plan). Inmutable durante shadow.

### P1.7 — Freeze integral del sistema

El freeze por decisión (`decision.py:DecisionFreezer`) y el content-hash por bundle se conservan; falta el freeze de sistema (referencia: `GEN2_SYSTEM_FREEZE.json`).

- Nuevo artefacto `SYSTEM_FREEZE` (JSON, inmutable, con hash propio) que ata: dataset/snapshots (hashes), feature schema+hash, versión de labels, definición de folds y embargo, normalizadores, modelos y calibradores (hashes), thresholds, Selection Policy (hash), reporte ECON (hash), universo+timeframe, entorno (python/numpy/versiones), commit de código, bundle final y criterios de promoción usados.
- Validación fail-closed: el runtime y el experimento verifican el freeze antes de operar con un bundle candidato/aprobado.
- Ciclo de vida de bundles con estados explícitos y transiciones válidas: `EXPERIMENTAL → CANDIDATE → SHADOW_APPROVED → LIVE_APPROVED` (este plan termina como máximo en `SHADOW_APPROVED`).
- **Regla dura nueva:** `load_model_bundle` rechaza `approved=true` con `metadata.trained=false`. El bundle de referencia actual (`aegis-offline-reference-v1`) se re-etiqueta en consecuencia (p. ej. `approved=false` + propósito referencia) — es el único cambio permitido sobre ese archivo y exige regenerar su content-hash.

---

## 5. P2 — Endurecimiento y mejoras

- **Evidencia persistente**: activar y probar la persistencia de la cadena SHA-256 (`config/brain.yaml: persistence_enabled` → true en entornos de experimento/shadow), recuperación tras reinicio (re-anclaje al último hash), no duplicación de outcomes (idempotencia por `decision_id`+tipo), verificación de cadena al arranque.
- **Golden values numéricos**: fixtures con valores esperados calculados a mano/externamente para features, transformaciones, capas y score.
- **Thresholds de régimen versionados**: los números mágicos de `classify_regime` (0.035, 0.70, ±0.002) pasan a configuración versionada con procedencia documentada.
- **Folds/embargo configurables**: eliminar la duplicación entre `training/dataset.py:walk_forward_splits` y los folds inline de `experiment.py:401–414`; una sola implementación parametrizada.
- **Lockbox semi-blind con presupuesto de queries**: reserva temporal declarada en el pre-registro del experimento, contador de consultas persistente (metodología de `GEN2_LOCKBOX_MANIFEST.json`).
- **Benchmark pareado contra Gen2** (ver §7).
- **Cobertura Python medida** (instalar herramienta de coverage es aceptable en fase de implementación; hoy no hay ninguna).
- **Determinismo entre procesos**: test que ejecuta el pipeline dos veces en procesos separados y exige igualdad bit-a-bit de dataset, artefacto y decisiones.
- **Compatibilidad de bundle**: tests de rechazo para cada combinación inválida (schema/feature-hash/universe/estado trained-approved).
- **Test de entorno**: registrar versiones en el freeze y verificar en carga (sustituye al `ENVIRONMENT_MISMATCH_VS_FREEZE` histórico; menos crítico con artefactos JSON, pero se conserva como defensa).
- **Performance y memoria**: presupuesto p95 < ciclo de 5m con margen (referencia actual medida: p95 ≈ 130 ms para 11 símbolos); test de regresión de latencia.
- **Documentación matemática**: cada fórmula portada con derivación, unidades y fuente histórica en los docs de `docs/implementation/`.

---

## 6. SHORT y LONG

**Alcance de todo este plan: SHORT + NO_TRADE.**

LONG: `RESEARCH_ONLY`, `SIDE_NOT_ENABLED`, `NOT_ELIGIBLE_FOR_SHADOW`, `NOT_ELIGIBLE_FOR_EXECUTION`.

- TypeScript: `allowedSides: ['SHORT']` en la configuración del gate (`OperationalContext.allowedSides` ya existe en `decision-gate.ts:7`; el gate ya emite `SIDE_NOT_ALLOWED`). En shadow, las decisiones LONG se registran como evidencia pero se marcan con esa razón.
- Python: el enum `TradeSide.LONG` se conserva en el dominio; la Selection Policy NO promueve candidatos LONG (reason code explícito, p. ej. `SIDE_NOT_ENABLED`); todos los criterios de promoción, reportes ECON y métricas se desglosan **por side**; ninguna métrica combinada LONG/SHORT puede aprobar un bundle SHORT (regla dura del experimento de Fase E — el reporte debe demostrar el split).
- Motivo (auditado): el camino promovido histórico era SHORT-only (`gen2_decision_loop.py` solo emite `CANDIDATE_SHORT`; labels y ECON1 son SHORT); el soporte LONG histórico fue investigación aparte sin freeze ni forward. Además, la cabeza SHORT actual es la negación exacta de la LONG (`experiment.py:370`), incapaz de especializarse.
- Cuando LONG se aborde (fuera de este plan), deberá pasar su propio ciclo completo: dataset, labels, features, modelos, calibración, ECON, threshold, freeze, forward y canary — separado y pre-registrado.

---

## 7. Benchmark pareado contra el sistema histórico

Los artefactos de `/home/jasan/Develop/aegis_gen2/` se usan como **benchmark científico vivo**, no como arquitectura a copiar.

Diseño de la comparación (Fase F):

- Mismos timestamps de decisión, mismos 11 símbolos, mismo lado SHORT, mismos datos canónicos.
- Por cada timestamp: decisión nueva (sistema clean-rebuild con su bundle candidato) vs decisión histórica registrada (`forward/forward_decisions.jsonl`: `tail_score`, `qmae_q90`, `eqm_score`, `vetoed_by_trrm`, `hypothetical_action`) — el histórico se lee de su evidencia, nunca se re-ejecuta ni se modifica.
- Comparar: tasa de NO_TRADE, distribución de scores, tail risk, QMAE, calidad, selección final, outcomes H12 (resueltos con velas finales), concentración, expectancy, drawdown.
- Reglas de integridad: no modificar el stream histórico; no contaminar su evidencia (los reportes del benchmark viven en el repo nuevo, p. ej. `reports/parity_benchmark/`, gitignorados si son voluminosos); el sistema nuevo no puede leer decisiones u outcomes futuros para decidir (alineación estrictamente causal por timestamp).
- Resultado esperado: un reporte pareado con verdicto descriptivo (no gate automático de promoción, pero sí evidencia obligatoria del expediente del candidato).

---

## 8. Fases del plan

Regla global: **no se puede saltar una fase porque los tests compilen.** Cada fase cierra con sus criterios de aceptación demostrados y con commits pequeños y reversibles. Cambios prohibidos en TODAS las fases: tocar Binance/credenciales, PM2, órdenes, kill switches, `execution.enabledByConfig`, autorización, el directorio `aegis_gen2/` (escritura), y push sin orden del owner.

### FASE A — Corrección de los cuatro P0

- **Objetivo:** cerrar P0.1–P0.4.
- **Prerrequisitos:** ninguno.
- **Archivos probables:** `src/aegis/training/experiment.py` (reescritura de la evaluación sobre el camino único), `src/aegis/models.py`, `src/aegis/layers.py`, `src/aegis/decision.py` (solo lo necesario para P0.1), módulo nuevo de datos canónicos (`src/aegis/data/` + tooling en `scripts/`), `config/models.yaml` (marcar thresholds placeholder), tests nuevos en `tests/unit/` y `tests/integration/`.
- **Cambios permitidos:** refactor del experimento; renombrados honestos (`qmae_mean` transitorio si aplica); creación del gate de finalidad; golden fixtures.
- **Cambios prohibidos:** entrenar candidatos; tocar el contrato TS salvo campos estrictamente necesarios; añadir features (eso es Fase B).
- **Pruebas:** test de paridad experimento↔runtime↔API (P0.1); test de fail-closed QMAE/calibración; test de rechazo de fuente no final; doble build bit-idéntico.
- **Entregables:** camino único de predicción; decisión documentada Opción 1 vs 2 de datos (P0.4); fixtures golden.
- **Criterio de aceptación:** los 4 criterios de cierre de §3 en verde.
- **Criterio de rechazo/detención:** si la paridad bit-a-bit no puede lograrse con tolerancia ≤1e-12 documentada, STOP y reportar la causa antes de continuar.
- **Riesgos:** reescribir la evaluación puede invalidar comparaciones con el experimento de Fase 2 (aceptado: aquel candidato ya fue rechazado).
- **Dependencia:** B–I dependen de A.

### FASE B — Feature schema v2 y labels SHORT V4

- **Objetivo:** P1.1 + P1.2.
- **Prerrequisitos:** Fase A (los tests de causalidad y golden se escriben contra el camino único).
- **Archivos probables:** `src/aegis/features.py` (v2 versionado), módulo de labels (`src/aegis/training/labels.py`), tests dedicados.
- **Pruebas:** causalidad por feature, golden values, ambigüedad hit/stop, gaps, resolución del `/4` de `momentum_acceleration_3_12`.
- **Entregables:** matriz feature-a-feature completa (PORTAR/REDISEÑAR/RECHAZAR con justificación y test), `aegis-features-v2` con hash, labels `aegis-labels-short-v4`.
- **Aceptación:** todas las features portadas con test de causalidad y golden; dimensionalidad final justificada; labels con golden tests.
- **Rechazo:** cualquier feature sin test causal no entra al esquema.
- **Riesgos:** redundancia/colinealidad — mitigar con el criterio de no-redundancia y con importancias en Fase C.
- **Dependencia:** C entrena sobre v2 + labels V4.

### FASE C — Modelos, calibración, QMAE y RV2

- **Objetivo:** P1.3 + cierre definitivo de P0.2/P0.3 con modelos reales.
- **Prerrequisitos:** A y B; datos canónicos disponibles (P0.4 resuelto).
- **Archivos probables:** `src/aegis/training/train.py` (competición), módulo de export/eval de árboles JSON, `src/aegis/models.py` (bundle v2 con bloques `calibration`/`qmae_quantile`/`tree_ensembles`), tests.
- **Pruebas:** paridad sklearn→export JSON (misma predicción con tolerancia declarada), cobertura conformal por fold/símbolo, ECE/Brier out-of-fold, determinismo entre procesos.
- **Entregables:** protocolo pre-registrado de competición (hipótesis y claves de ranking ANTES de correr), reporte de competición, artefactos inspeccionables.
- **Aceptación:** ganadores seleccionados por el protocolo pre-registrado; QMAE en banda de cobertura; calibración medida; export verificado.
- **Rechazo:** si ningún candidato supera los baselines con estabilidad (peor fold), se reporta honestamente `NOT_BEATEN` y se detiene el plan en C (no se fabrica un ganador).
- **Riesgos:** coste del evaluador de árboles propio — pagarlo; es la garantía anti-pickle/anti-entorno.
- **Dependencia:** D deriva umbrales de estos modelos.

### FASE D — ECON, Selection Policy y freeze integral

- **Objetivo:** P1.4 + P1.5 + P1.6 + P1.7.
- **Prerrequisitos:** C.
- **Archivos probables:** `src/aegis/training/econ.py` (nuevo), `src/aegis/layers.py` (semántica corregida, score re-definido y pre-registrado), `src/aegis/decision.py` (policy con umbral congelado + reason codes), módulo de freeze de sistema, `config/models.yaml` (thresholds derivados).
- **Pruebas:** replay ECON con golden trades; bootstrap reproducible por seed; policy `validate` con hard-stop; rechazo `approved && !trained`; unidades (test que impide mezclar score/probabilidad/fracción).
- **Entregables:** motor ECON, reporte ECON por escenarios/folds/símbolos/regímenes/side, `SELECTION_POLICY` congelada con hash, `SYSTEM_FREEZE`, capa de score corregida sin doble conteo.
- **Aceptación:** ECON independiente de labels de entrenamiento (verificable); umbral absoluto derivado y congelado; freeze validable fail-closed.
- **Rechazo/detención:** si ECON no muestra estrategia positiva robusta (peor fold, CI), STOP con veredicto honesto — no se pasa a E.
- **Dependencia:** E usa ECON+policy+freeze como criterios.

### FASE E — Experimento candidato pre-registrado

- **Objetivo:** producir (o rechazar) un bundle CANDIDATE real.
- **Prerrequisitos:** A–D íntegras. Datos con gate de finalidad. Pre-registro inmutable (config YAML committeada ANTES del run) con: ventanas, folds, criterios (se conservan los actuales: ≥100 señales test, ≥3/4 folds positivos, PF ≥1.05, expectancy > mejor baseline direccional, concentración ≤30%, sin leakage; añadidos: desglose por side, cobertura QMAE, ECE máximo, criterio de peor fold).
- **Pruebas:** el experimento corre por el camino único (test de Fase A vigente); doble ejecución reproducible.
- **Entregables:** reporte de experimento + bundle con estado `CANDIDATE` o `REJECTED_EXPERIMENT`; lockbox declarado con presupuesto de queries.
- **Aceptación:** clasificación honesta según criterios pre-registrados; ningún criterio relajado post-hoc.
- **Detención:** `REJECTED` ⇒ se documenta y se itera B–D con nuevo pre-registro; nunca se re-corre "hasta que pase" sin cambios de fondo declarados.
- **Dependencia:** F/G requieren bundle `CANDIDATE`.

### FASE F — Replay y benchmark pareado contra Gen2

- **Objetivo:** §7 completo.
- **Prerrequisitos:** E con bundle `CANDIDATE`.
- **Archivos probables:** herramienta offline de benchmark (`scripts/` o `src/aegis/training/benchmark.py`), reportes en `reports/parity_benchmark/`.
- **Pruebas:** alineación causal estricta (test que impide mirar futuro); no-escritura en `aegis_gen2/` (el tooling abre en modo lectura).
- **Entregables:** reporte pareado nuevo-vs-histórico con outcomes H12.
- **Aceptación:** reporte completo y reproducible; discrepancias explicadas.
- **Detención:** si el nuevo sistema es claramente inferior al benchmark en expectancy/estabilidad, el owner decide si G procede o se itera.
- **Dependencia:** G usa este expediente.

### FASE G — Shadow SHORT + NO_TRADE (no ejecutable)

- **Objetivo:** conectar el shadow read-only al servicio de velas coordinado, con el bundle `CANDIDATE`, evidencia persistente activada.
- **Prerrequisitos:** F; resolución del evento de cierre coordinado de 11 símbolos en TS (hoy inexistente — bloqueador conocido); `allowedSides=['SHORT']`; autorización del owner para iniciar el proceso shadow.
- **Cambios prohibidos:** cualquier camino hacia `createOrder`/sizing/brackets; `execution.enabledByConfig` permanece false; el gate sigue denegando todo con `SHADOW_MODE_NON_EXECUTING`.
- **Entregables:** proceso shadow corriendo, evidencia encadenada persistida, outcomes H12 resueltos vía `shadow.py`.
- **Aceptación:** N ciclos sin incidentes de integridad; cadena de evidencia verificada tras reinicio.
- **Detención:** cualquier violación de integridad de evidencia ⇒ parar y auditar.

### FASE H — Evidencia forward

- **Objetivo:** acumular evidencia forward madura (H12) del candidato en shadow, con maturación y resolución deterministas.
- **Prerrequisitos:** G estable.
- **Entregables:** stream forward propio + comparación pareada continua contra el stream Gen2 vivo; reportes periódicos.
- **Aceptación:** mínimos pre-registrados de volumen y ventana temporal (declarar en G; el histórico exigía mínimo forward pre-registrado antes de cualquier elegibilidad).
- **Detención:** drift de hashes, evidencia corrupta o divergencia experimento/shadow inexplicada.

### FASE I — Evaluación de posible canary

- **Objetivo:** SOLO evaluar elegibilidad; este plan no habilita live.
- **Prerrequisitos:** H con mínimos cumplidos; decisión explícita del owner.
- **Entregables:** expediente de elegibilidad (forward pareado, criterios, riesgos) con veredicto `ELIGIBLE`/`NOT_ELIGIBLE` y sin ninguna acción operacional.
- **Regla:** cualquier paso posterior (canary real) queda fuera de este documento y exige un plan propio.

---

## 9. Matriz de trazabilidad

| # | Hallazgo | Prioridad | Riesgo | Comp. histórico | Comp. nuevo | Solución | Fase | Archivos objetivo | Tests | Evidencia requerida | Criterio de cierre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Skew train/serving (raw-clip vs softmax/sigmoid; dos scores) | P0 | Promoción mide un sistema distinto del que decide | evaluador único `evaluate_symbol` | `experiment.py:311–318` vs `models.py:210–222`/`layers.py:96–103` | camino único de predicción | A | experiment.py, models.py, layers.py | paridad exp↔runtime↔API, golden fixtures | test en verde + fixtures | mismo bundle+batch ⇒ misma decisión |
| 2 | `qmae_q90` es media, no cuantil | P0 | Stop científico subestimado | `gen2_rv2_train.py:232–284` (quantile+conformal) | `experiment.py:207`, `train.py`, `layers.py:68` | QMAE cuantílico real (Opción A) | A(nombre)/C(modelo) | train.py, models.py, layers.py | cobertura conformal por fold/símbolo | banda [0.87,0.93] | ningún "q90" proviene de media |
| 3 | Probabilidades sin calibrar (sigmoid sobre ridge binario) | P0 | Thresholds sin significado | isotónica ECE 0.198→0.016 | `models.py:220–221` | calibración out-of-fold en bundle | A(contrato)/C(real) | train.py, models.py | ECE/Brier, aplicación idéntica, fail-closed | métricas out-of-fold | cabezas-probabilidad calibradas o vetadas |
| 4 | Fuente sin gate de finalidad; D3 reinterpretado como régimen | P0 | Entrenar sobre velas mid-bar (causa raíz F0.2) | `gen2_d3_build.py` gates G1–G7 | `candidate_experiment.yaml: source` SQLite vivo; `layers.py:classify_regime` | disciplina D3 de datos (Opción 1 o 2) + renombrar capa | A | src/aegis/data/, scripts/, layers.py | rechazo de vela no final; doble build bit-idéntico | manifests+hashes | Fase E solo sobre datos con gate |
| 5 | Pérdida de familias de features SHORT (~70 de 111) | P1 | Sin señal específica de SHORT | `compute_causal_features` | `features.py` (39) | `aegis-features-v2` selectivo | B | features.py | causalidad+golden por feature | matriz feature-a-feature | familias prioritarias portadas |
| 6 | Sin modelos no lineales ni competición; sin calibración | P1 | Capacidad predictiva inferior | RV2/EQM1 trainers | `train.py` ridge único | competición stability-first + export JSON | C | train.py, models.py | paridad export, determinismo | reporte de competición | ganador por protocolo pre-registrado |
| 7 | ECON degradado y autorreferencial | P1 | Edge económico no demostrado | `gen2_econ1_backtest.py` | `layers.py:80`, `experiment.py:337` | motor ECON replay real | D | training/econ.py | golden trades, bootstrap seed | reporte ECON | ECON independiente de labels |
| 8 | Umbral de selección 0.50 arbitrario | P1 | Selección sin base económica | `gen2_selection_policy.py` (0.01432240) | `models.yaml: selection_score` | umbral absoluto congelado + validate | D | decision.py, policy nueva | validate hard-stop | policy con hashes | umbral derivado de ECON |
| 9 | Labels solo terminales (sin MFE/path/ambigüedad) | P1 | Mide otra cosa que calidad de entrada | `short_quality_v4_labels.py` | `experiment.py:203–207` | labels V4 path-aware | B | training/labels.py | golden, ambigüedad, gaps | definiciones versionadas | labels V4 activos en C–E |
| 10 | Sin freeze de sistema ni chequeo de entorno | P1 | Cambios silenciosos entre piezas | `GEN2_SYSTEM_FREEZE.json` | solo content-hash por bundle | SYSTEM_FREEZE integral | D | freeze nuevo, runtime.py | validación fail-closed | freeze validable | freeze ata todas las piezas |
| 11 | Evidencia no persistente; sin forward propio | P1 | Cero evidencia acumulada | `forward/` (8.096 decisiones) | `brain.yaml: persistence_enabled: false` | persistencia + recuperación + shadow | D(cfg)/G–H | evidence.py, brain.yaml | reinicio sin pérdida, no duplicados | cadena verificada en disco | evidencia persistente en shadow |
| 12 | Score-producto con doble conteo (tail dos veces) y `d3_confidence` sin validar | P2 | Score sin semántica | score `s_eqm` con unidades | `layers.py:66–67,96–103` | score re-definido, gates separados | D | layers.py | golden de score, unidades | pre-registro del score | sin doble conteo; régimen fuera del score |
| 13 | Bundle referencia `approved:true` + `trained:false` | P2 | Trampa de integración | n/a | `aegis-offline-reference-v1.json` | regla dura + re-etiquetado | D | models.py, bundle ref | rechazo approved&&!trained | test en verde | combinación imposible |
| 14 | Cierre coordinado de 11 símbolos inexistente en TS | P2 | Bloquea shadow | n/a | informe Fase 2 §Shadow | evento coordinado en TS | G | TS candle service + brain | test TS de coordinación | shadow recibe ciclos | shadow conectado |
| 15 | `momentum_acceleration_3_12` con `/4` indocumentado; folds duplicados inline | P2 | Semántica ambigua; drift entre implementaciones | `momentum_accel_3_12` | `features.py:259`; `experiment.py:401–414` vs `dataset.py:76–94` | resolver semántica + unificar folds | B | features.py, dataset.py, experiment.py | golden + test único de folds | decisión documentada | una sola definición de cada cosa |
| 16 | Umbrales de régimen hard-coded; direction_threshold doble (0.10 vs 0.50); sin golden externos | P3 | Confusión y deriva | detector histórico con config | `layers.py:130–147`, configs | thresholds versionados + un solo direction_threshold | B/D | layers.py, configs | golden de régimen | procedencia documentada | sin números mágicos sin dueño |

---

## 10. Estado objetivo al cierre del plan

Python (`src/aegis/`):

- Arquitectura limpia actual intacta; cero red en el runtime científico.
- D3 = disciplina de integridad de datos (snapshots canónicos con gate de finalidad); el clasificador de régimen renombrado y fuera del score.
- `aegis-features-v2` causal con las familias SHORT recuperadas + cross-sectional nuevas.
- Labels SHORT V4 path-aware versionados.
- RV2 = TRRM calibrado (probabilidad real) + QMAE cuantílico conformal.
- EQM validado con targets separados (clean / net_quality).
- ECON independiente con replay de precios reales, escenarios de coste y bootstrap.
- Selection Policy con umbral absoluto congelado derivado de la economía validada, con modo validate.
- SYSTEM_FREEZE integral; ciclo de vida de bundles EXPERIMENTAL→CANDIDATE→SHADOW_APPROVED.
- Evidencia SHA-256 persistente con recuperación probada.
- Bundles inspeccionables (JSON, sin pickles), inferencia determinista, experimento = runtime.

TypeScript (`binance-futures-bot-ts`):

- Plataforma operacional única; brain client + manifest handshake; `allowedSides: ['SHORT']`.
- Shadow explícitamente no ejecutable (`SHADOW_MODE_NON_EXECUTING`); `execution.enabledByConfig=false`.
- Cero ciencia duplicada; cero Binance en Python.

Fuera del alcance de este plan: habilitar LONG, habilitar ejecución, canary real, y cualquier escritura en `/home/jasan/Develop/aegis_gen2/`.
