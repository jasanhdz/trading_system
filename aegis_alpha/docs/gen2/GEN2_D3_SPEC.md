# GEN2-D3 — Canonical Market Data & Dataset Specification

**Status:** FROZEN (approved by owner 2026-07-11; any change requires a new spec version)
**Spec version:** d3-spec-v1.1
**Author:** Fable (Principal Architect, Gen2)
**Date:** 2026-07-11
**Supersedes:** D2 dataset lineage (SQLite-sourced) — invalidated by partial-candle sensor defect (see `aegis_candle_finality_f03_*.md`)

## Amendments v1.1 (owner-approval conditions, binding)

**A1 — Frozen contract during D3.** Quedan congelados: builder de features, lista y orden de features, feature hash esperado, definición de labels, target `tail_risk_roe_030`, sampling, horizons, símbolos, splits conceptuales y parámetros del builder. Ninguna mejora de features ni de modelado dentro de D3. Si el feature hash Gen1 no es reproducible sobre datos canónicos, PROHIBIDO cambiar el contrato en silencio: la corrida se detiene y emite `D3_FEATURE_CONTRACT_BROKEN` con diagnóstico explícito de cuál de estas tres causas aplica: (a) el builder Gen1 dependía de datos/columnas no reproducibles; (b) el hash debe versionarse como d3-v2 (decisión del owner); (c) bug real en el builder.

**A2 — Lockbox Gen2 pre-registrado.** Antes de cualquier modelado debe existir `GEN2_LOCKBOX_MANIFEST.json` (§5.1) con: periodos de desarrollo/train/validation, ventana de confirmación histórica semi-ciega, inicio del forward lockbox intocado, embargo, presupuesto de consultas, contador actual (=0), mecanismo de registro de cada apertura, condiciones de desbloqueo y usos prohibidos. El manifest NO contiene labels agregados, prevalencias ni métricas del periodo forward. D3 puede recolectar datos forward pero no inspeccionar sus labels.

**A3 — ECON1 no se aprueba solo por profit factor.** El gate ECON1 del roadmap queda ampliado (ver GEN2_ROADMAP.md §ECON1): expectativa neta positiva por trade con fees+slippage pesimista+funding, drawdown dentro de límite pre-registrado, estabilidad multi-fold/multi-periodo, tamaño muestral suficiente, prueba de concentración (ningún trade explica fracción desproporcionada del PnL), superioridad vs random y vs dos baselines de reglas al mismo presupuesto, sensibilidad a costos, bootstrap/intervalos, y resultados por símbolo/mes/régimen. PF ≥ 1.5 es orientativo, no suficiente, y no se optimiza directamente.

**A4 — Formato de series canónicas.** El entorno de referencia (`venv_rocm62`) no incluye `pyarrow` y la regla "no instalar dependencias" prevalece sobre la preferencia de formato: las series canónicas se almacenan en **CSV con dtypes y formato de flotantes fijados** (repr de doble precisión completa), hasheadas sobre bytes. Si pyarrow se añade al entorno en el futuro, migrar exige spec v1.2.

---

## 0. Principio rector

Un dato solo es admisible en investigación Gen2 si:
1. proviene de una **fuente canónica inmutable** (klines finales de Binance Futures),
2. está cubierto por un **manifiesto con hash** que permite reconstruirlo bit a bit,
3. pasó **todas las puertas de validación** de esta especificación.

El SQLite operacional (`data/binance_candles.db`) queda **prohibido como fuente de research**. Su único rol futuro es sensor operacional del bot live y objeto de diagnóstico.

**Control científico de una variable:** D3 cambia ÚNICAMENTE el insumo de datos. El builder de features, la definición de labels, el muestreo y los parámetros quedan idénticos a D2. Cualquier mejora de features/labels es una fase posterior. Esto aísla el efecto del fix de sensor.

---

## 1. Fase científica

- **Objetivo:** producir el dataset canónico Gen2 (velas finales + features causales + labels V4/tail) totalmente reproducible, y demostrar que el bloqueo de paridad de F0.2 queda resuelto.
- **Hipótesis H1:** las velas finales re-descargadas de Binance son estables (re-fetch idéntico) → una fuente canónica es alcanzable.
- **Hipótesis H2:** con insumo canónico, la reconstrucción de features en una ventana de solapamiento alcanza paridad ≥ 99.9% (vs 66.2% en F0.2).
- **Hipótesis H3:** las labels de cola recomputadas sobre velas finales muestran prevalencia igual o mayor que las de D2 (dirección esperada del sesgo: las velas parciales *perdían* colas).
- **Criterio de éxito:** las tres hipótesis confirmadas + todas las puertas de §6 en verde + doble corrida bit-idéntica (§8).
- **Criterio de fracaso:** re-fetch de barras cerradas no idéntico (H1 falla → investigar antes de continuar); paridad < 99.9% (H2 falla → hay un segundo mecanismo de divergencia, PROHIBIDO avanzar); prevalencia cae >20% relativo (H3 invertida → el edge Gen1 era artefacto del sensor en mayor grado del estimado; hallazgo mayor, revisar antes de TRRM V2).
- **Rollback:** D3 no toca nada existente (directorios nuevos, append-only). Rollback = descartar el árbol `aegis_gen2/d3/<attempt_id>/`.
- **Puerta a la siguiente fase:** decisión `D3_CANONICAL_READY` emitida por el auditor de aceptación (§9), no por juicio humano.

---

## 2. Fuente de datos

- **Endpoint:** Binance USDT-M Futures, `GET https://fapi.binance.com/fapi/v1/klines` (público, sin API key). Es el mismo endpoint ya usado por F0.2 (`fetch_binance_klines` en `audit_recent_market_source_f02.py`) — reutilizar ese cliente.
- **Símbolos (11):** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, SUIUSDT, LTCUSDT, LINKUSDT.
- **Intervalo:** `5m` únicamente (idéntico a D2).
- **Rango objetivo:** 730 días hacia atrás desde el corte (§5). **Mínimo aceptable por símbolo: 350 días** (si un símbolo tiene menos historia futures, se incluye con lo disponible si ≥350d; si no, se excluye y se documenta).
- **Solo barras cerradas:** descartar toda barra con `close_time >= now_utc - 120s` (margen de seguridad). La última barra parcial jamás entra al almacén.
- **Paginación:** `limit=1500`, avanzar por `open_time`; verificar continuidad entre páginas.
- **Rate limiting:** ≤ 3 requests/segundo, backoff exponencial en HTTP 429/418 (base 2s, máx 60s), 3 reintentos por página; abortar la sesión de snapshot si un símbolo no completa.
- **Campos a almacenar (todos los que da el endpoint, sin descartar):** `open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume`. El builder consume un subconjunto; el raw guarda todo.
- **Estimación operativa:** 11 símbolos × 730d × 288 barras ≈ 2.3M barras ≈ ~1,550 requests ≈ 10–15 min con el rate limit indicado.

---

## 3. Almacenamiento, estructura y naming

Raíz de investigación Gen2 (fuera del repo, dentro del área research):

```
/home/jasan/Develop/aegis_gen2/
  d3/
    market_snapshots/
      <snapshot_id>/                     # snapshot_id = UTC stamp YYYYMMDDTHHMMSSZ
        BTCUSDT_5m.csv                   # una fila por barra, UTC, sin tz-suffix ambiguo
        ...
        LINKUSDT_5m.csv
        fetch_log.jsonl                  # cada request: url params, ts, filas, reintentos
        snapshot_manifest.json           # ver §4
        snapshot_manifest.sha256         # sha256 de cada CSV + del manifest
    canonical_series/
      <series_version>/                  # v1, v2... (v2 = extensión de rango, nunca edición)
        BTCUSDT_5m.parquet               # serie validada: dedup + gaps anotados + gates
        ...
        series_manifest.json
    datasets/
      <dataset_version>/                 # d3-v1, d3-v2...
        dense.csv
        strided.csv
        labels_summary.csv
        dataset_manifest.json
    reports/
      aegis_gen2_d3_<gate>_<stamp>.{md,json}
```

**Reglas duras:**
- `market_snapshots/*` es **append-only e inmutable**: nunca se edita ni borra un snapshot; datos nuevos = snapshot nuevo.
- `canonical_series/` y `datasets/` son **inmutables una vez referenciados** por cualquier corrida de entrenamiento o reporte.
- Formato: CSV en snapshots (legibilidad/diff), Parquet en series canónicas (dtypes fijos), CSV en datasets (compatibilidad con tooling E2 existente).
- Timestamps: **UTC naive ISO-8601** (`YYYY-MM-DD HH:MM:SS`) en todos los artefactos — una sola convención, verificada por gate de esquema.
- Ningún artefacto Gen2 puede escribirse bajo el repo, ni bajo `models/turbo/`, ni en rutas `active/` (reutilizar `validate_research_path`).

---

## 4. Manifiestos y contratos

### 4.1 `snapshot_manifest.json`
```json
{
  "schema": "gen2_d3_snapshot_v1",
  "snapshot_id": "<stamp>",
  "endpoint": "fapi.binance.com/fapi/v1/klines",
  "interval": "5m",
  "requested_range": {"start": "...", "end": "..."},
  "closed_bar_cutoff": "...",
  "symbols": {
    "BTCUSDT": {"rows": 0, "first": "...", "last": "...", "sha256": "..."}
  },
  "fetch_completed_at": "...",
  "tool": {"name": "...", "git_commit": "<sha del repo al correr>"}
}
```

### 4.2 `series_manifest.json`
Añade a lo anterior: snapshot(s) de origen, resultados de cada gate (§6) con números, filas excluidas y por qué, whitelist de gaps aceptados, y `series_sha256` por símbolo.

### 4.3 `dataset_manifest.json`
```json
{
  "schema": "gen2_d3_dataset_v1",
  "dataset_version": "d3-v1",
  "series_version": "v1",
  "builder": {"module": "build_trrm_causal_feature_dataset_d", "git_commit": "<sha>", "config_sha256": "<hash del config serializado>"},
  "feature_columns": [], "feature_hash": "<sha256 lista de features>",
  "expected_feature_hash_gen1": "fbb1fee2cf0c42d21c591169b25452eb65e932bf5bd76109ca8447a4dfd7057e",
  "labels": {"families": ["v4_path", "tail_roe_ladder", "qmae"], "config_sha256": "..."},
  "sampling": {"dense_rows": 0, "strided_rows": 0, "method": "uniform_time", "identical_to_d2_params": true},
  "splits": {"embargo_minutes": 120, "lockbox_policy": "ver seccion 5"},
  "row_counts": {}, "label_prevalence": {}, "prevalence_delta_vs_d2": {},
  "reproducibility": {"double_run_sha256_match": true},
  "decision": "D3_CANONICAL_READY | D3_PARTIAL | D3_REJECTED"
}
```

**Contrato de identidad:** un dataset se identifica por `(series_version, builder git_commit, config_sha256)`. Dos corridas con la misma identidad DEBEN producir bytes idénticos (§8). El `feature_hash` DEBE coincidir con el de Gen1 (mismas 111 features + one-hots de horizon); si no coincide, el builder cambió y la corrida se rechaza — D3 no permite cambios de features.

---

## 5. Cortes temporales, lockbox y jerarquía de evidencia

- **T_cut (corte de datos):** la última barra 5m cerrada al momento del snapshot aprobado.
- **Grid de entrenamiento/validación:** idéntico en mecánica a E2 (folds expansivos pre-lockbox, embargo 120 min).
- **Lockbox histórico Gen2:** `2026-04-27 → T_cut`, recomputado sobre velas canónicas. **Declaración honesta obligatoria en todo reporte:** esta ventana fue abierta por la Gen1 (E2/E2.1/E2.2); para Gen2 se degrada a **confirmación semi-ciega** — sirve para detectar regresiones graves, NO como evidencia primaria de aprobación.
- **Evidencia primaria Gen2 = forward:** todo dato posterior a la aprobación de esta spec es virgen por construcción y se acumula mientras se construyen TRRM V2/EQM V1. La promoción a freeze/shadow se juzga principalmente ahí.
- **Presupuesto de consultas:** el lockbox histórico admite **una** consulta por candidato final (TRRM V2 una, QMAE V2 una, EQM V1 una). Toda consulta se registra en el manifest del candidato. Consulta no registrada = candidato descalificado.

### 5.1 GEN2_LOCKBOX_MANIFEST.json (obligatorio antes de modelar — Amendment A2)

Ubicación: `/home/jasan/Develop/aegis_gen2/GEN2_LOCKBOX_MANIFEST.json`. Campos mínimos:
`schema`, `created_at_utc`, `historical_development_period` (train+validation, con sub-rangos), `semi_blind_historical_confirmation` (2026-04-27 → T_cut, declarada quemada por Gen1), `untouched_forward_start` (= aprobación de esta spec), `embargo_minutes` (120), `allowed_query_count` (por candidato), `current_query_count` (0), `query_log` (lista append-only: quién/cuándo/candidato/hash), `unlock_conditions` (candidato completo congelado por hash), `forbidden_uses` (selección de algoritmo, thresholds, calibración, features, sampling, ECON1 development). El forward lockbox permanece intocado durante D3, TRRM V2, QMAE V2, EQM V1, selección de algoritmos, calibración, thresholds y desarrollo de ECON1.

---

## 6. Puertas de validación (todas bloqueantes, en orden)

**G1 — Esquema:** columnas y dtypes exactos; timestamps UTC estrictamente crecientes; sin NaN en OHLCV; `high ≥ max(open, close)`, `low ≤ min(open, close)`, `volume ≥ 0`, `taker_buy_base ≤ volume`. Violación → fila listada y snapshot rechazado (no se "limpia" en silencio).

**G2 — Duplicados:** cero `open_time` duplicados por símbolo. Si dos fetches dan la misma barra con valores distintos → **fallo de G3, no un dedup**: se investiga, jamás se resuelve con `keep=last`.

**G3 — Finalidad (doble-fetch):** el mecanismo que mata la clase de bug de F0.2. Tras el fetch principal, esperar ≥15 min y re-descargar una muestra aleatoria del 5% de barras por símbolo (mín. 500) más las últimas 24h completas. **Exigir identidad bit a bit al 100%** en toda barra cerrada ≥10 min antes del primer fetch. Cualquier diferencia → snapshot rechazado + investigación (data revision del exchange es noticia mayor).

**G4 — Gaps:** grid esperado de 5m. Por símbolo: `gap_ratio ≤ 0.1%` del rango y ningún gap > 12 barras (1h) dentro de los últimos 365d, salvo whitelist explícita de mantenimientos documentados de Binance (con URL/announcement en el manifest). Exceso → símbolo excluido (D3_PARTIAL) o snapshot rechazado si afecta a >2 símbolos.

**G5 — Cobertura:** ≥350d por símbolo; primera/última barra registradas; los 11 símbolos presentes o exclusiones justificadas (máx. 2).

**G6 — Diagnóstico del sensor live (no bloqueante para D3, obligatorio de ejecutar):** correr `audit_candle_finality_f03` comparando SQLite vs el snapshot nuevo, para cuantificar la divergencia del sensor operacional y alimentar la decisión (del owner) de reparar el refresher. Se reporta; no bloquea D3.

**G7 — Paridad de replay (el cierre del bloqueo F0.2):** con la serie canónica v1 congelada, tomar un snapshot NUEVO e independiente de los últimos 7 días, reconstruir features con el builder congelado y comparar contra el dataset D3 en la ventana de solapamiento. **Aceptación: paridad de valores ≥ 99.9% con tolerancia rtol=1e-9** (idéntico insumo debe dar idéntico output; la tolerancia solo absorbe float I/O). Fallo → PROHIBIDO avanzar a TRRM V2.

**G8 — Sanidad de labels:** recomputar labels V4 + `tail_risk_roe_{015..035}` + `qmae` sobre la serie canónica; comparar prevalencias contra D2 por símbolo/horizonte. Esperado: prevalencia de cola igual o mayor (H3). Reportar tabla de deltas. Prevalencia que *cae* >20% relativo en ≥3 símbolos → detener y revisar (criterio de fracaso de fase).

---

## 7. Builders y labels (congelados de Gen1)

- **Builder de features:** `aegis_alpha/tools/build_trrm_causal_feature_dataset_d.py` en el commit vigente al aprobar la spec (fijar sha en el manifest). Prohibido tocarlo durante D3. Única adaptación permitida: la capa de I/O que lee de `canonical_series/` en vez de SQLite — como módulo NUEVO (`gen2` namespace) que importa las funciones de features del builder sin modificarlas.
- **Labels:** `short_quality_v4_labels.py` (path metrics V4) + escalera tail ROE + qmae, con la misma parametrización de D2 (leverage 20x, fees 4bps, slippage 1bps). Mismo config, hasheado.
- **Muestreo:** dense y strided con parámetros idénticos a D2 (uniform-in-time, mismos caps). El overlap ratio se reporta (informativo).
- **Horizontes:** 6, 12, 24 — h12 queda pre-registrado desde ya como horizonte operativo de Gen2 (los otros, diagnóstico).

---

## 8. Reproducibilidad

- **Doble corrida obligatoria:** el pipeline snapshot→series→dataset se ejecuta dos veces de punta a punta (segunda corrida desde el mismo snapshot); los sha256 de `dense.csv`, `strided.csv` y ambos manifests deben ser **idénticos**. Registrado en `dataset_manifest.reproducibility`.
- **Determinismo requerido:** sin dependencia de wall-clock en el contenido (los stamps van en nombres/manifests, no en filas); orden de procesamiento fijo (símbolos ordenados, timestamps ordenados); dtypes float64 explícitos; sin paralelismo no determinista.
- **Entorno registrado:** versión de Python, pandas, numpy, sklearn en el manifest (`/home/jasan/.venv_rocm62` es el entorno de referencia — la lección del unpickle E2/sklearn 1.8 queda codificada: **un solo entorno para todo Gen2**).

---

## 9. Auditoría de aceptación, reportes y decisión

Herramienta nueva: `audit_gen2_d3_acceptance.py`. Lee manifests + artefactos, re-verifica hashes, ejecuta/verifica G1–G8 y emite:

- `aegis_gen2_d3_acceptance_<stamp>.{md,json}` con: tabla de gates (pass/fail + números), cobertura por símbolo, tabla de prevalencias y deltas, resultado de paridad G7, doble-corrida, y **una** de:
  - **`D3_CANONICAL_READY`** — todos los gates verdes → puerta abierta a TRRM V2.
  - **`D3_PARTIAL`** — verde con ≤2 símbolos excluidos documentados → puede avanzar con universo reducido (decisión registrada).
  - **`D3_REJECTED`** — cualquier gate bloqueante rojo → no se avanza; investigación con el fallo específico.

La decisión la emite el auditor, no una persona. El owner solo puede *detener* un READY, nunca *promover* un REJECTED.

---

## 10. Tests mínimos (Codex debe entregarlos con la implementación)

1. Fetcher: paginación continua sin huecos/duplicados (fixture HTTP simulado); descarte correcto de la barra en formación (borde exacto del cutoff).
2. G1–G5: fixtures sintéticos con cada violación (dup, gap, high<close, barra parcial inyectada) → gate correcto en rojo.
3. G3: fixture donde el re-fetch difiere en 1 barra → snapshot rechazado.
4. Manifiestos: hash estable ante re-serialización; detección de manipulación (1 byte cambiado → verificación falla).
5. Builder I/O nuevo: sobre fixture de velas, produce exactamente las mismas features que el builder D sobre el mismo fixture (paridad interna 100%).
6. Doble corrida sobre fixture → bytes idénticos.
7. Labels: fixture con cola conocida → `tail_risk_roe_030` correcto; prevalencia del fixture exacta.
8. Auditor de aceptación: fixtures para READY / PARTIAL / REJECTED.

Todos ejecutables como scripts directos (patrón actual del repo), sin dependencias nuevas.

---

## 11. Política de actualización futura

- **Extensión de rango** (más historia o más reciente): nuevo snapshot + `canonical_series/v(n+1)` que DEBE contener `v(n)` como subconjunto bit-idéntico (gate automático de compatibilidad). Los datasets viejos jamás se regeneran in-place.
- **Nuevo símbolo:** solo vía spec amendment (versión de spec nueva).
- **Cadencia forward:** el collector Gen2 (fase F1) consumirá snapshots incrementales diarios con estos mismos gates G1–G4 en miniatura; G3 (doble-fetch) se convierte en el heartbeat semanal de paridad.

---

## 12. Fuera de alcance de D3 (explícito)

- Cambios de features, labels nuevas, targets nuevos → fases posteriores.
- Reparación del refresher/SQLite → decisión operativa del owner, fuera del pipeline research.
- Entrenamiento de cualquier modelo → TRRM V2 (fase siguiente).
- Cualquier escritura en rutas live/active/manifests del sistema actual.
