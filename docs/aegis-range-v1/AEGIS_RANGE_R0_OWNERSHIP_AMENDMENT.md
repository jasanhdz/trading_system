# Aegis Range Strategy V1 - R0 Ownership Amendment

**Fecha efectiva:** 2026-08-24

**Estado:** `AEGIS_RANGE_R0_OWNERSHIP_AMENDMENT`

**R0 cientifico:** `AEGIS_RANGE_R0_PREREGISTRATION_APPROVED`

**Fases autorizadas por esta enmienda:** ninguna fase experimental o de
implementacion.

## 1. Motivo y alcance

El R0 historico declaro que el preregistro y una futura implementacion TypeScript
research-only pertenecerian a `binance-futures-bot-ts`. Esa declaracion se
conserva intacta en el artefacto aprobado y en su commit original. Antes de R1 se
corrige exclusivamente el ownership para separar ciencia de runtime.

Esta enmienda:

- no reescribe retrospectivamente el R0 aprobado;
- no cambia ninguna formula, threshold, split, gate, lifecycle, grid, hipotesis,
  costo, fuente de datos ni politica E4;
- no observa ni ejecuta TRAIN, CALIBRATION, VALIDATION o HOLDOUT;
- no emite `PRE_VALIDATION_SPEC_FROZEN`;
- no autoriza R1, R2, backtest, shadow ni live;
- no modifica produccion, E4 o `RegimeEngineV2`.

La enmienda tiene precedencia sobre las declaraciones de ownership y ubicacion
futura de codigo del plan/R0 original. Todo contenido cientifico restante del
plan y preregistro conserva plena autoridad sin cambios.

## 2. Lineage historico preservado

Los originales fueron publicados en el repositorio hijo:

```text
repository: /home/jasan/Develop/trading_system/binance-futures-bot-ts
branch: work/entry-quality-evidence-20260726
plan lineage commit: bb034431e0ce05c8e0f978453c46dcff6efb981c
R0 artifact commit: 970b26c0c49b8ba7d0da7f90c898ddf30e96995a
```

El traslado al repositorio padre conserva byte-for-byte el plan y los cinco
artefactos R0. Los paths originales dejan de ser copias editables en el working
tree del hijo, pero permanecen recuperables en su historia Git. El hijo conserva
solo un README de redireccion.

El `r0_source_manifest.json` y el `r0_artifact_manifest.json` permanecen
intactos, incluidos sus paths y HEADs historicos. La presente enmienda y
`r0_ownership_manifest.json` son la capa auditable que resuelve esos valores
historicos sin falsificarlos.

## 3. Ownership canonico desde esta enmienda

### 3.1 Repositorio cientifico

```text
CANONICAL SCIENTIFIC REPOSITORY:
/home/jasan/Develop/trading_system

CANONICAL SCIENTIFIC DOCUMENTATION:
/home/jasan/Develop/trading_system/docs/aegis-range-v1
```

Pertenecen conceptualmente al repositorio padre:

- datasets y data lineage;
- documentacion y manifests cientificos;
- futura implementacion pura R1;
- futuro backtester R2;
- TRAIN y CALIBRATION;
- `PRE_VALIDATION_SPEC_FROZEN`;
- VALIDATION y HOLDOUT;
- resultados, reportes y freeze de estrategia.

La ubicacion reservada para una futura implementacion research-only es:

```text
/home/jasan/Develop/trading_system/sandbox/aegis_range_strategy_v1/
```

Si R1/R2 fueran autorizados por separado, ese root seguiria los patrones
existentes del padre con `src/`, `tests/`, `tools/`, `config/` y `artifacts/`.
Esta tarea no crea ninguno de esos paths ni archivos.

### 3.2 Repositorio runtime

```text
RUNTIME INTEGRATION REPOSITORY:
/home/jasan/Develop/trading_system/binance-futures-bot-ts
```

El repositorio hijo queda reservado para fases posteriores a evidencia
cientifica aprobada:

- port/adaptacion runtime;
- router de estrategia;
- SHADOW y telemetria operacional;
- integracion con E4, guards y hard safety;
- integracion con execution existente.

No es autoridad para modificar el contrato cientifico. Cualquier port futuro
debe demostrar parity determinista contra el artefacto cientifico congelado,
incluidos fixtures, decisiones, timestamps, IDs, fills y hashes, antes de poder
solicitar integracion operacional.

## 4. Estado experimental al emitir la enmienda

```text
train_executed: false
calibration_executed: false
validation_opened: false
holdout_opened: false
pre_validation_spec_frozen: false
r1_executed: false
r2_executed: false
```

No se leyeron outcomes, features, episodios, senales ni resultados economicos de
ninguna particion durante la migracion. VALIDATION y HOLDOUT permanecen sellados.

## 5. Contrato cientifico inalterado

Permanecen exactamente iguales, entre otros:

- timeframe 5m y causalidad;
- pivots `L=2/R=2`;
- clustering, touches y range episodes;
- rejection, entry, TP, SL, breakout y max hold;
- reentry y costos;
- grid de 384 candidatos;
- fechas TRAIN/CALIBRATION/VALIDATION/HOLDOUT;
- purge, embargo y bootstrap;
- jerarquia pooled/LONG/SHORT/majors/alts y Holm-Bonferroni;
- gate numerico;
- E4 frozen y politica de contaminacion;
- fuentes, manifests y hashes cientificos upstream.

Los hashes before/after de los seis documentos historicos se registran en
`r0_ownership_manifest.json` y deben coincidir.

## 6. Gitlink no resuelto

El padre registra el gitlink del hijo en:

```text
814a302885e1d07bfd27404ebb5e69a30acebcc5
```

El hijo estaba en `970b26c0c49b8ba7d0da7f90c898ddf30e96995a` al iniciar la
migracion. Se mantiene el estado:

```text
GITLINK_MISMATCH_RECORDED_NOT_RESOLVED
```

Esta enmienda no agrega `.gitmodules`, no actualiza el gitlink, no reescribe
historia y no normaliza la relacion entre repositorios.

## 7. Dependencia R2 preservada

```text
fundingRate materialized: 237/341
markPriceKlines materialized: 237/341
policy: BLOCK_R2_UNTIL_DOWNLOADED_AND_VERIFIED
```

No se descargan los 104 archivos faltantes por tipo en esta tarea.

## 8. Efecto de la enmienda

La secuencia de autoridad queda:

```text
R0 original aprobado
        |
        v
ownership amendment
        |
        v
scientific canonical home = trading_system
        |
        v
R1/R2 solo tras autorizacion futura y solo en trading_system
        |
        v
runtime port posterior en binance-futures-bot-ts con parity demostrada
```

El resultado de esta tarea vuelve a R0 para revision externa. No concede permiso
implicito para iniciar R1.
