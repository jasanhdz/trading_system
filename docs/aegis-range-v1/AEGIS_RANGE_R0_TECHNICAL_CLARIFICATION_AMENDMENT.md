# Aegis Range Strategy V1 - R0 Technical Clarification Amendment

**Fecha efectiva:** 2026-08-24

**Estado:** `AEGIS_RANGE_R0_TECHNICAL_CLARIFICATION_AMENDMENT`

**R0 cientifico:** `AEGIS_RANGE_R0_PREREGISTRATION_APPROVED`

**Ownership:** `AEGIS_RANGE_R0_OWNERSHIP_RESOLVED`

**Fases autorizadas por esta enmienda:** ninguna fase experimental o de
implementacion.

## 1. Motivo, alcance y precedencia

Esta enmienda puramente documental resuelve las nueve ambiguedades registradas en
`AEGIS_RANGE_R1_SPEC_AMBIGUITY_REPORT.md`. Tiene precedencia sobre el plan, el R0
original y la enmienda de ownership solo para las decisiones tecnicas enumeradas
en las secciones 2-10 de este documento.

Todo contrato no citado conserva su autoridad y contenido. En particular, esta
enmienda:

- no reescribe ningun artefacto historico R0;
- no cambia formulas, thresholds, grid, splits, hipotesis, costos, gate o E4,
  salvo las precisiones tecnicas expresamente congeladas aqui;
- no lee ni ejecuta TRAIN, CALIBRATION, VALIDATION o HOLDOUT;
- no emite `PRE_VALIDATION_SPEC_FROZEN`;
- no autoriza R1, R2, backtest, SHADOW o LIVE;
- no modifica produccion, E4 o `RegimeEngineV2`.

## 2. ATR14 raw y frontera `RangeAtr14V1`

Todas las decisiones Range que requieren ATR14 usan `atr14_raw`, calculado por la
frontera cientifica versionada `RangeAtr14V1` sobre exactamente las mismas 160
velas 5m cerradas, validas y contiguas entregadas al snapshot de
`RegimeEngineV2`.

`RangeAtr14V1` replica exactamente la semantica interna congelada en
`RegimeEngineV2.ts`:

```text
TR[i] = max(
  high[i] - low[i],
  abs(high[i] - close[i-1]),
  abs(low[i] - close[i-1])
)

ATR14 inicial = sum(TR[0..13]) / 14
ATR14 siguiente = (ATR14 anterior * 13 + TR actual) / 14
atr14_raw = ultimo ATR14 de la serie
```

La primera observacion TR corresponde a la segunda vela de la ventana. El
calculo usa operaciones IEEE-754 binary64 y devuelve el ultimo valor sin
redondeo ni cuantizacion. Esta frontera no modifica ni amplia la API de
`RegimeEngineV2`; es parte del futuro adapter cientifico Range.

Los indicadores ya publicados por `RegimeEngineV2` conservan su redondeo actual
a seis decimales. `atr14_raw` es la unica excepcion nueva y no se sustituye por
un indicador publicado redondeado.

## 3. Aritmetica, comparaciones y cuantizacion

Toda aritmetica de decisiones, incluidos medianas, promedios, tolerancias,
amplitude, midpoint, wick/body ratio, reward/risk, distancias, fills, TP, SL y
comparaciones contra thresholds, usa IEEE-754 binary64 sin cuantizacion
intermedia.

Las comparaciones inclusivas y estrictas permanecen exactamente como fueron
preregistradas y se aplican al resultado binary64 no cuantizado. No se introduce
epsilon implicito ni redondeo previo a una decision.

La cuantizacion decimal `ROUND_HALF_EVEN` a 12 lugares se aplica exclusivamente
al producir componentes decimales de IDs, del `thesis_feature_hash` y de sus
representaciones serializadas. La representacion canonica es una string decimal
con signo si corresponde, un digito entero minimo, punto decimal y exactamente
12 digitos decimales. Cero se serializa como `0.000000000000`; no se permite cero
negativo. `toFixed()` no es una operacion normativa para esta cuantizacion.

## 4. Invocacion y output del regime adapter

El regime adapter invoca `RegimeEngineV2.evaluate` con `market` ausente, que en
TypeScript equivale a `undefined`. No construye contexto BTC/ETH para esta
estrategia.

La frontera cientifica Range expone solo los campos de `RegimeEngineV2` usados
por el contrato R0 para blockers, score, telemetria preregistrada y tesis, mas
`atr14_raw` de `RangeAtr14V1`. Los campos dependientes de market que no figuran en
ese contrato quedan fuera de la frontera Range. La ausencia de E4 se representa
como JSON `null` donde el R0 exige persistirla.

## 5. Reemplazo del active pair

El paso 9 del orden de cierre siempre construye y ordena todos los pairs
elegibles, tambien cuando existe un episodio activo. Si el pair ganador tiene
IDs de soporte/resistencia distintos del active pair, el episodio activo termina
en ese mismo `decision_at` con reason `PAIR_REPLACED`.

La terminacion ejecuta el reset completo preregistrado del detector del simbolo.
El pair que produjo el reemplazo no sobrevive al reset, no nace un episodio nuevo
y no se evalua rejection ni senal Range en esa vela. Solo velas posteriores
pueden reconstruir estado y confirmar otro episodio.

## 6. `PENDING_ENTRY` en el open

`PENDING_ENTRY` solo puede existir estando flat. Su coexistencia con una posicion
abierta, una salida market pendiente o cualquier estado de salida de posicion es
invalida y termina el procesamiento con `STATE_INVARIANT_VIOLATION`; no se simula
un fill ni se consume quota.

En el orden global del open, el consumo de `PENDING_ENTRY` queda despues de las
tres etapas de salida ya preregistradas: salida market por breakout, salida
market por max hold y gaps TP/SL. Bajo la invariante flat, esas etapas no tienen
una posicion que procesar. El consumo de entry sigue exactamente este orden:

1. Verificar la invariante flat y la ausencia de estado de salida.
2. Evaluar solo las invalidaciones de open de R0 seccion 7.2 contra el raw open.
3. Calcular el fill candidato aplicando slippage adverso al raw open.
4. Calcular stop y target desde los niveles, ATR y parametros congelados en la
   senal que creo `PENDING_ENTRY`.
5. Evaluar contra el fill candidato target favorable, distancia bruta minima de
   42 bps y reward/risk bruto `>=1.0`.
6. Si todos los checks pasan, materializar el fill efectivo y la posicion,
   congelar la tesis y calcular IDs y `thesis_feature_hash`.
7. Solo despues del fill efectivo incrementar quota de episodio/lado y registrar
   el trade para reentry.

Una invalidacion o gate fallido cancela `PENDING_ENTRY`; no crea posicion, fill,
trade, cooldown ni consumo de quota. El high, low, close, volumen y regimen de la
vela abierta no participan en este procedimiento.

## 7. Rejection requiere counted touch

La condicion "touch valido en esa vela" de R0 se satisface solo si esa candle
produce un touch nuevo efectivamente contado por el paso 8. El contacto debe
pasar geometria, estado de rearmado, separacion minima de seis velas y toda regla
de elegibilidad del cluster, e incrementar su touch count.

Un contacto geometrico desarmado, demasiado cercano al touch anterior o que sea
continuacion del mismo contacto no puede emitir rejection.

## 8. Cooldown de reentry

El cooldown exige exactamente 12 cierres posteriores al evento de salida. Si la
salida se llena en el open de una vela, el close de esa misma vela es el cierre
posterior numero 1. La estrategia puede volver a emitir una senal elegible al
close posterior numero 12, para posible fill en el siguiente open. Los cierres
1-11 no son elegibles.

## 9. Recencia del pair

La tercera clave de orden de pairs queda definida como:

```text
pair_recency_at = min(
  last_counted_support_touch_at,
  last_counted_resistance_touch_at
)
```

Una mayor `pair_recency_at` gana. Los timestamps proceden exclusivamente de
touches nuevos efectivamente contados y causalmente activos.

## 10. Serializacion de `thesis_feature_hash`

El payload conserva exactamente los campos preregistrados en R0, con los ocho
parametros del candidato expandidos por nombre. No se agregan bindings ni campos
nuevos. Sus valores congelados de version y lineage son:

```text
schema_version = aegis-range-thesis-v1
typescript_git_head = bb034431e0ce05c8e0f978453c46dcff6efb981c
source_manifest_sha256 = 39cd5b8371ef4d193fcce22e6d6392ceaff8b802b52b4589d0c27cfa583a7704
split_manifest_sha256 = a67766d8ab446c657260550d37c55589d94cc11afba064ae3f043db803868c03
```

Las keys exactas, ordenadas lexicograficamente para la serializacion, son:

```text
ATR_entry
cluster_tolerance_atr
cost_scenario
decision_at
entry_available_at
entry_fill
max_adx
midpoint_at_entry
min_chop_risk
min_range_amplitude_pct
min_safety_volume_ratio
range_confidence_at_entry
range_confirmed_at
range_episode_id
range_id
regime_at_entry
rejection_min_wick_body_ratio
resistance_at_entry
schema_version
side
source_manifest_sha256
split_manifest_sha256
stop_at_entry
stop_buffer_atr
support_at_entry
symbol
tail_risk_score_at_entry
target_at_entry
target_buffer_atr
typescript_git_head
```

Antes de serializar se ordenan realmente por orden lexicografico de bytes UTF-8;
la lista anterior define el conjunto exacto, no un orden alternativo manual. El
objeto se codifica como JSON UTF-8 sin whitespace. Timestamps usan UTC ISO-8601
con milisegundos, enums e IDs son strings y todo decimal calculado usa la string
canonica `ROUND_HALF_EVEN` a 12 lugares definida en la seccion 3. La ausencia E4
es JSON `null` explicito. El hash final es SHA-256 de esos bytes JSON.

## 11. Estado preservado y efecto

```text
r1_implementation_created: false
r1_authorized: false
r2_backtest_executed: false
train_read_or_executed: false
calibration_read_or_executed: false
validation_opened: false
holdout_opened: false
pre_validation_spec_frozen: false
e4_modified: false
regime_engine_v2_modified: false
production_modified: false
fundingRate_materialized: 237/341
markPriceKlines_materialized: 237/341
r2_policy: BLOCK_R2_UNTIL_DOWNLOADED_AND_VERIFIED
gitlink_status: GITLINK_MISMATCH_RECORDED_NOT_RESOLVED
```

La aclaracion vuelve a revision externa. R1 permanece bloqueado hasta recibir
autorizacion separada basada en el hash aprobado de esta enmienda.
