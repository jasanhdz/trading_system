# Aegis Range Strategy V1 - R1 Specification Ambiguity Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R1_BLOCKED_BY_SPEC_AMBIGUITY`

**Rama:** `work/entry-quality-evidence-20260726`

**HEAD base:** `85b8bbeaad047d2903f5b66d378a79ec53714164`

## 1. Resultado de la auditoria previa

R1 fue autorizado externamente, pero la implementacion no comenzo porque el
contrato R0 no determina de forma unica varias transiciones y representaciones
que afectan decisiones, episodios o hashes. Elegir cualquiera de las alternativas
en codigo reescribiria implicitamente el preregistro.

Se leyeron completos los siete artefactos canonicos requeridos. Tambien se audito
read-only el codigo congelado:

| Archivo | SHA-256 |
|---|---|
| `RegimeEngineV2.ts` | `3726e28badfdba5acc81d87ccd3202fc43310a04d4b3cff2597f38acb2913134` |
| `RegimeEngineV2.types.ts` | `3b3972153f7c977d50ec864a5d8a4c4b3d8d2e73822453eaed1a25391211d10c` |
| `RegimeEngineV2.test.ts` | `80aa2619efdcb74fa8722f79ce62a01a4028bb213856f3a5b7fc0ea32e091cf1` |

No se leyo ningun dataset Range, funding file, outcome ni particion sellada. No
se creo el root `sandbox/aegis_range_strategy_v1/` ni codigo/tests R1.

## 2. Ambiguedades bloqueantes

### 2.1 ATR14 no tiene frontera ni precision congelada

**R0 afectado:** preregistro secciones 4.2, 5.2, 5.3, 6.2, 7.2, 7.3 y 7.4.

R0 exige ATR14 para clustering, touches, breakout, entry, TP, SL y thesis. El
motor calcula `atrSeriesValues` y `atr` en privado en
`RegimeEngineV2.ts:87-94`, mediante `atrSeries`/`wilderSeries` en
`RegimeEngineV2.ts:563-608`. Sin embargo, `RegimeEngineV2Indicators` no expone
ATR (`RegimeEngineV2.types.ts:64-106`) y el objeto publicado tampoco lo incluye
(`RegimeEngineV2.ts:189-232`).

R0 tampoco especifica si las comparaciones Range usan:

1. ATR raw de doble precision interno;
2. ATR redondeado a seis decimales como los indicadores publicados;
3. ATR cuantizado a 12 decimales con `ROUND_HALF_EVEN`;
4. una nueva salida versionada del adapter cientifico.

Estas opciones cambian tolerancias, asignacion de clusters, touches, breakout,
stops, targets y hashes en fronteras. Elegir una seria una decision cientifica
nueva.

### 2.2 Precision numerica de decisiones no esta congelada

**R0 afectado:** preregistro secciones 5.2-5.4, 6.1, 7.1-7.4, 8 y 9.

R0 fija `ROUND_HALF_EVEN` a 12 lugares para valores serializados en IDs/thesis,
pero no fija el dominio aritmetico ni el momento de cuantizacion para:

- mediana y promedio de dos pivots;
- tolerance y tau;
- amplitude y midpoint;
- wick/body ratio;
- reward/risk y distancia de 42 bps;
- TP/SL, breakout y fills;
- comparaciones exactas contra thresholds.

El motor TypeScript usa binary64 y `Number(value.toFixed(6))` para outputs
publicados (`RegimeEngineV2.ts:774-779`). Python `float`, `Decimal` y una
cuantizacion previa/posterior pueden emitir decisiones distintas en igualdad o
cerca de los limites. R1 no puede escoger una politica sin enmendar R0.

### 2.3 Argumento `market` del regime adapter no esta definido

**R0 afectado:** preregistro secciones 4.2-4.3 y requisito R1 Regime Adapter.

`RegimeEngineV2.evaluate` acepta `market` opcional
(`RegimeEngineV2.types.ts:141-146`). Ese argumento altera
`shortAdverseReboundRisk`, market confirmation, confidence y algunas ramas de
clasificacion (`RegimeEngineV2.ts:40-68`, `368-433`). R0 fija 160 candles, pero
no fija si el adapter debe:

1. invocar con `market=undefined`;
2. construir contexto causal BTC/ETH;
3. publicar ambos resultados;
4. excluir campos dependientes de market de la frontera cientifica.

Aunque varios hard blockers Range no usan market confirmation, R1 exige
reproducir la semantica del motor y el output completo no es unico sin esta
decision.

### 2.4 Reemplazo del pair contradice el orden de transicion

**R0 afectado:** preregistro secciones 4.5, 5.4 y 6.2.

El episodio termina cuando otro pair se vuelve ganador. Sin embargo, el orden
cerrado dispone:

1. evaluar estructura activa antes de insertar pivots nuevos;
2. insertar pivots y registrar touches en pasos posteriores;
3. construir/ordenar pairs en el paso 9;
4. en ese mismo paso, “mantener el pair activo” o confirmar uno si no hay
   episodio.

Un pivot/touch de la candle actual puede hacer ganador a otro pair despues del
chequeo estructural previo. R0 no determina si el episodio actual:

1. termina inmediatamente en ese close;
2. se mantiene y el reemplazo se evalua en el close siguiente;
3. impide que otro pair compita mientras esta activo.

Cada opcion cambia `range_episode_id`, `range_id`, senales y la prohibicion de
renacer en la misma candle.

### 2.5 Pending entry no tiene posicion exacta en el orden del open

**R0 afectado:** preregistro secciones 4.5, 7.2, 7.5 y 8.

R0 define `NEXT_BAR_OPEN` y sus invalidaciones, pero el orden obligatorio del
open enumera pending breakout exit, pending max-hold exit y gaps TP/SL; no ubica
el consumo de `PENDING_ENTRY`, el fill de entrada, la congelacion de thesis ni el
consumo de quota/reentry.

Las alternativas son procesar entry antes o despues de las acciones existentes,
o declarar que las invariantes hacen imposible toda coexistencia y formalizar
esa precondicion. Sin una regla explicita no existe una maquina de estados unica
ni un test de precedencia completo.

### 2.6 Rejection no define si requiere touch nuevo contado

**R0 afectado:** preregistro secciones 5.3 y 7.1.

Rejection exige “touch valido en esa candle”. El sistema de touches distingue
contacto geometrico de touch contado: un contacto puede estar desarmado, no haber
cumplido seis candles o ser continuacion de varias candles consecutivas.

R0 no determina si rejection requiere:

1. un evento touch nuevo efectivamente contado;
2. solo geometria de touch valida aunque el cluster siga desarmado;
3. geometria valida y separacion, pero sin incrementar touch count.

Esto cambia directamente `LONG/SHORT/NONE` y el numero de entradas por contacto.

### 2.7 Borde exacto del cooldown no esta definido

**R0 afectado:** preregistro seccion 7.5.

Se requieren 12 candles cerradas despues de cada salida. Si la salida ocurre en
el open, R0 no establece si el close de esa misma candle es la candle 1, ni si la
senal del close numero 12 ya es elegible o debe esperar al close 13.

Las convenciones posibles producen una diferencia de una candle en reentry y
pueden cambiar el maximo observable de trades por episodio.

### 2.8 Tie-break de recencia del pair no tiene formula

**R0 afectado:** preregistro seccion 5.4.

La tercera clave dice “touch mas reciente de la pareja”, pero no fija si es:

1. `max(last_support_touch, last_resistance_touch)`;
2. `min(last_support_touch, last_resistance_touch)`;
3. el instante mas reciente en que ambos lados estaban simultaneamente
   calificados.

Pairs distintos pueden ganar segun la formula, alterando episodios y senales.

### 2.9 `thesis_feature_hash` no es reproducible tras ownership

**R0 afectado:** preregistro seccion 7.2 y ownership amendment seccion 3.

El JSON exacto incluye `schema_version`, pero R0 no fija el valor de esa string.
Tambien exige `typescript_git_head`, mientras la enmienda traslada R1 al
repositorio Python padre y reserva TypeScript para R4+.

Debe congelarse si el binding usa:

1. el HEAD historico del motor TypeScript auditado;
2. el HEAD base del repositorio cientifico R1;
3. ambos como campos separados;
4. un hash de source bundle independiente de Git.

Cambiar nombres/campos o valores cambia todos los golden hashes y la futura
prueba de parity. R1 no puede inventar esos valores.

## 3. Enmienda minima requerida

Antes de reautorizar implementacion, una enmienda puramente documental debe
congelar exactamente:

1. fuente, exposicion y precision de ATR14;
2. politica aritmetica/comparaciones y puntos de cuantizacion;
3. argumentos exactos y frontera de output del regime adapter;
4. reemplazo de active pair despues de pivots/touches;
5. posicion y ownership de pending entry en el open;
6. counted-touch versus geometric-contact para rejection;
7. indexacion de las 12 candles de cooldown;
8. formula de recencia del pair;
9. `schema_version` y bindings de codigo/repositorio del thesis hash.

La enmienda puede resolver estas decisiones sin leer datos sellados ni outcomes.
R1 debera partir de su hash aprobado; no debe continuar sobre supuestos locales.

## 4. Estado preservado

```text
r1_implementation_created: false
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
```

No se modifico `binance-futures-bot-ts`, no se descargaron datos y no se hizo
commit.

`AEGIS_RANGE_R1_BLOCKED_BY_SPEC_AMBIGUITY`
