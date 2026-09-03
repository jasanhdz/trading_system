# Aegis Range Strategy V1 - R0 Preregistration Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R0_READY_FOR_REVIEW`

**Autoridad:** solo `R0_PREREGISTRATION_AUTHORIZED`

**No emitido:** `PRE_VALIDATION_SPEC_FROZEN`

Este documento cierra el contrato metodologico de R0. No autoriza R1, R2, TRAIN,
CALIBRATION, VALIDATION, HOLDOUT, SHADOW, live ni cambios de produccion. El plan
de autoridad sigue siendo `docs/AEGIS_RANGE_STRATEGY_V1_PLAN.md`.

## 1. Auditoria de rama y repositorios

### 1.1 Repositorio TypeScript

- Ruta: `/home/jasan/Develop/trading_system/binance-futures-bot-ts`
- Rama: `work/entry-quality-evidence-20260726`
- HEAD auditado: `bb034431e0ce05c8e0f978453c46dcff6efb981c`
- Commit HEAD: `bb03443 docs: close Aegis Range R0 methodology`
- Working tree antes de R0: limpio.
- Ultimos commits relevantes: `bb03443`, `4eb1035`, `6661e1e`, `7ba2121`,
  `967f612`, `f8a158a`, `e5fed4e`, `ac5d171`, `adbbcfc`, `814a302`.

### 1.2 Repositorio cientifico padre

- Ruta: `/home/jasan/Develop/trading_system`
- Rama: `work/entry-quality-evidence-20260726`
- HEAD auditado: `cfbde9bd47e9f29dfda35e9e9f05d57ca148f556`.
- El indice del padre registra `binance-futures-bot-ts` como gitlink a
  `814a302885e1d07bfd27404ebb5e69a30acebcc5`, mientras el repositorio hijo esta
  en `bb034431e0ce05c8e0f978453c46dcff6efb981c`.
- No existe una declaracion `.gitmodules` util para resolver ese gitlink.
- La modificacion visible del gitlink es consecuencia del HEAD independiente
  del hijo; R0 no la corrige ni la incorpora.

### 1.3 Frontera de ownership

- Este preregistro y la futura implementacion TypeScript research-only
  pertenecen al repositorio `binance-futures-bot-ts`.
- Los archivos historicos Binance y artefactos cientificos E4 existentes se
  consumen read-only desde el repositorio padre.
- R0 no mueve datos, no repara el gitlink y no crea codigo cientifico en el
  padre.
- Cualquier cambio futuro de ownership exige una decision separada antes de
  R1/R2 y no puede alterar los hashes fuente de este preregistro.

## 2. Inconsistencias registradas

1. `data/binance_candles.db` contiene 65 timestamps duplicados y 30 duplicados
   con OHLCV conflictivo. Su constructor historico pudo persistir la ultima vela
   parcial y no existe lineage checksum-backed por fila. Queda excluido como
   fuente canonica.
2. `RegimeEngineV2` no valida cierre, orden, duplicados ni continuidad. Su
   timestamp de salida representa el open time de la ultima vela, aunque la
   decision solo es valida al cierre. R0 impone un adapter causal futuro sin
   modificar el motor actual.
3. ADX, ATR y percentiles dependen del history length suministrado. R0 fija 160
   velas cerradas contiguas para eliminar ese grado de libertad.
4. No existe un fee schedule historico autoritativo por cuenta/VIP/BNB. R0 usa
   costos taker conservadores constantes y prohibe reclamar exactitud historica
   de cuenta.
5. Los estudios previos mezclan 10, 14, 20 y 30 bps y convenciones distintas de
   funding. Ninguno se hereda implicitamente.
6. No existe un periodo retrospectivo demostrado como E4 OOS end-to-end. Todo
   resultado historico Range+E4 queda sin autoridad de promocion.
7. Los manifiestos de funding y mark price describen 341 archivos por tipo,
   pero solo 237 por tipo estaban materializados localmente durante R0. Sus 341
   checksums esperados son conocidos; materializar y verificar los 104 restantes
   requerira autorizacion posterior y no forma parte de R0.
8. El repositorio padre y el TypeScript tienen commits y ownership
   independientes, con gitlink desfasado. Los hashes de ambos se registran de
   forma separada.

## 3. Contrato de datos

### 3.1 Universo

El universo es fijo y no seleccionable:

```text
BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT
ADAUSDT AVAXUSDT LINKUSDT SUIUSDT LTCUSDT
```

`config/universe.yaml` del repositorio padre es la referencia documental. Su
`symbol_set_hash` declarado es
`f6448e67daf1d017e16cc6b331f6494e97e178824474994fff08864303ccd348`.

### 3.2 Fuente canonica OHLCV

- Mercado: Binance USD-M futures.
- Dataset: klines mensuales de `data.binance.vision`.
- Granularidad fuente: 1m.
- Intervalo: `[2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z)`.
- Cobertura manifestada: 31 meses por 11 simbolos, 341 ZIP.
- Integridad: cada ZIP debe coincidir con su `expected_sha256` del manifiesto
  M1A antes de poder entrar al dataset derivado.
- Spot, aggTrades, `binance_candles.db`, live cache y colecciones de otros
  experimentos no pueden completar, sustituir ni corregir una vela primaria.

### 3.3 Agregacion causal 1m a 5m

El unico timeframe V1 es 5m para regimen, niveles, senal, entrada y salidas.

Para cada simbolo, una vela 5m con `open_time = T` usa exactamente las cinco
velas 1m con open times `T`, `T+1m`, `T+2m`, `T+3m` y `T+4m`, donde
`T mod 5m = 0`:

```text
open_5m   = open(T)
high_5m   = max(high(T..T+4m))
low_5m    = min(low(T..T+4m))
close_5m  = close(T+4m)
volume_5m = sum(volume(T..T+4m))
available_at = decision_at = T+5m
```

Se usa UTC y ventanas half-open. Los decimales fuente se parsean sin convertir
timestamps ni precios a precision binaria antes de validar. Una vela 1m
duplicada, ausente, fuera de orden, no finita, con OHLC inconsistente o con
checksum no verificado invalida la vela 5m. Una vela 5m invalida:

- no se rellena ni interpola;
- no genera decision ni fill;
- cierra el segmento continuo anterior;
- reinicia los 160 bars de warmup y todo estado de pivots/range;
- invalida cualquier trade que necesitara atravesar el gap, que se reporta como
  `DATA_INTEGRITY_EXCLUSION` y no entra a metricas economicas.

### 3.4 Funding y mark price

- Funding: archivos mensuales Binance USD-M `fundingRate` M1B.
- Valoracion en el evento: `markPriceKlines/1m` M1B.
- Ambos deben pasar sus checksums esperados.
- Si falta funding o mark price para cualquier evento que intersecta un trade,
  el trade es `FUNDING_DATA_MISSING` y queda excluido; funding cero no es una
  imputacion permitida.

Si cualquier trade recibe `DATA_INTEGRITY_EXCLUSION` o `FUNDING_DATA_MISSING`,
su `range_episode_id` completo queda `DATA_EXCLUDED`: ninguno de sus trades ni
labels entra a sample size, seleccion, bootstrap, metricas o gates. El conteo y
razon de episodios excluidos se reportan por separado. Cualquier exclusion en
CALIBRATION invalida ese candidato; cualquier exclusion en VALIDATION/HOLDOUT
hace fallar el gate de la particion.

### 3.5 Particiones selladas

Los limites son UTC y half-open:

| Particion | Inicio inclusive | Fin exclusive | Uso permitido |
|---|---|---|---|
| TRAIN | 2024-01-01T00:00:00Z | 2025-01-01T00:00:00Z | diseno/sanity bajo R2 |
| CALIBRATION | 2025-01-01T00:00:00Z | 2025-07-01T00:00:00Z | elegir 1 de 384 candidatos |
| VALIDATION | 2025-07-01T00:00:00Z | 2026-01-01T00:00:00Z | apertura unica tras freeze |
| HOLDOUT | 2026-01-01T00:00:00Z | 2026-08-01T00:00:00Z | sellado hasta gate validation |

R0 solo registra paths y checksums; no calcula features, episodios, senales ni
resultados de ninguna particion.

### 3.6 Purga, embargo y warmup

- El detector de niveles empieza vacio en cada particion. Pivots, clusters,
  touches y episodios de una particion anterior no cruzan la frontera.
- Las 160 velas anteriores al limite pueden alimentar exclusivamente el warmup
  de indicadores de regimen; no pueden crear niveles ni labels.
- Las primeras 48 horas de cada particion son embargo: se calculan indicadores
  y estado causal, pero no se confirma episodio ni se entra.
- Un episodio debe confirmar, operar y terminar antes del fin exclusive de su
  particion. Si sigue activo, tiene una orden pendiente o un trade abierto en la
  frontera, se purga el episodio completo de ambas particiones adyacentes.
- Un label de evaluacion cuyo horizonte exceda el fin de particion tambien purga
  el episodio.

## 4. Contrato causal y de regimen

### 4.1 Disponibilidad

- Todo calculo usa solo velas 5m cerradas con `available_at <= decision_at`.
- La decision se emite en el close timestamp, no en el open timestamp publicado
  actualmente por `RegimeEngineV2`.
- Una entrada/salida por senal cerrada usa el open de la siguiente vela 5m, que
  comparte timestamp con `decision_at` pero no se conoce hasta ejecutar el fill.
- Agregar datos posteriores no puede cambiar decisiones ya emitidas.
- Todos los niveles, ATR, target, stop y regimen se congelan al entrar.

### 4.2 Snapshot de `RegimeEngineV2`

Cada decision invoca conceptualmente el motor con exactamente las ultimas 160
velas 5m cerradas y contiguas, incluida la vela de decision. Se conservan sus
formulas existentes:

- ATR 14 y ADX 14;
- Choppiness 14 y `chopRisk`;
- Bollinger width 20;
- ATR y Bollinger percentile rank sobre hasta 120 observaciones;
- volume ratio contra las 20 velas cerradas anteriores;
- structure, range breakout, failed breakout y transition risk.

Historia menor de 160 es `INSUFFICIENT_HISTORY`, aunque el motor actual pudiera
producir algunos campos antes.

### 4.3 Hard blockers de entrada

Todos deben pasar en `decision_at`:

1. Segmento de 160 velas valido y continuo.
2. Regimen exacto en `{ACCUMULATION_RANGE, CHOP}`.
3. `transitionRisk != HIGH`.
4. `rangeBreakout == NONE`.
5. `ADX <= max_adx` del candidato.
6. `chopRisk >= min_chop_risk` del candidato.
7. `bollingerWidthPercentile < 0.45`.
8. `atrPercentile <= 0.80`.
9. `volumeRatio >= min_safety_volume_ratio` del candidato.
10. Episodio `OPERABLE_RANGE` activo, sin salida pendiente ni posicion abierta.
11. Reglas de nivel, rechazo, costos, reward/risk, cooldown y quota cumplidas.

`CHOP` por si solo nunca basta. Failed breakout count y breakout volume son
telemetria, no sustituyen blockers.

### 4.4 Score descriptivo

El score no permite operar, no participa en seleccion y no puede compensar un
blocker. Se registra en `[0,100]`:

```text
25 * min(1, min(support_touches, resistance_touches) / 4)
+ 20 * (1 - clamp(ADX / max_adx, 0, 1))
+ 20 * clamp((chopRisk - min_chop_risk) / (1 - min_chop_risk), 0, 1)
+ 15 * (1 - clamp(bollingerWidthPercentile, 0, 1))
+ 10 * clamp(volumeRatio / 1.5, 0, 1)
+ 10 * min(episode_age_hours / 24, 1)
```

### 4.5 Orden de transicion por vela

Para una vela 5m valida, ningun implementador puede reordenar estos pasos:

1. En el open se ejecutan salidas market pendientes y despues se evaluan gaps de
   TP/SL contra el estado congelado anterior.
2. Durante la vela se resuelven TP/SL resting, `adverse-first` si corresponde.
3. Al cierre se calcula el snapshot de regimen con las 160 velas cerradas.
4. Se actualizan max hold y breakout del trade contra la tesis congelada. Solo se
   programan salidas para el siguiente open; no se llenan en este cierre.
5. Para un episodio activo se actualiza su contador de breakout usando el ultimo
   snapshot publicado antes de esta vela. Si confirma, se termina y se resetea el
   detector; los pasos 6-10 se omiten para esta vela.
6. En cada cierre, haya o no pivot nuevo, expiran pivots y touches fuera del
   lookback y se recalculan medianas. Si el episodio activo pierde estructura,
   termina y se resetea; no puede renacer en esta misma vela.
7. Se insertan, en orden `pivot_at`, side y precio, los pivots cuyo `available_at`
   es este cierre; cada insercion recalcula su mediana.
8. La vela actual rearma o registra como maximo un touch por cluster elegible.
9. Se construyen y ordenan parejas. Se mantiene el par activo o se confirma uno
   nuevo si no hay episodio y no rige embargo.
10. Se publica el snapshot de niveles y despues se evaluan hard blockers y
    rejection. La misma vela que completa la confirmacion puede emitir rejection;
    la entrada continua siendo en `NEXT_BAR_OPEN`.

Los pasos de cierre usan exclusivamente el OHLCV ya cerrado. Una terminacion por
datos invalidos ocurre antes del paso 1 y no simula fills a traves del gap.

## 5. Pivots, niveles y touches

### 5.1 Pivots

- `L=2`, `R=2`, fijos y no calibrables.
- Pivot high en indice `i` si `high[i]` es estrictamente mayor que los highs de
  `i-2`, `i-1`, `i+1`, `i+2`.
- Pivot low analogo, estrictamente menor que los cuatro lows vecinos.
- Empates producen `NO_PIVOT`.
- `pivot_at` es el close timestamp de la vela `i`.
- `available_at` es el close timestamp de `i+2`.
- El pivot solo se inserta despues de ese cierre.

### 5.2 Clustering online

High pivots y low pivots nunca comparten cluster. El lookback movil es 7 dias
(2016 velas 5m) por `pivot_at`.

En cada `decision_at` se expiran observaciones antes de insertar pivots nuevos.
Para un pivot de precio `p` que queda disponible:

1. Ya se eliminaron todos los pivots y touches con
   `pivot_at/touch_at < decision_at-7d`, incluso si no aparece un pivot nuevo.
2. Cada centro se recalcula como la mediana de precios activos; con cantidad par
   se usa el promedio de los dos centrales.
3. La tolerancia de asignacion es
   `cluster_tolerance_atr * ATR14(available_at)`.
4. El pivot se asigna a un cluster del mismo lado si la distancia absoluta al
   centro es menor o igual a la tolerancia.
5. Si hay varios, gana menor distancia normalizada, despues cluster mas antiguo
   y finalmente `cluster_id` lexicografico.
6. Si no hay compatible, se crea uno. Los clusters no se fusionan entre si.

Si la expiracion deja cero pivots activos, se elimina definitivamente el cluster,
su ID, touches y estado de rearmado antes de procesar pivots nuevos. Con un solo
pivot conserva ID y mediana, pero no es elegible para touches ni parejas.

`cluster_id = sha256(symbol|side|first_pivot_at|first_pivot_price)` usando UTC
ISO-8601 y decimal fuente canonico. Un cluster requiere al menos dos pivots
activos para aceptar touches.

### 5.3 Touch

Sea `level` el centro causal actual y
`tau = cluster_tolerance_atr * ATR14(decision_at)`.

Un touch de soporte requiere:

```text
low <= level + tau
low >= level - 0.35 * ATR14
close > level
```

Un touch de resistencia requiere:

```text
high >= level - tau
high <= level + 0.35 * ATR14
close < level
```

Solo se cuentan touches ocurridos despues de que el cluster alcanzo dos pivots.
Varias velas del mismo contacto cuentan una vez. El contacto se rearma cuando:

- soporte: `low > level + 2*tau`;
- resistencia: `high < level - 2*tau`.

Ademas deben transcurrir al menos 6 velas cerradas desde el touch contado
anterior. Un touch pasado no se reevalua cuando cambia la mediana. Cada cluster
necesita dos touches activos y causalmente observados.

### 5.4 Pareja de niveles

Un par candidato requiere:

- cluster low como soporte y cluster high como resistencia;
- dos pivots y dos touches por lado;
- `support < close < resistance`;
- `amplitude = (resistance-support)/midpoint`;
- `midpoint = (support+resistance)/2`;
- `min_range_amplitude_pct <= amplitude <= 0.08`.

Si existen varios pares, se ordenan causal y deterministicamente por:

1. mayor minimo de touches entre lados;
2. mayor suma de touches;
3. touch mas reciente de la pareja;
4. `pair_first_eligible_at` mas antiguo;
5. IDs de soporte y resistencia lexicograficos.

No se examina el resultado economico para elegir el par.
`pair_first_eligible_at` es el primer `decision_at` continuo donde esos dos
cluster IDs pasan simultaneamente pivots, touches, orden y amplitud. Se borra si
el par deja de ser elegible y se crea de nuevo si recupera elegibilidad.

## 6. Lifecycle e identificadores

### 6.1 Inicio y version

Un episodio nace al cierre de la primera vela donde el par ganador pasa todos
los requisitos estructurales. Ese timestamp es `range_confirmed_at`.

```text
range_episode_id = sha256(
  symbol|5m|range_confirmed_at|support_cluster_id|resistance_cluster_id
)
```

El mismo par de cluster IDs conserva el episodio mientras sus medianas se
actualizan. Cada snapshot causal genera:

```text
range_id = sha256(
  range_episode_id|decision_at|support_12dp|resistance_12dp|midpoint_12dp
)
```

Los valores derivados para hashing usan decimal `ROUND_HALF_EVEN` a 12 lugares
decimales. El `range_id` de entrada y su tesis quedan congelados.

### 6.2 Fin del episodio

El episodio termina por el primer evento disponible:

1. `CONFIRMED_BREAKOUT`;
2. 48 horas desde `range_confirmed_at`;
3. cluster de soporte o resistencia cae por debajo de dos pivots/touches activos;
4. amplitud sale de `[min_range_amplitude_pct, 0.08]`;
5. otro par pasa a ser ganador;
6. gap/integridad de datos;
7. frontera de split.

Al terminar se descartan pivots, clusters y touches del detector del simbolo.
Solo velas posteriores al timestamp final pueden construir el siguiente
episodio. Un fallo temporal de ADX, regimen, volumen o transition risk bloquea
entradas, pero no termina por si mismo la estructura.

`CONFIRMED_BREAKOUT` del episodio es independiente de que exista un trade. Para
cada cierre se usa el snapshot causal del episodio publicado al cierre anterior:

```text
UP_OUTSIDE   si close > resistance_prev + 0.10 * ATR_prev
DOWN_OUTSIDE si close < support_prev    - 0.10 * ATR_prev
```

Dos cierres consecutivos con la misma direccion confirman `UP_BREAKOUT` o
`DOWN_BREAKOUT`; un cierre dentro o que cambia de direccion reinicia el contador
a uno/cero segun corresponda. Los niveles no se recalculan antes de esta prueba.
El episodio termina al segundo cierre. El trade, si existe, conserva su propia
regla adversa basada en niveles de entrada y su salida causal independiente.

### 6.3 Labels de evaluacion

- `false_range = true` si ocurre `UP_BREAKOUT` o `DOWN_BREAKOUT` del episodio
  dentro de las primeras 12 velas cerradas (60 minutos) posteriores a
  `range_confirmed_at`, haya o no trade.
- `breakout_loss = true` si un trade termina por breakout y su retorno neto es
  negativo.
- Los labels solo evaluan episodios ya confirmados; nunca retroceden para borrar
  o crear entradas.

## 7. Rejection, entrada y tesis

### 7.1 Vela de rechazo

Se evalua despues del cierre. Sea `body = abs(close-open)` y
`body_floor = max(body, 0.01*ATR14)`.

LONG requiere conjuntamente:

- touch de soporte valido en esa vela;
- `close > open`;
- `support < close <= midpoint`;
- `(min(open,close)-low)/body_floor >= rejection_min_wick_body_ratio`.

SHORT es simetrico:

- touch de resistencia valido;
- `close < open`;
- `midpoint <= close < resistance`;
- `(high-max(open,close))/body_floor >= rejection_min_wick_body_ratio`.

No se usa RSI ni volumen de entrada adicional. `min_entry_volume_ratio=0` queda
fijo y deshabilitado. Una vela ambigua no genera senal.

### 7.2 Entrada

- La senal queda disponible en `decision_at`.
- La orden se ejecuta en `NEXT_BAR_OPEN` con slippage adverso.
- No hay entrada si falta la siguiente vela, su open no es valido, cae fuera del
  mismo split, `entry_at-range_confirmed_at >=48h`, el episodio termino al cierre
  anterior, o el raw open no cumple `support_at_entry < open < resistance_at_entry`.
  Esas son las unicas invalidaciones evaluadas en el open; no se usa high, low,
  close, volumen ni regimen de la vela que acaba de abrir.
- El target debe quedar favorable respecto al fill efectivo.
- Distancia bruta fill-target debe ser al menos 42 bps.
- Reward/risk bruto respecto a stop y target debe ser `>=1.0`.
- Al entrar se congelan episodio, `range_id`, soporte, resistencia, midpoint,
  ATR, stop, target, regimen, score, costos y hashes de tesis.

`thesis_feature_hash` es SHA-256 del JSON UTF-8, sin whitespace y con keys
ordenadas lexicograficamente, que contiene exactamente:

```text
schema_version, symbol, side, range_episode_id, range_id,
decision_at, entry_available_at, entry_fill,
support_at_entry, resistance_at_entry, midpoint_at_entry, ATR_entry,
range_confirmed_at, stop_at_entry, target_at_entry,
regime_at_entry, range_confidence_at_entry, tail_risk_score_at_entry,
los ocho parametros del candidato, cost_scenario,
source_manifest_sha256, split_manifest_sha256, typescript_git_head
```

Timestamps usan UTC ISO-8601 con milisegundos. Enums/IDs son strings, ausencia E4
es JSON `null`, y todo decimal calculado usa string `ROUND_HALF_EVEN` a 12
lugares. `entry_fill` es el fill efectivo posterior a slippage. No se incluyen
campos opcionales o dependientes de resultados.

### 7.3 TP y SL

Con `ATR_entry` y parametros del candidato seleccionado:

```text
LONG target = midpoint - target_buffer_atr * ATR_entry
LONG stop   = support  - stop_buffer_atr   * ATR_entry

SHORT target = midpoint + target_buffer_atr * ATR_entry
SHORT stop   = resistance + stop_buffer_atr * ATR_entry
```

TP y SL son resting durante el backtest, pero todos sus fills pagan taker fee y
slippage conservador. Nunca se ensanchan ni mejoran despues de entrar.

### 7.4 Breakout y max hold

Breakout del trade LONG-adverso requiere dos cierres consecutivos
`close < support_at_entry - 0.10*ATR_entry`. Breakout SHORT-adverso requiere dos
cierres consecutivos `close > resistance_at_entry + 0.10*ATR_entry`. Queda
confirmado al segundo cierre y sale en `NEXT_BAR_OPEN`.

`breakout_volume_ratio=1.5` se registra solo para segmentacion y no cambia la
confirmacion. Un cierre que vuelve dentro reinicia el contador.

Max hold es 144 velas, exactamente 12 horas desde el fill de entrada. Al cierre
de la vela 144 se programa salida en el open siguiente.

### 7.5 Reentry

- Una posicion maxima por simbolo.
- Maximo dos trades por `range_episode_id`.
- Maximo un LONG y un SHORT por episodio.
- Cooldown de 12 velas cerradas (60 minutos) despues de cada salida.
- No hay scale-in, pyramiding ni entrada simultanea.
- Terminar el episodio cancela toda reentrada.

## 8. Fills, gaps y precedencia

Para cada vela despues de entrar, la precedencia exacta es:

1. salida market pendiente por breakout del cierre anterior;
2. salida market pendiente por max hold del cierre anterior;
3. gaps en TP/SL al open;
4. TP/SL intrabar, con `adverse-first` si ambos fueron alcanzables;
5. al cierre, actualizar contadores de breakout y max hold para el open siguiente.

Si breakout y max hold nacen en el mismo cierre, breakout gana como exit reason.

Gap LONG:

- `open <= stop`: trigger base `open`, nunca el stop mejor;
- `open >= target`: trigger base `target`, sin mejora favorable.

Gap SHORT es simetrico. Una salida market pendiente llena al open aunque el
rango intrabar posterior toque otro nivel. El slippage adverso se aplica despues
de determinar el trigger base:

```text
BUY fill  = base_price * (1 + slippage_bps/10000)
SELL fill = base_price * (1 - slippage_bps/10000)
```

## 9. Costos

### 9.1 Baseline obligatorio

- Fee de entrada: 5 bps taker.
- Fee de salida: 5 bps taker.
- Slippage de entrada: 2 bps adverso.
- Slippage de salida: 2 bps adverso.
- Roundtrip aproximado sin movimiento/funding: 14 bps.
- Sin maker rebate, VIP, descuento BNB ni fee negativo.

### 9.2 Stress preregistrado

| Escenario | Fee por lado | Slippage por lado | Roundtrip aproximado |
|---|---:|---:|---:|
| BASELINE | 5 bps | 2 bps | 14 bps |
| STRESS_20 | 5 bps | 5 bps | 20 bps |
| STRESS_30 | 5 bps | 10 bps | 30 bps |

El baseline decide la inferencia primaria; `STRESS_20` forma parte del gate y
`STRESS_30` es diagnostico obligatorio.

BASELINE fija la poblacion, entry eligibility, timestamps, triggers y exit
reasons. `STRESS_20` y `STRESS_30` reprician exactamente esos mismos trades con
su slippage por lado y el mismo fee/funding; no vuelven a ejecutar blockers,
reward/risk, distancia minima, senales ni lifecycle. Un trade aceptado por
BASELINE nunca desaparece de stress y uno rechazado nunca aparece.

### 9.3 Retorno y funding

El fill efectivo de entrada define unit notional: `qty=1/entry_fill`. Para lado
`s=+1` LONG y `s=-1` SHORT:

```text
gross_return = s * (exit_fill-entry_fill) / entry_fill
fee_return = 0.0005 * (1 + exit_fill/entry_fill)
funding_return = -s * sum(rate_j * mark_price_j/entry_fill)
net_return = gross_return - fee_return + funding_return
```

Se incluyen eventos funding con timestamps en `(entry_fill_at, exit_fill_at]`.
Un evento exactamente en la entrada se excluye; uno exactamente en la salida se
incluye. `mark_price_j` es el close de la vela mark-price 1m cuyo
`open_time = funding_at-60s` y `available_at = funding_at`. El funding timestamp
debe estar alineado a minuto UTC. Si esa vela exacta no existe o no esta cerrada,
aplica `FUNDING_DATA_MISSING`; no se usa la vela que abre en `funding_at`.

## 10. Grid cerrado y seleccion

### 10.1 Unica familia V1

No hay busqueda abierta de formulas ni familias alternativas. TRAIN puede
verificar plumbing y falsar esta familia, pero no agregar parametros o variantes
sin revisar R0 antes de abrir CALIBRATION.

CALIBRATION puede seleccionar exactamente un candidato del producto cartesiano:

| Parametro | Valores permitidos |
|---|---|
| `cluster_tolerance_atr` | `[0.20, 0.30]` |
| `min_range_amplitude_pct` | `[0.0125, 0.0200, 0.0300]` |
| `rejection_min_wick_body_ratio` | `[1.0, 1.5]` |
| `stop_buffer_atr` | `[0.35, 0.50]` |
| `target_buffer_atr` | `[0.00, 0.10]` |
| `max_adx` | `[20, 25]` |
| `min_chop_risk` | `[0.62, 0.70]` |
| `min_safety_volume_ratio` | `[0.50, 0.75]` |

Total: `2*3*2*2*2*2*2*2 = 384` candidatos. Todos los demas valores de este
documento son fijos/singleton. Esta misma configuracion se aplica a los 11
simbolos y ambos lados.

### 10.2 Objetivo de seleccion

Cada candidato debe tener cero episodios `DATA_EXCLUDED`, al menos 100 episodios
operados y 20 trades por lado en CALIBRATION. Entre candidatos elegibles se
maximiza el percentil 5 del bootstrap de expectativa neta media por episodio bajo
BASELINE.

Empates con diferencia absoluta `<1e-6` se resuelven, en orden, por:

1. mayor expectativa neta `STRESS_20`;
2. menor maximum drawdown normalizado;
3. menor cantidad de trades;
4. serializacion JSON lexicograficamente menor de parametros.

No se selecciona por simbolo, lado, mes, regimen ni resultado E4. Si ningun
candidato cumple elegibilidad, el resultado es `NO_CALIBRATION_CANDIDATE` y no
se abre VALIDATION.

## 11. Hipotesis y multiplicidad

### 11.1 Contrato elegido

Se elige el contrato global: la unica autoridad de promocion es la estrategia
pooled de 11 simbolos con ambos lados y un unico candidato.

```text
H0 primaria: media del retorno neto por episodio <= 0
H1 primaria: media del retorno neto por episodio > 0
alpha: 0.05, unilateral
```

Un episodio operado es una observacion primaria. Su outcome es la suma de los
retornos netos de sus uno o dos trades. Episodios sin trade quedan en tasas de
estructura/abstencion, no en la media economica.

### 11.2 Secundarias

- LONG.
- SHORT.
- Majors: `BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT`.
- Alts: `XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, SUIUSDT, LTCUSDT`.

Los cuatro p-values unilaterales forman una familia Holm-Bonferroni con
`familywise alpha=0.05`, valida bajo dependencia. Ninguna secundaria rescata un
fallo primario. Simbolo individual, mes, regime exacto y cualquier subgrupo no
listado son `HYPOTHESIS_GENERATION_ONLY`.

Para LONG/SHORT, el outcome de episodio es la suma de trades de ese lado y solo
entran episodios que lo operaron. Para majors/alts se usa el outcome completo de
episodios de sus simbolos. Para cada secundaria se centra cada outcome restando
su media observada, se ejecuta el mismo moving-block bootstrap sincronizado y se
calcula:

```text
p = (1 + count(bootstrap_centered_mean >= observed_mean)) / 10001
```

Holm ordena `p(1)<=...<=p(4)` y compara secuencialmente con
`0.05/4, 0.05/3, 0.05/2, 0.05`; al primer fallo, esa hipotesis y las restantes no
se rechazan. La inferencia primaria usa ademas su lower bound preregistrado y no
comparte esta familia.

## 12. Bootstrap y metricas

- 10,000 replicas.
- Seed fija: `20260824`.
- Moving blocks de 7 dias UTC.
- Los starts candidatos son cada dia UTC que satisface
  `split_start <= start <= split_end-7d`; cada bloque completo vive dentro de la
  misma particion y nunca puede leer un episodio adyacente.
- Cada bloque conserva conjuntamente todos los simbolos y episodios cuyo
  `range_confirmed_at` cae en `[block_start, block_start+7d)`.
- Se muestrean bloques completos con reemplazo hasta cubrir la duracion de la
  particion; si sobra longitud sintetica, solo se usan los primeros dias
  necesarios del ultimo bloque. El truncamiento es por posicion dentro del
  bloque muestreado, no por timestamp fuera del split.
- Todos los trades de un episodio permanecen juntos.
- La dependencia transversal contemporanea se conserva al muestrear el mismo
  bloque temporal para los 11 simbolos.
- Intervalos primarios: percentile bootstrap unilateral 95%. Para cualquier
  quantile se ordenan `n` valores y se usa nearest-rank
  `Q(p)=x[ceil(p*n)]` con indices desde uno, sin interpolacion.

Metricas exactas:

- Expectancy: media de `episode_net_return`.
- Profit factor: suma de trade net returns positivos dividida por valor absoluto
  de la suma de negativos; infinito si no hay negativos.
- CVaR95: ordenar ascendente los `n` episode returns y promediar exactamente los
  primeros `max(1,ceil(0.05*n))`; no se agregan empates fuera de ese rank.
- Pseudo-equity: trades ordenados por fill timestamp y simbolo, cada retorno
  aporta `net_return/11`; no representa sizing live.
- Pseudo-equity empieza en `E_0=1` y evoluciona aditivamente como
  `E_k=E_(k-1)+net_return_k/11`. Para cada k,
  `P_k=max(E_0,...,E_k)` y `DD_k=(P_k-E_k)/P_k`; maximum drawdown es
  `max_k(DD_k)`. Si `E_k<=0`, maximum drawdown se fija en 100%.
- Breakout-loss rate: breakout losses / trades cerrados.
- False-range rate: false ranges / episodios confirmados.

## 13. Gate numerico congelado

VALIDATION pasa solo si cumple conjuntamente bajo el candidato congelado:

1. Cero episodios `DATA_EXCLUDED`, al menos 100 episodios operados y 20 trades
   por lado.
2. Expectativa neta BASELINE positiva y lower bound bootstrap unilateral 95%
   estrictamente mayor que cero.
3. Profit factor BASELINE `>=1.10`.
4. Expectativa neta `STRESS_20 > 0` y profit factor `STRESS_20 > 1.00`.
5. Maximum drawdown de pseudo-equity `<=10%`.
6. CVaR95 por episodio `>=-1.5%`.
7. Breakout-loss rate `<=20%`.
8. False-range rate `<=35%`.
9. Al menos 4 de los 6 meses tienen expectativa neta positiva.
10. LONG, SHORT, majors y alts tienen expectativa puntual no negativa.
11. Al menos 7 de 11 simbolos tienen expectativa puntual no negativa.
12. Ningun simbolo aporta mas de 35% y ningun mes mas de 50% de la suma de PnL
    positivo. Si el PnL pooled no es positivo, el gate falla antes.

Para el punto 12, `G_g` es la suma de `episode_net_return` BASELINE de todos los
episodios cuyo `range_confirmed_at` pertenece al simbolo/mes `g`. La contribucion
es `max(G_g,0) / sum_h(max(G_h,0))`, calculada por separado para los 11 simbolos
y para los meses de la particion. No se usan gross wins ni trades individuales.

Solo si VALIDATION pasa se abre HOLDOUT una vez. HOLDOUT aplica el mismo gate,
salvo estabilidad temporal: al menos 4 de sus 7 meses deben ser positivos. No se
retunea, excluye ni especializa nada entre ambas aperturas. Un fallo en cualquier
condicion termina V1 como `REJECTED`.

## 14. Politica E4

### 14.1 Primario

Range puro no usa E4 para seleccionar poblacion. Si existe un score historico se
registra como telemetria sin filtrar, ordenar ni elegir thresholds.

### 14.2 Secundario

E4 permanece congelado:

```text
guard: e4_tail_risk
model: E4_V1_FROZEN
BLOCK si score >= 0.4522452210875323
```

Se preregistran dos lecturas descriptivas, sin tuning:

- Range + E4 frozen: conserva `score < 0.4522452210875323`.
- Range compatibility estricta: conserva `score < 0.40`.

El unico OOS model-only documentado de E4 es 2023-11-07 a 2023-12-06 y no
intersecta este dataset. La politica de guard fue evaluada con datos que llegan a
2026-08-01 y el freeze operacional es posterior. Por ello todos los resultados
retrospectivos secundarios V1 se etiquetan obligatoriamente:

```text
CONTAMINATED_COMPATIBILITY_DIAGNOSTIC
NO_PROMOTION_AUTHORITY
```

No pueden alterar seleccion, gate ni veredicto primario. Evidencia E4 OOS
end-to-end solo puede comenzar prospectivamente despues del freeze operacional
y pertenece a una fase autorizada posterior.

## 15. Freeze y reproducibilidad

### 15.1 Manifiestos R0

- `docs/aegis-range-v1/r0_source_manifest.json`
- `docs/aegis-range-v1/r0_split_manifest.json`
- `docs/aegis-range-v1/r0_production_diff.json`
- `docs/aegis-range-v1/r0_artifact_manifest.json`

El source manifest fija hashes de manifests upstream, universo, codigo conceptual
y baselines de produccion. El split manifest fija limites, embargo y purga. El
artifact manifest fija los hashes de los entregables R0 sin intentar un hash
autorreferencial.

### 15.2 Protocolo futuro

1. R1/R2 requieren aprobacion explicita posterior.
2. Antes de TRAIN se materializan fuentes faltantes y se verifica cada checksum.
3. TRAIN no puede leer CALIBRATION, VALIDATION ni HOLDOUT para debugging visual.
4. CALIBRATION se abre una vez para seleccionar dentro de los 384 candidatos.
5. Se publican hashes del dataset derivado, codigo R1/R2, candidato, resultados
   TRAIN/CALIBRATION y este preregistro.
6. Solo entonces puede emitirse `PRE_VALIDATION_SPEC_FROZEN`.
7. VALIDATION y HOLDOUT verifican hashes antes de su apertura unica.

Un cambio a formula, universo, split, costo, grid, lifecycle, gate o inferencia
invalida el preregistro y exige una nueva version antes de observar datos
sellados.

## 16. Limites de implementacion

R0 no creo ni modifico:

- `RangeDetectorV1.ts`, `RangeLevelsV1.ts`, `RangeSafetyV1.ts`,
  `RangeSignalV1.ts`, `RangeBreakoutV1.ts`;
- tests ejecutables ni backtester;
- `TradingService.ts`, execution, orders, leverage, sizing o brackets;
- `AegisSymbolMode`, `AegisRegimeGuard.ts` o `regime_config.live.yaml`;
- modelo, threshold, API, FeatureBridge, precompute o adapter E4;
- PM2, procesos live ni produccion.

## 17. Checklist R0

- [x] Timeframe unico 5m.
- [x] Pivots `L=2/R=2`, empates y `available_at`.
- [x] Clustering online, touches y amplitud.
- [x] Lifecycle, `range_episode_id` y `range_id`.
- [x] Rejection, entry, TP, SL, breakout y max hold.
- [x] Resting fills, adverse-first, gaps y precedencia.
- [x] Fees, slippage, funding y stress.
- [x] Reentry, cooldown y maximo de trades.
- [x] Splits, warmup, embargo y purga.
- [x] Grid cerrado de 384 candidatos y seleccion.
- [x] Hipotesis, subgrupos y Holm-Bonferroni.
- [x] Bootstrap temporal sincronizado de 7 dias.
- [x] Gate numerico VALIDATION/HOLDOUT.
- [x] Primario Range puro y secundario E4 contaminado.
- [x] Fuentes y hashes documentales.
- [x] Inconsistencias y frontera de repositorios.
- [x] Diff documental que confirma produccion intacta.

R0 queda listo para revision externa. No existe ningun permiso implicito para
continuar a R1/R2.
