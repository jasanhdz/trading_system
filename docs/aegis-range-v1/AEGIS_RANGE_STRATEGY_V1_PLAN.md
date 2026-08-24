# Aegis Range Strategy V1

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_STRATEGY_V1_DESIGN_APPROVED`

**Fase autorizada:** `R0_PREREGISTRATION_AUTHORIZED`

**Fases no autorizadas todavía:** R1, R2, `VALIDATION`, `HOLDOUT`, integración SHADOW y live

**Objetivo:** Evaluar si los rechazos causales en soportes y resistencias de rangos laterales operables producen expectativa neta positiva después de costos.

## 1. Decisiones Inmutables

### 1.1 Nombre y separación de responsabilidades

La estrategia se llama **Aegis Range Strategy V1**. Sus componentes de investigación usan nombres como `RangeDetectorV1`, `RangeLevelsV1`, `RangeSignalV1` y `RangeBreakoutV1`.

El nombre E4 queda reservado exclusivamente para el Tail Risk Guard existente:

- Guard: `e4_tail_risk`
- Endpoint: `/ml-v2/e4_tail_risk`
- Modelo: `E4_V1_FROZEN`
- Threshold congelado: `0.4522452210875323`

Range no puede modificar el modelo, threshold, `FeatureBridge`, API, precompute, adapter, decisiones ni evidencia de E4. Cuando Range eventualmente genere una oportunidad, el E4 Tail Risk Guard existente evaluará ese lado y conservará autoridad para bloquearlo.

### 1.2 Autoridad operacional y estrategia son dimensiones separadas

`AegisSymbolMode` permanece exactamente:

```typescript
type AegisSymbolMode = 'OFF' | 'SHADOW' | 'LIVE';
```

No se añadirá `RANGE`. El modo responde **si el símbolo puede operar**; el router de estrategia responderá **qué estrategia corresponde al estado de mercado**.

```text
                 SYMBOL MODE
          OFF / SHADOW / LIVE
                    |
                    v
               Market State
                    |
          +---------+---------+
          |         |         |
          v         v         v
      MOMENTUM    RANGE     UNSAFE
          |         |
          v         v
      estrategia  Aegis Range
       existente  Strategy V1
          |         |
          +----+----+
               |
               v
        E4 Tail Risk frozen
               |
               v
          Hard Safety
               |
               v
       Approval Boundary
               |
               v
       ejecución existente
```

La integración operacional de este router no pertenece a `RESEARCH_ONLY`.

### 1.3 Motor de régimen autorizado

Range reutilizará conceptualmente `RegimeEngineV2`. No se añadirá `RANGE_BOUND` al guard legacy `AegisRegimeGuard`.

`RegimeEngineV2` ya expone, entre otros:

- `ADX` y pendiente de ADX
- Choppiness y `chopRisk`
- Percentil de ATR
- Percentil del ancho de Bollinger
- Volume ratio
- Estructura de mercado
- Range breakout
- Failed breakouts
- Compresión previa al breakout
- Breakout follow-through
- Transition risk
- Régimen `ACCUMULATION_RANGE`

La clasificación actual de `ACCUMULATION_RANGE` ocurre cuando:

```text
chopRisk >= 0.62
AND structure == MIXED
AND bollingerWidthPercentile < 0.45
```

`CHOP` no equivale a rango seguro. `ACCUMULATION_RANGE` o un contexto no tendencial solo crean un **candidato**; `RangeDetectorV1` debe demostrar además que existen niveles causales, repetibles y operables.

## 2. Hipótesis Preregistrada

### 2.1 Hipótesis principal

> Cuando `RegimeEngineV2` identifica un entorno no tendencial y existe un rango causalmente confirmado con soporte y resistencia repetidos, comprar un rechazo cerca del soporte y vender un rechazo cerca de la resistencia produce expectativa neta positiva después de costos.

### 2.2 Flujo experimental

```text
Closed candles
      |
      v
RegimeEngineV2
      |
      v
RangeDetectorV1
      |
      v
RangeLevelsV1
      |
      v
RangeSafetyV1
      |
      v
RangeSignalV1
      |
      v
Historical outcome
```

La primera investigación no modificará `TradingService`, ejecución, órdenes, leverage, sizing, brackets, `AegisSymbolMode` ni `regime_config.live.yaml`.

### 2.3 Jerarquía de hipótesis

La hipótesis primaria es una estrategia Range agrupada sobre los 11 símbolos. Las especializaciones se preregistran así:

- **Primaria:** pooled 11-symbol Range.
- **Secundaria 1:** LONG.
- **Secundaria 2:** SHORT.
- **Secundaria 3:** majors, con membresía exacta definida en R0.
- **Secundaria 4:** alts, con membresía exacta definida en R0.

Las hipótesis secundarias requieren control de multiplicidad preregistrado. Los resultados por símbolo, mes y régimen son diagnósticos de estabilidad, no permisos automáticos para seleccionar ganadores.

Una especialización descubierta después de observar `VALIDATION`, por ejemplo “solo LONG en BTC y ETH”, queda etiquetada como `HYPOTHESIS_GENERATION_ONLY`. No tiene autoridad de promoción y requiere un experimento futuro con datos no observados.

R0 debe elegir explícitamente uno de estos contratos antes de abrir datos fuera de TRAIN:

1. Estrategia global que debe pasar agregada y estable.
2. LONG y SHORT como hipótesis independientes con multiplicity control.
3. Símbolos o regímenes como hipótesis independientes con criterios y multiplicity control propios.

El contrato elegido, su objetivo de selección y su corrección de multiplicidad forman parte del hash de `PRE_VALIDATION_SPEC_FROZEN`.

## 3. Invariantes de Causalidad

Estas reglas son hard blockers del experimento, no factores de confianza:

1. Solo se usan velas cerradas disponibles en `decision_at`.
2. Ningún indicador, nivel, régimen o etiqueta puede usar datos posteriores a `decision_at`.
3. Cada pivot o extremo local debe almacenar `pivot_at` y `available_at`.
4. Para un pivot con `L` velas a la izquierda y `R` a la derecha, el pivot en `T` solo está disponible después del cierre de `T + R`.
5. Al construir niveles en `T`, solo pueden utilizarse pivots con `available_at <= T`.
6. El rango se reconstruye en cada `T` con la información disponible hasta `T`; está prohibido descubrir un rango retrospectivamente y simular entradas anteriores a su confirmación.
7. Una vela de rechazo debe estar cerrada antes de emitir la señal.
8. La entrada histórica se ejecuta en `NEXT_BAR_OPEN`, incluyendo costos y slippage configurados.
9. Un breakout de dos cierres solo se confirma al terminar el segundo cierre; la salida histórica se ejecuta en `NEXT_BAR_OPEN`.
10. El volumen completo de una vela OHLCV solo está disponible al cierre. No se simulan cierres intrabar con volumen final conocido retrospectivamente.
11. Una variante futura basada en eventos o WebSocket será un experimento separado y no se mezclará con el backtest OHLCV.
12. Al abrir una operación se congela la tesis del rango; ningún recálculo posterior puede ensanchar el stop o mejorar retrospectivamente el target.

### 3.1 Snapshot congelado de entrada

Cada operación de investigación almacenará como mínimo:

```text
range_episode_id
range_id
decision_at
entry_available_at
support_at_entry
resistance_at_entry
midpoint_at_entry
range_confirmed_at
stop_at_entry
target_at_entry
regime_at_entry
range_confidence_at_entry
tail_risk_score_at_entry
thesis_feature_hash
```

`range_episode_id` es la unidad estadística primaria. Un episodio puede contener varios trades anidados que comparten régimen, niveles y estructura. `range_id` identifica la versión causal concreta de niveles usada por una tesis; R0 debe fijar cuándo una actualización de niveles conserva o inicia un nuevo episodio.

## 4. Componentes Research-Only

### 4.1 `RangeDetectorV1`

Responsabilidades:

- Consumir velas cerradas y el snapshot causal de `RegimeEngineV2`.
- Distinguir `RANGE_CANDIDATE`, `OPERABLE_RANGE` y `NOT_OPERABLE`.
- Separar invariantes obligatorios de un score descriptivo de confianza.
- Rechazar contextos con breakout activo, transition risk incompatible, historia insuficiente o niveles no confirmados.

No usará una regla ciega de “3 de 4”. Los requisitos estructurales son hard blockers; los indicadores auxiliares solo pueden contribuir a confianza o segmentación.

### 4.2 `RangeLevelsV1`

Responsabilidades:

- Crear pivots causales con `pivot_at` y `available_at`.
- Agrupar niveles compatibles sin utilizar información futura.
- Contar toques confirmados y rechazos por lado.
- Estimar soporte, resistencia, midpoint, amplitud y antigüedad.
- Generar un `range_id` determinista y auditable.

La primera variante propuesta para investigación puede usar `L=2, R=2`, pero ese valor deberá quedar preregistrado antes de abrir `VALIDATION`. Un pivot en `T` no existe para el algoritmo hasta `T+2`.

### 4.3 `RangeSafetyV1`

Responsabilidades:

- Confirmar que el candidato posee soporte y resistencia repetibles.
- Diferenciar `CHOP` desordenado de rango operable.
- Rechazar niveles demasiado estrechos después de costos.
- Rechazar breakouts, alta transición, datos incompletos y políticas de riesgo incumplidas.
- Mantener el experimento primario libre de E4 como selector de población.
- Registrar tail risk solo como telemetría para el experimento secundario Range+E4.

### 4.4 `RangeSignalV1`

Responsabilidades:

- Emitir `LONG`, `SHORT` o `NONE` después del cierre de una vela de rechazo.
- Registrar `decision_at` y ordenar ejecución simulada en `NEXT_BAR_OPEN`.
- Congelar el snapshot de tesis al entrar.
- Mantener LONG y SHORT como cohortes separadas.

Definición inicial que debe formalizarse en R0:

- Rechazo LONG: la vela toca o penetra la zona causal de soporte y cierra nuevamente dentro del rango.
- Rechazo SHORT: la vela toca o penetra la zona causal de resistencia y cierra nuevamente dentro del rango.

Los requisitos de cuerpo, wick, distancia al nivel, RSI u otras confirmaciones no se asumirán como definitivos hasta preregistrarlos.

### 4.5 `RangeBreakoutV1`

Responsabilidades:

- Detectar ruptura alcista o bajista respecto al snapshot congelado de la operación.
- Separar breakout confirmado por cierres de breakout asistido por volumen.
- Registrar el momento exacto en que la evidencia queda disponible.
- Simular salida causal en `NEXT_BAR_OPEN`.

Variante OHLCV inicial:

```text
primer cierre fuera del rango
        |
segundo cierre fuera del rango
        |
breakout confirmado al cierre
        |
exit NEXT_BAR_OPEN
```

## 5. Registro de Fórmulas R0

Antes de escribir thresholds definitivos se documentarán y congelarán las fórmulas siguientes. Cada fórmula tendrá versión, lookback, tratamiento de datos insuficientes y timestamp de disponibilidad.

### 5.1 Fórmulas ya disponibles en `RegimeEngineV2`

Se reutilizarán sus implementaciones causales cuando sean adecuadas, sin duplicarlas arbitrariamente:

- ADX de 14 periodos y pendiente de la serie.
- ATR de 14 periodos.
- `atrPercentile`: percentile rank del ATR actual dentro de hasta 120 observaciones recientes.
- Bollinger width de 20 periodos.
- `bollingerWidthPercentile`: percentile rank del ancho actual dentro de hasta 120 observaciones recientes.
- `volumeRatio`: volumen de la vela actual dividido por el promedio de las 20 velas cerradas anteriores.
- Choppiness de 14 periodos.
- Estructura, range breakout, failed breakouts y transition risk existentes.

### 5.2 Fórmulas que deben preregistrarse antes del experimento

- **Timeframe:** timeframe exacto de régimen, niveles, señal y ejecución. R0 debe elegir una única variante inicial; no se permite probar después 5m, 15m o combinaciones hasta encontrar la mejor.
- **Range amplitude:** fórmula exacta y denominador (`midpoint`, soporte u otro).
- **Zone tolerance:** distancia permitida alrededor del nivel, preferiblemente normalizada por ATR o precio.
- **Touch:** penetración o aproximación válida, separación temporal mínima y regla para evitar contar varias velas del mismo contacto.
- **Rejection candle:** touch, cierre dentro del rango, cuerpo, wick y momento de disponibilidad.
- **Range duration:** desde qué evento empieza y cómo se trata una actualización de niveles.
- **Range confirmation:** mínimos de toques por lado y orden temporal permitido.
- **False range:** etiqueta causal de fallo usada para evaluación, no para entradas previas.
- **Breakout:** número de cierres, distancia mínima fuera del rango y tratamiento del buffer.
- **Take profit:** midpoint exacto o buffer causal antes del midpoint.
- **Stop loss:** distancia exacta respecto al nivel congelado, expresada en ATR, porcentaje u otra unidad preregistrada.
- **Max hold:** duración exacta y momento causal de la salida.
- **Reentry:** si se permiten reentradas, cooldown y máximo de trades por `range_episode_id`.
- **Range episode lifecycle:** inicio, actualización, expiración, invalidación y condición de nuevo episodio.
- **Split boundary policy:** purga de episodios que atraviesen fronteras de partición.
- **Costs:** fees, slippage y funding aplicados por símbolo y periodo.

Ninguna fórmula o threshold puede elegirse observando `VALIDATION` o `HOLDOUT`.

### 5.3 Checklist de cierre de R0

R0 no termina hasta congelar:

```text
timeframe exacto
pivots exactos
clustering exacto
touch exacto
rejection exacto
entry exacta
TP exacto
SL exacto
breakout exacto
max hold exacto
costos exactos
reentry policy
range episode definition
TRAIN dates
CALIBRATION dates
VALIDATION dates
HOLDOUT dates
threshold grids
selection objective
side/symbol/regime policy
multiplicity control
bootstrap block size
E4 OOS policy
dataset/config/code hashes
```

El artefacto R0 completo permite solicitar autorización separada para R1/R2. El estado `PRE_VALIDATION_SPEC_FROZEN` solo se emite después de ejecutar TRAIN/CALIBRATION bajo ese preregistro. Si falta cualquiera de estos cierres, R1/R2 y especialmente `VALIDATION` siguen sin autorización.

## 6. Políticas y Thresholds Separados

### 6.1 E4 congelado

```text
E4 Tail Risk:
BLOCK si score >= 0.4522452210875323
```

### 6.2 Separación científica de Range y E4

El descubrimiento del edge usa dos experimentos separados:

```text
EXPERIMENTO PRIMARIO
Range puro
sin E4 como selector de población
        |
        v
¿existe edge después de costos?

EXPERIMENTO SECUNDARIO
Range congelado + E4 frozen
        |
        v
¿E4 mejora o empeora Range?
```

E4 no forma parte del gate primario para descubrir si Range posee edge. `tail_risk_score_at_entry` puede conservarse como telemetría, pero no filtrará la población primaria.

Antes del experimento secundario debe demostrarse que sus fechas son genuinamente OOS respecto al entrenamiento y calibración de E4. Si no puede demostrarse, el resultado se etiqueta:

```text
CONTAMINATED_COMPATIBILITY_DIAGNOSTIC
NO_PROMOTION_AUTHORITY
```

En una integración live futura, E4 seguirá siendo obligatorio. Esta separación solo evita atribuir a Range un edge producido por un filtro potencialmente entrenado sobre el mismo periodo.

### 6.3 Política Range de tail risk

`range_max_tail_risk_score = 0.40` permanece como hipótesis secundaria específica de compatibilidad Range+E4. No reemplaza ni altera el threshold E4, no filtra el experimento primario y solo puede calibrarse en `TRAIN/CALIBRATION` antes de `PRE_VALIDATION_FREEZE`.

### 6.4 Volumen

Se prohíbe el ambiguo `volume_threshold`. Se separan:

```yaml
min_safety_volume_ratio: PREREGISTER
min_entry_volume_ratio: PREREGISTER
breakout_volume_ratio: PREREGISTER
```

- `min_safety_volume_ratio`: liquidez/actividad mínima para considerar operable el rango.
- `min_entry_volume_ratio`: confirmación de la vela de entrada, si la investigación demuestra que aporta valor.
- `breakout_volume_ratio`: evidencia de ruptura, independiente de la entrada.

Los valores anteriores `0.5`, `1.5` y `2.0` quedan retirados del plan hasta preregistro y calibración válidos.

## 7. Diseño del Backtest

### 7.1 Alcance

- Los 11 símbolos configurados: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, SUI y LTC contra USDT.
- Múltiples meses y regímenes; 30 días solo sirven como smoke test técnico.
- LONG y SHORT evaluados por separado.
- Resultados por símbolo, mes, régimen y partición.
- Retornos normalizados por unit notional durante la investigación de edge.
- Sin leverage ni `position_fraction` operacional en R1/R2.
- El smoke test de 30 días usa exclusivamente datos pertenecientes a `TRAIN` o datos sintéticos.
- Está prohibido usar directa o indirectamente `CALIBRATION`, `VALIDATION` o `HOLDOUT` para smoke tests o debugging visual.

### 7.2 Unidad estadística y range episodes

La unidad estadística primaria es `range_episode_id`; los trades son observaciones anidadas dentro del episodio.

Reglas obligatorias:

1. Un `range_episode_id` puede contener múltiples trades, pero no se contarán como experimentos independientes.
2. Un episodio completo pertenece a una sola partición.
3. Un mismo episodio no puede caer parcialmente en TRAIN y parcialmente en `CALIBRATION`, `VALIDATION` o `HOLDOUT`.
4. Los episodios que atraviesen fronteras de split se purgan según una política preregistrada.
5. Reentradas, cooldown y máximo de trades por episodio deben fijarse en R0.
6. Bootstrap, intervalos de confianza y estimación de N efectivo agrupan por episodio y por bloques temporales.
7. La correlación transversal entre símbolos contemporáneos debe conservarse en el remuestreo.

### 7.3 Particiones cronológicas

```text
HISTORICAL DATA
      |
      +-- TRAIN
      |     descubrimiento y descarte de ideas
      |
      +-- CALIBRATION
      |     thresholds dentro de grids preregistrados
      |
      +-- PRE_VALIDATION_FREEZE
      |     fórmulas, thresholds, políticas y hashes inmutables
      |
      +-- VALIDATION
      |     evaluación única sin retuning
      |
      +-- HOLDOUT
            sellado hasta decisión final
```

Reglas:

1. TRAIN permite diseño, comparación de fórmulas y selección de familias.
2. `CALIBRATION` solo permite elegir thresholds dentro de grids definidos antes de abrirla.
3. Al terminar `CALIBRATION` se crea `PRE_VALIDATION_SPEC_FROZEN` con hashes de código, datos, configuración y especificación.
4. Después de `PRE_VALIDATION_FREEZE`, ninguna fórmula, threshold, política de reentrada, salida, especialización o gate puede cambiar.
5. `VALIDATION` se abre una sola vez con cero decisiones adaptativas.
6. Después de abrir `VALIDATION`, no se reajustan parámetros para volver a declarar éxito.
7. `HOLDOUT` permanece sellado hasta que `VALIDATION` pase el gate congelado.
8. R3 verifica los hashes congelados antes de abrir `HOLDOUT` una sola vez.
9. Se usará block-bootstrap temporal con tamaño de bloque preregistrado.
10. No se tratarán BTC, ETH y SOL en el mismo instante como observaciones independientes.
11. Las ventanas deberán incluir warmup causal sin filtrar información del futuro.

### 7.4 Modelo de ejecución histórica

- Señal generada únicamente al cierre de la vela.
- Entrada por señal en `NEXT_BAR_OPEN`.
- Salida por breakout confirmado en `NEXT_BAR_OPEN` después del segundo cierre.
- Salida por max hold en `NEXT_BAR_OPEN` después de cumplirse el límite.
- TP y SL son órdenes resting y se simulan al tocarse intrabar conforme a la política de fill preregistrada; no esperan al siguiente open.
- Fees, slippage y funding explícitos.
- Si TP y SL pueden tocarse dentro de la misma vela sin resolución intrabar, resolver `adverse-first`.
- Si el mercado abre atravesando el stop, aplicar un fill conservador de gap preregistrado, nunca el stop teórico mejor.
- La precedencia exacta entre TP, SL, breakout y max hold debe quedar congelada en R0.
- No usar el mejor precio de la vela para simular ejecución.
- Registrar operaciones rechazadas y razones de abstención.

## 8. Métricas y Gate de Éxito

### 8.1 Métricas primarias

- Gross expectancy.
- Net expectancy after costs.
- Profit factor.
- Maximum drawdown.
- MFE y MAE.
- CVaR / tail loss.
- Breakout-loss rate.
- False-range rate.
- Estabilidad temporal por mes.
- Consistencia por símbolo y régimen.

### 8.2 Métricas secundarias

- Win rate.
- Sharpe ratio.
- Número de trades.
- Trades por semana.
- Duración media.

No se exige una cuota de operaciones. La abstención es un resultado válido. Un win rate alto no compensa expectativa neta negativa o pérdidas de ruptura desproporcionadas.

### 8.3 Gate primario

La promoción requiere conjuntamente:

1. Expectativa neta positiva después de costos.
2. Profit factor aceptable y estable fuera de muestra.
3. Drawdown y CVaR compatibles con la política de riesgo.
4. Estabilidad temporal; el resultado no puede depender de un solo mes o símbolo.
5. Breakout-loss rate y false-range rate controlados.
6. Evidencia suficiente en `VALIDATION`, seguida de confirmación en `HOLDOUT` sellado.

Los valores numéricos finales del gate deben preregistrarse antes de abrir `VALIDATION`.

## 9. Sizing y Portfolio

El plan retira `position_fraction: 0.15`. El runtime momentum live utiliza normalmente `0.01` y contiene `max_position_fraction: 0.01` en sus safety caps.

La investigación separa dos preguntas:

1. ¿Range posee edge con retornos normalizados por unit notional?
2. Si posee edge, ¿cómo debe dimensionarse dentro del portfolio actual?

Sizing, leverage, correlación, concurrencia y presupuesto de riesgo solo se estudiarán después de congelar y validar el edge. No se autoriza exposición live durante `RESEARCH_ONLY`.

## 10. Fases de Trabajo

### PHASE R0 — Preregistration

- Es la única fase autorizada actualmente.
- Auditar la rama y commits actuales.
- Documentar fórmulas, lookbacks y timestamps de disponibilidad.
- Definir hard blockers y score descriptivo.
- Definir costos y política de fills ambiguos.
- Definir particiones cronológicas y sellar `HOLDOUT`.
- Fijar timeframe, range episodes, reentradas y lifecycle completo de operaciones.
- Definir familias y grids de thresholds sin mirar fuera de TRAIN.
- Definir jerarquía de hipótesis, multiplicity control y objetivo de selección.
- Definir política OOS de E4.

**Entregable:** preregistro R0 completo, todavía sin autorizar R1/R2.

### PHASE R1 — Pure Research

**Estado:** no autorizada todavía; requiere aprobación explícita del entregable R0.

Crear componentes puros y testeables:

```text
RangeDetectorV1.ts
RangeLevelsV1.ts
RangeSafetyV1.ts
RangeSignalV1.ts
RangeBreakoutV1.ts
```

Agregar tests de causalidad, especialmente:

- Un pivot no aparece antes de `available_at`.
- Agregar velas futuras no cambia decisiones históricas ya emitidas.
- El rango no existe antes de `range_confirmed_at`.
- La vela de rechazo genera entrada únicamente en la vela siguiente.
- El segundo cierre de breakout genera salida únicamente en la vela siguiente.
- Los niveles y stops congelados no se ensanchan tras la entrada.

**Prohibido en R1:** modificar runtime live o ejecución.

### PHASE R2 — Backtest

**Estado:** no autorizada todavía; requiere R1 aprobada y tests causales completos.

- Crear backtester research-only.
- Ejecutar smoke test de 30 días solo sobre TRAIN o datos sintéticos para validar plumbing.
- Ejecutar experimento completo con múltiples meses y los 11 símbolos.
- Usar TRAIN para fórmulas/familias y `CALIBRATION` solo para thresholds dentro de grids preregistrados.
- Reportar métricas por lado, símbolo, mes y régimen.

### PRE_VALIDATION_FREEZE — Freeze obligatorio

Ocurre después de TRAIN/CALIBRATION y antes de abrir `VALIDATION`:

- Congelar fórmulas, thresholds, timeframe y lifecycle.
- Congelar reentry policy y range episode definition.
- Congelar jerarquía de hipótesis, multiplicity control y gate de éxito.
- Congelar política E4 OOS y clasificación de diagnósticos contaminados.
- Publicar hashes de código, configuración, dataset, splits y preregistro.
- Emitir el estado `PRE_VALIDATION_SPEC_FROZEN`.

Sin este estado, `VALIDATION` no puede ejecutarse.

### PHASE R3 — Validation y Holdout

Solo después de `PRE_VALIDATION_SPEC_FROZEN`:

- Verificar que los hashes coinciden antes de cada ejecución.
- Abrir `VALIDATION` una única vez, sin ajustes posteriores.
- Evaluar el gate congelado sin seleccionar post-hoc lados, símbolos o regímenes.
- Solo si `VALIDATION` pasa, abrir `HOLDOUT` una única vez.
- Rechazar la estrategia si no mantiene expectativa, estabilidad y control de cola.

### PHASE R4 — Shadow Integration

Solo después de R3:

- Añadir `aegis.range_strategy` en configuración no-live.
- Añadir routing de estrategia independiente de `AegisSymbolMode`.
- Reutilizar `RegimeEngineV2`.
- Pasar oportunidades por E4 Tail Risk congelado y hard safety.
- Emitir telemetría y decisiones SHADOW sin órdenes ni cambios de ejecución.

### PHASE R5 — Evidencia Prospectiva

- Ejecutar shadow prospectivo durante una ventana preregistrada.
- Comparar señales, fills simulados, costos y rupturas con el backtest.
- No retocar parámetros durante la ventana.

Solo después de evidencia prospectiva suficiente podrá proponerse una integración operacional controlada. Esa propuesta requerirá revisión y aprobación separadas.

## 11. Arquitectura Objetivo Posterior a Research

```text
                     Market candles
                           |
                           v
                    RegimeEngineV2
                           |
              +------------+------------+
              |                         |
       Momentum environment       Range candidate
              |                         |
              v                         v
       existing strategy         RangeDetectorV1
                                        |
                                 RangeLevelsV1
                                        |
                                 RangeSafetyV1
                                        |
                                 RangeSignalV1
                                        |
                            LONG / SHORT / NONE
                                        |
              +-------------------------+
              v
          E4 Tail Risk
       FROZEN, SIN CAMBIOS
              |
          ALLOW/BLOCK
              |
              v
         hard safety
              |
              v
      approval boundary
              |
              v
    existing execution path
```

Esta arquitectura es un objetivo condicionado a R0-R5; no autoriza implementación live.

## 12. Alcance Autorizado Ahora

Solo está autorizado R0 documental. Puede crear o actualizar:

- El preregistro R0 de Aegis Range Strategy V1.
- Manifiestos documentales de datasets y splits, sin abrir `VALIDATION` ni `HOLDOUT`.
- Registro de fórmulas, grids candidatos, costos, políticas y hashes.
- Auditorías read-only del código y datos existentes.

R0 no autoriza todavía archivos de implementación como `RangeDetectorV1.ts`, tests ejecutables ni backtester. Eso pertenece a R1/R2 y requiere aprobación posterior del preregistro completo.

No están autorizados:

- `TradingService.ts`
- Ejecución u órdenes
- Leverage o sizing live
- Brackets
- `AegisSymbolMode`
- `AegisRegimeGuard.ts`
- `regime_config.live.yaml`
- E4 Tail Risk, su threshold o su infraestructura
- Componentes Range ejecutables antes de aprobar R1
- Backtests antes de aprobar R2
- Abrir `VALIDATION` o `HOLDOUT`
- Deploy o procesos PM2

## 13. Entregables de R0

Antes de solicitar autorización para R1, entregar:

1. Timeframe exacto para régimen, niveles, señales y fills.
2. Fórmulas exactas, lookbacks y tratamiento de datos insuficientes.
3. Pivots, clustering, touches y `available_at` exactos.
4. Definición exacta de `range_episode_id`, `range_id` y fronteras de episodios.
5. Rejection, entry, TP, SL, breakout y max hold exactos.
6. Precedencia exacta de fills, `adverse-first` y gap policy.
7. Costos, slippage y funding exactos.
8. Reentry policy, cooldown y máximo de trades por episodio.
9. Fechas exactas de TRAIN, `CALIBRATION`, `VALIDATION` y `HOLDOUT`.
10. Grids de thresholds que `CALIBRATION` podrá seleccionar.
11. Objetivo de selección y gate de promoción.
12. Jerarquía de hipótesis y multiplicity control.
13. Tamaño de bloques para bootstrap temporal.
14. Política E4 OOS y tratamiento de diagnósticos contaminados.
15. Política de purga para episodios que atraviesan splits.
16. Hashes de código, datos, configuración y especificación.
17. Lista de inconsistencias encontradas.
18. Diff que confirme que producción no fue modificada.

El preregistro R0 aprobado será la base de `PRE_VALIDATION_SPEC_FROZEN`, pero el freeze final solo se emite después de TRAIN/CALIBRATION y antes de `VALIDATION`.

## 14. Prompt Autorizado para R0

```text
Revisa la rama y los commits actuales. Vamos a ejecutar únicamente PHASE R0 — PREREGISTRATION de Aegis Range Strategy V1, NO "E4 Range".

No crear todavía RangeDetectorV1, RangeLevelsV1, RangeSafetyV1, RangeSignalV1, RangeBreakoutV1, tests ejecutables ni backtester. No ejecutar TRAIN, CALIBRATION, VALIDATION o HOLDOUT.

E4 significa exclusivamente el Tail Risk Guard congelado existente (e4_tail_risk, threshold 0.4522452210875323). No modificar su modelo, threshold, FeatureBridge, API, precompute, adapter ni decisiones.

No modificar TradingService, ejecución, órdenes, leverage, sizing, brackets, AegisSymbolMode, AegisRegimeGuard, regime_config.live.yaml, PM2 ni producción.

Usa RegimeEngineV2 como fuente conceptual existente para ACCUMULATION_RANGE, CHOP, ADX, choppiness, ATR percentile, Bollinger width percentile, volume ratio, range breakout, failed breakouts y transition risk. No añadir RANGE_BOUND al guard legacy.

Produce un preregistro documental que cierre exactamente:
- timeframe de régimen, niveles, señal y ejecución;
- pivots L/R y available_at;
- clustering, touches, range amplitude y range episode lifecycle;
- rejection candle, entry NEXT_BAR_OPEN, TP, SL, breakout y max hold;
- resting TP/SL, adverse-first, gap policy y precedencia de fills;
- costos, slippage y funding;
- reentry, cooldown y máximo de trades por range episode;
- fechas TRAIN/CALIBRATION/VALIDATION/HOLDOUT y política de purga;
- threshold grids y objetivo de selección;
- jerarquía pooled/LONG/SHORT/majors/alts y multiplicity control;
- bootstrap temporal por range episode;
- experimento primario Range puro sin E4 como selector;
- experimento secundario Range+E4 y demostración OOS de E4;
- hashes de código, datos, configuración y especificación.

TRAIN podrá seleccionar fórmulas y familias. CALIBRATION solo podrá seleccionar thresholds dentro de grids preregistrados. Después se emitirá PRE_VALIDATION_SPEC_FROZEN. VALIDATION y HOLDOUT no están autorizados ahora.

El smoke test futuro de 30 días solo podrá usar TRAIN o datos sintéticos.

Entrega únicamente documentación, manifiestos y auditoría read-only. Al terminar informa cualquier decisión que no pueda congelarse sin evidencia adicional; no la resuelvas mirando VALIDATION.
```
