# Aegis Turbo multi-symbol LIVE summary - 2026-05-08

Este documento resume lo realizado desde el onboarding masivo de nuevos simbolos
Turbo hasta el estado actual con todos los simbolos configurados en LIVE.

## Estado actual

Simbolos Aegis Turbo configurados:

| Symbol | Estado actual | Leverage configurado | Nota |
| --- | --- | ---: | --- |
| ETHUSDT | LIVE | 20x | Operacion abierta/atascada desde 2026-05-07 segun observacion operativa. |
| BTCUSDT | LIVE | 20x | Abriendo y gestionando con estado propio. |
| SOLUSDT | LIVE | 20x | Antes OFF/SHADOW, ahora LIVE. |
| BNBUSDT | LIVE | 15x | LIVE multi-symbol. |
| XRPUSDT | LIVE | 20x | Override subido a 20x. |
| DOGEUSDT | LIVE | 10x | Tier mas conservador. |
| ADAUSDT | LIVE | 20x | Override subido a 20x. |
| AVAXUSDT | LIVE | 20x | Override subido a 20x. |
| LINKUSDT | LIVE | 20x | Override subido a 20x. |
| SUIUSDT | LIVE | 8x | Tier mas conservador. |
| LTCUSDT | LIVE | 15x | LIVE multi-symbol. |

Confirmaciones importantes:

- Todos los simbolos nuevos pasaron de observabilidad SHADOW a ejecucion LIVE por decision operativa.
- No se activo push remoto.
- El bot TS ya compila despues del refactor multi-symbol.
- La suite TS completa pasa: `112 tests passed`.
- BTCUSDT abrio una entrada real despues del refactor:
  `BTCUSDT LONG @ 79266.60 qty=0.0090 lev=20x score=0.612`.
- Los simbolos con score alto como `LINKUSDT`, `SUIUSDT`, `AVAXUSDT` y `ADAUSDT`
  estaban siendo bloqueados por un bug de daily loss, ya corregido en codigo.

## Onboarding de simbolos

Se agregaron estos 10 simbolos al universo Aegis Turbo:

```text
BTCUSDT
SOLUSDT
BNBUSDT
XRPUSDT
DOGEUSDT
ADAUSDT
AVAXUSDT
LINKUSDT
SUIUSDT
LTCUSDT
```

Trabajo realizado:

- Validacion contra Binance Futures para confirmar que los simbolos existen y son operables.
- Limpieza previa de datos no-ETH/no deseados cuando fue necesario para evitar historicos parciales o corruptos.
- Descarga/backfill de candles por simbolo usando `scripts/update_candles.py --symbol SYMBOL`.
- Validacion de continuidad de candles, duplicados y gaps.
- Entrenamiento Turbo por simbolo para ventanas `7d`, `14d` y `30d`.
- Generacion de modelos `long_edge` y `short_edge` por ventana.
- Generacion de snapshots frescos por simbolo.
- Validacion de `/ml-v2/predict` por symbol.
- Confirmacion de fallback seguro: si faltan artefactos, Aegis responde HOLD y no ejecuta.

## PM2 refresh

Se decidio dividir el refresh de snapshots en varios procesos para evitar que un
refresh multi-symbol completo se acerque demasiado a la ventana de freshness de
900s.

Procesos relevantes:

```text
04-Aegis-Turbo-Refresh-A
05-Aegis-Turbo-Refresh-B
06-Aegis-Turbo-Refresh-C
```

Observacion actual:

- `04-Aegis-Turbo-Refresh-A` esta online.
- `06-Aegis-Turbo-Refresh-C` esta online.
- `05-Aegis-Turbo-Refresh-B` aparecio `stopped` en PM2 durante la revision.

Pendiente operativo:

- Levantar/revisar `05-Aegis-Turbo-Refresh-B` para que el lote asignado no quede
  sin snapshots frescos.
- Confirmar con:

```bash
pm2 list
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/turbo_snapshot_status.py \
  --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT
```

## Cambios de configuracion

Archivo principal:

```text
binance-futures-bot-ts/regime_config.live.yaml
```

Cambios realizados:

- Se agregaron los 10 simbolos nuevos.
- `SOLUSDT` paso de `OFF` a `SHADOW` durante onboarding y luego a `LIVE`.
- Todos los simbolos quedaron en `LIVE` por decision operativa.
- Se agregaron overrides de leverage por simbolo.
- Se mantuvo `ETHUSDT` en 20x.
- Se subieron a 20x: `XRPUSDT`, `SOLUSDT`, `AVAXUSDT`, `LINKUSDT`, `ADAUSDT`.

Leverage actual esperado:

```yaml
ETHUSDT: 20
BTCUSDT: 20
SOLUSDT: 20
BNBUSDT: 15
XRPUSDT: 20
DOGEUSDT: 10
ADAUSDT: 20
AVAXUSDT: 20
LINKUSDT: 20
SUIUSDT: 8
LTCUSDT: 15
```

## Bugs encontrados y fixes aplicados

### 1. Guard de un solo LIVE

Problema:

- `ConfigLoader` bloqueaba mas de un simbolo LIVE.
- Esto era correcto para la fase SHADOW, pero impedia pasar todos los simbolos a LIVE.

Fix:

- Se removio el bloqueo duro de mas de un LIVE.
- Se actualizaron tests para permitir multiples LIVE.

Archivos:

```text
binance-futures-bot-ts/src/infra/config/ConfigLoader.ts
binance-futures-bot-ts/src/infra/config/ConfigLoader.aegis-symbols.test.ts
```

### 2. BotState global bloqueaba todos los simbolos salvo ETH

Problema:

- Aunque todos los simbolos estaban en LIVE, el bot seguia usando un solo
  `BotState` global.
- Como ETH estaba en `LONG_RIDE`, los demas simbolos eran saltados con:

```text
aegis_skip_manage_position_global_state_symbol_mismatch
```

Impacto:

- BTC, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, SUI y LTC podian escanear,
  pero no se gestionaban correctamente como LIVE independientes.

Fix:

- Se agrego estado escopado por simbolo con `stateForSymbol(symbol)`.
- `TradingService` ahora usa estado por simbolo en:
  - gate de entrada
  - apertura de posicion
  - confirmacion de posicion
  - brackets
  - trailing
  - cierre
  - emergency close
- `FsStateStore` ahora soporta `forSymbol(symbol)`.
- Telegram `/positions` y `/brackets` leen estado por simbolo.
- Se agrego migracion del estado global legacy al primer simbolo LIVE.
- Se agrego attach de posiciones reales abiertas en Binance hacia estado por simbolo.

Archivos:

```text
binance-futures-bot-ts/src/app/ports/StateStore.ts
binance-futures-bot-ts/src/infra/logging/FsStateStore.ts
binance-futures-bot-ts/src/app/services/TradingService.ts
binance-futures-bot-ts/src/app/telegram/TelegramCommandHandlers.ts
binance-futures-bot-ts/src/app/services/TradingService.aegis-live.test.ts
```

Validacion:

```text
allows a LIVE BTC entry while ETH has its own active symbol state
```

### 3. Falso daily_loss_stop_reached despues de abrir BTC

Problema:

- Tras abrir BTC, simbolos con buen score eran bloqueados:

```text
LINKUSDT score ~0.91/0.92 -> daily_loss_stop_reached
SUIUSDT score ~0.94/0.96 -> daily_loss_stop_reached
AVAXUSDT score ~0.89 -> daily_loss_stop_reached
ADAUSDT score ~0.88 -> daily_loss_stop_reached
```

Causa:

- `getUSDTBalance()` devuelve `availableBalance`, no equity total.
- Al abrir BTC, el margen aislado usado redujo el disponible.
- El bot interpreto esa reduccion de disponible como perdida diaria.

Fix:

- El daily loss ahora se calcula usando:
  - `equityTotal`, si esta disponible
  - si no, `walletBalance`
  - si no, fallback a `availableBalance`
- `availableBalance` sigue usandose para sizing.
- Se agrego `lastDailyPnlPct` al runtime snapshot.

Validacion:

```text
does not treat isolated margin usage as daily loss when equity is unchanged
```

### 4. Refresh multi-symbol demasiado pesado

Problema:

- Refrescar todos los simbolos en un solo proceso puede acercarse demasiado a
  la ventana de freshness de 900s.

Fix operativo:

- Se decidio dividir refresh en 2 o 3 procesos PM2.
- Cron recomendado: no depender de un solo refresh largo.

Pendiente:

- Confirmar que todos los procesos estan online, especialmente
  `05-Aegis-Turbo-Refresh-B`.

### 5. Latencia de `/ml-v2/predict`

Observacion:

- Hubo warnings de latencia en predicciones Aegis, tipicamente entre
  ~700ms y ~1500ms por simbolo en algunos momentos.

Estado:

- No se observo crash por esto.
- API siguio respondiendo.
- Es un punto a monitorear ahora que todos los simbolos estan LIVE.

## Estado operativo de ETH

ETHUSDT esta en LIVE y se reporta como atascado en una operacion desde
2026-05-07.

Riesgo:

- ETH puede seguir ocupando margen y afectar disponibilidad para otros simbolos.
- Aunque el estado por simbolo ya evita bloquear otros mercados, ETH requiere
  revision manual de:
  - posicion real en Binance
  - brackets SL/TP
  - estado persistido de `ETHUSDT`
  - razon por la cual no cerro por TP/SL/trailing/time limit

Comandos utiles:

```bash
pm2 logs 01-Trading-Bot --lines 100
```

Telegram:

```text
/positions
/brackets
/signal ETHUSDT
/status
/risk
```

## Estado de tests

Despues de los fixes:

```text
npm run build: OK
npm test: 16 files passed, 112 tests passed
```

Tests relevantes agregados/actualizados:

```text
permits validation when more than one symbol is LIVE
allows a LIVE BTC entry while ETH has its own active symbol state
does not treat isolated margin usage as daily loss when equity is unchanged
```

## Archivos modificados en TS

Cambios actuales relevantes en `binance-futures-bot-ts`:

```text
regime_config.live.yaml
src/app/ports/StateStore.ts
src/app/services/TradingService.ts
src/app/services/TradingService.aegis-live.test.ts
src/app/telegram/TelegramCommandHandlers.ts
src/infra/config/ConfigLoader.ts
src/infra/config/ConfigLoader.aegis-symbols.test.ts
src/infra/logging/FsStateStore.ts
```

## Pendientes recomendados

1. Reiniciar `01-Trading-Bot` para aplicar el fix de daily loss si todavia no se aplico.
2. Revisar `05-Aegis-Turbo-Refresh-B`, que aparecio detenido.
3. Confirmar que ya no aparece:

```text
aegis_skip_manage_position_global_state_symbol_mismatch
```

4. Confirmar que ya no hay falsos bloqueos:

```text
daily_loss_stop_reached
```

cuando el equity no haya caido realmente.

5. Revisar ETHUSDT manualmente porque sigue atascado desde 2026-05-07.
6. Confirmar que cada simbolo LIVE tenga brackets correctos despues de abrir.
7. Monitorear margen disponible, equity y exposicion total porque todos los
   simbolos estan en LIVE.

## Lectura rapida

La fase empezo como multi-symbol observability y termino en multi-symbol LIVE.
El problema principal no fue Aegis Python ni los modelos: los modelos y snapshots
funcionaron. Los bloqueos principales estuvieron en el bot TS:

- primero, un guard que impedia mas de un LIVE;
- despues, un `BotState` global que hacia que ETH bloqueara el resto;
- finalmente, un daily loss calculado con `availableBalance`, que confundia
  margen usado con perdida real.

Con los fixes actuales, el bot queda preparado para que cada simbolo LIVE tenga
su propio estado, sus propias probabilidades/scores, sus propios brackets y su
propia gestion de posicion.
