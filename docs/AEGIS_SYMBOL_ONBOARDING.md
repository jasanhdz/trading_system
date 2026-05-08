# Aegis Symbol Onboarding

Este flujo prepara un símbolo nuevo en SHADOW. No habilita LIVE.

## 1. Seleccionar Símbolo

Validar antes de incorporarlo:

- Liquidez suficiente en Binance Futures.
- Spread estable durante las horas de operación.
- Volumen real consistente.
- Mínimos de Binance compatibles con el sizing actual.
- Correlación y exposición respecto al símbolo LIVE existente.

## 2. Descargar Velas

Actualizar velas para el símbolo nuevo:

```bash
python scripts/update_candles.py --symbol BTCUSDT
```

El script guarda velas con `symbol` en la DB y filtra duplicados.

## 3. Generar Datasets Turbo

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/build_turbo_symbol_dataset.py --symbol BTCUSDT --windows 7d,14d,30d
```

Esto genera snapshots bajo:

```text
aegis_alpha/data/processed/turbo/BTCUSDT/
```

## 4. Generar Modelos Turbo

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/train_turbo_symbol_models.py --symbol BTCUSDT --windows 7d,14d,30d
```

Esto genera:

- `long_edge_7d`
- `short_edge_7d`
- `long_edge_14d`
- `short_edge_14d`
- `long_edge_30d`
- `short_edge_30d`

## 5. Refrescar Snapshots

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbol BTCUSDT
```

Para refrescar varios símbolos:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols ETHUSDT,BTCUSDT
```

## 6. Validar Inferencia

```bash
curl -s -X POST http://127.0.0.1:8001/ml-v2/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTCUSDT"}'
```

Si faltan artefactos, la respuesta debe ser `HOLD` con `reason=missing_turbo_artifacts_for_symbol`.

## 7. Agregar a YAML

```yaml
symbols:
  BTCUSDT:
    enabled: true
    mode: SHADOW
```

No usar `LIVE` para un segundo símbolo hasta que exista estado portfolio-safe.

## 8. Reiniciar Servicios Cuando Aplique

Si cambió código Python:

```bash
pm2 restart 02-Aegis-API
```

Si cambió YAML o código TS:

```bash
pm2 restart 01-Trading-Bot
```

No ejecutar `pm2 save` salvo que se quiera persistir el cambio de procesos.

## 9. Validar Telegram

Usar:

```text
/signal BTCUSDT
/signals
/status
/config
```

`/signals` debe incluir símbolos LIVE y SHADOW. OFF puede aparecer en `/status` y `/config`, pero no se escanea.

## 10. Dejar en SHADOW

Mantener 24-72 horas en SHADOW antes de considerar LIVE.

Criterios mínimos para pasar a LIVE:

- Snapshots frescos.
- Señales suficientes.
- Sin estado stale frecuente.
- Distribución de score razonable.
- MFE/MAE favorable.
- BotState portfolio-safe implementado.
- Risk manager portfolio activo.

## PM2 Refresher Multi-Symbol

Comando recomendado:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols ETHUSDT,BTCUSDT
```

Si existe `04-Aegis-Turbo-Refresh`, cambiar su comando de:

```text
--symbol ETHUSDT
```

a:

```text
--symbols ETHUSDT,BTCUSDT
```

## Mass Shadow Onboarding

Símbolos incorporados para observación SHADOW:

```text
BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT
```

Reglas de seguridad:

- `ETHUSDT` sigue siendo el único símbolo `LIVE`.
- Todos los símbolos nuevos quedan `SHADOW`.
- Los símbolos `SHADOW` pueden escanearse, loggearse y aparecer en Telegram, pero no ejecutan órdenes.
- Multi-symbol `LIVE` requiere refactor de portfolio state, multi-position recovery, portfolio risk manager y bracket guard multi-position.

Validar símbolos contra Binance USD-M Futures antes de procesar:

```bash
/home/jasan/.venv_rocm62/bin/python - <<'PY'
# Validar con ccxt/binance fapi exchangeInfo y guardar reporte en
# aegis_alpha/logs/symbol_onboarding/symbol_validation_<timestamp>.json
PY
```

Actualizar velas por símbolo:

```bash
for symbol in BTCUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT SUIUSDT LTCUSDT; do
  /home/jasan/.venv_rocm62/bin/python scripts/update_candles.py --symbol "$symbol"
done
```

Para reconstrucciones históricas completas, hacer backup de `data/binance_candles.db` antes de limpiar rows de un símbolo. No borrar datos de `ETHUSDT`.

Entrenar Turbo por símbolo:

```bash
for symbol in BTCUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT SUIUSDT LTCUSDT; do
  /home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/train_turbo_symbol_models.py --symbol "$symbol" --windows 7d,14d,30d
done
```

Refrescar snapshots:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/refresh_turbo_snapshots.py \
  --mode features-only \
  --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT
```

Validar freshness:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/turbo_snapshot_status.py \
  --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT
```

Validar inferencia sin ejecución:

```bash
curl -s http://127.0.0.1:8001/ml-v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT"}'
```

Validación Telegram:

```text
/signals
/signal BTCUSDT
/signal SOLUSDT
/config
/status
```

Refresher PM2 de un solo proceso, solo para cargas pequenas o validacion manual:

```bash
pm2 delete 04-Aegis-Turbo-Refresh
pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 04-Aegis-Turbo-Refresh \
  --cron "*/5 * * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT
pm2 save
```

Si el refresh completo tarda mas de 240 segundos, usar lotes. En la corrida de onboarding masivo, un refresh unico de 11 simbolos tardo ~746s y no es operativo para el freshness window de 900s.

Configuracion por lotes aplicada para mantener ETH/BTC/SOL en cadencia de 5 minutos y los lotes de 4 simbolos en 7 minutos. Duraciones medidas: A ~198s, B ~291-301s, C ~259s.

```bash
pm2 delete 04-Aegis-Turbo-Refresh

pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 04-Aegis-Turbo-Refresh-A \
  --cron "0-59/5 * * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols ETHUSDT,BTCUSDT,SOLUSDT

pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 05-Aegis-Turbo-Refresh-B \
  --cron "1-59/7 * * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT

pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 06-Aegis-Turbo-Refresh-C \
  --cron "4-59/7 * * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT

pm2 save
```
