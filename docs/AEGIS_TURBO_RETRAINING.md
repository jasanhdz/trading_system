# Aegis Turbo Retraining

Este documento describe el retrainer programado de Aegis Turbo para mantener
modelos por simbolo actualizados sin bloquear inference ni tocar ejecucion live.

## Snapshot refresh vs retrain

Snapshot refresh:

- Actualiza candles recientes.
- Reconstruye features/snapshots `.npz`.
- No entrena modelos nuevos.
- Corre frecuentemente, cada 5 a 10 minutos segun lote PM2.
- Mantiene `freshness.is_fresh=true` para `/ml-v2/predict`.

Retrain:

- Actualiza candles.
- Reconstruye datasets recientes.
- Entrena modelos Turbo nuevos por simbolo.
- Valida artefactos candidatos.
- Promueve modelos nuevos solo si pasan sanity checks.
- Debe correr mucho menos frecuente que snapshot refresh.

No se reentrena cada minuto porque:

- El entrenamiento consume CPU/RAM y compite con inference.
- Los modelos no necesitan cambiar por cada vela de 5m.
- Promover modelos demasiado frecuente aumenta ruido y riesgo operativo.
- El refresh de snapshots ya mantiene las features en tiempo real.

## Script

Archivo:

```text
aegis_alpha/tools/run_turbo_scheduled_retrain.py
```

Ejemplo manual:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/run_turbo_scheduled_retrain.py \
  --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT \
  --mode safe \
  --promote-if-valid
```

Prueba de un simbolo:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/run_turbo_scheduled_retrain.py \
  --symbols BTCUSDT \
  --mode safe \
  --promote-if-valid
```

Opciones:

```text
--symbols SYMBOL1,SYMBOL2
--mode safe
--promote-if-valid
--skip-existing-fresh true|false
--max-symbols-per-run N
```

## Flujo seguro

Por cada simbolo:

1. Ejecuta `scripts/update_candles.py --symbol SYMBOL`.
2. Entrena modelos `long` y `short` para ventanas `7d`, `14d`, `30d`.
3. Escribe artefactos en:

```text
aegis_alpha/models/turbo/SYMBOL/candidates/YYYYMMDDTHHMMSSZ/
```

4. Valida:

- archivos existen
- `joblib.load` funciona
- estimador existe
- sample count minimo
- features finitas, sin NaN/inf
- predict local retorna valores finitos
- snapshot refresh por simbolo funciona
- snapshot status queda fresh

5. Si pasa y se uso `--promote-if-valid`, promueve a:

```text
aegis_alpha/models/turbo/SYMBOL/active/
```

6. Si ya habia active, lo mueve a:

```text
aegis_alpha/models/turbo/SYMBOL/backups/YYYYMMDDTHHMMSSZ/
```

7. Escribe manifest atomico:

```text
aegis_alpha/models/turbo/SYMBOL/active_manifest.json
```

Si falla, no toca `active/` y continua con el siguiente simbolo.

## Inference

La prioridad de carga de modelos Turbo es:

1. `aegis_alpha/models/turbo/SYMBOL/active_manifest.json`
2. `aegis_alpha/models/turbo/SYMBOL/active/`
3. `aegis_alpha/models/turbo/SYMBOL/`
4. rutas legacy ETH si `symbol=ETHUSDT`

El cache de modelos verifica `mtime_ns`, por lo que un modelo promovido puede
recargarse cuando cambie el archivo activo.

## Reports

Cada corrida escribe:

```text
aegis_alpha/logs/turbo_retrain/turbo_retrain_YYYYMMDDTHHMMSSZ.json
aegis_alpha/logs/turbo_retrain/turbo_retrain_YYYYMMDDTHHMMSSZ.md
```

Campos relevantes:

```text
symbols_requested
symbols_succeeded
symbols_failed
promoted_symbols
per_symbol.SYMBOL.candidate_dir
per_symbol.SYMBOL.active_dir
per_symbol.SYMBOL.backup_dir
per_symbol.SYMBOL.validation
per_symbol.SYMBOL.snapshot_validation
```

## Lock

El retrainer usa:

```text
/tmp/aegis_turbo_retrain.lock
```

Si otro proceso sigue vivo, sale con:

```text
retrain already running
```

Si el lock es viejo y el PID ya no existe, lo limpia con warning.

## PM2 recomendado

Cada 12 horas:

```bash
pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 07-Aegis-Turbo-Retrain \
  --cron "20 */12 * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/run_turbo_scheduled_retrain.py \
     --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT \
     --mode safe \
     --promote-if-valid
```

Modo mas agresivo, cada 6 horas:

```bash
pm2 start /home/jasan/.venv_rocm62/bin/python \
  --name 07-Aegis-Turbo-Retrain \
  --cron "20 */6 * * *" \
  --no-autorestart \
  --cwd /home/jasan/Develop/trading_system \
  -- aegis_alpha/tools/run_turbo_scheduled_retrain.py \
     --symbols ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT \
     --mode safe \
     --promote-if-valid
```

No usar un cron demasiado frecuente: snapshot refresh ya cubre freshness.

## Rollback

Para rollback manual de un simbolo:

1. Identificar backup:

```bash
ls -ltr aegis_alpha/models/turbo/BTCUSDT/backups
```

2. Detener retrainer si esta corriendo.

3. Mover active actual a una carpeta temporal y restaurar backup:

```bash
mv aegis_alpha/models/turbo/BTCUSDT/active aegis_alpha/models/turbo/BTCUSDT/active_bad_$(date -u +%Y%m%dT%H%M%SZ)
cp -a aegis_alpha/models/turbo/BTCUSDT/backups/YYYYMMDDTHHMMSSZ aegis_alpha/models/turbo/BTCUSDT/active
cp aegis_alpha/models/turbo/BTCUSDT/active/active_manifest.json aegis_alpha/models/turbo/BTCUSDT/active_manifest.json
```

4. Validar:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/turbo_snapshot_status.py --symbols BTCUSDT
curl -s http://127.0.0.1:8001/ml-v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT"}' | jq '.aegis.turbo'
```

## Checklist operativo

Antes de activar PM2:

```bash
/home/jasan/.venv_rocm62/bin/python -m py_compile \
  aegis_alpha/tools/run_turbo_scheduled_retrain.py \
  aegis_alpha/turbo/turbo_signal.py \
  aegis_alpha/tools/refresh_turbo_snapshots.py \
  aegis_alpha/tools/turbo_snapshot_status.py \
  scripts/update_candles.py
```

Luego probar un simbolo:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/run_turbo_scheduled_retrain.py \
  --symbols BTCUSDT \
  --mode safe \
  --promote-if-valid
```

Validar:

```bash
ls aegis_alpha/models/turbo/BTCUSDT/active
cat aegis_alpha/models/turbo/BTCUSDT/active_manifest.json | jq .
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/turbo_snapshot_status.py --symbols BTCUSDT
```

Confirmaciones de seguridad:

- No toca bot TS.
- No toca YAML live.
- No toca `.env`.
- No ejecuta ordenes reales.
- No modifica PM2 automaticamente.
- No hace push remoto.
