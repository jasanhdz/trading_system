# GEN2 — Operations Runbook, Deployment Checklist & Recovery Procedures

**Status:** ops-runbook-v1.0 (2026-07-14) · Candidate `gen2-20260711T202935Z` · Owner: Jasan
**Principio rector:** todo es fail-closed. Ante la duda, NO operar. El kill switch nunca se rearma solo; desarmarlo es SIEMPRE una acción humana deliberada.

Rutas usadas abajo:
- `REPO` = `/home/jasan/Develop/trading_system`
- `GEN2` = `/home/jasan/Develop/aegis_gen2`
- `CDIR` = `GEN2/live_canary/gen2-20260711T202935Z`
- `PY` = `/home/jasan/.venv_rocm62/bin/python` — **SIEMPRE este intérprete**: es el registrado en `GEN2_SYSTEM_FREEZE.json` (pandas 3.0.2/sklearn 1.8.0). En este host existe un segundo venv (`REPO/.venv_rocm62`, pandas 2.3.3) cuyo pandas NO puede cargar los pickles congelados; el decision loop ahora lo rechaza en frío con `ENVIRONMENT_MISMATCH_VS_FREEZE` antes de despicklear.

---

## 1. Deployment Checklist (pre-canary, ejecutar en orden, humano)

Cada ítem debe dar el resultado esperado ANTES de pasar al siguiente. Si alguno falla: detenerse.

| # | Acción | Comando | Resultado esperado |
|---|--------|---------|--------------------|
| 1 | Suite Python | `for f in REPO/aegis_alpha/tests/test_gen2_*.py; do $PY $f; done` | todos `: OK` |
| 2 | Suite TS | `cd REPO/binance-futures-bot-ts && npx vitest run src/gen2` | todos verdes |
| 3 | Phase O pausada | `$PY REPO/aegis_alpha/tools/gen2_canary_core.py --mode conflict-audit` | `phase_o_new_entries_paused: true` |
| 4 | Sin exposición Phase O | mismo comando | `open_position: false`, `conflict: false` |
| 5 | Wallet experimental fondeada | Binance UI / private read | ≥ $100 USDT disponibles, isolated |
| 6 | Escribir contrato | `$PY REPO/aegis_alpha/tools/gen2_operational_contract.py --mode-op write-contract --mode experimental --initial-equity <EQUITY_REAL>` | JSON del contrato, sin `CONTRACT_INCOHERENT` |
| 7 | Levantar bridge (dry-run primero) | `cd REPO/binance-futures-bot-ts && GEN2_BRIDGE_SECRET=<SECRET> npx ts-node src/gen2/gen2_bridge_main.ts` | `{"gen2_bridge":"LISTENING", "execution_enabled":false, "phase_o_allow_orders":false}` |
| 8 | Probar dry-run end-to-end | `GEN2_BRIDGE_SECRET=<SECRET> $PY REPO/aegis_alpha/tools/gen2_decision_loop.py --live` (SIN arm token) | 0 órdenes; razones `CANARY_UNARMED_NO_TOKEN` |
| 9 | Monitor limpio | `GEN2_BRIDGE_SECRET=<SECRET> $PY REPO/aegis_alpha/tools/gen2_ops_monitor.py` | `"healthy": true`, exit 0 |
| 10 | Habilitar ejecución real | reiniciar bridge con `GEN2_EXECUTION_ENABLED=true` + credenciales en env | `execution_enabled: true` |
| 11 | ARM (humano, último paso) | `$PY REPO/aegis_alpha/tools/gen2_operational_contract.py --mode-op create-arm-token --candidate-id gen2-20260711T202935Z --expiry-hours 24 --allowed-symbols ADAUSDT` | token con `max_orders: 1` |
| 12 | Watcher | `GEN2_BRIDGE_SECRET=<SECRET> $PY REPO/aegis_alpha/tools/gen2_decision_loop.py --watch --live --interval-seconds 300` | heartbeat.json actualizándose |
| 13 | Tras la 1ª orden | `$PY REPO/aegis_alpha/tools/gen2_canary_exec.py --mode second-opinion` (credenciales read-only en env) | `RECONCILED`, 0 incidentes |

Reglas del primer armado: 1 orden máxima, 1 símbolo, expiry ≤24h. No re-armar hasta revisar fills, brackets, second opinion y outcome de la primera orden.

## 2. Runbook 3AM (síntoma → acción)

**Regla 0: si hay CUALQUIER duda sobre exposición → PASO DE EMERGENCIA:**
```
# 1) matar bridge (bloquea órdenes nuevas):
GEN2_BRIDGE_SECRET=<SECRET> $PY -c "import aegis_alpha.tools.gen2_bridge_client as b; print(b.post_kill('gen2-20260711T202935Z','operator-emergency'))"
# 2) kill local (bloquea el cerebro):
$PY REPO/aegis_alpha/tools/gen2_canary_core.py --mode kill --reason operator-emergency
# 3) verificar exposición real en Binance UI. Cerrar posición MANUALMENTE en la UI si es necesario.
```
El sistema NUNCA cierra posiciones automáticamente; cerrar es decisión humana.

| Síntoma | Diagnóstico | Acción |
|---|---|---|
| Monitor exit 1 | leer `alerts` en `CDIR/monitor_report.json` | fila correspondiente abajo |
| `BRIDGE_UNREACHABLE` | proceso TS caído / puerto | ver §3.2 (TS crash). Sin bridge NO salen órdenes (fail-closed); paper sigue |
| `KILL_SWITCH_ENGAGED` (cualquiera) | leer razón: `cat CDIR/KILL_SWITCH` y `logs/gen2_bridge/BRIDGE_KILL` | investigar la causa raíz ANTES de considerar desarme (§4) |
| Orden llena pero `BRACKET_FAILED` | kill ya engancha solo | verificar stop en Binance UI; si no existe, poner stop MANUAL en UI o cerrar manual. Nunca desarmar sin bracket confirmado |
| `RECONCILIATION_FAIL_CLOSED` | leer último registro de `CDIR/reconciliations.jsonl` | con exposición: kill ya enganchado; conciliar a mano contra Binance UI. Sin exposición: investigar duplicado/huérfana antes de seguir |
| Heartbeat parado | watcher muerto | §3.1 (Python crash) |
| `FORWARD_EVIDENCE_STALLED` | snapshot/fetch fallando | revisar incidentes `WATCH_CYCLE_FAILED` en `CDIR/incidents/incidents.jsonl`; suele ser red/Binance (§3.4) |
| Decisiones = 0 muchas horas | ¿mercado sin velas nuevas? no | revisar `loop_last_cycle.json` y `no_decision` reasons |
| `NEW_INCIDENTS_SINCE_LAST_CHECK` | `tail CDIR/incidents/incidents.jsonl` | clasificar; cualquier incidente de ejecución con posición → Regla 0 |

## 3. Recovery Procedures

### 3.1 Python crash / watcher muerto
Estado en disco, nada en memoria. Recuperación: relanzar el watcher (checklist #12). El checkpoint de eventos, `loop_state.json` (dedup por vela), `events_seen.json` y los streams son idempotentes: no habrá dobles decisiones ni doble ingesta. Verificar después con el monitor.

### 3.2 TS bridge crash
Sin bridge no hay órdenes (el loop registra `BRIDGE_STATUS_UNAVAILABLE` y sigue en paper). Recuperación: relanzar `gen2_bridge_main.ts` con el MISMO `GEN2_STATE_DIR` — `bridge_state.json` restaura idempotencia (una orden repetida devuelve `DUPLICATE`) y la secuencia de eventos continúa. Si había orden en vuelo: correr second-opinion (checklist #13) antes de re-armar nada.

### 3.3 VPS reboot
Orden de arranque: (1) bridge TS → (2) monitor (`gen2_ops_monitor.py`, debe salir healthy o explicar por qué no) → (3) watcher. El kill switch es un archivo: sobrevive el reboot; si estaba enganchado, sigue enganchado (correcto). El arm token expira solo (no re-armar automáticamente).

### 3.4 Binance outage / internet outage
Todo degrada a fail-closed: fetch de velas falla → `WATCH_CYCLE_FAILED` (incidente, el watcher sigue); bridge no puede ejecutar → `EXCHANGE_ERROR` REJECTED; monitor alerta `BINANCE_PUBLIC_UNREACHABLE`. **Riesgo real: una posición abierta con stop en el exchange está protegida por el STOP_MARKET (vive en Binance, no aquí)** — por eso el bracket es obligatorio y su fallo engancha kill. Al volver la conectividad: monitor → second-opinion → revisar outcomes.

### 3.5 Ejecución parcial (fill sin bracket / crash entre fill y bracket)
El bridge ya engancha kill (`FILLED_BUT_BRACKET_FAILED_KILL_ENGAGED`). Si el crash fue ANTES del ack: la orden puede existir en Binance sin registro local → la second opinion la detecta como `ORPHAN_ORDER_ON_EXCHANGE` / `POSITION_WITHOUT_LOCAL_ORDER` y engancha kill con exposición. Acción humana: confirmar posición/stop en UI, decidir cierre manual, documentar en `incidents.jsonl` (append manual permitido con `type: OPERATOR_NOTE`).

### 3.6 Fallo de reconciliación
`RECONCILIATION_FAIL_CLOSED` + exposición = kill automático. NO desarmar hasta que un humano explique CADA incidente (duplicated fill, huérfana, leverage/margen/balance). La reconciliación es read-only: correrla de nuevo es siempre seguro.

### 3.7 Checkpoint corrupto
- `events_checkpoint.json` corrupto → bórralo: la próxima descarga re-trae desde 0 y `events_seen.json` dedupa todo (at-least-once seguro).
- `events_seen.json` corrupto → restaurar de git NO (no está en git): borrarlo re-ingiere eventos; los streams tendrán duplicados de eventos pero `record_trade_result` se re-aplicaría → **NO borrar con PnL ya contabilizado**: en su lugar, engage kill, reconstruir a mano desde `events_outbox.jsonl` del bridge (fuente de verdad, append-only) comparando `ts_sequence`.
- `loop_state.json` corrupto → borrarlo solo re-emite decisiones de la última vela (dedup del resolver por key evita doble outcome).
- `bridge_state.json` corrupto → PARAR: sin él se pierde idempotencia. Restaurar del propio archivo `.tmp` si existe; si no, NO enviar órdenes hasta second-opinion limpia; re-crear con `processed` vacío solo tras confirmar 0 órdenes abiertas en Binance.
- `risk_state.json` corrupto → engage kill, reconstruir pérdidas desde `fills.jsonl` (`POSITION_CLOSED.realized_pnl`), luego decidir desarme.

## 4. Desarme del kill switch (única vía)
1. Causa raíz identificada y escrita como `OPERATOR_NOTE` en `CDIR/incidents/incidents.jsonl`.
2. Second-opinion `RECONCILED` y 0 posiciones abiertas.
3. Borrar a mano `CDIR/KILL_SWITCH` y/o `logs/gen2_bridge/BRIDGE_KILL` (ambos si ambos).
4. Token nuevo (el viejo habrá expirado o consumido) — checklist #11.
No existe ningún comando de desarme en el código: es intencional.

## 5. Rollback
- **Rollback operativo** (parar canary): Regla 0 (§2) + dejar expirar el token. El sistema queda en paper-only indefinidamente; no hay nada más que revertir porque los modos viven en un archivo de contrato.
- **Rollback de modo**: escribir contrato SAFE (`--mode safe --force` sin posición abierta) → el token EXPERIMENTAL queda inválido automáticamente (mode mismatch).
- **Rollback de código**: `git revert` del commit ofensivo (no rewrite). El candidate/freeze no se toca nunca en rollback.

## 6. Monitoreo continuo
- `gen2_ops_monitor.py --expect-watch-running` cada 15–30 min (cron del operador o manual). Exit 1 = atención.
- Alertas acumuladas: `CDIR/alerts.jsonl` (append-only). Reporte vivo: `CDIR/monitor_report.json`.
- Streams append-only auditables: `forward_decisions.jsonl`, `live_orders.jsonl`, `fills.jsonl`, `brackets.jsonl`, `reconciliations.jsonl`, `incidents/incidents.jsonl`, `forward_outcomes.jsonl`, `alerts.jsonl`, `contract_history.jsonl`, bridge `events_outbox.jsonl`.

## 7. Límites conocidos (aceptados para el canary, documentados)
- El monitor no envía Telegram/email por sí mismo; el operador ejecuta o programa el cron (decisión: cero secretos nuevos en Python).
- El time-exit H12 lo ejecuta el bridge TS (scheduler de 30s, corre incluso con kill switch porque reduce riesgo). El PnL del time-exit es un **estimado** (entry fill vs close fill, sin fees); PnL desconocido (stop/manual) genera `POSITION_CLOSED_UNKNOWN_PNL_REQUIRES_OPERATOR` — el operador registra el PnL real de Binance y ajusta `risk_state.json` a mano si difiere materialmente.
- `available_balance` del status del bridge requiere `execution_enabled: true` (sin exchange port devuelve null → el sizing se niega, fail-closed).
- El bridge no consume el user-data websocket de Binance: los stops ejecutados en el exchange se descubren por time-exit skip o second-opinion, no en tiempo real. Aceptado para el canary (1 orden, revisión humana); si se escala, es la primera mejora.
