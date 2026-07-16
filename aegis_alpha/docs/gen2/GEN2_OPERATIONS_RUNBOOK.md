# GEN2 — Operations Runbook, Deployment Checklist & Recovery Procedures

**Status:** ops-runbook-v3.0 (2026-07-16, autonomous) · Candidate `gen2-20260711T202935Z` · Owner: Jasan

## 9. Autonomous operation (config = source of truth, PM2 = arm)

**The model (v3):** there is no contract, arm token, or manual re-arming. A single file, `gen2_config.yaml`, is the entire operational source of truth. **`pm2 start` = ARMED, `pm2 stop` = DISARMED.** A running watcher IS the authorization.

**`gen2_config.yaml`** (at `/home/jasan/Develop/aegis_gen2/`, gitignored; template in `docs/gen2/gen2_config.example.yaml`): mode, capital.initial_equity, symbols, execution_enabled, telegram_enabled, optional risk_overrides. The mode preset supplies the safe leverage/loss caps; overrides are coherence-validated and **fail-closed** (an incoherent config refuses to trade). Science (candidate, hashes, frozen threshold) is NOT editable here.

**To operate:** edit `gen2_config.yaml` (set `execution_enabled: true` when ready) → `pm2 restart gen2-bridge gen2-watcher`. That's it. To pause: `pm2 stop gen2-watcher` (or `execution_enabled: false` + restart).

**Automatic audit (replaces contracts):**
- Every start writes `startup_audit.jsonl` (timestamp, commit, candidate, config checksum, hashes, mode, symbols, environment).
- A live edit is detected and diffed → `config_changes.jsonl` with `CONFIGURATION_CHANGED` (old/new per field) and a Telegram WARNING; an edit that makes the config INVALID is flagged CRITICAL and the last valid config keeps running (fail-closed).

**Kill switch under autonomous operation (smart re-arm):** kill switches are UNCHANGED — any critical event (bracket failure, reconciliation mismatch, orphan/unprotected position, hash/env/candidate mismatch, equity floor) stops new orders, protects capital, fires Telegram + incident. The switch file persists across restart. On `pm2 restart` the watcher runs the **startup gauntlet** (science hashes + env + selection policy + config coherence + Phase O paused + **no open position** per the bridge). The kill clears **only if the gauntlet passes clean** — so a restart can never resume into a broken/unsafe state. If the cause is unresolved (e.g. a position still open), the kill stays and Telegram says exactly why. Resolve the cause, then `pm2 restart`.

**What was removed (and why it's safe):** arm token (PM2-alive is the authorization; its candidate/hash binding is already covered by the freeze/env/selection-policy hard-stops; its order cap is covered by `max_orders_per_day`/`max_concurrent`/`max_orders_per_cycle`), the derived contract JSON (config is the single source), and `consume_order`. The blast radius is bounded by the capital caps (daily 10% / total 25% / equity floor 75%), which trip long before the daily order cap.

---


## 0. Modo always-on (PM2) — el estado normal del sistema

Desde ops-runbook-v2.0 los servicios Gen2 corren bajo PM2 (`ecosystem.gen2.config.js` en el repo TS) y se restauran solos tras reboot/apagón (systemd `pm2-jasan.service` + `pm2 save`, ya configurados):

| Proceso | Qué hace | Política |
|---|---|---|
| `gen2-bridge` | bridge HTTP TS (único ejecutor) | autorestart, backoff 5s, 400M |
| `gen2-watcher` | decision loop `--watch --live` cada 300s (paper hasta armar) | autorestart, backoff 15s, 1500M, lock singleton |
| `gen2-monitor` | ops monitor `--expect-watch-running` | cron `*/15`, exit 1 visible en `pm2 ls` como errored |
| `gen2-reconciler` | second-opinion vs exchange | cron `*/30` (necesita keys read-only en `.env.gen2`) |

Secretos: `binance-futures-bot-ts/.env.gen2` (gitignored, 0600). `GEN2_EXECUTION_ENABLED=false` por defecto — el bridge valida en dry-run hasta que el owner lo cambie deliberadamente y reinicie `gen2-bridge`.

**Checklist de despliegue corto (el definitivo):**
1. Encender el servidor.
2. Esperar ~2 min a que PM2 restaure todo (`pm2 ls`: gen2-bridge y gen2-watcher `online`).
3. Verificar monitor: `pm2 logs gen2-monitor --lines 5 --nostream` o correr el monitor a mano → `healthy: true`.
4. (Solo para operar en real) editar `.env.gen2`: `GEN2_EXECUTION_ENABLED=true` + keys → `pm2 restart gen2-bridge` → crear arm token (§1 paso 11).
5. Esperar señal. Nada más.

Logs separados en `binance-futures-bot-ts/logs/gen2/{bridge,watcher,monitor,reconciler}.{out,err}.log`.

**Semántica de reinicios (verificada en vivo):** reinicio del watcher no re-emite decisiones (estado por símbolo por vela, lock singleton); reinicio del bridge no duplica órdenes (`bridge_state.json` + ids deterministas + dedup del exchange); el token se consume ANTES de enviar (un crash jamás deja orden aceptada con token vivo); apagón a mitad de un append deja como mucho una línea rota que todos los lectores saltan; snapshots se podan a los últimos 12.
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

## 8. Selection Policy (umbral congelado)

**Qué es.** La regla que decide si un ciclo abre o se abstiene. Vive en `GEN2_SELECTION_POLICY.json` (junto al freeze) y es parte de la ciencia congelada.

**Flujo por ciclo:** 11 símbolos → veto TRRM (abstiene si `tail ≥ 0.1017`) → EQM score de los supervivientes → **ranking global** → se toma el mejor → se compara contra el **umbral absoluto congelado** (`threshold_value = 0.01432240`). Si el mejor lo supera → abre. Si no → `NO_DECISION` con motivo `BELOW_FROZEN_EQM_THRESHOLD`, se registra en paper (`selection_outcomes.jsonl`) y espera la siguiente vela.

**Por qué NO usamos `score > 0`.** El cero del score de EQM (reg_component) **no es** el breakeven económico — es un valor arbitrario. Usar `>0` no corresponde a nada que ECON1 haya validado.

**Por qué NO usamos best-of-cycle sin filtro.** Abrir "el mejor de cada ciclo" aunque todos sean negativos toma trades de expectativa negativa que ECON1 nunca midió; sangraría capital y contaminaría la pregunta científica "¿aguanta el edge?".

**Por qué el umbral congelado.** ECON1 validó tomar el **top-decil por EQM score** del pool veto-retenido (`budget_n = round(0.10 × keep_n)`, TOPK=0.10, edge +$0.140/trade). El umbral es exactamente ese top-decil expresado como un número absoluto: `quantile_0.90(s_eqm | supervivientes-del-veto del dev H12 congelado)`. Es **market-independent**, no se recalcula, no se adapta.

**Qué garantiza científicamente.** El canary abre **solo** los trades de alta calidad que ECON1 midió, reproduciendo su régimen de selección. Cuando el mercado no ofrece señal top-decil (como ahora — mejor score ~+0.002 < 0.0143), **abstenerse es lo correcto**, no un fallo.

**Comandos:**
- `gen2_selection_policy.py --mode freeze` — congela el umbral (inmutable; verifica todos los hashes vs freeze).
- `gen2_selection_policy.py --mode validate-selection-policy` — **recomputa desde cero** y compara (drift debe ser 0); hard stop si algo cambió (dataset/EQM/candidate/threshold). Correr antes de cada re-armado.
- `gen2_decision_loop.py --dry-run-cycle` — un ciclo paper mostrando ranking, scores, umbral, ganador, abre/no-abre y motivo. Sin órdenes.

**Fail-closed:** el loop carga el umbral verificando candidate + los 4 hashes contra el freeze en cada arranque; un policy ausente o con hash distinto es **hard stop** del watcher (marcadores `SELECTION_POLICY_*`), igual que un mismatch de modelo — nunca opera con una política de selección no validada.

## 7. Límites conocidos (aceptados para el canary, documentados)
- El monitor no envía Telegram/email por sí mismo; el operador ejecuta o programa el cron (decisión: cero secretos nuevos en Python).
- El time-exit H12 lo ejecuta el bridge TS (scheduler de 30s, corre incluso con kill switch porque reduce riesgo). El PnL del time-exit es un **estimado** (entry fill vs close fill, sin fees); PnL desconocido (stop/manual) genera `POSITION_CLOSED_UNKNOWN_PNL_REQUIRES_OPERATOR` — el operador registra el PnL real de Binance y ajusta `risk_state.json` a mano si difiere materialmente.
- `available_balance` del status del bridge requiere `execution_enabled: true` (sin exchange port devuelve null → el sizing se niega, fail-closed).
- El bridge no consume el user-data websocket de Binance: los stops ejecutados en el exchange se descubren por time-exit skip o second-opinion, no en tiempo real. Aceptado para el canary (1 orden, revisión humana); si se escala, es la primera mejora.
