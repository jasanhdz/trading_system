# GEN2 Live Canary — Trabajo pendiente para Codex

**Estado al handoff (2026-07-12):** spec congelada (`GEN2_F1_LIVE_CANARY_SPEC.md`), safety core implementado y testeado (`gen2_canary_core.py`: dual-stream records, risk caps, kill switch persistente, arm token humano, auditoría de conflicto Phase O, gauntlet fail-closed). Canary inicializado y **DESARMADO**. Decisión vigente: `LIVE_CANARY_CONFLICT_WITH_EXISTING_STRATEGY` (Phase O activa + wallet única de $16.24). **Ninguna orden puede enviarse hasta resolver el conflicto y crear el arm token humano.**

## Bloqueos operativos previos (decisión del owner, NO de Codex)
1. Pausar nuevas entradas Phase O (tarea operativa separada; sin cierre automático de posiciones).
2. Fondear/segregar wallet experimental (≥$100 recomendado; $16 está bajo min-notional) con isolated margin.

## Pendiente de implementación (en orden, un commit por punto, mismos estándares Gen2: spec-first ya cubierta, tests fixture-based, fail-closed, sin secretos, identidad git "Jasan Hernández <jasanhdzb@gmail.com>", sin push)
1. **`gen2_canary_exec.py` — order adapter** (commit `feat: add Gen2 canary execution and reconciliation`):
   - Credenciales SOLO por env vars en runtime (jamás leer/escribir .env ni serializar claves).
   - Flujo: `canary_eligibility()` → G5 execution-readiness (exchangeInfo filters: minNotional/step/tick, balance, leverage ≤5x isolated, reloj) → orden MARKET reduceonly=false con `newClientOrderId=GEN2-<candidate>-<signal_id>` (dedup por id) → confirmar fill real → **G6 brackets**: SL reduceOnly (riesgo 0.5% del cap) + salida temporal H12 programada; si brackets no confirman en ≤60s → cerrar posición a mercado + `CRITICAL_EXECUTION_FAILURE` + `engage_kill_switch()`.
   - Registrar en `live_orders.jsonl`/`fills.jsonl`/`brackets.jsonl`: acks, fill price/qty, latencia (t_envío→t_ack→t_fill), fees reales, slippage vs precio teórico del paper stream.
   - `orders_sent` +1 en `risk_state.json` SOLO tras ack.
2. **Reconciler** (mismo commit o siguiente): cada corrida compara posiciones/órdenes del exchange vs `live_orders.jsonl`; huérfana/duplicada/qty-mismatch → incidente + kill switch. Append a `reconciliations.jsonl`.
3. **Integración dual-stream** en `gen2_system_freeze.py --mode collect`: por cada decisión, `record_paper_decision()` SIEMPRE; si `canary_eligibility()` pasa → exec adapter. El paper stream nunca depende del live.
4. **Parity heartbeat + cost auditor + daily report** (commit `feat: ...`): comparar decisión research vs inputs usados por el adapter (rtol 1e-9) → `parity_reports/`; modeled-vs-realized fees/slippage/funding → `cost_reports/`; resumen diario → `daily_reports/` con las métricas operativas de la spec.
5. **Maduración de outcomes F1**: job que, cumplido H12, etiqueta cada decisión paper/live con velas canónicas finales (usar fetcher gen2_d3_snapshot; gate de finalidad) y acumula la comparación pareada (paper vs live vs diagnósticos random/rules/TRRM-only — los diagnósticos jamás generan órdenes).
6. **Gauntlet de tests restante** (commit `test: add Gen2 live canary safety gauntlet`): partial fill, rejected order, timeout/disconnect (mock), invalid quantity, insufficient balance, restart mid-trade, corrupted state, replay del adapter contra exchange mock, no-secret-serialization (grep del árbol de artifacts), no auto-scaling (notional constante), no auto-rearm.

## Criterios que NO puede cambiar Codex
Candidate `gen2-20260711T202935Z` y todos sus hashes; sizing/caps de la spec; semántica fail-closed; doble stream; NO SCALING; gate técnico `LIVE_CANARY_TECHNICALLY_VALIDATED` (≥20 órdenes, 100% brackets, 0 huérfanas/duplicadas, paridad completa) que NO demuestra edge; decisiones válidas de fase (lista en spec). Si algo exige cambiar el candidate → detenerse y reportar.
