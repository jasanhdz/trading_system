# GEN2-F1 + LIVE CANARY — Specification

**Status:** FROZEN — f1canary-spec-v1.0 (2026-07-12)
**Candidate:** `gen2-20260711T202935Z` (freeze `GEN2_SYSTEM_FREEZE.json`; TRRM 69c03e12…, EQM 77887b7c…, D3 86be5a15…). Cualquier cambio de componente congelado invalida el candidate, detiene el canary y reinicia el forward.
**Postura declarada:** ECONOMICALLY_PROMISING_BUT_STRUCTURALLY_UNCERTAIN (EQM_REJECTED predictivo + ECON_READY económico — tensión visible, prohibido reinterpretar pocos trades como validación).

## Hipótesis
- **H-F1:** las decisiones forward maduras (velas canónicas finales, H12) reproducen el signo económico del desarrollo. **H0-F1:** expectancy forward ≤ 0.
- **H-CAN:** el stack congelado puede ejecutar ≥20 órdenes reales con paridad research/live completa, brackets 100%, cero huérfanas/duplicadas y pérdida acotada determinista. **H0-CAN:** la ejecución diverge de la decisión research o los límites no se pueden garantizar.

## Contrato de capital experimental
- `experimental_capital_cap`: **$15.00 USDT** (< balance actual $16.24; nunca el balance completo).
- **BLOQUEO ACTUAL:** la wallet es única y compartida con Phase O (activa). Con margen compartido no existe límite de pérdida determinista del experimento → `LIVE_CANARY_CONFLICT_WITH_EXISTING_STRATEGY`. Además $16 está bajo el min-notional típico. Requisitos para desbloquear: (a) pausar nuevas entradas Phase O (tarea operativa separada, sin cierre automático de posiciones), (b) fondear la wallet experimental a ≥$100 o aceptar universo reducido a símbolos con min-notional ≤$5, (c) isolated margin.

## Sizing pre-registrado (al desbloquear)
Riesgo/op 0.5% del cap; daily loss 2%; total canary 5%; ≤1 posición nueva/30min; ≤1 simultánea; leverage mínimo del instrumento (≤5x); sin martingala/averaging/compounding/scaling; notional congelado.

## Gates G1–G8 (todos bloqueantes antes de cualquier orden)
G1 integridad de artifacts/hashes/commit/entorno; G2 finalidad de datos (solo velas cerradas canónicas, paridad semanal); G3 paridad research/live por oportunidad (mismos inputs→misma decisión, rtol 1e-9); G4 integridad de oportunidad (signal_id único, causal, SHORT H12, sin duplicados/conflictos); G5 execution readiness (filters, step/tick, balance, leverage, reloj); G6 brackets (entrada confirmada + SL reduceOnly + salida temporal confirmados en ≤60s o neutralizar + CRITICAL_EXECUTION_FAILURE + pausa); G7 límites de pérdida no alcanzados; G8 **arm token humano** válido.

## Fail-closed
Cualquier incertidumbre (features faltantes, hash mismatch, score no finito, vela no final, balance/posición/bracket ambiguo, timeout, state corrupto, límite no comprobable) → `NO_TRADE` con razón exacta registrada. Nunca RETAIN implícito.

## Doble stream obligatorio
`paper_decisions.jsonl` (decisión congelada + precio teórico causal + costos asumidos + outcome al madurar, independiente del live) y `live_orders.jsonl`/`fills.jsonl`/`brackets.jsonl`/`reconciliations.jsonl` (acks, fills, latencia, fees, funding, slippage, errores). Nunca mezclados. Persistencia append-only en `aegis_gen2/live_canary/<candidate_id>/`, escrituras atómicas, sin secretos.

## Arm token (dos pasos, humano)
Paso 1: sistema READY y desarmado (este trabajo). Paso 2: SOLO el usuario crea el token:
```
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/gen2_canary_core.py --mode create-arm-token \
  --candidate-id gen2-20260711T202935Z --capital-cap 15 --expiry-hours 72 \
  --allowed-symbols ADAUSDT,DOGEUSDT --max-orders 5
```
El token incluye checksum ligado al candidate y expira; sin token válido = no orders; sin re-arme automático; un token gastado (max_orders) no se reutiliza.

## Pausa automática (cualquiera → desarmar + bloquear entradas + preservar/reconciliar posiciones + incidente + re-arme humano)
daily/total/drawdown/consecutive-loss caps; bracket failure; orden duplicada; posición huérfana; parity mismatch; fallo de finalidad; hash mismatch; state corrupto; reconciliación discrepante; slippage >5× modelado; fees inesperadas; posición > límite; conflicto Phase O; inestabilidad API sostenida.

## Kill switch
Archivo persistente `KILL_SWITCH` en el directorio del canary + verificación al inicio de cada evaluación; fail-closed tras reinicio; comando manual documentado; independiente de PM2; sin re-arme automático.

## Evidencia y no-mezcla
F1: ≥8 semanas, ≥300 decisiones maduras, ≥2 regímenes, comparación pareada paper/live/baselines-diagnóstico (los diagnósticos jamás generan órdenes). Gate técnico intermedio `LIVE_CANARY_TECHNICALLY_VALIDATED` (≥20 órdenes, 100% brackets, 0 huérfanas/duplicadas, paridad y reconciliación completas, límites intactos) — no demuestra edge ni permite scaling. NO SCALING en toda la fase.
