# GEN2 — Operational Modes Specification (SAFE / EXPERIMENTAL)

**Status:** FROZEN — opmodes-spec-v1.0 (2026-07-12). Supersede el Capital Contract V2 (rechazado en revisión de arquitectura) y unifica TODOS los contratos de riesgo en una sola fuente de verdad.

## Principio
Los modelos científicos (D3, TRRM V2, QMAE V2, EQM1, hashes, thresholds, feature store, candidate `gen2-20260711T202935Z`) están CONGELADOS e **idénticos en ambos modos**. Un cambio de modo cambia únicamente parámetros de ejecución operativa. Cambiar cualquier componente científico invalida el candidate y exige nueva generación.

## Fuente única de verdad
`aegis_gen2/live_canary/<candidate_id>/operational_contract.json` — escrito por `gen2_operational_contract.py`, consumido por core (risk gate), exec (sizing) y decision loop. `DEFAULT_LIMITS` y `operational_manifest_v2` quedan eliminados. Un solo arm token (schema v2 + campo `mode` cubierto por checksum). Un solo chequeo de Phase O (en core; exec lo importa). El token es válido SOLO para el modo con el que fue creado.

## SAFE (contrato de producción futuro)
- sizing: **notional fijo en USD** (`fixed_notional_usd`, default $25), sin compounding
- leverage: ≤ **5x** (default 3x), isolated
- riesgo/trade ≤ **1%** del equity inicial del experimento; daily loss cap 2%; total cap 5%; equity floor 95%
- max 1 posición simultánea; ≤1 nueva/30min; sin martingala/averaging/pyramiding
- coherencia obligatoria (validada por código): pérdida esperada por stop < daily_cap < total_cap

## EXPERIMENTAL (solo fase de aprendizaje, wallet experimental)
Objetivo declarado: acelerar validación de arquitectura/bridge/forward — **no** validar edge con menos evidencia.
- sizing: fracción de balance ≤ **0.25** (no 0.50), compounding permitido y declarado
- leverage: ≤ **10x** (default 10x), isolated — nota de arquitecto: 15–20x del V2 rechazado; 10x ya implica ~riesgo 3.75% equity/trade con stop 1.5%
- daily loss cap 10% del equity inicial; total cap 25%; equity floor **75%** (no 50%)
- max 1 posición; ≤1 nueva/30min; primeras armas `max_orders=1`
- prohibido igual que SAFE: martingala, averaging, pyramiding, aumentar tras pérdidas
- **sunset clause:** el modo EXPERIMENTAL expira al alcanzar `LIVE_CANARY_TECHNICALLY_VALIDATED` (≥20 órdenes técnicamente limpias) o 60 días, lo que ocurra primero; después solo SAFE.

## Cambio de modo
`--mode safe|experimental` al escribir el contrato. El cambio: regenera contrato + exige token nuevo (el token viejo queda inválido por mode mismatch) + queda registrado en `contract_history.jsonl`. Nunca cambia modelos/hashes/candidatos/thresholds/features/training. Cambiar de modo con posición abierta: prohibido (fail-closed).

## Fail-closed transversal
Cualquier fallo (contrato ausente/corrupto, incoherencia de caps, token de otro modo, posición abierta durante switch, Phase O no pausada, kill switch) → NO_TRADE con razón registrada.
