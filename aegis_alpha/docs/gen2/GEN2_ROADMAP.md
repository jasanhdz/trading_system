# GEN2 — Roadmap maestro

**Status:** PROPOSED (se congela junto con GEN2_D3_SPEC.md)
**Regla del proceso:** cada fase recibe su propia especificación completa (como GEN2_D3_SPEC) ANTES de implementarse. Este roadmap fija el orden y los gates entre fases; no sustituye a las specs. Ninguna fase arranca sin la decisión de aceptación de la anterior emitida por su auditor.

```
GEN2-D3            dataset canónico                       [spec: GEN2_D3_SPEC.md]
   │  gate: D3_CANONICAL_READY (o PARTIAL documentado)
   ▼
GEN2-RV2           TRRM V2 + QMAE V2 (una sola fase)
   │  gate: RV2_VETO_READY
   ▼
GEN2-EQM1          Entry Quality Model V1
   │  gate: EQM1_EDGE_CONFIRMED
   ▼
GEN2-ECON1         backtest económico en USDT netos
   │  gate: ECON1_POSITIVE_EXPECTANCY
   ▼
GEN2-FREEZE        congelación del trío + política
   │  gate: manifiesto de freeze verificado
   ▼
GEN2-F1            forward research (collector, sin enforcement)
   │  gate: F1_FORWARD_CONFIRMED (evidencia mínima pre-registrada)
   ▼
GEN2-SHADOW        shadow paralelo contra el sistema live (decisiones hipotéticas)
   │  gate: SHADOW_CONFIRMED + aprobación explícita del owner
   ▼
GEN2-LIVE-CANDIDATE  capital mínimo, kill-switch, auto-pausa por PnL real
```

## Justificación del orden

- **RV2 antes que EQM1:** el veto define la población sobre la que EQM se evalúa; evaluar EQM sin veto mide un sistema que nunca existirá. Además TRRM V2/QMAE V2 comparten dataset, gauntlet y tooling E2 ya probado → fase corta que valida D3 en el camino.
- **TRRM V2 y QMAE V2 juntos:** mismo insumo, mismos folds, mismos reportes; separarlos duplica burocracia sin ganancia de control. QMAE aporta el quantile q90 conformal que la política de veto y el sizing futuro consumen.
- **ECON1 como fase propia (no un apéndice de EQM):** la conversión de métricas de label a dinero (fees + slippage pesimista + funding + distribución de drawdown + baselines de reglas con el mismo presupuesto de trades) es donde mueren la mayoría de los sistemas. Merece gate propio: PF ≥ 1.5 sin gestión, expectancy positiva neta, max loss acotado, y **superar dos baselines simples de reglas**. Si EQM no supera al baseline de reglas, el candidato pasa a ser "reglas + veto TRRM V2" y EQM vuelve a research (el roadmap no se detiene: cambia de vehículo).
- **F1 antes de SHADOW:** forward pasivo acumula la evidencia virgen (la primaria de Gen2, dado que el lockbox histórico está degradado); shadow añade el contraste operacional contra el sistema actual.
- **Evidencia mínima pre-registrada para F1:** ≥50 eventos de cola (ideal 100) y ≥8 semanas cruzando ≥2 regímenes; bandas de rechazo y calibración definidas en el freeze. Sin cumplirse, no hay SHADOW aunque los números "se vean bien".

## Criterios transversales (aplican a toda fase)

1. Selección solo pre-lockbox; lockbox histórico = 1 consulta por candidato final, registrada.
2. Evidencia primaria = forward Gen2 (posterior a la aprobación de las specs).
3. Todo artefacto con manifest + sha256; doble corrida bit-idéntica donde aplique.
4. Un solo entorno de ejecución (venv_rocm62) registrado en manifests.
5. Reportes con scope/population/engine etiquetados (ley desde la auditoría Fable-A).
6. Fail-closed: ausencia de datos/historia → no operar, nunca "permitir por defecto".
7. El owner puede detener cualquier avance; nunca promover un candidato rechazado por gates.

## Paralelo permitido (no bloqueante)

- Reparación del refresher + backfill del SQLite (operativo, owner) — mejora el sistema live actual y el futuro shadow.
- Pausa de entradas Phase O SHORT (operativo, owner) — recomendada desde la primera revisión; reduce pérdida esperada y limpia el entorno para SHADOW.
- Shadow Committee V2 y regime features → candidatos post-GEN2-F1, solo si EQM1 deja gaps identificados.
