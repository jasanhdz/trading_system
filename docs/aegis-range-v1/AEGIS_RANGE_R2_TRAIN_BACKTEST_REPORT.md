# AEGIS Range R2 TRAIN Backtest Report

## 1. Estado y conclusion

**Estado:** `AEGIS_RANGE_R2_TRAIN_BACKTEST_COMPLETE_REJECT_NO_EDGE`

El experimento TRAIN se completo sobre los 11 simbolos y los 384 candidatos
preregistrados. Ningun candidato produjo expectativa neta positiva ni profit
factor superior a 1 en `BASELINE`. El stress de costos empeora monotonicamente
el resultado y tampoco deja candidatos positivos.

La hipotesis operativa evaluada no demuestra edge neto en TRAIN bajo el grid,
el lifecycle y el modelo de costos congelados. No se selecciona candidato, no
se autoriza promocion y no existe justificacion empirica para abrir
`CALIBRATION`, `VALIDATION` o `HOLDOUT`.

Este resultado rechaza la configuracion investigada; no prueba que toda posible
estrategia de rangos sea inviable.

## 2. Autoridad y lineage

- Commit base: `5ecfa7009f6cfe3f700cdc1607aacb15ce245614`.
- R1 immutable: `42/42 MATCH`.
- R1 manifest SHA-256:
  `2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c`.
- Dataset logical SHA-256:
  `587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8`.
- Post-R1 revalidation manifest SHA-256:
  `9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1`.
- Politica funding/mark: `MONTHLY_PRIMARY_DAILY_GAP_FILL_V1`.
- Reporte PHASE A SHA-256:
  `b21fc0e85286c09e5ac53a12a30965858aaa15858464847ec9486509a94274c5`.
- PHASE A: `AEGIS_RANGE_LINEAGE_HASH_RECONCILIATION_PASS`.

La unica particion abierta fue TRAIN. Los flags registrados fueron
`TRAIN=true`, `CALIBRATION=false`, `VALIDATION=false` y `HOLDOUT=false`.

## 3. Alcance ejecutado

- Ventana TRAIN: `[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`.
- Embargo inicial: 48 horas; primera confirmacion admisible en
  `2024-01-03T00:00:00Z`.
- Timeframe: 5 minutos.
- Simbolos: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT,
  ADAUSDT, AVAXUSDT, LINKUSDT, SUIUSDT y LTCUSDT.
- Grid: `384/384`, sin pruning ni seleccion adaptativa.
- Escenarios: `BASELINE`, `STRESS_20` y `STRESS_30`.
- Los escenarios stress reprician exactamente los trades BASELINE; no generan
  una poblacion distinta.
- Funding aplicado en el intervalo `(entry_fill_at, exit_fill_at]`.
- Unidad estadistica primaria: `range_episode_id`.

## 4. Implementacion research-only

El backtester externo delega las decisiones al `RangeEngineV1` y al lifecycle
R1 mediante observers que llaman `super()`. No se duplico ni modifico el motor
R1. `RegimeEngineV2` se evaluo por un bridge TypeScript batch y se materializo
en caches causales con ventana exacta de 160 velas.

El primer precompute se detuvo antes de ejecutar candidatos porque el bridge
ignoraba escrituras parciales del pipe de stdout. Fue un defecto de IPC del
nuevo runner, no una divergencia economica ni un defecto R1. Se reemplazo por
escritura asincrona con backpressure y se agrego una regresion de 2,200 velas.
Despues del fix, el cache BTC produjo las `105,249` decisiones esperadas y el
run oficial completo finalizo con `AEGIS_RANGE_R2_TRAIN_RUN_COMPLETE`.

## 5. Evidencia oficial

La unica fuente oficial es:

`sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a/`

| Artefacto | Filas | SHA-256 |
|---|---:|---|
| `run_manifest.json` | n/a | `5f62022f35fb38de174e6f7c573397d1c1ceebc75d76f7d848260c35456012b8` |
| `candidate_metrics.json` | 384 | `12f72be45420099d7ab0a56524ca934e791dfbaa9da0c0add87277d7939b656f` |
| `episodes.jsonl.gz` | 1,457,408 | `82989a83a68935ed44866afb2f5904e703c81e27b50acde5fe1c2fabd6af5270` |
| `trades.jsonl.gz` | 22,016 | `125f31dcb1bf27e6f183bbbb02da901a5133847e577196a9bd6a59be42cd4537` |

Los 11 shards coinciden con sus hashes declarados. `run_b` fue una replica
interrumpida, no posee manifest final y no se usa como evidencia. No se volvera
a ejecutar sin autorizacion explicita.

## 6. Auditoria integral

Se recorrieron todos los episodios y trades oficiales.

- Artefactos principales: `3/3 MATCH`.
- Shards por simbolo: `11/11 MATCH`.
- Metricas por candidato: `384/384`.
- Episodios no purgados: `1,456,704`.
- Episodios purgados en split boundary: `704`.
- Trades purgados: `0`.
- Trades asociados a un episodio existente: `22,016/22,016`.
- Errores de conteo candidato/episodio/trade: `0`.
- Errores en `net = gross - fees + funding`: `0`.
- Violaciones de orden `BASELINE > STRESS_20 > STRESS_30`: `0`.
- Funding events cobrados: `2,872`; violaciones de `(entry, exit]`: `0`.
- Violaciones de la particion TRAIN: `0`.
- Cierres intrabar en la vela de entrada: `2,448` (11.12%); son validos y
  tienen `entry_at == exit_at`.
- Tests Python finales: `89/89 PASS`.
- Tests TypeScript frozen: `17/17 PASS`.
- R1 al cierre: `42/42 MATCH`.

La causalidad se sostiene por el procesamiento secuencial, snapshots cacheados
por timestamp, warmup exacto y los tests que prueban que agregar velas futuras
no cambia decisiones historicas. Los replay hashes por simbolo y candidato
quedaron registrados en el manifest oficial.

## 7. Poblacion observada

- Episodios operados: `22,016`.
- Abstencion agregada: `98.49%`.
- Rango de abstencion por candidato: `97.43%` a `99.11%`.
- Trades por candidato: 28 a 91.
- LONG: 9,868 (44.82%).
- SHORT: 12,148 (55.18%).
- Win rate BASELINE: 25.40%.
- Win rate STRESS_30: 25.18%.
- Holding mediano: 25 minutos.
- Holding maximo: 720 minutos.

Distribucion de salidas:

| Reason | Trades | Porcentaje |
|---|---:|---:|
| `STOP` | 16,280 | 73.95% |
| `TARGET` | 5,032 | 22.86% |
| `MAX_HOLD` | 560 | 2.54% |
| `TRADE_BREAKOUT` | 144 | 0.65% |

La perdida dominante se materializa mediante stops, no mediante la salida
confirmada de breakout. El `breakout_loss_rate` igual a cero no compensa la
frecuencia elevada de stops.

## 8. Distribucion de candidatos

La expectativa se expresa por episodio operado; en esta ejecucion cada episodio
operado contiene un trade.

| Escenario | Min | Mediana | Max | Expectativa positiva | PF > 1 |
|---|---:|---:|---:|---:|---:|
| BASELINE | -0.2747% | -0.1607% | -0.0220% | 0/384 | 0/384 |
| STRESS_20 | -0.3347% | -0.2207% | -0.0820% | 0/384 | 0/384 |
| STRESS_30 | -0.4348% | -0.3207% | -0.1821% | 0/384 | 0/384 |

El candidato descriptivamente mejor es `C371`, empatado economicamente con
`C369`. El tie-break determinista del manifest publica `C371`. El candidato
mediano descriptivo es `C364` y el peor es `C093`. Estas etiquetas tienen
`NO_SELECTION_AUTHORITY`.

Solo existen 184 firmas economicas distintas entre 384 candidatos. En
particular, `min_chop_risk=0.62` y `0.70` producen distribuciones identicas en
TRAIN, por lo que ese eje fue no vinculante en esta muestra.

## 9. Mejor candidato descriptivo: C371

Parametros:

```json
{
  "cluster_tolerance_atr": 0.3,
  "max_adx": 20.0,
  "min_chop_risk": 0.7,
  "min_range_amplitude_pct": 0.03,
  "min_safety_volume_ratio": 0.75,
  "rejection_min_wick_body_ratio": 1.5,
  "stop_buffer_atr": 0.5,
  "target_buffer_atr": 0.0
}
```

- Episodios confirmados: 2,838.
- Episodios operados/trades: 28.
- Abstencion: 99.01%.
- LONG/SHORT: 11/17.
- Salidas: 20 STOP, 7 TARGET y 1 MAX_HOLD.
- Win rate BASELINE: 28.57%.
- Holding mediano: 27.5 minutos; medio: 97.9; maximo: 720.
- False-range rate: 45.31%.
- Funding events: 5.

| Metrica | BASELINE | STRESS_20 | STRESS_30 |
|---|---:|---:|---:|
| Gross expectancy | 0.0770% | 0.0170% | -0.0831% |
| Net expectancy | -0.0220% | -0.0820% | -0.1821% |
| Profit factor | 0.9495 | 0.8283 | 0.6684 |
| Episode CVaR95 | -0.9223% | -0.9823% | -1.0823% |
| Pseudo-equity max drawdown | 0.6216% | 0.6981% | 0.8257% |
| Total net return | -0.6151% | -2.2963% | -5.0987% |

En BASELINE, la expectativa gross de 7.70 bps no cubre aproximadamente 10 bps
de fees por round trip. Funding aporta solo 0.10 bps por trade, dejando -2.20
bps netos. El edge bruto marginal desaparece antes de cualquier stress.

La estabilidad tampoco pasa:

- 8 meses negativos y 3 positivos entre 11 meses con trades; junio no opera.
- 5 simbolos negativos y 6 positivos en BASELINE.
- Noviembre concentra 80.07% de las contribuciones mensuales positivas.
- XRPUSDT concentra 55.47% de las contribuciones positivas por simbolo.
- LONG expectancy: +0.0651%; SHORT expectancy: -0.0783%.
- Majors expectancy: -0.1491%; alts expectancy: +0.0204%.

Estas diferencias por lado y subuniverso son diagnosticas post hoc y no otorgan
autoridad para seleccionar solo LONG, alts, XRPUSDT o noviembre.

## 10. Lectura economica

Los datos no muestran un candidato cercano a un gate defendible:

1. Todos los candidatos fallan expectativa neta BASELINE.
2. Todos los candidatos fallan profit factor BASELINE.
3. El mejor resultado depende de una poblacion pequena de 28 trades y presenta
   concentracion temporal y por simbolo.
4. Los stresses degradan el resultado de forma material y monotona.
5. La cola permanece negativa, con CVaR95 de -0.92% incluso para `C371`.
6. La alta abstencion no evita una tasa de STOP de 73.95% en las entradas
   aceptadas.

La conclusion no depende de fijar thresholds finales del gate: expectativa neta
positiva y profit factor superior a 1 son condiciones primarias cualitativas ya
incumplidas por `384/384` candidatos.

## 11. Limitaciones de evidencia

- `candidate_metrics.json` publica pseudo-equity drawdown, no un backtest de
  portfolio con sizing, concurrencia o leverage.
- MFE, MAE y Sharpe no se materializaron como campos del artefacto. El win rate
  se reconstruyo exactamente desde `trades.jsonl.gz`; MFE/MAE no pueden
  reconstruirse sin la trayectoria intratrade por barra.
- Los paths de shards en el manifest oficial son absolutos y atan ese manifest
  al workspace actual. Los hashes de contenido y artefactos relativos siguen
  siendo verificables.
- TRAIN permite diagnostico, no confirmacion fuera de muestra.

Estas limitaciones no revierten el rechazo: las dos condiciones economicas mas
basicas ya fallan de manera uniforme.

## 12. Frontera de fase

- `candidate_selected=false`.
- `calibration_opened=false`.
- `pre_validation_spec_frozen=false`.
- `validation_opened=false`.
- `holdout_opened=false`.
- No hubo cambios live ni ejecucion de ordenes.

**Decision final:** detener la linea experimental congelada en TRAIN. Cualquier
nueva formula, filtro, lifecycle o familia seria una hipotesis nueva que exige
preregistro y evidencia independiente; no debe presentarse como ajuste del
experimento completado.
