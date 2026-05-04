# Aegis Alpha Whitepaper

**Estado:** arquitectura inicial v0.1.0  
**Propósito:** documento vivo de arquitectura, responsabilidades de carpetas y razón de ser de cada módulo.

## 1. Tesis Del Proyecto

Aegis Alpha reemplaza la línea experimental Phantom V30 como una plataforma ML de señales de trading. La filosofía cambia de “un agente RL descubre todo” a una arquitectura por capas:

1. Datos y features limpios.
2. Regímenes de mercado explícitos.
3. Behavior Cloning prudente.
4. Refinamiento con PPO/RL.
5. Coliseo Survivor por riesgo, fees, regímenes y señal.
6. SignalQ para medir calidad de señal.
7. API de inferencia estable para el bot TypeScript.

El principio central es:

```text
Primero proteger capital. Después buscar edge. Luego escalar.
```

En el estado actual, Aegis ya expone inferencia compatible con el bot TS, pero todavía no tiene champion entrenado. Por diseño, mientras no exista modelo productivo, responde `IDLE` defensivo.

## 2. Estado Actual De Runtime

PM2 apunta el servicio de inferencia a:

```text
02-Aegis-API -> aegis_alpha/inference/server.py
```

Endpoints vivos:

```text
GET  /health
POST /predict
POST /ml-v2/predict
POST /ml-v2/exit_signal
```

Compatibilidad con el bot TypeScript:

El bot sigue llamando `http://127.0.0.1:8001/ml-v2/predict`. Aegis mantiene ese contrato y devuelve los campos legacy:

```text
long_prob
short_prob
neutral_prob
close_prob
consensus_level
meta_verdict
smart_leverage
features
```

Además incluye un bloque nuevo `aegis` con metadata completa:

```text
model_name
model_version
raw_action
gated_action
probs
signal_quality
risk_context
regime
metadata
```

## 3. Distribución De Carpetas

```text
aegis_alpha/
├── bc/
├── coliseum/
├── configs/
├── data/
├── docs/
├── env/
├── features/
├── inference/
├── logs/
├── models/
├── rl/
├── tests/
└── tools/
```

### `bc/`

Especialización: Behavior Cloning prudente.

Esta capa debe crear la política base que evita que PPO nazca caótico. No busca el trader perfecto; busca hábitos sanos:

```text
mucho IDLE
entradas simétricas LONG/SHORT
no overtrade
respetar cooldown
cerrar cuando el setup se invalida
```

Archivos:

- `labeler.py`: contiene reglas heurísticas para etiquetar acciones `IDLE/LONG/SHORT/CLOSE` a partir de features, estado de posición, hold steps y flat steps. Es la base del teacher prudente.
- `train_bc.py`: entrypoint placeholder para entrenar el modelo BC prudente. Debe evolucionar hacia la creación de `models/bc/aegis_bc_prudent.zip`.
- `evaluate_bc.py`: scaffold para evaluar el BC antes de dejar que PPO lo refine.
- `policies/`: espacio reservado para políticas BC auxiliares o variantes.

### `coliseum/`

Especialización: evaluación profesional y promoción de modelos.

El Coliseo no debe promover por PnL bruto. Debe medir supervivencia, riesgo, fees, direccionalidad, consistencia por régimen y SignalQ.

Archivos:

- `survivor_rules.py`: reglas duras iniciales de supervivencia: balance mínimo, DD, worst DD, fees y overtrade.
- `promotion.py`: utility inicial para rankear candidatos con net profit, drawdown, consistencia de régimen, fees, dominance y signal bonus.
- `signal_quality.py`: resumen básico de SignalQ: top probability promedio, porcentaje de señales >65 y gap long/short.
- `evaluator.py`: scaffold del evaluador de modelos. Aquí debe vivir el runner multi-window por regímenes.
- `reports.py`: formateo simple de reportes del Coliseo.

### `configs/`

Especialización: configuración declarativa.

Archivos:

- `base.yaml`: configuración base del sistema: símbolo, timeframe, DB, risk budget, gates y rutas de modelo.
- `production.yaml`: configuración de inferencia productiva: puerto, host, logging y gates.
- `curriculum_5x.yaml`: hiperparámetros iniciales de curriculum 5x para PPO.
- `bc_prudent.yaml`: configuración objetivo del BC prudente: dataset, mezcla de acciones y entrenamiento.

### `data/`

Especialización: insumos y datasets de Aegis.

Subcarpetas:

- `raw/`: datos crudos exportados o snapshots.
- `processed/`: datasets procesados listos para BC/RL.
- `regimes/`: ventanas o etiquetas de régimen.

Hoy solo contiene `.gitkeep`; no hay datasets Aegis versionados todavía.

### `docs/`

Especialización: documentación viva.

Archivos:

- `AEGIS_ALPHA_WHITEPAPER.md`: este documento. Debe actualizarse cada vez que cambie arquitectura, contratos, carpetas, endpoints, criterios de promoción o flujo de entrenamiento.

### `env/`

Especialización: MDP, reglas de riesgo y ejecución simulada.

Archivos:

- `aegis_env.py`: entorno determinista inicial. Implementa `reset`, `step`, equity, tracking de fees, opens, closes, invalid actions y max drawdown. Es una base simple para smoke tests, BC y futuro Coliseo.
- `action_mask.py`: reglas de acciones válidas. Si está flat, permite LONG/SHORT solo con `flat_steps >= min_flat_steps`; si está en trade, permite CLOSE solo con `hold_steps >= min_hold_steps`; por ahora no permite flips.
- `risk_engine.py`: funciones puras de riesgo y posición: abrir posición, cerrar posición, calcular ROE y notional.
- `reward.py`: reward inicial limpio, basado en cambio de equity, penalización de entrada e invalid action. Es deliberadamente simple para evitar la complejidad tóxica acumulada en Phantom.

### `features/`

Especialización: construcción de features y división de datos.

Archivos:

- `feature_builder.py`: genera las 21 features base usadas por Aegis: returns, RSI, EMAs, CVD, slopes, accelerations, ADX, trend efficiency y volatility regime.
- `regime_detector.py`: detector simple de régimen: `trend_up`, `trend_down`, `chop`, `high_vol`, `compression`, `mixed`.
- `dataset_splits.py`: plan inicial de splits walk-forward: train, validation windows y holdout.

### `inference/`

Especialización: API productiva y contrato con TypeScript.

Archivos:

- `server.py`: FastAPI principal. Expone `/health`, `/predict`, `/ml-v2/predict` y `/ml-v2/exit_signal`.
- `schemas.py`: modelos Pydantic para request/response. Define respuesta Aegis nueva y respuesta legacy compatible con el bot.
- `model_loader.py`: wrapper de carga de modelo. Si no existe champion en `models/champion/aegis_champion.zip`, devuelve predicción defensiva `IDLE`.
- `gates.py`: reglas de gating de señal: top probability mínima, gap long/short mínimo y thresholds más estrictos para `chop`.

### `logs/`

Especialización: estado y reportes runtime.

Subcarpetas:

- `coliseum/`: reportes futuros de evaluaciones.
- `telegram/`: logs futuros de mensajes/alertas.

### `models/`

Especialización: artefactos ML.

Subcarpetas:

- `champion/`: modelo productivo promovido, esperado como `aegis_champion.zip`.
- `challengers/`: modelos candidatos PPO.
- `bc/`: modelos de Behavior Cloning, esperado como `aegis_bc_prudent.zip`.
- `archive/`: backups históricos de champions y challengers.

Actualmente no hay champion Aegis; por eso inferencia responde `IDLE`.

### `rl/`

Especialización: entrenamiento PPO/RL.

Archivos:

- `policy.py`: `AegisTransformerExtractor`, extractor ligero Transformer 32D/1 layer compatible con observaciones `market` y `account`.
- `curriculum.py`: helper inicial para entropy por linaje (`champion`, `bc`, `fresh`).
- `callbacks.py`: scaffold para callbacks RL.
- `train_challenger.py`: scaffold del entrenamiento de un challenger.
- `trainer.py`: scaffold del orquestador de entrenamiento.

### `tests/`

Especialización: smoke tests y regresiones mínimas.

Archivos:

- `test_action_mask.py`: pruebas de action masking: cooldown para entradas y hold mínimo para cierre.

Nota: el entorno actual no tiene `pytest` instalado, por eso las pruebas se validaron ejecutando las funciones directamente.

### `tools/`

Especialización: herramientas operativas.

Archivos:

- `backtest_model.py`: scaffold para backtests.
- `export_champion.py`: scaffold para exportar/promover champion.
- `inspect_policy.py`: scaffold para inspección de política.
- `cleanup_checkpoints.py`: scaffold para limpieza de checkpoints.

## 4. Contrato De Inferencia

### `/health`

Respuesta esperada:

```json
{
  "status": "healthy",
  "service": "aegis_alpha",
  "model_loaded": false,
  "model_name": "aegis_alpha",
  "model_version": "0.1.0"
}
```

### `/ml-v2/predict`

Respuesta legacy compatible:

```json
{
  "symbol": "ETHUSDT",
  "long_prob": 0.005,
  "short_prob": 0.005,
  "neutral_prob": 0.99,
  "close_prob": 0.0,
  "consensus_level": 0.99,
  "meta_verdict": "AEGIS_ALPHA_IDLE",
  "smart_leverage": 5.0,
  "features": {
    "cvd_z": 0.0,
    "cvd_slope": 0.0,
    "weakness": 0.0,
    "volatility_z": 0.0
  }
}
```

Mientras no haya champion, la respuesta defensiva es correcta. El bot puede seguir vivo y el risk manager externo no recibe señales falsas.

## 5. Filosofía De Seguridad

Aegis nunca debe operar solo porque existe un modelo. Toda señal debe pasar por gates.

Reglas iniciales:

```text
Si no hay champion cargado -> IDLE
Si top_prob < 0.65 -> IDLE
Si long_short_gap < 0.15 -> IDLE
Si régimen es chop -> exigir top_prob >= 0.72 y gap >= 0.20
Si datos incompletos -> IDLE
```

La inferencia debe preferir falsos negativos antes que falsos positivos. Una señal perdida es aceptable; una señal mala con dinero real no.

## 6. Roadmap Técnico

### Bloque 1: Core

Estado: iniciado.

Completado:

```text
estructura de carpetas
config base
feature builder inicial
regime detector inicial
env determinista inicial
action mask
risk engine
API compatible
PM2 apuntando a Aegis
```

Pendiente:

```text
tests formales con pytest instalado
integración real DB -> feature window en inference
metrics completas de episode
```

### Bloque 2: BC Prudente

Estado: implementado en v0.2.0 como primera base entrenable; walk-forward agregado en v0.2.1.

Completado:

```text
tools/build_bc_dataset.py
bc/labeler.py prudente
bc/train_bc.py supervisado
bc/evaluate_bc.py
tools/evaluate_bc_walkforward.py
models/bc/aegis_bc_prudent.zip
```

Artefactos actuales:

```text
aegis_alpha/data/processed/bc_prudent_dataset.npz
aegis_alpha/models/bc/aegis_bc_prudent.zip
```

Distribución del dataset v0.2.0:

```text
Samples: 231,444
IDLE: 84.0%
LONG: 4.1%
SHORT: 3.9%
CLOSE: 8.0%
```

Resultado de evaluación inicial del BC en ventana de 4,032 pasos:

```text
Balance: $20.65
Net: +$0.65
P95 DD: 6.95%
Max DD: 9.95%
Opens: 190
Direction dominance: 73.7%
Invalid actions: 70
Fees: $5.14
```

Lectura: el BC v0.2.0 sirve como baseline de investigación porque evita dominancia extrema y no nace de random. El walk-forward v0.2.1 muestra que no es apto para promoción: pierde en múltiples ventanas históricas por frecuencia/fees/ruido. No es señal productiva todavía.

### Bloque 3: PPO Sobre BC

Pendiente:

```text
train_challenger.py funcional
trainer.py funcional
carga de BC como Challenger B
linaje Champion vs BC
```

### Bloque 4: Coliseo Por Regímenes

Pendiente:

```text
ventanas bull/bear/chop/high_vol
worst-window metrics
promotion rules finales
reportes persistentes
```

### Bloque 5: Shadow Trading

Pendiente:

```text
guardar señales Aegis
medir MFE/MAE real
calibrar SignalQ
comparar señales con ejecución TS
```

## 7. Protocolo De Actualización Del Whitepaper

Este archivo debe actualizarse cuando ocurra cualquiera de estos cambios:

```text
se crea, elimina o renombra una carpeta
se crea, elimina o renombra un archivo importante
cambia un endpoint de inferencia
cambia el shape de respuesta consumido por TypeScript
cambia el risk budget
cambia action masking
cambia reward
cambia el set de features
cambian reglas de BC labeler
cambian reglas de Coliseo/Survivor
cambia el flujo de PM2/ecosystem
cambia la ruta esperada de modelos
```

Regla operativa:

```text
Todo PR o cambio funcional en Aegis debe incluir actualización de este whitepaper si altera arquitectura, contrato o comportamiento.
```

## 8. Decisión De Diseño Actual

Phantom queda congelado como experimento histórico.

Aegis Alpha queda como nueva línea:

```text
Aegis Alpha =
Behavior Cloning prudente
+ PPO refinador
+ Coliseo Survivor
+ SignalQ
+ Inference API segura
+ TypeScript risk manager
```

El sistema actual ya es seguro para runtime porque responde `IDLE` sin champion. La próxima prioridad es construir el primer BC prudente real y evaluarlo antes de cualquier entrenamiento PPO.

## 9. Changelog Vivo

### v0.2.0 - Behavior Cloning Prudente

Fecha: 2026-05-02.

Cambios:

```text
Se implementó el pipeline BC completo:
- tools/build_bc_dataset.py
- bc/labeler.py
- bc/train_bc.py
- bc/evaluate_bc.py
```

`tools/build_bc_dataset.py` genera el dataset supervisado desde candles ETHUSDT 5m, construye features 21D, calcula régimen, simula estado básico de cuenta/posición, etiqueta acciones prudentes y guarda metadata de evaluación futura:

```text
market: (N, 64, 21)
account: (N, 6)
actions: IDLE/LONG/SHORT/CLOSE
timestamp
price
regime
future_return_3/6/12
mfe_12
mae_12
```

El builder aplica downsampling controlado de IDLE para conservar labels válidos por cooldown sin dejar que el entrenamiento colapse a IDLE puro. El target actual es `target_idle_pct=0.84`.

`bc/labeler.py` define el maestro heurístico prudente. Reglas principales:

```text
si no hay cooldown suficiente -> IDLE
si está flat -> LONG/SHORT solo con tendencia, eficiencia, ADX y CVD no contradictorio
si está en posición -> CLOSE por stop controlado, invalidación, take-profit debilitado o max hold
si hay duda -> IDLE
```

`bc/train_bc.py` entrena una política supervisada compatible con `AegisTransformerExtractor` y guarda el modelo SB3 PPO inicializado por BC. El mejor resultado actual se entrenó sin oversampling (`sampler_power=0.0`) usando pesos de clase suaves.

`bc/evaluate_bc.py` ejecuta el modelo en `AegisEnv` y reporta:

```text
balance
net
p95_dd
max_dd
opens
long_opens
short_opens
manual_closes
invalid_actions
fees
avg_hold_steps
avg_flat_steps
direction_dominance
top_prob_avg
signals_gt_65_pct
long_short_gap_avg
last_regime
```

Cambio de arquitectura importante: el vector `account (6,)` ahora es explícito para cooldown y dirección:

```text
0 equity_ratio
1 signed_exposure
2 roe
3 position_side
4 hold_steps / 288
5 flat_steps / 288
```

Razón: la versión anterior no exponía `flat_steps`; el BC no podía aprender cuándo una entrada era inválida por cooldown. Este cambio redujo fuertemente acciones inválidas durante evaluación.

Estado de producción:

```text
No hay champion Aegis activo.
La API sigue respondiendo IDLE defensivo.
El bot TypeScript no fue modificado.
El modelo BC queda como semilla para PPO, no como señal productiva.
```

### v0.2.1 - BC Walk-Forward Evaluation

Fecha: 2026-05-02.

Cambios:

```text
Se agregó tools/evaluate_bc_walkforward.py.
Evalúa el BC prudente sobre múltiples ventanas históricas de 4,032 steps.
Incluye ventanas recientes, aleatorias y seleccionadas por régimen disponible.
Guarda reportes JSON en logs/coliseum/.
No modifica inference.
No promueve champion automáticamente.
```

Archivo de reporte generado:

```text
aegis_alpha/logs/coliseum/bc_walkforward_20260502T221905Z.json
```

Cada ventana reporta:

```text
start/end timestamp
regime dominante
balance/net
p95_dd/max_dd
opens/long_opens/short_opens
manual_closes
invalid_actions
fees
avg_hold_steps/avg_flat_steps
direction_dominance
top_prob_avg
signals_gt_65_pct
long_short_gap_avg
```

Resumen del primer walk-forward:

```text
window_count: 14
median_balance: $16.23
p25_balance: $15.42
worst_balance: $13.44
median_p95_dd: 21.55%
worst_max_dd: 38.83%
median_fees: $4.25
dominance_median: 67.36%
dominance_max: 77.89%
dominance_gt_80_pct: 0.0%
dominance_gt_95_pct: 0.0%
```

Lectura:

```text
El BC v0.2.0 no pasa walk-forward.
La direccionalidad es sana, pero el modelo sobreopera y pierde por fees/ruido en ventanas históricas.
Se conserva como baseline de investigación, no como champion ni señal productiva.
El siguiente ajuste debe atacar frecuencia, hold/flat efectivo, SignalQ/gap y selección de entradas antes de PPO.
```

### v0.2.2 - Selective BC Experiments

Fecha: 2026-05-02.

Cambios:

```text
Se agregaron variantes offline de labeler:
- conservative
- edge
- ultra

Se agregaron filtros opcionales de future MFE/MAE para construir labels LONG/SHORT.
Estos filtros solo existen en tools/build_bc_dataset.py y quedan prohibidos en inference.
No se tocó inference.
No se promovió champion.
```

Artefactos generados:

```text
aegis_alpha/data/processed/bc_conservative_dataset.npz
aegis_alpha/data/processed/bc_edge_dataset.npz
aegis_alpha/data/processed/bc_ultra_dataset.npz

aegis_alpha/models/bc/aegis_bc_conservative.zip
aegis_alpha/models/bc/aegis_bc_edge.zip
aegis_alpha/models/bc/aegis_bc_ultra.zip

aegis_alpha/logs/coliseum/bc_walkforward_conservative_20260502T235004Z.json
aegis_alpha/logs/coliseum/bc_walkforward_edge_20260502T235052Z.json
aegis_alpha/logs/coliseum/bc_walkforward_ultra_20260502T235142Z.json
aegis_alpha/logs/coliseum/bc_walkforward_comparison_20260502T235142Z.json
```

`tools/evaluate_bc_walkforward.py` ahora agrega métricas de calidad de trade:

```text
avg_return_per_trade
win_rate
profit_factor
avg_win
avg_loss
fees_per_trade
entry_count
entry_mfe_avg/median
entry_mae_avg/median
```

Resumen comparativo walk-forward:

```text
conservative:
  median_balance: $16.36
  p25_balance: $15.08
  worst_balance: $12.41
  median_p95_dd: 20.11%
  worst_max_dd: 37.93%
  median_fees: $4.82
  dominance_median: 63.24%
  dominance_gt_80_pct: 7.14%
  median_entry_count: 212.5
  median_avg_return_per_trade: -0.0747%
  median_win_rate: 36.90%
  median_profit_factor: 0.58

edge:
  median_balance: $15.52
  p25_balance: $14.38
  worst_balance: $12.85
  median_p95_dd: 24.67%
  worst_max_dd: 35.87%
  median_fees: $4.96
  dominance_median: 75.00%
  dominance_gt_80_pct: 21.43%
  median_entry_count: 224.0
  median_avg_return_per_trade: -0.0889%
  median_win_rate: 35.49%
  median_profit_factor: 0.52

ultra:
  median_balance: $15.36
  p25_balance: $13.61
  worst_balance: $11.13
  median_p95_dd: 22.93%
  worst_max_dd: 45.56%
  median_fees: $4.93
  dominance_median: 100.00%
  dominance_gt_80_pct: 100.00%
  median_entry_count: 224.0
  median_avg_return_per_trade: -0.0937%
  median_win_rate: 33.71%
  median_profit_factor: 0.49
```

Lectura:

```text
El ranking compuesto favorece conservative por median_balance, p25_balance, worst_balance, DD, fees y trade quality.
Ninguna variante pasa como champion: las tres mantienen retorno medio por trade negativo y profit_factor < 1.
Ultra queda descartado como candidato inmediato por dominance_median 100% y worst_max_dd 45.56%.
Edge mejora tamaño de dataset, pero sobreopera y pierde contra conservative en balance, drawdown y calidad de trade.
Conservative queda como mejor baseline offline v0.2.2, no como señal productiva.
```

### v0.3.0 - Edge Model Supervisado

Fecha: 2026-05-03.

Cambios:

```text
Se agregó pipeline offline de edge supervisado:
- tools/build_edge_dataset.py
- edge/train_edge_model.py
- edge/evaluate_edge_deciles.py

El modelo no participa en inference.
No se promueve champion.
No se ejecuta PPO.
```

Artefactos generados:

```text
aegis_alpha/data/processed/edge_dataset_v030.npz
aegis_alpha/models/edge/aegis_edge_model_v030.joblib
aegis_alpha/logs/edge/edge_train_report_v030.json
aegis_alpha/logs/edge/edge_decile_report_v030.json
aegis_alpha/logs/edge/edge_decile_report_v030_fine.json
```

Dataset:

```text
samples: 437,341
features: 168 compact window features
horizon: 12 velas
eval_horizon: 24 velas
profit_threshold: 0.30%
risk_threshold: 0.30%
round_trip_fee: commission + slippage por entrada/salida

LONG good: 118,203 (27.03%)
SHORT good: 115,324 (26.37%)
NO TRADE: 203,814 (46.60%)
avg long net return: -0.0978%
avg short net return: -0.1022%
```

Modelo:

```text
sklearn HistGradientBoostingClassifier para LONG success
sklearn HistGradientBoostingClassifier para SHORT success
sklearn HistGradientBoostingRegressor para LONG net return
sklearn HistGradientBoostingRegressor para SHORT net return
split cronologico: 80% train / 20% holdout
holdout: 87,469 muestras
holdout start: 2025-07-03 01:40:00
holdout end: 2026-05-02 18:40:00
```

Métricas holdout:

```text
LONG classifier:
  roc_auc: 0.5806
  average_precision: 0.3322
  positive_rate: 28.27%

SHORT classifier:
  roc_auc: 0.5789
  average_precision: 0.3249
  positive_rate: 28.00%

baseline holdout:
  long_avg_return_after_fees: -0.0989%
  long_win_rate: 40.89%
  long_profit_factor: 0.65
  short_avg_return_after_fees: -0.1011%
  short_win_rate: 39.56%
  short_profit_factor: 0.65
```

Evaluación por deciles:

```text
Por probabilidad de success:
  LONG top 10%:
    count: 8,747
    avg_return_after_fees: -0.0900%
    win_rate: 48.18%
    profit_factor: 0.74

  SHORT top 10%:
    count: 8,747
    avg_return_after_fees: -0.1243%
    win_rate: 45.82%
    profit_factor: 0.66

Por retorno esperado LONG:
  top 5%:
    count: 4,374
    avg_return_after_fees: -0.0632%
    win_rate: 49.27%
    profit_factor: 0.84

  top 2%:
    count: 1,750
    avg_return_after_fees: +0.0528%
    win_rate: 52.57%
    profit_factor: 1.13

  top 1%:
    count: 875
    avg_return_after_fees: +0.1133%
    win_rate: 53.60%
    profit_factor: 1.25

  top 0.5%:
    count: 438
    avg_return_after_fees: +0.2061%
    win_rate: 54.11%
    profit_factor: 1.40

Por retorno esperado SHORT:
  top 10% a top 0.5% siguen negativos.
```

Lectura:

```text
El Edge Model v0.3.0 encuentra señal predictiva débil, no suficiente como política general.
Los clasificadores de success no producen edge rentable en top decile.
El regresor de retorno esperado LONG sí aísla una cola pequeña rentable entre top 0.5% y top 2%.
SHORT no muestra edge aprovechable.
El grid simple LONG/SHORT por probabilidad no pasa gate mínimo; las únicas slices viables son LONG por expected_return.
Esto no habilita champion ni PPO todavía, pero sí justifica v0.3.1 orientado a LONG-only edge-gated BC o validación walk-forward específica de esa cola.
```

### v0.3.1 - Long Edge Gate Validation

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_gate.py.
Carga aegis_alpha/models/edge/aegis_edge_model_v030.joblib.
Evalúa política secuencial LONG-only sobre ventanas walk-forward.
No usa SHORT como entrada.
Respeta RiskConfig Aegis:
  leverage 5x
  position_fraction 0.25
  hard_stop_roe 15%
  min_hold_steps 6
  min_flat_steps 12
No toca inference.
No promueve champion.
```

Política evaluada:

```text
flat:
  LONG si expected_return_long >= gate

en posición:
  CLOSE por hard_stop
  CLOSE por take_profit simple
  CLOSE por max_hold
  CLOSE por deterioro de edge
  si no, IDLE
```

Gates:

```text
top 2.0% expected_return_long threshold: 0.0007201
top 1.0% expected_return_long threshold: 0.0010184
top 0.5% expected_return_long threshold: 0.0014292
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_gate_validation_20260503T002808Z.json
```

Resumen walk-forward por gate:

```text
top 2%:
  median_balance: $20.15
  p25_balance: $19.41
  worst_balance: $17.90
  median_pf: 1.71
  p25_pf: 0.69
  profitable_window_pct: 57.14%
  median_trades: 11.0
  worst_max_dd: 16.10%
  median_avg_return_per_trade: +0.1001%
  median_exposure_time: 2.62%

top 1%:
  median_balance: $20.08
  p25_balance: $19.61
  worst_balance: $17.70
  median_pf: 1.51
  p25_pf: 0.54
  profitable_window_pct: 57.14%
  median_trades: 4.0
  worst_max_dd: 15.39%
  median_avg_return_per_trade: +0.1060%
  median_exposure_time: 1.10%

top 0.5%:
  median_balance: $20.00
  p25_balance: $20.00
  worst_balance: $18.55
  median_pf: 0.00
  p25_pf: 0.00
  profitable_window_pct: 35.71%
  median_trades: 1.0
  worst_max_dd: 11.45%
  median_avg_return_per_trade: 0.0000%
  median_exposure_time: 0.21%
```

Lectura:

```text
El gate top 2% es el mejor candidato por rentabilidad y cantidad de trades.
Top 1% reduce exposición y trades, pero no mejora robustez: peor worst_balance y p25_pf.
Top 0.5% queda demasiado escaso; muchas ventanas no operan y la mediana PF colapsa a 0.
La señal LONG edge-gated supera claramente al BC heurístico en mediana, fees y frecuencia.
Todavía no pasa como champion: p25_pf < 1, profitable_window_pct solo 57.14% y existe una ventana con DD ~16%.
Siguiente paso razonable: v0.3.2 con validación de estabilidad por régimen y calibración de salida, antes de BC/PPO.
```

### v0.3.2 - Regime Stability + Exit Calibration

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_gate_grid.py.
Usa el gate principal top 2% expected_return_long.
Mantiene política LONG-only.
No usa SHORT.
No toca inference.
No promueve champion.
```

Grid evaluado:

```text
max_hold: 6, 12, 18, 24
take_profit: 0.15%, 0.25%, 0.35%, 0.50%
stop_loss: -0.15%, -0.25%, -0.35%
edge_deterioration:
  expected_return_long < 0
  expected_return_long < top 5%
  expected_return_long < top 10%

total configs: 144
ventanas por config: 14
entry gate top 2% threshold: 0.00072011
```

Reportes generados:

```text
aegis_alpha/logs/edge/long_edge_gate_grid_20260503T003602Z.json
aegis_alpha/logs/edge/long_edge_regime_report_20260503T003602Z.json
```

Baseline de comparación v0.3.1 top 2%:

```text
p25_pf: 0.6946
worst_balance: $17.90
profitable_window_pct: 57.14%
median_trades: 11.0
```

Mejor configuración del grid:

```text
config_id: mh18_tp0p50_sl0p25_edge_lt_top10
max_hold: 18
take_profit: 0.50%
stop_loss: -0.25%
edge_deterioration: expected_return_long < top 10%

median_balance: $20.05
p25_balance: $19.40
worst_balance: $17.62
median_pf: 1.06
p25_pf: 0.65
profitable_window_pct: 57.14%
median_trades: 11.5
worst_max_dd: 16.43%
median_avg_return_per_trade: +0.0161%
median_exposure_time: 1.80%
```

Resultado del criterio de éxito:

```text
configs que mejoran simultáneamente:
  p25_pf
  worst_balance
  profitable_window_pct
  sin reducir demasiado trades

resultado: 0 / 144
```

Top configs por ranking interno:

```text
1. mh18_tp0p50_sl0p25_edge_lt_top10
   p25_pf: 0.65
   worst_balance: $17.62
   profitable_window_pct: 57.14%
   median_trades: 11.5

2. mh24_tp0p50_sl0p25_edge_lt_top10
   p25_pf: 0.65
   worst_balance: $17.62
   profitable_window_pct: 50.00%
   median_trades: 11.5

3. mh18_tp0p50_sl0p35_edge_lt_0
   p25_pf: 0.64
   worst_balance: $17.82
   profitable_window_pct: 50.00%
   median_trades: 11.5
```

Reporte por régimen de la mejor config:

```text
trend_down:
  trade_count: 130
  avg_return: -0.1565%
  win_rate: 47.69%
  profit_factor: 0.64
  drawdown_contribution: 76.38%

mixed:
  trade_count: 47
  avg_return: +0.0198%
  win_rate: 46.81%
  profit_factor: 1.07
  drawdown_contribution: 17.46%

chop:
  trade_count: 7
  avg_return: +0.1775%
  win_rate: 57.14%
  profit_factor: 1.93
  drawdown_contribution: 1.79%

high_vol:
  trade_count: 2
  avg_return: +0.0425%
  win_rate: 50.00%
  profit_factor: 1.10
  drawdown_contribution: 1.12%

trend_up:
  trade_count: 7
  avg_return: -0.0119%
  win_rate: 42.86%
  profit_factor: 0.97
  drawdown_contribution: 3.25%

compression:
  trade_count: 0
```

Lectura:

```text
El criterio de éxito v0.3.2 no se cumplió.
La calibración de salidas no mejoró robustez frente al v0.3.1 top 2%.
La mejor config reduce la mediana de PF y empeora worst_balance.
El problema principal es régimen trend_down: concentra 67% de los trades de la mejor config y 76% de la contribución de pérdidas.
El edge LONG existe, pero no es estable a través de regímenes.
La siguiente mejora no debe ser más grid de salidas; debe ser regime gating explícito para bloquear LONG en trend_down y revalidar.
```

### v0.3.3 - Regime-Gated Long Edge Validation

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_regime_gate.py.
Usa baseline v0.3.1:
  LONG-only
  top 2% expected_return_long
  threshold 0.00072011
No usa SHORT.
No toca inference.
No promueve champion.
```

Variantes evaluadas:

```text
A) allow_all_except_trend_down
B) allow_trend_up_mixed_chop_high_vol
C) allow_mixed_chop_high_vol
D) allow_mixed_chop
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_regime_gate_20260503T004330Z.json
```

Baseline v0.3.1 top 2%:

```text
p25_pf: 0.6946
worst_balance: $17.90
profitable_window_pct: 57.14%
median_trades: 11.0
```

Resultados por variante:

```text
allow_mixed_chop_high_vol:
  median_balance: $20.19
  p25_balance: $20.06
  worst_balance: $19.76
  median_pf: 8.76
  p25_pf: 1.25
  profitable_window_pct: 78.57%
  median_trades: 4.5
  worst_max_dd: 8.27%
  median_avg_return_per_trade: +0.1912%
  median_exposure_time: 0.94%
  allowed_count: 80
  blocked_count: 533
  skipped_trend_down_count: 519

allow_all_except_trend_down:
  median_balance: $20.24
  p25_balance: $20.01
  worst_balance: $19.88
  median_pf: 8.54
  p25_pf: 1.13
  profitable_window_pct: 78.57%
  median_trades: 4.5
  worst_max_dd: 8.27%
  median_avg_return_per_trade: +0.2066%
  median_exposure_time: 0.97%
  allowed_count: 83
  blocked_count: 520
  skipped_trend_down_count: 520

allow_trend_up_mixed_chop_high_vol:
  median_balance: $20.24
  p25_balance: $20.01
  worst_balance: $19.88
  median_pf: 8.54
  p25_pf: 1.13
  profitable_window_pct: 78.57%
  median_trades: 4.5
  worst_max_dd: 8.27%
  median_avg_return_per_trade: +0.2066%
  median_exposure_time: 0.97%
  allowed_count: 83
  blocked_count: 520
  skipped_trend_down_count: 520

allow_mixed_chop:
  median_balance: $20.19
  p25_balance: $20.01
  worst_balance: $19.76
  median_pf: 121.26
  p25_pf: 1.12
  profitable_window_pct: 78.57%
  median_trades: 4.5
  worst_max_dd: 4.13%
  median_avg_return_per_trade: +0.1912%
  median_exposure_time: 0.94%
  allowed_count: 78
  blocked_count: 557
  skipped_trend_down_count: 522
```

Criterio de éxito:

```text
Mejorar contra v0.3.1 top 2%:
  p25_pf
  worst_balance
  profitable_window_pct

resultado: 4 / 4 variantes cumplen.
```

Lectura:

```text
Regime gating resuelve el principal fallo detectado en v0.3.2.
Bloquear trend_down mejora p25_pf de 0.69 a 1.13-1.25 y worst_balance de $17.90 a $19.76-$19.88.
El costo es una reducción fuerte de frecuencia: median_trades baja de 11.0 a 4.5 por ventana.
La mejor variante por ranking es allow_mixed_chop_high_vol porque tiene el mayor p25_pf.
La variante más conservadora por drawdown es allow_mixed_chop, con worst_max_dd 4.13%, pero menos allowed_count.
Esto todavía no es champion: hay pocas operaciones por ventana y debe validarse con más ventanas/splits antes de producción.
Pero v0.3.3 sí confirma que el edge LONG era regime-dependent, no inexistente.
```

### v0.3.4 - Long Edge Robustness Expansion

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_robustness.py.
Usa la mejor variante v0.3.3:
  LONG-only
  allowed_regimes: mixed, chop, high_vol
No usa SHORT.
No toca inference.
No promueve champion.
```

Expansión evaluada:

```text
ventanas: 96
fuentes:
  recientes
  aleatorias
  por régimen
  consecutivas / no solapadas

thresholds:
  top 1.5%
  top 2.0%
  top 2.5%
  top 3.0%

fee model:
  1.0x
  1.5x
  2.0x
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_robustness_20260503T005306Z.json
```

Criterio de éxito:

```text
window_count >= 50
profitable_window_pct >= 65%
p25_pf >= 1.0
worst_balance >= $19.0
worst_max_dd <= 12%
no colapsar con fee model 1.5x
```

Resultado:

```text
passes_any: false
passes_fee_1_5x: false
```

Mejor configuración por ranking:

```text
config_id: top_3pct_fee1x
window_count: 96
median_balance: $20.30
p25_balance: $19.98
worst_balance: $15.72
median_pf: 3.08
p25_pf: 0.85
profitable_window_pct: 70.83%
median_trades: 9.5
median_trades_per_month: 20.65
worst_max_dd: 24.95%
median_avg_return_per_trade: +0.2038%
median_exposure_time: 2.26%
```

Mejor configuración con fee 1.5x:

```text
config_id: top_3pct_fee1.5x
median_balance: $20.20
p25_balance: $19.88
worst_balance: $15.36
median_pf: 2.50
p25_pf: 0.64
profitable_window_pct: 65.63%
median_trades: 9.5
median_trades_per_month: 20.65
worst_max_dd: 26.16%
median_avg_return_per_trade: +0.1537%
median_exposure_time: 2.26%
```

Top 2% baseline-style bajo robustez:

```text
top_2pct_fee1x:
  median_balance: $20.24
  p25_balance: $20.00
  worst_balance: $15.43
  median_pf: 4.89
  p25_pf: 0.81
  profitable_window_pct: 70.83%
  median_trades: 5.0
  median_trades_per_month: 10.87
  worst_max_dd: 24.42%

top_2pct_fee1.5x:
  median_balance: $20.18
  p25_balance: $20.00
  worst_balance: $15.17
  median_pf: 3.84
  p25_pf: 0.58
  profitable_window_pct: 68.75%
  median_trades: 5.0
  median_trades_per_month: 10.87
  worst_max_dd: 25.46%
```

Lectura:

```text
El resultado v0.3.4 no pasa robustez.
La mediana sigue siendo positiva, pero la cola mala reaparece al ampliar a 96 ventanas.
El p25_pf queda por debajo de 1 en todos los thresholds y costos.
El worst_balance cae a $15-$16 y worst_max_dd sube a 21%-27%.
El modelo tampoco cumple la condición de no colapsar con costos 1.5x.
La señal v0.3.3 era real en el conjunto pequeño, pero no suficientemente robusta para avanzar a champion o PPO.
Siguiente trabajo: ampliar validación temporal por splits purgados o rediseñar el Edge Model con mejores features/targets; no promover ni conectar a inference.
```

### v0.3.5 - Risk Guard Robustness

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_risk_guard.py.
Usa la mejor base de v0.3.4:
  LONG-only
  expected_return_long top 3%
  allowed_regimes: mixed, chop, high_vol
  fee multipliers: 1.0x y 1.5x
No toca inference.
No promueve champion.
```

Guards evaluados:

```text
max_window_loss_pct: 3%, 5%, 7%
pause_after_loss_steps: 12, 24, 48
pause_after_2_losses_steps: 48, 96
max_trades_per_day: 1, 2, 3
configs: 108
ventanas: 96
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_risk_guard_20260503T012735Z.json
```

Baseline v0.3.4 top_3pct_fee1x:

```text
p25_pf: 0.85
worst_balance: $15.72
worst_max_dd: 24.95%
profitable_window_pct: 70.83%
```

Criterio de éxito:

```text
worst_balance >= $18.50
worst_max_dd <= 15%
p25_pf >= 1.0
profitable_window_pct >= 65%
median_trades >= 5
```

Resultado:

```text
passes_any: false
```

Mejor configuración:

```text
config_id: loss7_pause48_pause2_48_maxday3_fee1x
fee_multiplier: 1.0x
max_window_loss_pct: 7%
pause_after_loss_steps: 48
pause_after_2_losses_steps: 48
max_trades_per_day: 3

median_balance: $20.22
p25_balance: $19.89
worst_balance: $19.19
median_pf: 3.01
p25_pf: 0.88
profitable_window_pct: 66.67%
median_trades: 8.0
median_trades_per_month: 17.39
worst_max_dd: 9.29%
median_avg_return_per_trade: +0.1880%
median_exposure_time: 1.87%
skipped_by_guard: 7,383
stopped_trade_count: 0
```

Comparación contra v0.3.4 top_3pct_fee1x:

```text
p25_pf: 0.88 vs 0.85
worst_balance: $19.19 vs $15.72
worst_max_dd: 9.29% vs 24.95%
profitable_window_pct: 66.67% vs 70.83%
median_trades: 8.0 vs 9.5
```

Mejores variantes con max_trades_per_day=1:

```text
loss5_pause12_pause2_48_maxday1_fee1x:
  worst_balance: $18.59
  worst_max_dd: 8.22%
  p25_pf: 0.82
  profitable_window_pct: 71.88%
  median_trades: 5.5

loss3_*_maxday1_fee1.5x:
  worst_balance: $19.02
  worst_max_dd: 5.4%
  p25_pf: 0.57
  profitable_window_pct: 66.67%
  median_trades: 5.0
```

Lectura:

```text
Risk guards sí reducen la cola de pérdida: worst_balance mejora de $15.72 a $19.19 y worst_max_dd baja de 24.95% a 9.29%.
El costo es menor profitable_window_pct y algo menos de frecuencia.
El criterio completo no se cumple porque p25_pf no llega a 1.0; el mejor queda en 0.88.
Con fee 1.5x la robustez de PF empeora más, aunque algunos guards mantienen worst_balance y DD controlados.
La señal LONG edge-gated necesita mejorar calidad predictiva, no solo controles de riesgo.
No hay champion; no hay PPO; no hay inference.
```

## v0.4.7 - Edge Deterioration Guard

Se probó un guard específico para cierres por deterioro del edge sobre el candidate congelado de v0.4.2, manteniendo:

```text
LONG-only
expected_return_long top 3%
allowed regimes: mixed, chop, high_vol
risk guard: loss7_pause48_pause2_48_maxday3
dynamic sizing: full 0.25, reduced 0.125, meta_high 0.60
fee multipliers: 1.0x y 1.25x
```

Mejor variante:

```text
A_pause48_after_loss @ fee 1.0x
median_balance: 20.1588
p25_balance: 19.9928
worst_balance: 18.3378
median_pf: 1.8815
p25_pf: 0.7638
profitable_window_pct: 69.44%
median_trades: 7.0
trades_per_month: 15.22
worst_max_dd: 8.32%
edge_deterioration_closes: 1127
losing_edge_deterioration_closes: 336
skipped_after_deterioration: 1254
full_size_trades: 602
reduced_size_trades: 666
```

Conclusión:

```text
El guard de deterioro sí recorta entradas después de cierres malos, pero no arregla la cola OOS.
La mejor variante sigue por debajo de worst_balance >= 19.00 y worst_max_dd <= 8%, y también falla profitable_window_pct >= 70%.
Las variantes con pausas más largas empeoran peor-balance y drawdown; las reglas de close_ratio tampoco levantan p25_pf.
La conclusión práctica es que el problema sigue siendo la señal de entrada y no solo la disciplina de salida.
```

## v0.5.0 - Signal Lab + Multi-Horizon Edge Research

La rama v0.4 se congeló como `RESEARCH_SIGNAL` porque la estrategia Long Edge mejoró la mediana y la disciplina, pero no aguantó la cola OOS. El problema quedó concentrado en `worst_balance`, `p25_pf` y fragilidad bajo fee stress. Por eso no se promovió champion.

Signal Lab abre una nueva fase: investigar familias de señales y horizontes en vez de seguir reparando una sola política.

Se añadieron:

```text
signals/common.py
signals/horizon_targets.py
signals/signal_registry.py
tools/build_signal_lab_dataset.py
signals/train_signal_models.py
signals/evaluate_signal_deciles.py
tools/evaluate_signal_combinations.py
```

Horizontes investigados:

```text
h6, h12, h24, h48
```

Se entrenaron modelos para:

```text
long_edge_h6/h12/h24/h48
short_edge_h6/h12/h24/h48
long_failure_risk_h12/h24/h48
```

Hallazgos iniciales:

```text
long_edge_h48 mostró el mejor top-3% PF en holdout, pero la cola OOS siguió débil.
La combinación long_edge_h12 top 3% dio el mejor OOS de la primera tanda, con p25_pf > 1.0, pero worst_balance y worst_max_dd quedaron fuera de criterio.
Ninguna combinación pasó como champion.
```

Estado final:

```text
v0.5.0 es investigación, no producción.
La rama Long Edge v0.4 queda congelada como benchmark de research signal.
No hay señales activas.
No hay champion nuevo.
```

## v0.5.1 - H12/H48 Failure Analysis + Tail-Risk Targets

Fecha: 2026-05-04.

Se agregó una capa de análisis de cola sobre Signal Lab v0.5.0. El objetivo fue explicar por qué `A_long_edge_h12_top3` tenía buen OOS medio pero fallaba en `worst_balance` y `worst_max_dd`, y probar filtros tail-risk más específicos sin tocar inference, sin promover champion y sin ejecutar PPO.

Artefactos:

```text
aegis_alpha/tools/analyze_signal_failures.py
aegis_alpha/tools/evaluate_horizon_agreement.py
aegis_alpha/signals/train_tail_risk_models.py
aegis_alpha/logs/signals/signal_failure_analysis_v051.json
aegis_alpha/logs/signals/tail_risk_train_v051.json
aegis_alpha/logs/signals/horizon_agreement_report_v051.json
aegis_alpha/models/signals/aegis_long_tail_risk_h12_v051.joblib
aegis_alpha/models/signals/aegis_long_tail_risk_h24_v051.joblib
aegis_alpha/models/signals/aegis_long_tail_risk_h48_v051.joblib
```

Targets nuevos:

```text
tail_loss_gt_0p20
tail_loss_gt_0p35
mae_gt_mfe
edge_deterioration_loss
regime_shift_loss
```

Entrenamiento tail-risk:

```text
long_tail_risk_h12:
  roc_auc: 0.553
  average_precision: 0.432
  positive_rate: 39.56%

long_tail_risk_h24:
  roc_auc: 0.543
  average_precision: 0.450
  positive_rate: 42.14%

long_tail_risk_h48:
  roc_auc: 0.526
  average_precision: 0.455
  positive_rate: 43.74%
```

Failure analysis sobre `A_long_edge_h12_top3`, `long_edge_h48_top3` y `H_ensemble_h12h24_risk50`:

```text
total_trades: 661
loss_count: 339

losses_by_combo:
  A_long_edge_h12_top3: 173
  long_edge_h48_top3: 103
  H_ensemble_h12h24_risk50: 63

losses_by_regime:
  mixed: 313
  chop: 20
  high_vol: 6

losses_by_vol_bucket:
  low: 276
  normal: 50
  elevated: 10
  compressed: 3

losses_by_exit_reason:
  edge_deterioration: 245
  hard_stop: 79
  max_hold: 15

h12/h48 agreement false on losses: 233
h48_score > h12_score on losses: 279
```

Evaluación OOS con 144 ventanas y fee 1.0x/1.25x:

```text
baseline A_h12_top3 fee 1.0x:
  median_balance: 20.37
  p25_balance: 20.03
  worst_balance: 18.15
  median_pf: 2.39
  p25_pf: 1.11
  profitable_window_pct: 77.78%
  median_trades: 8.5
  worst_max_dd: 12.35%
  avg_return_per_trade: 0.208%
  exposure: 1.60%

best ranked E_h12_top3_tail50 fee 1.0x:
  median_balance: 20.35
  p25_balance: 20.04
  worst_balance: 18.21
  median_pf: 3.61
  p25_pf: 1.27
  profitable_window_pct: 79.17%
  median_trades: 7.0
  worst_max_dd: 10.90%
  avg_return_per_trade: 0.216%
  exposure: 1.26%

defensive F_h12_top3_tail30 fee 1.0x:
  worst_balance: 19.13
  worst_max_dd: 7.10%
  p25_pf: 0.52
  profitable_window_pct: 70.83%
  median_trades: 5.0
```

Lectura:

```text
El filtro tail_risk_h12 bottom50 mejora contra A_long_edge_h12_top3 en worst_balance, worst_max_dd, p25_pf y profitable_window_pct, conservando median_trades >= 5.
No alcanza el ideal de worst_balance >= 19.00 y worst_max_dd <= 9%; esa zona solo aparece con tail30, pero tail30 destruye p25_pf.
La cola mala sigue dominada por mixed + low volatility y salidas edge_deterioration/hard_stop.
El failure-risk específico mejora la selección OOS como filtro, aunque su AUC aislado sigue flojo.
No hay champion; no hay PPO; no hay inference.
```

## v0.5.2 - Tail-Risk Calibration + Reproducibility Fix

Fecha: 2026-05-04.

v0.5.1 dejó una tensión clara:

```text
tail50:
  mejora p25_pf y profitable_window_pct, pero deja peor cola.

tail30:
  mejora worst_balance y DD, pero rompe p25_pf.
```

v0.5.2 calibró el punto medio entre esos extremos y corrigió reproducibilidad de los modelos tail-risk. El runtime activo usa:

```text
sklearn_version: 1.8.0
```

Se reentrenaron modelos tail-risk con sufijo v052:

```text
aegis_alpha/models/signals/aegis_long_tail_risk_h12_v052.joblib
aegis_alpha/models/signals/aegis_long_tail_risk_h24_v052.joblib
aegis_alpha/models/signals/aegis_long_tail_risk_h48_v052.joblib
```

Reporte:

```text
aegis_alpha/logs/signals/tail_risk_train_v052.json
```

Rangos:

```text
train:
  2022-03-06 05:40:00 -> 2025-07-03 13:55:00
holdout:
  2025-07-03 14:00:00 -> 2026-05-03 10:05:00
```

Métricas holdout:

```text
long_tail_risk_h12:
  roc_auc: 0.553
  average_precision: 0.432
  positive_rate: 39.56%

long_tail_risk_h24:
  roc_auc: 0.543
  average_precision: 0.450
  positive_rate: 42.14%

long_tail_risk_h48:
  roc_auc: 0.526
  average_precision: 0.455
  positive_rate: 43.74%
```

Calibración OOS:

```text
aegis_alpha/tools/evaluate_tail_risk_calibration.py
aegis_alpha/logs/signals/tail_risk_calibration_v052.json
```

Se evaluaron:

```text
hard_tail30/35/40/45/50
dynamic_tail30_50
conservative_tail35_50
ultra_defensive_tail30_45

fees:
  1.0x
  1.25x
```

Mejor balanceado:

```text
config_id: conservative_tail35_50
sizing_mode: conservative_dynamic_sizing
fee_multiplier: 1.0x

tail <= 35%:
  full size 0.25
35% < tail <= 50%:
  reduced size 0.10
tail > 50%:
  no trade

median_balance: 20.29097
p25_balance: 20.02397
worst_balance: 19.20109
median_pf: 3.23979
p25_pf: 1.24341
profitable_window_pct: 77.78%
median_trades: 7.5
trades_per_month: 16.31
worst_max_dd: 6.53%
median_avg_return_per_trade: 0.2167%
exposure_time: 1.40%
full_size_trades: 761
reduced_size_trades: 621
skipped_by_tail_risk: 1185
```

Mejor profit-quality:

```text
config_id: hard_tail45
fee_multiplier: 1.0x
p25_pf: 1.29023
profitable_window_pct: 79.86%
median_balance: 20.37140
worst_balance: 18.40321
worst_max_dd: 9.95%

Lectura:
  gana calidad/rentabilidad, pero no pasa worst_balance ni DD.
```

Mejor defensiva cercana:

```text
config_id: ultra_defensive_tail30_45
fee_multiplier: 1.0x
worst_balance: 19.16292
worst_max_dd: 5.85%
p25_pf: 1.29023
profitable_window_pct: 78.47%
median_trades: 7.0
median_balance: 20.18078

Lectura:
  defensa excelente, pero queda bajo median_balance >= 20.25.
```

Decisión:

```text
conservative_tail35_50 cumple todos los criterios de candidato fuerte:
  median_balance >= 20.25
  p25_balance >= 20.00
  worst_balance >= 19.00
  p25_pf >= 1.0
  profitable_window_pct >= 75%
  median_trades >= 5
  worst_max_dd <= 9%
```

Se congeló candidate offline:

```text
aegis_alpha/models/strategy_candidates/aegis_h12_tail_risk_candidate_v052.json
status: OFFLINE_CANDIDATE_NOT_LIVE
```

Razones para no estar live:

```text
needs_shadow_validation
needs_fee_slippage_live_validation
not_promoted_to_champion
```

Próximos pasos:

```text
Validación shadow sin órdenes reales.
Stress con fees/slippage live.
Reentrenar o exportar long_edge_h12 en el mismo runtime sklearn si se quiere eliminar también el warning del modelo edge v050.
No hay champion; no hay PPO; no hay inference; no hay señales reales activadas.
```

## v0.4.6 - Score Floor + Low-Vol Mixed Block

Se añadió un filtro de score floor sobre el candidate congelado de v0.4.2 y se probó un bloqueo adicional de `mixed + low_vol`, con fee stress `1.0x` y `1.25x` sobre las mismas 144 ventanas OOS.

Variants evaluadas:

```text
A) block meta_score < 0.60
B) block mixed + low_vol
C) A + B
D) C + reduced_size 0.10 for 0.60 <= meta_score < 0.70
```

Mejor resultado:

```text
variant: A_score_floor
fee: 1.0x
median_balance: 20.2621
p25_balance: 19.8487
worst_balance: 16.4425
median_pf: 2.0013
p25_pf: 0.7851
profitable_window_pct: 67.36%
median_trades: 12.0
worst_max_dd: 19.91%
median_avg_return_per_trade: 0.1828%
median_exposure_time: 2.74%
```

Conclusión:

```text
El score floor mejora la calidad media, pero la cola sigue rota.
El bloqueo mixed + low_vol no arregla el peor balance ni el drawdown.
Ninguna variante cumple el criterio OOS: worst_balance >= 19.10, worst_max_dd <= 7%, profitable_window_pct >= 75%, median_trades >= 5 y p25_pf > 0.8786.
La señal sigue concentrando pérdida en trend_down y mixed, con edge_deterioration como salida perdedora dominante.
```

## Aegis Alpha v0.4.1 - Adaptive Meta-Filter / Defensive Mode

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_long_edge_adaptive_meta.py.
Base usada:
  LONG-only
  expected_return_long top 3%
  allowed_regimes: mixed, chop, high_vol
  risk guard: loss7_pause48_pause2_48_maxday3_fee1x
Meta-filter:
  aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib
No toca inference.
No promueve champion.
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_adaptive_meta_20260503T033742Z.json
```

Benchmarks:

```text
v0.3.6:
  p25_pf: 0.8786
  worst_balance: $19.19
  worst_max_dd: 9.29%
  profitable_window_pct: 66.67%
  median_trades: 8.0

v0.4.0 threshold 0.65:
  p25_pf: 2.22
  worst_balance: $18.01
  worst_max_dd: 10.44%
  profitable_window_pct: 79.17%
  median_trades: 4.0
```

Variantes evaluadas:

```text
A_loss_defensive_48:
  p25_pf: 0.8786
  worst_balance: $19.10
  worst_max_dd: 9.29%
  profitable_window_pct: 66.67%
  median_trades: 8.5
  skipped_by_meta: 0

A_loss_defensive_72:
  p25_pf: 0.78
  worst_balance: $18.51
  worst_max_dd: 9.44%
  profitable_window_pct: 68.75%
  median_trades: 8.0
  skipped_by_meta: 95

B_drawdown_defensive_2pct:
  p25_pf: 0.72
  worst_balance: $17.74
  worst_max_dd: 11.72%
  profitable_window_pct: 66.67%
  median_trades: 7.0
  skipped_by_meta: 288

B_drawdown_defensive_3pct:
  p25_pf: 0.73
  worst_balance: $17.78
  worst_max_dd: 11.54%
  profitable_window_pct: 67.71%
  median_trades: 8.0
  skipped_by_meta: 170

C_two_losses_last5:
  p25_pf: 0.76
  worst_balance: $17.72
  worst_max_dd: 11.82%
  profitable_window_pct: 69.79%
  median_trades: 7.0
  skipped_by_meta: 319

D_soft55_defensive65:
  p25_pf: 1.36
  worst_balance: $17.96
  worst_max_dd: 11.05%
  profitable_window_pct: 77.08%
  median_trades: 6.0
  skipped_by_meta: 646

E_dynamic_sizing:
  p25_pf: 0.8786
  worst_balance: $19.12
  worst_max_dd: 6.55%
  profitable_window_pct: 75.00%
  median_trades: 8.5
  reduced_size_trades: 580

D_plus_E_soft_dynamic:
  p25_pf: 1.37
  worst_balance: $18.02
  worst_max_dd: 12.50%
  profitable_window_pct: 77.08%
  median_trades: 6.0
  skipped_by_meta: 502
  reduced_size_trades: 357
```

Criterio v0.4.1:

```text
worst_balance >= $19.00
worst_max_dd <= 10%
median_trades >= 6
profitable_window_pct >= 70%
p25_pf > 0.8786
```

Resultado:

```text
passes_any: false
beats_v036_p25_pf_count: 2
best_by_rank: D_plus_E_soft_dynamic
```

Lectura:

```text
El modo defensivo con meta-filter sí mejora p25_pf cuando se aplica como filtro suave/estricto, pero vuelve a abrir la cola de pérdida.
D_soft55_defensive65 y D_plus_E_soft_dynamic superan p25_pf del benchmark, pero fallan worst_balance y worst_max_dd.
E_dynamic_sizing es la variante más defensiva operacionalmente: mantiene worst_balance sobre $19, baja DD a 6.55%, sube profitable_window_pct a 75% y conserva trades; pero no mejora p25_pf sobre v0.3.6.
La defensa por drawdown y pérdidas recientes no funciona bien: filtra entradas después del daño y reduce PF de cola.
La conclusión es clara: dynamic sizing ayuda a supervivencia, meta-filter ayuda a trade quality, pero combinarlos todavía no resuelve simultáneamente cola y p25_pf.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.4.0 - Long Edge Meta-Filter

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/build_long_edge_candidate_dataset.py.
Se agregó edge/train_meta_filter.py.
Se agregó tools/evaluate_long_edge_meta_filter.py.
Base usada:
  LONG-only
  expected_return_long top 3%
  allowed_regimes: mixed, chop, high_vol
  risk guard: loss7_pause48_pause2_48_maxday3_fee1x
No toca inference.
No promueve champion.
```

Artefactos generados:

```text
aegis_alpha/data/processed/long_edge_candidates_v040.npz
aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib
aegis_alpha/logs/edge/long_edge_candidate_dataset_v040.json
aegis_alpha/logs/edge/long_edge_meta_filter_train_v040.json
aegis_alpha/logs/edge/long_edge_meta_filter_20260503T015840Z.json
```

Dataset de candidatos:

```text
candidates: 982
candidate_win_rate: 71.28%
candidate_avg_return: +0.2256%
features:
  expected_return_long
  long_success_prob
  expected_return_short
  edge_gap
  regime
  volatility/trend/CVD compact features
  simulated_trade_return
  win/loss
  mfe/mae
```

Entrenamiento meta-filter:

```text
model: sklearn.HistGradientBoostingClassifier
target: candidate LONG profitable net
chronological holdout:
  roc_auc: 0.534
  average_precision: 0.604
  positive_rate: 54.47%
```

Benchmark v0.3.6:

```text
p25_pf: 0.8786
worst_balance: $19.19
worst_max_dd: 9.29%
profitable_window_pct: 66.67%
median_trades: 8.0
```

Resultado thresholds:

```text
threshold 0.50:
  median_balance: $20.31
  worst_balance: $17.82
  worst_max_dd: 12.54%
  p25_pf: 1.31
  profitable_window_pct: 76.04%
  median_trades: 7.0

threshold 0.55:
  median_balance: $20.33
  worst_balance: $17.79
  worst_max_dd: 11.88%
  p25_pf: 1.37
  profitable_window_pct: 77.08%
  median_trades: 6.0

threshold 0.60:
  median_balance: $20.27
  worst_balance: $17.78
  worst_max_dd: 11.71%
  p25_pf: 2.14
  profitable_window_pct: 79.17%
  median_trades: 5.5

threshold 0.65:
  median_balance: $20.29
  worst_balance: $18.01
  worst_max_dd: 10.44%
  p25_pf: 2.22
  profitable_window_pct: 79.17%
  median_trades: 4.0

threshold 0.70:
  median_balance: $20.20
  worst_balance: $17.88
  worst_max_dd: 12.75%
  p25_pf: 1.52
  profitable_window_pct: 78.13%
  median_trades: 4.0
```

Criterio v0.4.0:

```text
worst_balance >= $19.00
worst_max_dd <= 10%
median_balance >= $20.10
profitable_window_pct >= 65%
median_trades >= 5
p25_pf >= 0.95, ideal >= 1.0
```

Resultado:

```text
passes_any: false
hits_p25_pf_target_count: 5
hits_p25_pf_ideal_count: 5
best_by_rank: threshold 0.65
```

Lectura:

```text
El meta-filter consigue lo que v0.3.6 no conseguía: p25_pf supera 1.0 en todos los thresholds evaluados.
También mejora profitable_window_pct frente al benchmark, pasando de 66.67% a 76-79%.
Pero no cumple robustez total: worst_balance cae por debajo de $19.00 y worst_max_dd supera 10%.
El threshold 0.60 es el mejor compromiso de frecuencia: p25_pf 2.14 con median_trades 5.5, pero worst_balance $17.78 y DD 11.71%.
El threshold 0.65 es el mejor por ranking de calidad: p25_pf 2.22 y worst_balance $18.01, pero median_trades baja a 4.0 y DD queda en 10.44%.
La señal del meta-filter separa calidad media, pero todavía no controla los eventos de cola.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.3.6 - Risk Guard Fine Tuning + Fee Stress

Fecha: 2026-05-03.

Cambios:

```text
Se extendió tools/evaluate_long_edge_risk_guard.py.
Parte de la mejor configuración v0.3.5:
  LONG-only
  expected_return_long top 3%
  allowed_regimes: mixed, chop, high_vol
  max_window_loss_pct alrededor de 7%
  pause_after_loss_steps alrededor de 48
  pause_after_2_losses_steps alrededor de 48
  max_trades_per_day 3
No toca inference.
No promueve champion.
```

Grid fino evaluado:

```text
max_window_loss_pct: 5%, 6%, 7%, 8%
pause_after_loss_steps: 24, 48, 72, 96
pause_after_2_losses_steps: 48, 96, 144
max_trades_per_day: 2, 3
fee_multiplier: 1.0x, 1.25x, 1.5x
configs: 288
ventanas: 96
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_risk_guard_20260503T013626Z.json
```

Baseline v0.3.5 best:

```text
p25_pf: 0.88
worst_balance: $19.19
worst_max_dd: 9.29%
profitable_window_pct: 66.67%
median_trades: 8.0
```

Criterio v0.3.6:

```text
worst_balance >= $19.00
worst_max_dd <= 10%
median_balance >= $20.10
profitable_window_pct >= 65%
median_trades >= 6
p25_pf >= 0.95, ideal >= 1.0
```

Resultado:

```text
passes_any: false
hits_p25_pf_target_count: 0
hits_p25_pf_ideal_count: 0
```

Mejor configuración:

```text
config_id: loss7_pause48_pause2_48_maxday3_fee1x
fee_multiplier: 1.0x
max_window_loss_pct: 7%
pause_after_loss_steps: 48
pause_after_2_losses_steps: 48
max_trades_per_day: 3

median_balance: $20.22
p25_balance: $19.89
worst_balance: $19.19
median_pf: 3.01
p25_pf: 0.88
profitable_window_pct: 66.67%
median_trades: 8.0
median_trades_per_month: 17.39
worst_max_dd: 9.29%
median_avg_return_per_trade: +0.1880%
median_exposure_time: 1.87%
skipped_by_guard: 7,383
```

Variantes destacadas:

```text
loss8_pause48_pause2_48_maxday3_fee1x:
  resultado equivalente al mejor v0.3.5
  no mejora p25_pf

loss7_pause24_pause2_48_maxday3_fee1x:
  median_balance: $20.26
  worst_balance: $18.95
  p25_pf: 0.86
  profitable_window_pct: 68.75%
  median_trades: 8.0
```

Lectura:

```text
El fine tuning no encontró una variante superior al mejor risk guard v0.3.5.
El control de riesgo se mantiene fuerte: worst_balance queda sobre $19.00 y worst_max_dd bajo 10% en la mejor variante.
El bloqueo real sigue siendo p25_pf: ninguna configuración alcanzó 0.95 y ninguna alcanzó 1.0.
El fee stress 1.25x y 1.5x degrada la calidad del percentil bajo y no produce una configuración robusta superior.
La conclusión se mantiene: los guards reducen cola de pérdida, pero no crean edge estadístico suficiente.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.4.2 - Dynamic Sizing Calibration + Fee Stress

Fecha: 2026-05-03.

Cambios:

```text
Se extendió tools/evaluate_long_edge_adaptive_meta.py.
Se agregó el modo --experiment dynamic-sizing-grid.
Base usada:
  LONG-only
  expected_return_long top 3%
  allowed_regimes: mixed, chop, high_vol
  risk guard: loss7_pause48_pause2_48_maxday3
  full_size: 0.25
Meta-filter:
  aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib
No toca inference.
No promueve champion.
```

Grid evaluado:

```text
reduced_size: 0.05, 0.075, 0.10, 0.125, 0.15
meta_high_threshold: 0.60, 0.65, 0.70
meta_low_threshold: none, 0.50, 0.55
fee_multiplier: 1.0x, 1.25x, 1.5x
configs: 135
ventanas: 96
```

Reporte generado:

```text
aegis_alpha/logs/edge/long_edge_dynamic_sizing_grid_20260503T041147Z.json
```

Benchmark v0.4.1 E_dynamic_sizing:

```text
p25_pf: 0.8786
worst_balance: $19.12
worst_max_dd: 6.55%
profitable_window_pct: 75.00%
median_trades: 8.5
```

Criterio v0.4.2:

```text
Principal:
  worst_balance >= $19.10
  worst_max_dd <= 7%
  profitable_window_pct >= 75%
  median_trades >= 7
  median_balance >= $20.15

Secundario:
  p25_pf >= 0.88
  ideal p25_pf >= 0.95
```

Resultado:

```text
passes_primary_count: 4
hits_p25_pf_secondary_count: 90
hits_p25_pf_ideal_count: 75
primary_plus_secondary_count: 0
```

Mejor configuración por criterio principal:

```text
config_id: full25_reduced0p125_high0p60_lownone_fee1x
fee_multiplier: 1.0x
full_size: 0.25
reduced_size: 0.125
meta_high_threshold: 0.60
meta_low_threshold: none

median_balance: $20.27
p25_balance: $20.02
worst_balance: $19.18
median_pf: 2.93
p25_pf: 0.87857
profitable_window_pct: 76.04%
median_trades: 8.5
median_trades_per_month: 18.48
worst_max_dd: 6.55%
median_avg_return_per_trade: +0.1880%
median_exposure_time: 1.90%
reduced_size_trades: 472
full_size_trades: 508
```

Otras configuraciones que pasan criterio principal:

```text
full25_reduced0p125_high0p60_lownone_fee1p25x:
  fee_multiplier: 1.25x
  median_balance: $20.21
  worst_balance: $19.12
  worst_max_dd: 6.68%
  profitable_window_pct: 75.00%
  median_trades: 8.5
  p25_pf: 0.81

full25_reduced0p125_high0p65_lownone_fee1x:
  median_balance: $20.24
  worst_balance: $19.12
  worst_max_dd: 6.55%
  profitable_window_pct: 75.00%
  median_trades: 8.5
  p25_pf: 0.87857

full25_reduced0p125_high0p70_lownone_fee1x:
  median_balance: $20.16
  worst_balance: $19.11
  worst_max_dd: 5.56%
  profitable_window_pct: 75.00%
  median_trades: 8.5
  p25_pf: 0.87857
```

Lectura:

```text
La calibración confirma que reduced_size 0.125 es el punto más estable para supervivencia.
El mejor resultado mejora ligeramente E_dynamic_sizing en worst_balance, median_balance y profitable_window_pct, manteniendo DD bajo 7%.
El p25_pf queda prácticamente igual al benchmark v0.4.1, pero por valor exacto no alcanza el umbral secundario >= 0.88.
Aplicar meta_low_threshold mejora p25_pf, pero rompe la cola: worst_balance cae a la zona $17.8-$18.5 y DD sube.
Fee 1.25x conserva una configuración que pasa el criterio principal, pero degrada p25_pf a 0.81.
Fee 1.5x no conserva el criterio principal.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.4.3 - Candidate Freeze

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/export_strategy_candidate.py.
Se congeló la mejor configuración v0.4.2 como candidate offline.
No toca inference.
No promueve champion.
```

Candidate exportado:

```text
aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json
```

Contenido congelado:

```text
edge model path: aegis_alpha/models/edge/aegis_edge_model_v030.joblib
meta-filter path: aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib
entry gate: top 3%
allowed regimes: mixed, chop, high_vol
risk guard: loss7_pause48_pause2_48_maxday3
dynamic sizing: full=0.25, reduced=0.125, meta_high=0.60, meta_low=null
status: OFFLINE_CANDIDATE
```

Freeze basis:

```text
best_config_id: full25_reduced0p125_high0p60_lownone_fee1x
median_balance: $20.27
p25_balance: $20.02
worst_balance: $19.18
worst_max_dd: 6.55%
profitable_window_pct: 76.04%
median_trades: 8.5
```

Lectura:

```text
La mejor configuración v0.4.2 se congeló sin cambios funcionales.
Este paso no hace inferencia ni promoción; solo deja un artefacto estable para validación OOS.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.4.4 - Candidate OOS Evaluation

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/evaluate_strategy_candidate_oos.py.
Se evaluó el candidate congelado con nuevas semillas, 144 ventanas y ventanas mensuales consecutivas cuando hubo datos.
Fee multipliers evaluados: 1.0x y 1.25x.
No toca inference.
No promueve champion.
```

Reporte generado:

```text
aegis_alpha/logs/edge/strategy_candidate_oos_20260503T090317Z.json
```

OOS selection:

```text
window_count: 144
seeds: 6101, 7331
sources: monthly + recent + non_overlap + random_seed:6101 + random_seed:7331
```

Benchmark v0.4.2 congelado:

```text
p25_pf: 0.8786
worst_balance: $19.18
worst_max_dd: 6.55%
profitable_window_pct: 76.04%
median_trades: 8.5
```

Resultado OOS:

```text
fee 1.0x:
  median_balance: $20.16
  p25_balance: $19.99
  worst_balance: $18.34
  median_pf: 1.88
  p25_pf: 0.76
  profitable_window_pct: 69.44%
  median_trades: 7.0
  worst_max_dd: 8.32%
  median_avg_return_per_trade: +0.1251%
  exposure_time: 1.43%
  full_size_trades: 602
  reduced_size_trades: 666

fee 1.25x:
  median_balance: $20.14
  p25_balance: $19.99
  worst_balance: $18.32
  median_pf: 1.61
  p25_pf: 0.69
  profitable_window_pct: 68.75%
  median_trades: 6.5
  worst_max_dd: 8.39%
  median_avg_return_per_trade: +0.1001%
  exposure_time: 1.40%
  full_size_trades: 597
  reduced_size_trades: 667
```

Criterio OOS:

```text
worst_balance >= $19.00
worst_max_dd <= 8%
profitable_window_pct >= 70%
median_balance >= $20.10
median_trades >= 5
fee 1.25x worst_balance >= $18.75
```

Resultado:

```text
passes_any: false
best_fee: 1.0x
```

Lectura:

```text
El candidate congelado no sostuvo la cola OOS.
La pérdida principal está en worst_balance y en profitable_window_pct: 1.0x queda en $18.34 y 69.44%, 1.25x queda en $18.32 y 68.75%.
El DD sí queda cerca del umbral, pero no compensa la caída de balance y frecuencia.
El candidate sirve como freeze estable, pero todavía no está listo para champion ni para tocar inference.
No hay champion; no hay PPO; no hay inference.
```

## Aegis Alpha v0.4.5 - OOS Failure Analysis

Fecha: 2026-05-03.

Cambios:

```text
Se agregó tools/analyze_strategy_candidate_failures.py.
Se cargaron:
  models/strategy_candidates/aegis_long_edge_dynamic_v042.json
  logs/edge/strategy_candidate_oos_20260503T090317Z.json
Se re-simularon las 144 ventanas OOS del fee 1.0x para materializar trades detallados.
No toca inference.
No promueve champion.
```

Reporte generado:

```text
aegis_alpha/logs/edge/oos_failure_analysis_20260503T091718Z.json
```

Cola OOS:

```text
top balance failures:
  monthly:2025-02
  random_seed:6101
  monthly:2024-12

top dd failures:
  monthly:2025-02
  random_seed:6101
  non_overlap

top pf failures:
  monthly:2022-12
  random_seed:7331
  non_overlap
```

Resumen de pérdidas:

```text
total_trades: 1268
losses_full_size: 100
losses_reduced_size: 282

losses_by_regime:
  mixed: 347
  chop: 29
  high_vol: 6

losses_by_exit_reason:
  edge_deterioration: 336
  hard_stop: 35
  max_hold: 11

losses_by_score_bucket:
  <0.55: 241
  [0.55,0.60): 41
  [0.60,0.65): 29
  [0.65,0.70): 37
  >=0.70: 34

losses_by_volatility_bucket:
  low: 356
  normal: 15
  high: 6
  compressed: 5
```

Recomendaciones automáticas:

```text
Reducir size o bloquear meta_score < 0.60; ahí se concentra demasiada pérdida.
Apretar salida por edge deterioration o pausar antes de entrar en condiciones frágiles.
```

Lectura:

```text
La peor cola OOS no está concentrada en high_vol sino en mixed con vol_bucket low.
El meta_score bajo 0.60 domina la pérdida, y edge_deterioration es la salida perdedora principal.
El tamaño reducido ayuda, pero no elimina la cola; la señal sigue entrando demasiado en estados de baja calidad.
La conclusión práctica es que la política congelada necesita un bloqueador más fuerte en score bajo y una gestión de salida más temprana cuando el edge se degrada.
No hay champion; no hay PPO; no hay inference.
```
