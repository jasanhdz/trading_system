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
