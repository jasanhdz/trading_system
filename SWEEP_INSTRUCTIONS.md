# 🔬 Instrucciones: Hyperparameter Sweep

## ¿Qué es esto?

Un sistema para probar **diferentes configuraciones** de tus modelos y encontrar la que da mejores resultados.

## 📊 Modos Disponibles

### 1. `fast` - Rápido (Recomendado para Empezar)
- **Experimentos**: 9
- **Tiempo**: ~3-4 horas
- **Qué prueba**: Solo diferentes objetivos (target_return + prediction_horizon)
- **Uso**: Primera exploración rápida

### 2. `balanced` - Balanceado
- **Experimentos**: 27
- **Tiempo**: ~8-10 horas
- **Qué prueba**: Objetivos + tamaño del modelo (hidden_dim)
- **Uso**: Si `fast` muestra resultados prometedores

### 3. `thorough` - Exhaustivo
- **Experimentos**: 81
- **Tiempo**: ~24-30 horas (1+ día)
- **Qué prueba**: Objetivos + arquitectura completa (hidden_dim + dropout)
- **Uso**: Solo si tienes buenos resultados en `balanced`

## 🚀 Cómo Ejecutar

### Modo Fast (Recomendado Ahora)

```bash
cd /home/jasan/Develop/trading_system

# Limpiar cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Ejecutar sweep
.venv_rocm62/bin/python scripts/hyperparameter_sweep.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50 \
    --mode fast
```

### Modo Balanced (Si Fast Funciona)

```bash
.venv_rocm62/bin/python scripts/hyperparameter_sweep.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50 \
    --mode balanced
```

### Modo Thorough (Solo Si Tienes Buenos Resultados)

```bash
.venv_rocm62/bin/python scripts/hyperparameter_sweep.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50 \
    --mode thorough
```

## 📁 Dónde se Guardan Los Resultados

Cada experimento crea una carpeta:
```
experiments/
└── BTCUSDT_15m_tr0.002_ph3_hd192_dr0.35_lr0.0003/
    ├── config.json                         # Configuración usada
    ├── results.json                        # Métricas obtenidas
    ├── production_training_results.json    # Resultados detallados
    ├── meta.json                          # Metadata del modelo
    └── training.log                        # Log completo
```

Además, se crea un resumen general:
```
experiments/BTCUSDT_15m_sweep.json  # Resumen de TODOS los experimentos
```

## 📊 Cómo Analizar Resultados

### Mientras Corre

Ver progreso:
```bash
tail -f experiments/*/training.log | grep "Epoch\|✅"
```

Ver resumen parcial:
```bash
cat experiments/BTCUSDT_15m_sweep.json | jq '.'
```

### Cuando Termine

El script imprime automáticamente:
- **Top 5 por Macro F1**: Mejor rendimiento general
- **Top 5 por Long F1**: Mejor para detectar largos
- **Recomendación**: La mejor configuración encontrada

También puedes buscar manualmente:
```bash
# Encontrar configuraciones con Long F1 > 0.30
cat experiments/BTCUSDT_15m_sweep.json | jq '.[] | select(.metrics.avg_long_f1 > 0.30)'

# Encontrar configuraciones con Short F1 > 0.30
cat experiments/BTCUSDT_15m_sweep.json | jq '.[] | select(.metrics.avg_short_f1 > 0.30)'

# Ordenar por PnL
cat experiments/BTCUSDT_15m_sweep.json | jq 'sort_by(.metrics.avg_pnl) | reverse'
```

## 🎯 Qué Buscar en Los Resultados

### Benchmarks Mínimos

Para que una configuración sea **prometedora**:
- ✅ `avg_long_f1` > 0.30
- ✅ `avg_short_f1` > 0.30
- ✅ `avg_test_f1` (macro) > 0.35
- ✅ `avg_pnl` > 0.0 (positivo)

### Flags Rojas 🚩

**NO uses** una configuración si:
- ❌ `avg_long_f1` < 0.10 (no detecta largos)
- ❌ `avg_short_f1` < 0.10 (no detecta cortos)
- ❌ `avg_pnl` < -0.01 (pierde dinero consistentemente)

## ⚙️ Ajustar El Grid De Búsqueda

Si quieres probar otros parámetros, edita `scripts/hyperparameter_sweep.py`:

```python
# En la función generate_experiments(), añade más valores:

if mode == "fast":
    target_returns = [0.002, 0.003, 0.004, 0.005]  # Añadir 0.004
    prediction_horizons = [2, 3, 4, 6]  # Añadir 2
    # ... etc
```

Parámetros que puedes variar:
- `target_return`: Umbral de retorno (0.001 - 0.010)
- `prediction_horizon`: Períodos hacia adelante (2 - 12)
- `sequence_length`: Ventana de contexto (24 - 96)
- `hidden_dim`: Tamaño del modelo (64 - 512)
- `dropout`: Regularización (0.1 - 0.5)
- `lr`: Learning rate (1e-5 - 1e-3)
- `batch_size`: Tamaño del batch (32 - 256)
- `gap_multiplier`: Gap temporal (1 - 3)

## 🔄 Ejecutar en Segundo Plano

Si quieres dejar el sweep corriendo y desconectarte:

```bash
# Con nohup
nohup .venv_rocm62/bin/python scripts/hyperparameter_sweep.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50 \
    --mode fast \
    > logs/sweep_btc_15m.log 2>&1 &

# Ver PID
echo $!

# Monitorear
tail -f logs/sweep_btc_15m.log
```

## 💡 Estrategia Recomendada

### Día 1: Fast Sweep
1. Ejecutar modo `fast` para BTC y SOL
2. Analizar resultados
3. Si alguna configuración tiene Long/Short F1 > 0.30 → **CONTINUAR**
4. Si ninguna pasa 0.30 → **REPLANTEAR ENFOQUE**

### Día 2-3: Balanced Sweep (Si Fast Funcionó)
1. Ejecutar modo `balanced` con las mejores configuraciones de `fast`
2. Buscar Macro F1 > 0.40

### Día 4-7: Thorough + Backtest (Si Balanced Funcionó)
1. Ejecutar modo `thorough`
2. Backtest con costos reales
3. Paper trading

## ❓ FAQ

**P: ¿Puedo detener el sweep a la mitad?**
R: Sí, Ctrl+C lo detiene. Los resultados parciales se guardan en `sweep.json`.

**P: ¿Puedo ejecutar varios sweeps en paralelo?**
R: Sí, pero en **símbolos diferentes** para evitar colisiones.

**P: ¿Cuánta RAM/VRAM necesito?**
R: ~8GB VRAM por experimento. Con tu RX 6600 (8GB), puedes correr 1 a la vez.

**P: ¿Qué hago si todos los experimentos dan mal?**
R: Revisar el `MODEL_IMPROVEMENT_PLAN.md` - puede que necesites feature engineering o cambiar de enfoque.

---

**Última actualización**: 2024-12-04 14:00 UTC
