# 🚀 Guía de Mejoras del Modelo para Producción

Esta guía explica paso a paso cómo usar las mejoras implementadas en el sistema de trading.

## 📋 Índice

1. [¿Qué Mejoras se Implementaron?](#qué-mejoras-se-implementaron)
2. [Guía Rápida de Uso](#guía-rápida-de-uso)
3. [Opción 1: Entrenamiento Completo Automático](#opción-1-entrenamiento-completo-automático)
4. [Opción 2: Paso a Paso Manual](#opción-2-paso-a-paso-manual)
5. [Interpretación de Resultados](#interpretación-de-resultados)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 ¿Qué Mejoras se Implementaron?

### 1. **Features de Régimen de Mercado** (35 nuevas features)
   - **Fuerza de tendencia**: ADX, consistencia de tendencia
   - **Volatilidad**: Regímenes alto/bajo, Bollinger Band squeeze
   - **Microestructura**: Presión compradora/vendedora, distancia de VWAP
   - **Sesiones**: Asiática, Europea, US (importante para 5m/15m)
   - **Patrones temporales**: Día de semana, efectos de lunes/fin de semana
   - **Momentum**: Cruces de medias, ROC, aceleración

   **Total ahora**: ~99 features (antes 64)

### 2. **Arquitectura Mejorada del Modelo**
   - **DeepTemporalNet**: Versión más profunda
     - 3 capas LSTM (antes 2)
     - Hidden dim: 192 (antes 128)
     - Dense layers: 384→256→128 (antes 256→128)
     - 8 attention heads (antes 4)
     - Mejor regularización con dropout escalonado

   - **TransformerNet**: Alternativa basada en Transformer
   - **HybridCNNLSTM**: Combinación CNN + LSTM

### 3. **Split con Validación**
   - Ahora: **Train (60%) / Validation (20%) / Test (20%)**
   - Antes: Solo Train / Test
   - Validación se usa para early stopping
   - Test nunca visto durante entrenamiento

### 4. **Focal Loss**
   - Mejor manejo del desbalanceo de clases
   - Se enfoca en ejemplos difíciles
   - Reduce importancia de ejemplos fáciles

### 5. **Configuraciones Optimizadas**
   - `sequence_length`: 48 (antes 24) = 4 horas de contexto
   - `prediction_horizon`: 6 (antes 12) = 30 minutos adelante
   - `target_return`: 0.005 (antes 0.002) = 0.5% (señales más fuertes)
   - `max_history_days`: 1100 (usa todos tus datos disponibles)
   - `learning_rate`: 3e-4 (antes 5e-4)
   - `batch_size`: 128 (antes 256)

### 6. **Ensemble de Modelos**
   - Entrena 5 modelos con diferentes configuraciones
   - Promedia predicciones con pesos optimizados
   - Reduce overfitting y mejora robustez

### 7. **Integración con XGBoost**
   - Agrega modelo de gradient boosting al ensemble
   - Captura patrones diferentes a redes neuronales
   - Combina predicciones LSTM + XGBoost

---

## 🚀 Guía Rápida de Uso

### Opción 1: Entrenamiento Completo Automático (Recomendado)

```bash
# Entrenar TODO (modelo individual + ensemble + híbrido) en ambos timeframes
python scripts/test_btc_models.py \
    --mode all \
    --timeframes 5m,15m \
    --symbol BTCUSDT \
    --device cuda \
    --epochs 200 \
    --ensemble-epochs 150 \
    --n-models 5
```

**Tiempo estimado**: 6-12 horas (dependiendo de GPU)

**Qué hace**:
1. ✓ Entrena modelo individual en BTC 5m (200 epochs)
2. ✓ Entrena modelo individual en BTC 15m (200 epochs)
3. ✓ Entrena ensemble de 5 modelos en BTC 5m
4. ✓ Entrena ensemble de 5 modelos en BTC 15m
5. ✓ Crea ensemble híbrido (LSTM + XGBoost) en 5m
6. ✓ Crea ensemble híbrido (LSTM + XGBoost) en 15m
7. ✓ Compara todos los resultados

---

### Opción 2: Entrenamiento Selectivo

#### Solo Modelo Individual

```bash
# Entrenar solo modelo mejorado individual
python scripts/test_btc_models.py \
    --mode single \
    --timeframes 5m \
    --epochs 200
```

**Tiempo estimado**: 2-3 horas

#### Solo Ensemble

```bash
# Entrenar ensemble (requiere más tiempo)
python scripts/test_btc_models.py \
    --mode ensemble \
    --timeframes 5m \
    --n-models 5 \
    --ensemble-epochs 150
```

**Tiempo estimado**: 4-6 horas

#### Solo Híbrido (requiere ensemble previo)

```bash
# Agregar XGBoost a ensemble existente
python scripts/test_btc_models.py \
    --mode hybrid \
    --timeframes 5m
```

**Tiempo estimado**: 30-60 minutos

---

## 📖 Opción 2: Paso a Paso Manual

### Paso 1: Entrenar Modelo Individual Mejorado

```bash
python scripts/train_production_ready.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --epochs 200 \
    --batch-size 128 \
    --lr 3e-4 \
    --sequence-length 48 \
    --prediction-horizon 6 \
    --target-return 0.005 \
    --max-history-days 1100 \
    --device cuda
```

**Parámetros clave**:
- `--sequence-length 48`: 4 horas de contexto (48 × 5min)
- `--prediction-horizon 6`: Predice 30 minutos adelante (6 × 5min)
- `--target-return 0.005`: Solo señales de movimientos > 0.5%
- `--max-history-days 1100`: Usa todos tus 3+ años de datos

**Archivos generados**:
```
models/advanced/BTCUSDT/5m/
├── best_model_fold1.pt
├── best_model_fold2.pt
├── best_model_fold3.pt
├── best_model_fold4.pt
├── best_model_fold5.pt
├── scaler.pkl
├── feature_selector.pkl
└── production_training_results.json
```

### Paso 2: Entrenar Ensemble de Modelos

```bash
python scripts/train_ensemble.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --n-models 5 \
    --epochs 150 \
    --device cuda
```

**Qué hace**:
- Entrena 5 modelos con diferentes configuraciones
- Cada modelo usa diferente:
  - Seed aleatorio
  - Arquitectura (hidden_dim, lstm_layers)
  - Sequence length (48, 64, 96)
- Optimiza pesos basado en rendimiento de validación

**Archivos generados**:
```
models/advanced/BTCUSDT/5m/ensemble/
├── ensemble_model_1.pt
├── ensemble_model_2.pt
├── ensemble_model_3.pt
├── ensemble_model_4.pt
├── ensemble_model_5.pt
├── ensemble_metadata.json
├── scaler.pkl
└── feature_selector.pkl
```

### Paso 3: Agregar XGBoost (Ensemble Híbrido)

```bash
python scripts/train_hybrid_ensemble.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --ensemble-dir models/advanced/BTCUSDT/5m/ensemble \
    --device cuda
```

**Qué hace**:
- Carga el ensemble de redes neuronales
- Entrena modelo XGBoost en las mismas features
- Optimiza pesos: `w_neural * pred_neural + w_xgb * pred_xgb`
- Busca la mejor combinación

**Archivos generados**:
```
models/advanced/BTCUSDT/5m/ensemble/hybrid/
├── xgboost_model.pkl
└── hybrid_metadata.json
```

---

## 📊 Interpretación de Resultados

### Métricas Clave a Revisar

#### 1. **Accuracy** (Precisión General)
```
< 40%  : ❌ Muy débil (apenas mejor que aleatorio)
40-45% : ⚠️  Débil (necesita mejoras)
45-50% : ✓  Aceptable (puede ser útil)
50-55% : ✓✓ Bueno (listo para paper trading)
> 55%  : ✓✓✓ Excelente (muy prometedor)
```

#### 2. **Macro F1** (Balance entre clases)
```
< 35%  : ❌ Modelo no funciona
35-40% : ⚠️  Débil
40-45% : ✓  Aceptable
45-50% : ✓✓ Bueno
> 50%  : ✓✓✓ Excelente
```

#### 3. **F1 por Clase**
```
Long F1:  > 45% ✓  (puede predecir longs)
Short F1: > 40% ✓  (puede predecir shorts)
```

**⚠️ CRÍTICO**: Si Short F1 < 30%, el modelo NO puede predecir shorts correctamente. No usar en producción.

#### 4. **Comparación de Modelos**

Ejemplo de buenos resultados:
```
Modelo Individual:  F1 = 0.42
Ensemble Neural:    F1 = 0.46  (+4%)
Híbrido:            F1 = 0.48  (+6%)
```

Si el ensemble no mejora o empeora:
- Puede haber overfitting
- Modelos muy similares entre sí
- Probar con más diversidad (más n_models o diferentes architecturas)

### Revisar Archivos de Resultados

#### Modelo Individual
```bash
cat models/advanced/BTCUSDT/5m/production_training_results.json
```

Busca:
```json
{
  "results": [
    {
      "fold": 1,
      "test_metrics": {
        "accuracy": 0.45,
        "macro_f1": 0.42,
        "long_f1": 0.48,
        "short_f1": 0.35
      }
    }
  ]
}
```

#### Ensemble
```bash
cat models/advanced/BTCUSDT/5m/ensemble/ensemble_metadata.json
```

#### Híbrido
```bash
cat models/advanced/BTCUSDT/5m/ensemble/hybrid/hybrid_metadata.json
```

---

## 🆚 Comparar 5m vs 15m

```bash
# Ver resultados lado a lado
python scripts/test_btc_models.py --mode all --timeframes 5m,15m
```

**Esperado**:
- **15m generalmente tiene mejor F1** (menos ruido)
- **5m tiene más trades** pero menor win rate
- **Para producción**: Usa el timeframe con F1 > 0.45

**Decisión**:
```
Si 5m F1 < 0.40 y 15m F1 > 0.45:
  → Usa 15m como primario
  → 5m solo para timing de entrada

Si ambos 5m y 15m > 0.45:
  → Usa confirmación multi-timeframe
  → Señal de 5m confirmada por 15m

Si ambos < 0.40:
  → Necesitas más mejoras antes de producción
```

---

## 🎯 Criterios para Producción

### ✅ Lista de Verificación

Antes de usar en producción con dinero real:

- [ ] **Modelo**: F1 > 0.45 en walk-forward validation
- [ ] **Short prediction**: Short F1 > 0.40 (crítico!)
- [ ] **Timeframe**: Elegido basado en métricas (5m o 15m)
- [ ] **Paper trading**: Mínimo 2 semanas con Sharpe > 1.0
- [ ] **Leverage**: Reducido a 10-20x (no 50x)
- [ ] **Risk limits**: Portfolio heat < 100%
- [ ] **Monitoring**: Grafana + alertas configuradas
- [ ] **Backtesting**: Sharpe > 1.5, Max DD < 20%

---

## 🐛 Troubleshooting

### Problema 1: CUDA Out of Memory

```bash
# Reducir batch size
python scripts/train_production_ready.py \
    --batch-size 64 \
    ...
```

O entrenar en CPU (más lento):
```bash
python scripts/train_production_ready.py \
    --device cpu \
    ...
```

### Problema 2: ImportError (xgboost)

```bash
pip install xgboost
```

### Problema 3: "Not enough data"

Verifica que tienes datos suficientes:
```bash
python -c "
from data.storage.database_manager import db_manager
df = db_manager.get_ohlcv_data('BTC/USDT:USDT', '5m')
print(f'Registros: {len(df)}')
print(f'Días: {(df.index[-1] - df.index[0]).days}')
"
```

Si tienes < 1000 días, reduce `--max-history-days`:
```bash
python scripts/train_production_ready.py \
    --max-history-days 730 \
    ...
```

### Problema 4: Modelos no mejoran

**Síntomas**: F1 < 0.35 en todos los modelos

**Soluciones**:

1. **Aumentar target_return** (señales más fuertes):
```bash
--target-return 0.008  # 0.8% en lugar de 0.5%
```

2. **Aumentar sequence_length** (más contexto):
```bash
--sequence-length 96  # 8 horas en lugar de 4
```

3. **Probar 15m en lugar de 5m**:
```bash
--timeframe 15m
```

4. **Revisar distribución de clases**:
```python
from ml.advanced_models.dataset import load_sequence_dataset, AdvancedDatasetConfig

config = AdvancedDatasetConfig(
    symbol='BTC/USDT:USDT',
    timeframe='5m',
    target_return=0.005,
)

_, labels, _, _ = load_sequence_dataset(config)

import numpy as np
print(np.bincount(labels))
# Debe ser relativamente balanceado
# Ejemplo bueno: [10000, 8000, 7500]  (neutral, long, short)
# Ejemplo malo:  [50000, 2000, 1000]  (muy desbalanceado)
```

Si está muy desbalanceado, aumenta `target_return`.

### Problema 5: Entrenamiento muy lento

Para entrenar más rápido:
```bash
# Reducir epochs
--epochs 100 \

# Menos folds (menos robusto pero más rápido)
# Editar en el código: n_splits=3 en lugar de 5

# Entrenar solo mejores modelos del ensemble
--n-models 3 \

# Usar modelo más pequeño
# En el código, cambiar hidden_dim=128 en lugar de 192
```

---

## 📈 Próximos Pasos Después del Entrenamiento

### 1. Analizar Resultados
```bash
# Comparar modelos
python scripts/test_btc_models.py --mode all --timeframes 5m,15m
```

### 2. Si F1 > 0.45: Paper Trading
```bash
# Configurar bot en modo testnet
cd binance_futures_bot_py
IS_TESTNET=1 python main.py
```

Monitorear por 2 semanas:
- Win rate
- Sharpe ratio
- Maximum drawdown
- Profit factor

### 3. Si F1 < 0.40: Más Experimentación
- Probar diferentes `target_return` (0.008, 0.010)
- Probar diferentes `sequence_length` (64, 96)
- Probar TransformerNet en lugar de DeepTemporalNet
- Agregar más features personalizadas

### 4. Producción (solo si todo está bien)
- Reduce leverage a 10x
- Configura circuit breakers
- Setup Grafana monitoring
- Empieza con capital pequeño (2-5% del portfolio)

---

## 📞 Resumen de Comandos

### Entrenamiento Rápido (Todo Automático)
```bash
# BTC 5m y 15m: individual + ensemble + híbrido
python scripts/test_btc_models.py --mode all --timeframes 5m,15m
```

### Solo Modelo Individual
```bash
python scripts/train_production_ready.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --epochs 200 \
    --device cuda
```

### Solo Ensemble
```bash
python scripts/train_ensemble.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --n-models 5 \
    --device cuda
```

### Solo Híbrido
```bash
python scripts/train_hybrid_ensemble.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --ensemble-dir models/advanced/BTCUSDT/5m/ensemble
```

---

## 🎓 Explicación Técnica de las Mejoras

### ¿Por qué estas mejoras funcionan?

1. **Features de régimen**: El modelo ahora sabe si está en tendencia o rango. En tendencia, momentum funciona. En rango, mean reversion funciona.

2. **Arquitectura profunda**: Más capas = más capacidad de aprendizaje de patrones complejos. Residual connections previenen vanishing gradients.

3. **Validación**: Early stopping en validación previene overfitting. Test set da estimación real de rendimiento futuro.

4. **Focal Loss**: Clases minoritarias (especialmente shorts) reciben más atención. Ejemplos difíciles se priorizan sobre fáciles.

5. **Ensemble**: Diferentes modelos capturan diferentes aspectos. Promediando se reduce variance. "Wisdom of crowds" en ML.

6. **XGBoost**: Tree-based models capturan interacciones no-lineales que redes neuronales pueden perder. Complementario a LSTM.

7. **Configuración**: Más contexto (48 vs 24), horizonte más corto (6 vs 12), señales más fuertes (0.5% vs 0.2%) = mejor signal-to-noise ratio.

---

## 📚 Referencias

- **Focal Loss**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **Attention Mechanism**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- **Ensemble Methods**: [Dietterich, 2000](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)
- **Walk-Forward Analysis**: [Pardo, 2008](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies,+2nd+Edition-p-9780470128015)

---

**¡Buena suerte con tu trading! 🚀📈**
