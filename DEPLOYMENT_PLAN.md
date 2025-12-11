# Plan de Despliegue - Sistema ML Institucional

## Estado Actual
✅ Pipeline de entrenamiento "Institutional Grade" implementado
✅ Eliminación de data leakage (per-fold scaling/selection)
✅ Métricas completas (Classification + Regression MSE/MAE + PnL Implícito)
✅ Predictor robusto con ensemble de pipelines
✅ Reproducibilidad garantizada (seeds + determinismo)

## Fase 1: Validación End-to-End (EN PROGRESO)

### 1.1 Smoke Test ⏳
```bash
.venv_rocm62/bin/python scripts/train_production_ready.py \
    --symbol BTCUSDT --timeframe 5m --epochs 5
```
**Objetivo**: Verificar que el pipeline completo funciona sin errores
**Duración estimada**: 10-15 minutos
**Artefactos generados**:
- `models/advanced/BTCUSDT/5m/best_model_fold{1-5}.pt`
- `models/advanced/BTCUSDT/5m/scaler_fold{1-5}.pkl`
- `models/advanced/BTCUSDT/5m/feature_selector_fold{1-5}.pkl`
- `models/advanced/BTCUSDT/5m/meta.json`

### 1.2 Validación del Predictor 📋
```bash
.venv_rocm62/bin/python scripts/validate_predictor.py
```
**Objetivo**: Confirmar que el predictor carga correctamente y genera predicciones
**Verificaciones**:
- ✓ Carga de artefactos (modelos, scalers, selectors, meta)
- ✓ Alineación de features
- ✓ Predicción simple (última vela)
- ✓ Predicción por lotes (batch)
- ✓ Salida de probabilidades y regresión

## Fase 2: Backtesting con Costos Reales

### 2.1 Actualizar Backtest Script
**Archivo**: `scripts/backtest_production_model_v2.py`

**Mejoras necesarias**:
1. **Costos de Trading**:
   ```python
   TRADING_COSTS = {
       'maker_fee': 0.0002,  # 0.02% Binance Maker
       'taker_fee': 0.0004,  # 0.04% Binance Taker
       'slippage': 0.0005,   # 0.05% slippage estimado
   }
   ```

2. **Umbrales de Decisión**:
   ```python
   THRESHOLDS = {
       'long_entry': 0.60,   # Probabilidad mínima para Long
       'short_entry': 0.60,  # Probabilidad mínima para Short
       'min_return': 0.003,  # Retorno mínimo predicho (0.3%)
   }
   ```

3. **Métricas de Trading**:
   - PnL Neto (después de costos)
   - Sharpe Ratio
   - Max Drawdown
   - Win Rate
   - Profit Factor
   - Número de trades

### 2.2 Ejecutar Backtest
```bash
.venv_rocm62/bin/python scripts/backtest_production_model_v2.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --start-date 2024-11-01 \
    --end-date 2024-12-04 \
    --initial-capital 10000
```

**Criterios de Aprobación**:
- Sharpe Ratio > 1.0
- Max Drawdown < 15%
- Win Rate > 45%
- PnL Neto positivo después de costos

## Fase 3: Entrenamiento Completo Multi-Símbolo

### 3.1 Símbolos Prioritarios
```
BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, LINKUSDT
```

### 3.2 Configuración de Entrenamiento
```bash
# Usar dispatch_training.py para distribuir en GPUs
.venv_rocm62/bin/python scripts/dispatch_training.py \
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT LINKUSDT \
    --timeframe 5m \
    --epochs 200 \
    --batch-size 128
```

**Duración estimada**: 4-6 horas (con 2 GPUs)

## Fase 4: Monitoreo en Vivo (Paper Trading)

### 4.1 Configurar Servicio ML
**Archivo**: `services/ml_probability_service.py`

**Actualizaciones necesarias**:
1. Usar `AdvancedPredictor` en lugar del predictor antiguo
2. Implementar caché de predicciones (TTL: 5 minutos)
3. Añadir health checks
4. Logging de todas las predicciones

### 4.2 Lanzar Servicio
```bash
pm2 start services/ml_probability_service.py \
    --name ml_service \
    --interpreter .venv_rocm62/bin/python \
    --log logs/ml_service.log
```

### 4.3 Monitoreo
- Dashboard de métricas en tiempo real
- Alertas si la confianza promedio cae < 40%
- Tracking de PnL simulado vs real

## Fase 5: Producción Real (Trading en Vivo)

### 5.1 Criterios de Activación
- ✓ Backtest con Sharpe > 1.5 en últimos 30 días
- ✓ Paper trading exitoso por 7 días
- ✓ Max Drawdown en paper < 10%
- ✓ Sistema de stop-loss automático probado

### 5.2 Configuración Inicial
- Capital inicial: $1,000 - $5,000
- Tamaño de posición: 2% del capital por trade
- Stop-loss: 1.5% por posición
- Take-profit: 3% por posición

### 5.3 Escalado Gradual
- Semana 1-2: $1,000 capital
- Semana 3-4: $2,500 capital (si PnL > 5%)
- Mes 2+: Escalar según performance

## Checklist de Seguridad

### Pre-Producción
- [ ] Smoke test completado sin errores
- [ ] Predictor validado end-to-end
- [ ] Backtest con costos muestra rentabilidad
- [ ] Paper trading activo y monitoreado
- [ ] Sistema de alertas configurado

### Producción
- [ ] API keys en variables de entorno (no hardcoded)
- [ ] Rate limiting implementado
- [ ] Circuit breakers para fallos de API
- [ ] Backup automático de modelos
- [ ] Logs rotados y archivados
- [ ] Monitoreo de uso de GPU/CPU/RAM

## Próximos Pasos Inmediatos

1. ⏳ **Esperar a que termine el smoke test** (5-10 min restantes)
2. 🧪 **Ejecutar `validate_predictor.py`**
3. 📊 **Actualizar script de backtest con costos**
4. 🚀 **Ejecutar backtest en datos recientes**
5. 📈 **Analizar resultados y decidir si proceder a entrenamiento completo**

---

**Última actualización**: 2024-12-04 07:02 UTC
**Estado**: Smoke Test en progreso (Fold 1/5)
