# 🚀 Instrucciones para Ejecutar el Smoke Test

## ✅ Estado Actual del Código

Todas las correcciones han sido aplicadas y verificadas:
- ✅ Fix 1: `train_labels` correctamente definido
- ✅ Fix 2: Verificación de `all_returns` corregida  
- ✅ Fix 3: Verificación de `all_reg_preds` corregida
- ✅ Cache de Python limpiado automáticamente

## 📋 Pasos para Ejecutar

### Opción 1: Ejecutar el Smoke Test (Recomendado)

Desde tu terminal con acceso a GPUs:

```bash
cd /home/jasan/Develop/trading_system
./scripts/run_smoke_test.sh
```

Este script:
- 🔍 Detecta automáticamente NVIDIA o AMD
- 🔧 Configura variables de entorno (ROCm si es necesario)
- 🧹 Limpia el cache de Python
- 🚀 Ejecuta 5 épocas de entrenamiento
- 📝 Guarda log en `/tmp/smoke_test.log`

### Opción 2: Verificar Correcciones Primero

Si quieres confirmar que el código tiene todos los fixes:

```bash
./scripts/verify_fixes.sh
```

### Opción 3: Ejecución Manual

Si prefieres ejecutar manualmente:

**Para NVIDIA:**
```bash
.venv_cuda/bin/python scripts/train_production_ready.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --epochs 5 \
    --device cuda
```

**Para AMD ROCm:**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0

.venv_rocm62/bin/python scripts/train_production_ready.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --epochs 5 \
    --device cuda
```

## 📊 Monitoreo del Progreso

Mientras el entrenamiento corre:

```bash
# Ver progreso en tiempo real
tail -f /tmp/smoke_test.log

# O desde el log principal
tail -f logs/trading_system.log
```

## 🎯 Qué Esperar

El smoke test debería:
1. ✅ Cargar ~316k muestras con 94 features
2. ✅ Crear 5 folds walk-forward
3. ✅ Para cada fold:
   - Seleccionar features (per-fold, sin leakage)
   - Escalar datos (per-fold)
   - Entrenar 5 épocas
   - Guardar modelo, scaler, selector
4. ✅ Generar artefactos en `models/advanced/BTCUSDT/5m/`

**Duración estimada**: 5-10 minutos (depende de la GPU)

## 📁 Artefactos Generados

Después del smoke test, deberías tener:

```
models/advanced/BTCUSDT/5m/
├── best_model_fold1.pt
├── best_model_fold2.pt
├── best_model_fold3.pt
├── best_model_fold4.pt
├── best_model_fold5.pt
├── scaler_fold1.pkl
├── scaler_fold2.pkl
├── scaler_fold3.pkl
├── scaler_fold4.pkl
├── scaler_fold5.pkl
├── feature_selector_fold1.pkl
├── feature_selector_fold2.pkl
├── feature_selector_fold3.pkl
├── feature_selector_fold4.pkl
├── feature_selector_fold5.pkl
├── meta.json
└── production_training_results.json
```

## 🧪 Siguiente Paso: Validación del Predictor

Una vez completado el smoke test:

```bash
.venv_cuda/bin/python scripts/validate_predictor.py
# O con ROCm:
.venv_rocm62/bin/python scripts/validate_predictor.py
```

Esto verificará que:
- ✅ El predictor carga correctamente todos los artefactos
- ✅ Las features se alinean correctamente
- ✅ La inferencia funciona (simple y batch)
- ✅ No hay errores de shape o tipo

## ⚠️ Troubleshooting

### Error: "GPU NO DISPONIBLE"
- Verifica que ejecutas desde una terminal con acceso a GPUs
- Confirma con `nvidia-smi` o `rocm-smi`
- No ejecutes desde entornos sandbox/containers sin GPU passthrough

### Error: "NameError: name 'train_labels' is not defined"
- Ejecuta `./scripts/verify_fixes.sh` para confirmar correcciones
- Limpia cache: `find . -name "*.pyc" -delete`
- Re-ejecuta el smoke test

### Error: "HIP error: invalid device function" (ROCm)
- Confirma `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- Verifica que `LD_LIBRARY_PATH` incluye ROCm libs
- Considera usar NVIDIA si está disponible

## 📞 Soporte

Si encuentras problemas:
1. Revisa `/tmp/smoke_test.log` para el error exacto
2. Verifica que todas las correcciones están presentes: `./scripts/verify_fixes.sh`
3. Confirma acceso a GPU: `nvidia-smi` o `rocm-smi`

---

**Última actualización**: 2024-12-04 07:42 UTC
**Estado**: Código corregido y verificado ✅
