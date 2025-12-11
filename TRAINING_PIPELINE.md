# Training Pipeline Overview (Production Trainer)

Esta guía describe cómo entrena el sistema los modelos de trading (script `scripts/train_production_ready.py`) y resume el flujo de datos, ingeniería de features, arquitectura del modelo y ciclo de entrenamiento/evaluación.

## Datos y targets
- Fuente: OHLCV desde `data.storage.database_manager.db_manager.get_ohlcv_data` para el símbolo/timeframe.
- Historial: se recorta a `max_history_days` (por defecto 1100 días) y se garantiza al menos `sequence_length + prediction_horizon + min_records`.
- Target de clasificación (3 clases): futuro retorno `R = (close[t+h]/close[t]) - 1`.
  - `long` si `R >= target_return` (0.5% por defecto)
  - `short` si `R <= -target_return`
  - `neutral` en otro caso.
- Target de regresión: el mismo retorno futuro (horizonte `prediction_horizon`). Actualmente no se usa en la loss porque no se pasa a `SequenceDataset` en este script.

## Ingeniería de features (`ml/nn_pattern/features.py`)
- Genera ~94 features brutas: momentum (RSI, ROC, CCI), tendencia (MAs, MACD, ADX, Aroon, SAR), volatilidad (Bandas de Bollinger, Keltner, ATR, hist vol), volumen (OBV, VPT, CMF, MFI), features custom (retornos log, vol rolling, atr_pct, price_location, volume_flow) y régimen de mercado (35 features adicionales).
- Normalizaciones específicas antes del escalado global:
  - MAs y momentum a porcentajes del precio.
  - Bandas/Canales → posición relativa y ancho en %; se eliminan valores absolutos.
  - SAR como distancia relativa en %.
  - Volúmenes (SMA) como ratio vs volumen medio 20; OBV/AD/VPT a z-score rolling; volume_flow normalizado por volumen medio.
  - Limpieza de inf/NaN, forward-fill y dropna final.
- Output: `feature_frame` y lista de columnas finales (algunas columnas se reemplazan/eliminan en la normalización).

## Selección y escalado (por fold)
- Feature Selection: dentro de cada fold, fit en train (muestra de hasta 20k filas) y transform en val/test; se guarda `feature_selector_fold{i}.pkl`.
- Escalado: `StandardScaler` fit en train del fold, aplicado a val/test; se guarda `scaler_fold{i}.pkl`.

## Splits y dataset
- Split walk-forward con validación (`walk_forward_split_with_validation`):
  - Ratios por defecto: 60% train, 20% val, 20% test repartidos en 5 folds (ventanas deslizantes).
  - Sin gap entre conjuntos.
- `SequenceDataset`: construye ventanas de longitud `sequence_length` (48 pasos) alineadas con etiquetas ya shift-eadas; puede aplicar ruido gaussiano a las secuencias de train (augmentación).
- DataLoader: `num_workers=2`, `pin_memory=True`, shuffle en train.

## Arquitectura (`ml/advanced_models/improved_architecture.py::DeepTemporalNet`)
- Proyección de entrada (Linear + LayerNorm + ReLU + dropout).
- LSTM apilado (3 capas, bidireccional, hidden_dim 192 por defecto), con dropout interno desactivado en ROCm salvo `FORCE_LSTM_DROPOUT=1`.
- LayerNorm tras LSTM.
- Atención multihead (8 heads) con residual + LayerNorm.
- Denso profundo con bloques residuales y LayerNorm; dropout escalonado.
- Heads: clasificador (3 clases) y regressor opcional (no usado en la loss actual).
- Inicialización explícita (orthogonal para LSTM, Kaiming para densas, constantes para norms/bias).

## Entrenamiento (estado actual)
- Reproducibilidad: semilla global (42) para torch/numpy/python, determinismo CuDNN forzado.
- Loss: `MultiTaskLoss` con FocalLoss y pesos de clase dinámicos por fold; regresión activa (peso 0.2) en train; evaluación no reporta métricas de regresión.
- Optimizer/Scheduler: AdamW (lr 3e-4, wd 1e-4) + warmup (15) + cosine; grad clip 1.0.
- AMP: ON en CUDA, OFF en ROCm; GradScaler cuando aplica.
- Early stopping: paciencia 30 en macro-F1 de validación; guarda mejor modelo por fold.
- Métricas: accuracy, macro-F1/precision/recall y por clase (no PnL ni métricas de trading).
- Logging/artefactos: modelos `best_model_fold{i}.pt`, scaler/selector por fold, meta.json y results JSON.

## Inferencia (`ml/advanced_models/predictor.py`)
- Intención: cargar ensemble por fold (modelo + scaler + selector) y promediar probs/regresión.
- Riesgo actual: el predictor no inicializa `sequence_length`, `selected_features` ni `model_config` desde meta; puede fallar si esos campos no están en meta.json. Verificar y completar inicialización antes de usar en producción.

## Ejecución (dispatcher)
- `scripts/dispatch_training.py` asigna jobs a GPUs (ROCm usa `.venv_rocm62`, NVIDIA `.venv_cuda`), setea env ROCm (`HSA_OVERRIDE_GFX_VERSION`, `LD_LIBRARY_PATH`, `HIP_VISIBLE_DEVICES`) y limita batches (96). AMD usa AMP off; NVIDIA AMP on.
- Walk-forward se repite por símbolo/timeframe; logs en `logs/multi_gpu/*.log`.

## Estado de riesgos
- Data leakage: mitigado (selector/escalado per-fold).
- Cabeza de regresión: se entrena en train, pero no se reporta en eval; confirmar uso downstream.
- Clase desbalanceada: pesos dinámicos por fold + FocalLoss; revisar valores si cambian las clases.
- Reproducibilidad: seed y determinismo configurados; AMP puede introducir ligeras variaciones en CUDA.
- Validación temporal: sin gap entre conjuntos; si hay autocorrelación fuerte, considerar gap.
- Métricas de trading: no se calculan PnL/costos/umbral; métricas siguen siendo de clasificación.
- Inferencia: predictor requiere completar inicialización desde meta para ser robusto.

## Sugerencias de mejora
1) Completar predictor: leer `sequence_length`, `selected_features` y `model_config` desde meta y validar shapes; añadir manejo robusto de ensembles sin esos campos.
2) Reportar métricas de regresión (MSE/MAE) y/o quitar la cabeza si no se usa en producción.
3) Añadir métricas de trading (PnL con costos, drawdown) y calibración de umbrales para decisiones long/short.
4) Opcional: gap temporal entre train/val/test para reducir leakage cercano en series autocorrelacionadas.
5) Revisar/ajustar pesos focal por fold según distribución real; explorar α/γ.
6) HPO ligero (batch, lr, dropout, hidden_dim) y backtests con costos.
7) Considerar más determinismo en CUDA (desactivar AMP) si se requiere reproducibilidad estricta.
