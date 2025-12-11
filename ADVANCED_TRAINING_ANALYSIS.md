# Analisis del pipeline de entrenamiento

Este documento resume como se entrenan los modelos temporales en `scripts/train_advanced_model.py` y evalua su solidez para uso institucional.

## Flujo end-to-end
- Carga de OHLCV desde la base (`db_manager.get_ohlcv_data`) con maximos de historia y muestras configurables.
- Ingenieria de features en `ml/nn_pattern/features.py` (indicadores de momentum, tendencia, volatilidad, volumen, rasgos de regimen). Se normalizan indicadores basados en precios/volumen antes de limpiar NaN.
- Generacion de targets en `ml/advanced_models/dataset.py`: retornos a `prediction_horizon` barras; etiquetas 3 clases (neutral/long/short) con umbral `target_return`; objetivo de regresion = retorno continuo.
- Opcional: seleccion de features (mutual information por defecto) sobre un subconjunto; reduccion a `n_features_to_select`.
- Creacion de `SequenceDataset` con ventanas deslizantes de largo `sequence_length`, alineando cada secuencia con su etiqueta ya desplazada; puede aplicar ruido gaussiano para data augmentation.
- Escalado: en walk-forward se ajusta `StandardScaler` solo con train de cada fold; en el entrenamiento final se ajusta con todos los datos (ver riesgos abajo).
- Entrenamiento:
  - Walk-forward (opcional) usando `walk_forward_split_with_validation` (60% train inicial, 20% val, ventanas de test progresivas). Early stopping y ReduceLROnPlateau se basan en F1 macro de validacion. Se calculan pesos de clase del fold.
  - Entrenamiento final (siempre) con split cronologico 70/15/15 (train/val/test) sobre el dataset completo.
  - Posible ensamble: se entrenan varios modelos con seeds distintos y se promedian logits/regresion.
- Modelos soportados: `AdvancedTemporalNet` (LSTM bidireccional + atencion + backbone denso residual), `TemporalConvNet`, `TransformerNet`, `DeepTemporalNet` opcional. Cabezas de clasificacion (3 clases) y regresion (retorno).
- Perdida: `MultiTaskLoss` combina Focal Loss (con pesos de clase y `gamma=2`) y MSE; usa ponderacion aprendida via log-varianza para balancear tareas. Se aplica clipping de gradiente (norm 1.0) y AdamW con weight decay.
- Evaluacion: `evaluate_model` calcula CE, accuracy, F1 macro, precision/recall por clase, AP por clase de direccionalidad y MSE/MAE para regresion. Walk-forward agrega promedios; split final reporta test set.
- Artefactos guardados en `models/advanced/<symbol>/<timeframe>/`: `model.pt` (estado o ensemble), `scaler.pkl`, `feature_selector.pkl` (si aplica) y `meta.json` (config + metricas). Resultados de walk-forward se escriben en `walk_forward_results.json`.

## Fortaleza y brechas
- Fortalezas:
  - Validacion walk-forward que respeta temporalidad y usa early stopping por F1 macro.
  - Multi-tarea (clasificacion + regresion) con Focal Loss y pesos de clase para manejar desbalance.
  - Arquitectura moderna con atencion, residual y opciones TCN/Transformer; ensamble disponible.
  - Semillas fijadas y determinismo de cudnn desactivado para reproducibilidad.
  - Normalizacion de features basada en precios relativos evita magnitudes dependientes del activo.
- Brechas/riesgos:
  - **Fuga de datos** en entrenamiento final: el `StandardScaler` y la seleccion de features se ajustan usando todo el dataset antes de definir train/val/test, introduciendo leakage. En walk-forward el escalado es correcto; el split final no.
  - La seleccion de features se realiza previo a cualquier split; aun en walk-forward se eligen con todas las etiquetas, lo que puede sobreestimar el rendimiento.
  - No hay registro de experimentos (MLflow/W&B), versionado de datasets ni control de semilla por ejecucion en los metadatos mas alla de `meta.json`.
  - El scheduler ReduceLROnPlateau observa solo F1 macro; la regresion no participa en la señal de parada.
  - No hay calibracion de probabilidades ni validacion sobre drawdown/ratio de ganancias en un backtest integrado; las metricas son puramente de clasificacion.
  - La particion final 70/15/15 es cronologica pero usa `SubsetRandomSampler`, mezclando el orden temporal en los batches; no es critico pero rompe un poco el esquema secuencial.
  - No hay control de drift o monitoreo de distribuciones para reentrenos periodicos.

## Recomendaciones para robustez institucional
1) Corregir leakage: ajustar scaler y selector solo con train de cada split (val/test transformados). Para el entrenamiento final, recalcular seleccion de features y scaler dentro del subset train antes de validar/testear.
2) Consolidar pipeline reproducible: registrar hashes de datos/consultas, semilla, versiones de librerias y config en un sistema de tracking (MLflow/W&B) junto con artefactos y metricas.
3) Validar targets: evaluar sensibilidad a `target_return` y `prediction_horizon`; agregar busqueda sistematica de hiperparametros (Optuna) y documentar la configuracion elegida.
4) Mejorar evaluacion: incluir PR-AUC por clase, curva calibracion y metrica orientada a trading (PnL simulado, Sharpe, max drawdown) en el reporte; integrar backtest en el script de entrenamiento para evitar discrepancias.
5) Entrenamiento final coherente: usar dataloaders ordenados temporalmente (o sampler secuencial) para que batches reflejen drift; mantener el escalado/seleccion solo con train. Si se entrena un ensamble, variar seeds y si es posible arquitectura o bootstrap de muestras.
6) Robustez operativa: validar que `db_manager` entregue datos completos; añadir chequeos de data sufficiency, manejo de outliers y tests unitarios para `SequenceDataset`/splits (evitando solapamiento train-val-test).
7) Despliegue: guardar version de `feature_names` y `selected_features` en el meta, incluir prueba de carga en `test_predictor_load.py` que verifique compatibilidad de shapes y que el modelo produce probabilidades calibradas.

## Valoracion
El pipeline muestra buenas practicas (walk-forward, multitarea con Focal Loss, ensamble, atencion) y puede producir modelos competitivos. Sin embargo, las fugas de datos en el entrenamiento final y la seleccion de features, junto con la falta de trazabilidad y evaluacion orientada a negocio, lo alejan de un estandar institucional. Corrigiendo el leakage, incorporando seguimiento de experimentos y metrica financiera de validacion, el sistema estaria mucho mas cerca de modelos robustos listos para produccion.
