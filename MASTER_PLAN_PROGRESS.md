# 🏗️ Bitácora del Plan Maestro de Ingeniería (Trading Institucional)

**Estado General:** 🟡 EN PROCESO
**Versión del Sistema:** 2.1 (Legacy ADA + BTC Filter + Data Collector V2)
**Fecha de Inicio:** 2025-12-20

---

## 🗺️ Mapa de Ruta y Progreso

### 🟢 Fase 1: La Refinería de Datos (Data Refinery)
*Objetivo: Capturar datos de microestructura que los modelos actuales no ven.*

- [x] **Diseño de Base de Datos V2:** SQLite independiente para datos de alta frecuencia.
- [x] **Implementación del Colector:** Script `market_data_collector.py` capturando Order Book (OBI), Spread, Funding Rates.
- [x] **Despliegue:** Servicio `data-collector-v2` corriendo en PM2.
- [ ] **Validación de Datos:** Verificar integridad tras 1 semana de recolección.
- [ ] **Feature Engineering V2:** Crear pipeline para transformar estos datos crudos en tensores para PyTorch.

### 🟡 Fase 2: El Motor de Inferencia (Model Engine)
*Objetivo: Cambiar cómo aprenden los modelos (de "Adivinar" a "Ganar").*

- [x] **Infraestructura Optuna:** Script base para optimización bayesiana creado y validado.
- [x] **Lección Aprendida (Exp. 1):** Optimizar `Accuracy` no garantiza rentabilidad.
- [x] **Implementación Sharpe Loss:** Función de pérdida diferenciable implementada.
- [x] **Optimización Sharpe:** Script `optimize_hyperparameters.py` actualizado para maximizar Sharpe Ratio.

### 🟢 Fase 3: La Fábrica de Modelos (Model Factory)
*Objetivo: Diversidad cognitiva mediante múltiples arquitecturas (El Comité de Sabios).*

- [x] **LSTM Profundo:** Arquitectura base (`improved_architecture.py`).
- [x] **TCN (Temporal Convolutional Network):** Implementada (`tcn_model.py`) para patrones locales.
- [x] **Transformer (TFT):** Implementado (`transformer_model.py`) para atención global.
- [x] **XGBoost Wrapper:** Implementado (`tabular_model.py`) para análisis tabular rígido.
- [x] **Ensemble Manager:** Orquestador de votación.
- [x] **Meta-Labeling:** Entrenar modelo secundario para filtrar falsos positivos.

### 🔴 Fase 4: Robustez y Automatización
*Objetivo: El sistema se cuida solo.*

- [x] **Filtro de Tendencia BTC:** Implementado en `strategy-runner.ts`. Bloquea operaciones contra la tendencia macro.
- [ ] **Pipeline CI/CD ML:** Re-entrenamiento automático semanal.

---

## 📅 Próximos Pasos (Agenda)

### 🎯 Hito: "El Despertar de los Sabios"
**Fecha Objetivo:** 28 de Diciembre de 2025 (1 Semana de Datos)

1.  **Validación de Datos:** Verificar integridad de `market_data_v2.db`.
2.  **Entrenamiento Piloto:** Entrenar el Ensemble con los primeros 10k registros.
3.  **Evaluación:** Si Sharpe > 1.5 en validación, considerar despliegue en modo "Paper Trading".

---

## 📝 Diario de Hallazgos y Decisiones

### 2025-12-20: El Fracaso del "Accuracy"
- **Experimento:** Se usó Optuna para maximizar Accuracy en ADAUSDT.
- **Resultado:** Se logró 47% Accuracy (vs 33% random), pero el modelo resultante era demasiado tímido (0 trades con confianza > 0.8).
- **Lección:** El Accuracy es una métrica vanidosa en finanzas. Un modelo puede acertar el 60% de las veces ganando 1 centavo y perder el 40% perdiendo 1 dólar.
- **Corrección:** Implementar `SharpeLoss` para que el gradiente descienda hacia la rentabilidad, no hacia la precisión.

### 2025-12-19: La Importancia del "Veto de BTC"
- **Hallazgo:** Los trades fallidos de ADA/AVAX coincidían casi siempre con una señal fuerte opuesta en BTC.
- **Solución:** Se implementó un bloqueo duro. Si BTC tiene prob > 0.55 en contra, no se opera.

---

## ⚙️ Estado Técnico Actual
*   **Bot:** Operativo (Legacy Model).
*   **Colector:** Activo (Recogiendo ~10k registros/día).
*   **GPUs:** AMD RX 6600 configuradas y funcionando con ROCm.
