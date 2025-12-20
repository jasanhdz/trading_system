# Plan Maestro de Ingeniería: Sistema de Trading Institucional Autónomo (v3.0)

## Visión
Transformar el sistema actual en una infraestructura de trading algorítmico de grado institucional, capaz de **auto-optimizarse**, adaptarse a regímenes de mercado cambiantes y gestionar el riesgo mediante consenso de modelos y datos alternativos.

---

## Pilar 1: La Refinería de Datos (Feature Engineering 2.0)
*El modelo es tan bueno como los datos que consume. OHLCV ya no es suficiente.*

### 1.1. Microestructura de Mercado (Order Book)
No solo ver el precio, sino la *intención* detrás del precio.
*   **Order Book Imbalance (OBI):** Diferencia entre volumen de Bids y Asks en el tope del libro (Top 10/20 levels).
*   **Spread & Depth:** Liquidez real disponible. Si la liquidez desaparece, la volatilidad explota.
*   **Trade Flow:** Volumen agresivo (Taker) de compra vs venta. ¿Quién está empujando el precio?

### 1.2. Datos de Derivados (Sentimiento)
*   **Funding Rates:** El costo de mantener posiciones. Funding negativo alto = Posible Short Squeeze.
*   **Open Interest (OI):** ¿Está entrando dinero nuevo o saliendo?
    *   Precio sube + OI sube = Tendencia fuerte.
    *   Precio sube + OI baja = Debilidad (Short covering).
*   **Liquidations:** "Combustible" del mercado. Detectar cascadas de liquidación.

### 1.3. Datos Inter-Market (Correlaciones)
*   **Beta de BTC:** Calcular la correlación dinámica (rolling correlation) de ADA/AVAX con BTC en tiempo real.
*   **Dominancia:** BTC.D y USDT.D como filtros de régimen.

---

## Pilar 2: El Motor de Inferencia (Arquitectura de Modelos)
*Abandonar el concepto de "Un Modelo para Todo".*

### 2.1. Ensemble Learning ("El Consejo de Sabios")
En lugar de confiar en una sola red neuronal, crearemos un comité.
*   **Miembros del Comité:**
    1.  **LSTM Profundo:** (El actual) Bueno para secuencias temporales.
    2.  **TCN (Temporal Convolutional Network):** Excelente para detectar patrones locales y rupturas.
    3.  **XGBoost / LightGBM:** (Sobre features tabulares) Excelente para reglas de decisión rígidas.
    4.  **Transformer (TFT):** Para capturar dependencias a largo plazo y atención.
*   **Mecanismo de Votación:**
    *   **Soft Voting:** Promedio ponderado de probabilidades.
    *   **Veto:** Si un modelo detecta riesgo extremo, anula a los demás.

### 2.2. Función de Pérdida Financiera (Sharpe Loss)
*   **Problema Actual:** `CrossEntropy` optimiza la *precisión* (acertar dirección).
*   **Solución:** `Differentiable Sharpe Ratio Loss`.
*   **Objetivo:** Entrenar a la red para maximizar el ratio Retorno/Riesgo directamente. Penaliza la volatilidad de los retornos, no solo el error direccional.

### 2.3. Meta-Labeling ("El Filtro de Calidad")
*   **Modelo Primario:** Dice "¿Long o Short?".
*   **Modelo Secundario (Meta):** Dice "¿Debo operar esta señal?".
*   **Input del Meta-Modelo:** Volatilidad, hora del día, spread, confianza del modelo primario, estado de BTC.
*   **Resultado:** Filtra los falsos positivos en mercados laterales.

---

## Pilar 3: La Fábrica de Modelos (Auto-ML & MLOps)
*El sistema debe mejorarse a sí mismo sin intervención humana constante.*

### 3.1. Optimización Bayesiana (Optuna)
*   Dejar de adivinar hiperparámetros (`lr`, `layers`, `hidden_dim`).
*   **Implementación:** Un script que corre semanalmente, prueba 100 combinaciones usando búsqueda bayesiana (aprende de los intentos anteriores) y encuentra la configuración matemática perfecta para el mercado actual.

### 3.2. Pipeline de Re-entrenamiento Continuo (CI/CD for ML)
*   **Trigger:** Semanal o cuando el rendimiento (Sharpe) cae por debajo de un umbral (Drift Detection).
*   **Proceso:**
    1.  Descargar nuevos datos.
    2.  Ejecutar Optuna (búsqueda rápida).
    3.  Entrenar Ensemble.
    4.  Validar contra Backtest (Walk-Forward).
    5.  Si el Nuevo Modelo > Viejo Modelo → Desplegar automáticamente.

---

## Hoja de Ruta de Implementación (Roadmap)

### Fase 1: Cimientos de Datos (Semana 1-2)
- [ ] Implementar colector de Order Book y Funding Rates (Binance API).
- [ ] Crear base de datos para almacenar estos features de alta frecuencia (TimescaleDB o Parquet optimizado).
- [ ] Actualizar `dataset.py` para ingerir estos nuevos features.

### Fase 2: Optimización del Motor (Semana 3)
- [ ] Implementar **Optuna** en `train_production_ready.py`.
- [ ] Implementar **Sharpe Loss** customizada en PyTorch.
- [ ] Validar mejora de rendimiento en ADA/AVAX.

### Fase 3: Robustez (Semana 4)
- [ ] Crear arquitectura de **Ensemble** (Entrenar 3 modelos distintos y promediar).
- [ ] Implementar **Meta-Labeling** (Modelo secundario XGBoost sobre los errores del primario).

### Fase 4: Automatización (Semana 5+)
- [ ] Script de auto-evaluación y re-entrenamiento (`auto_train.py`).
- [ ] Dashboard de monitoreo de salud de modelos (Drift monitoring).

---

## Requerimientos Técnicos
*   **Hardware:** Las 2x AMD RX 6600 son suficientes para entrenar, pero necesitaremos optimizar la VRAM (Gradient Checkpointing) si usamos Ensembles grandes.
*   **Software:** PyTorch, Optuna, CCXT Pro (para Websockets de Order Book), Pandas TA.

---
**Filosofía:** "No buscamos predecir el futuro, buscamos probabilidades asimétricas favorables gestionadas por una ingeniería superior."
