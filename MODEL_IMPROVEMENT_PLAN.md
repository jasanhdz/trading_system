# Plan de Excelencia en Modelado Algorítmico (Roadmap to Excellence)

Este documento describe las estrategias avanzadas para elevar el rendimiento de los modelos de trading más allá de la configuración estándar, buscando consistencia, robustez y rentabilidad superior (Sharpe > 2.0).

## 1. Ensemble Learning (El "Consejo de Sabios")
En lugar de depender de un solo oráculo, creamos un comité de expertos.

*   **Concepto:** Entrenar múltiples modelos con arquitecturas o semillas diferentes y combinar sus predicciones.
*   **Implementación:**
    *   Entrenar 5 instancias de `DeepTemporalNet` con diferentes semillas aleatorias.
    *   Entrenar modelos heterogéneos: 1 LSTM, 1 GRU, 1 Transformer (TFT), 1 XGBoost.
    *   **Votación:** Tomar la media de las probabilidades (Soft Voting) o requerir consenso (Hard Voting).
*   **Beneficio:** Reduce la varianza y filtra el ruido. Si un modelo "alucina" una señal, los otros lo corrigen.
*   **Costo:** Aumenta linealmente el tiempo de entrenamiento e inferencia.

## 2. Feature Engineering de Nueva Generación (Datos Alternativos)
Los modelos actuales son "ciegos" a lo que no sea precio y volumen.

*   **Micro-estructura de Mercado:**
    *   **Order Book Imbalance:** Presión real de compra/venta en el libro de órdenes.
    *   **Trade Flow:** Diferencia entre volumen taker de compra vs venta.
*   **Datos de Derivados:**
    *   **Funding Rates:** Sentimiento del mercado (si es muy positivo, el mercado está sobre-apalancado en long).
    *   **Open Interest:** Interés abierto y su variación (entrada/salida de capital).
    *   **Liquidations:** Picos de liquidaciones suelen marcar suelos/techos locales.
*   **Correlaciones:**
    *   Features de BTC y ETH como input para las altcoins (Beta hedging implícito).

## 3. Optimización Bayesiana de Hiperparámetros (Optuna)
Dejar de adivinar los parámetros y usar matemáticas para encontrarlos.

*   **Herramienta:** Optuna o Ray Tune.
*   **Espacio de Búsqueda:**
    *   Learning Rate (log scale).
    *   Hidden Dimensions (64 a 512).
    *   Num Layers (2 a 6).
    *   Dropout (0.1 a 0.5).
    *   Sequence Length (24 a 96 velas).
*   **Proceso:** Ejecutar 50-100 trials pequeños para encontrar la configuración óptima para *cada* moneda individualmente.
*   **Beneficio:** Adapta el modelo a la "personalidad" única de cada activo (volatilidad, ruido).

## 4. Meta-Labeling (El "Filtro de Calidad")
Un segundo cerebro que decide cuándo operar.

*   **Modelo Primario:** Dice "¿Dirección? Long o Short". (Alta sensibilidad).
*   **Modelo Secundario:** Dice "¿Probabilidad de Acierto? Sí o No". (Alta precisión).
*   **Input Secundario:** Volatilidad, hora del día, spread, confianza del modelo primario.
*   **Resultado:** El modelo secundario aprende a filtrar las señales falsas del primario en condiciones de mercado adversas (ej. rango lateral estrecho).

## 5. Función de Pérdida Personalizada (Sharpe Loss)
Entrenar al modelo para ganar dinero, no para acertar la dirección.

*   **Problema:** `CrossEntropy` optimiza la clasificación (acertar la clase), pero no distingue entre un acierto de +0.1% y uno de +5%.
*   **Solución:** Implementar `Differentiable Sharpe Ratio Loss`.
*   **Efecto:** El modelo es penalizado fuertemente si falla en movimientos grandes, y recompensado más por capturar tendencias fuertes que por ruido.

## 6. Aprendizaje Continuo (Online Learning)
El mercado cambia, el modelo debe adaptarse.

*   **Estrategia:** Re-entrenamiento incremental semanal o diario.
*   **Fine-tuning:** Tomar el modelo base y entrenarlo por 1-2 epochs con los datos de la última semana antes de cada sesión de trading.
*   **Beneficio:** Mantiene el modelo sincronizado con el régimen de mercado actual (alcista, bajista, lateral).

---
**Próximo Paso Recomendado:**
Si los modelos actuales (V2) no alcanzan los objetivos, implementar **Optuna (Punto 3)** es la ruta más eficiente en términos de costo-beneficio antes de añadir complejidad de datos o arquitecturas.
