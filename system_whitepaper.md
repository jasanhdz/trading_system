# 🏛️ El Sistema de Trading de 4 Pilares: Whitepaper Técnico

**Versión:** 2.1 (Híbrido Sage-Ninja)
**Fecha:** 28 de Diciembre, 2025
**Estado:** Producción

---

## 1. Resumen Ejecutivo

Este documento detalla la arquitectura de nuestro Sistema de Trading Algorítmico de Alta Frecuencia. El sistema opera sobre una **Arquitectura de 4 Pilares**, diseñada para maximizar el "alpha" (retorno) mientras gestiona estrictamente el riesgo mediante un híbrido de Machine Learning (ML) y Lógica Determinista.

**Filosofía Central:** "Entrada Agresiva, Salida Inteligente."
Utilizamos señales de ML con umbrales bajos para capturar movimientos de tendencia tempranos, protegidos por un protocolo de salida "Ninja" de múltiples capas que asegura ganancias y corta pérdidas basándose en cambios de probabilidad en tiempo real.

---

## 2. Los 4 Pilares de la Arquitectura

### 🏛️ Pilar I: Recolección de Datos e Ingeniería de Características
*   **Objetivo:** Alimentar los modelos de ML con datos de mercado de alta fidelidad.
*   **Mecanismo:**
    *   **Mecanismo de "Snapshots" (La Verdad del Mercado):**
        *   A diferencia de sistemas tradicionales que usan velas OHLCV (que pueden estar obsoletas por minutos), nuestro sistema captura una **Instantánea del Libro de Órdenes cada 10 segundos**.
        *   **¿Qué contiene una Instantánea?**
            1.  **OBI (Order Book Imbalance):** Calculamos la presión de compra/venta en los primeros 20 niveles de profundidad. Fórmula: `(VolCompra - VolVenta) / (VolCompra + VolVenta)`.
            2.  **Spread Dinámico:** Medimos la distancia entre el mejor Bid y Ask para detectar liquidez.
            3.  **Profundidad (Depth):** Suma total de volumen disponible para absorber órdenes de mercado.
            4.  **Volumen Taker:** Diferenciamos agresivamente entre quién *inicia* la operación (Comprador vs Vendedor) para detectar urgencia.
    *   **Almacenamiento:** Base de datos SQLite (`market_data_v2.db`) con indexación por `timestamp` para recuperación instantánea de secuencias históricas.

### 🏛️ Pilar II: El "Consejo de Sabios" (Ensamble de Modelos)
*   **Objetivo:** Predecir la dirección del precio mediante el consenso de múltiples inteligencias artificiales.
*   **Arquitectura:**
    *   **El Consejo:** No dependemos de un solo algoritmo. Utilizamos un **Ensamble de Votación Ponderada** compuesto por 4 arquitecturas distintas, cada una experta en un área:
        1.  **XGBoost (El Estadístico):** Experto en datos tabulares y niveles de precios exactos.
        2.  **LSTM (El Historiador):** Red Neuronal Recurrente que recuerda secuencias temporales largas.
        3.  **TCN (El Analista Técnico):** Red Convolucional Temporal que detecta patrones visuales en el gráfico.
        4.  **Transformer (El Visionario):** Mecanismo de atención que detecta relaciones complejas no lineales.
    *   **Mecanismo de Consenso:**
        *   Cada modelo emite su voto (Probabilidad Long/Short).
    *   **Pipeline de Entrenamiento (La Escuela de los Sabios):**
        *   **Ventana Deslizante (Rolling Window):** No entrenamos una vez y olvidamos. Cada día, el sistema toma los últimos 30 días de datos, descarta lo viejo y reentrena desde cero. Esto permite que los modelos se adapten a la "personalidad" cambiante del mercado (ej. de alcista a lateral).
        *   **Etiquetado Inteligente (Triple Barrier Method):**
            *   No enseñamos al modelo a predecir "si el precio sube". Le enseñamos a predecir "si el precio subirá lo suficiente para cubrir comisiones y riesgo antes de tocar el Stop Loss".
            *   Usamos el Ratio de Sharpe para definir si una operación fue "exitosa" en el pasado.
        *   **Validación Cruzada Temporal:** Para evitar que el modelo "haga trampa" memorizando el futuro, usamos un esquema de validación donde el set de prueba es siempre *posterior* al set de entrenamiento.
    *   **Entradas del Modelo (13 Características):**
        *   **Libro de Órdenes:** `bid_depth`, `ask_depth`, `obi_5`, `obi_10`, `obi_20`, `micro_price`, `bid_ask_spread`.
        *   **Derivados:** `funding_rate`, `open_interest`.
        *   **Volumen Agresivo:** `taker_buy_vol`, `taker_sell_vol`.
        *   **Calculadas:** `buy_sell_ratio`, `depth_imbalance`.
        *   **Deep Learning (LSTM/TCN/Transformer):** Reciben una **Secuencia de Tensores (Batch, 12, 13)** que representa los últimos 2 minutos de "película" del mercado (12 snapshots de 10s).
    *   **Salida:** Una "Súper-Probabilidad" unificada (0.0 - 1.0) que representa la convicción del Consejo.

### 🏛️ Pilar III: El Servicio de Inferencia REST (Python)
*   **Objetivo:** Servir predicciones del modelo al bot de trading con latencia sub-milisegundo.
*   **Tecnología:** FastAPI (Python).
*   **Función:**
    *   Expone endpoints (ej. `/predict`) que el Bot de Trading consulta.
    *   Carga los últimos modelos entrenados (`.joblib`) en memoria.
    *   Realiza el cálculo de características en tiempo real sobre las solicitudes entrantes.

### 🏛️ Pilar IV: El Bot de Ejecución (TypeScript)
*   **Objetivo:** Ejecutar operaciones, gestionar el riesgo y manejar eventos del ciclo de vida.
*   **Tecnología:** Node.js / TypeScript.
*   **Lógica Central (`StrategyRunner.ts`):**
    *   **El Cerebro de Ejecución (`tick()`):**
        1.  **Sincronización:** Cada 1000ms, el bot despierta.
        2.  **Consulta al Oráculo:** Envía los datos actuales al Servicio Python y recibe la probabilidad (ej. `LONG: 0.42`).
        3.  **Evaluación de Umbral Dinámico:**
            *   El bot consulta `thresholds_config.json`.
            *   Si el activo es volátil (ej. DOGE), el umbral exigido será alto (0.50).
            *   Si el activo es estable (ej. BNB), el umbral será bajo (0.30).
            *   *Lógica:* `Si Probabilidad > Umbral -> INTENCIÓN DE COMPRA`.
        4.  **Gestión de Posición (Ninja Protocol):**
            *   **Capa 1: Pánico (Reacción Rápida):** Si la probabilidad del modelo cae por debajo de 0.50 (se vuelve en contra), cerramos inmediatamente. No esperamos al Stop Loss. "Si la razón de entrada desaparece, la posición desaparece".
            *   **Capa 2: Agotamiento (Neutralidad):** Si el mercado se pone lateral (Probabilidad Neutral > 0.60) y ya ganamos >5%, cerramos. Mejor pájaro en mano.
            *   **Capa 3: Trailing Stop (Dejar Correr):** Si el precio sube, subimos el Stop Loss detrás de él.
                *   Ganancia > 10%: Stop a distancia del 40%.
                *   Ganancia > 30%: Stop a distancia del 20%.
                *   Ganancia > 50%: Stop a distancia del 10% (Asegurar victoria).

---

## 3. Flujo del Sistema: El Ciclo de Vida de una Operación

1.  **La Chispa:** Binance WebSocket envía una actualización de operación para `ETHUSDT`.
2.  **El Cerebro (El Consejo):** El Servicio Python procesa la secuencia de snapshots y consulta a los 4 modelos (XGBoost, LSTM, TCN, Transformer).
    *   *Votación:* XGBoost (0.45), LSTM (0.39), TCN (0.44), Transformer (0.40).
    *   *Consenso:* El `EnsembleManager` pondera los votos y emite el veredicto final. Resultado: `Probabilidad LONG: 0.42`.
3.  **El Portero:** El Bot TypeScript ve `0.42` > Umbral `0.40`.
4.  **La Entrada:** El Bot envía orden `MARKET BUY` a Binance.
6.  **La Vigilancia:** El Bot monitorea la posición cada segundo.
    *   *Escenario A:* El precio sube un 8%. El Bot activa el Trailing Stop.
    *   *Escenario B:* El precio se estanca. La Probabilidad ML cae a `0.45` (Neutral). El Bot cierra con pequeña ganancia.
    *   *Escenario C:* El mercado cae. La Probabilidad ML cambia a `SHORT 0.55`. El Bot activa **Salida de Pánico** para limitar la pérdida.

---

## 4. Logros y Rendimiento

*   **Disponibilidad:** 99.9% de tiempo de actividad mediante gestión de procesos PM2.
*   **Latencia:** <50ms desde la ingesta de datos hasta la ejecución de la orden.
*   **Adaptabilidad:** Transición exitosa de "Umbrales Estáticos" a "Umbrales Dinámicos" (MlConfigWatcher), permitiendo optimización por activo.
*   **Gestión de Riesgo:** La "Salida Ninja" ha reducido significativamente el drawdown cortando tempranamente las "Posiciones Zombi" (operaciones estancadas).

## 5. Recomendaciones y Hoja de Ruta Futura

1.  **Modo Sniper (Q1 2026):** Reemplazar Entradas a Mercado con Órdenes Limit para capturar el spread (Comisiones Maker).
2.  **Meta-Etiquetado:** Entrenar un modelo secundario para predecir la *fiabilidad* del modelo primario, filtrando falsos positivos en mercados agitados.
3.  **Tamaño por Volatilidad:** Implementar el Criterio de Kelly para dimensionar posiciones basándose en la convicción (apostar más fuerte en alta probabilidad, menos en baja).

---

**Aviso de Confidencialidad:** Este documento contiene detalles arquitectónicos propietarios del Sistema de Trading.
