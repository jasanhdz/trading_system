# 🧠 Phantom AI: Reinforcement Learning Evolution Whitepaper

**Autor:** Antigravity (Quant / RL Lead)  
**Fecha:** Abril 2026  
**Objetivo del Documento:** Proveer una radiografía histórica y arquitectónica de cómo se ha programado, mutado y evaluado el "Horno de IAs" (El Coliseo) desde su concepción en la V30 hasta su refinamiento de Francotirador en la V33. 

---

## 1. La Arquitectura del Coliseo (El Horno)

Nuestro sistema de entrenamiento no es un modelo estándar; es una simulación darwiniana a la que llamamos **El Coliseo**. 

### ¿Cómo funciona la Ingeniería?
*   **Vectorización de Matrices (NumPy):** En lugar de correr lentos bucles de Python, el `matrix_env.py` simula **2048 canales de trading paralelos** ejecutando operaciones tensoriales nativas en C++. Esto permite simular meses de datos en segundos.
*   **Torneo de Supervivencia:** Usamos `matrix_trainer.py` para aislar el entrenamiento en Subprocesos de GPU. Mientras una IA mutante (Retador) se entrena durante 16 Millones de pasos, el árbitro en la CPU toma al Campeón Invvicto y lo hace pelear contra el mutante. Si el Mutante gana en PnL y mantiene un Riesgo (Drawdown) inferior al 95%, es coronado como el nuevo Campeón.
*   **Evaluación P25:** Para evitar que un modelo suicida ascienda por pura suerte con una semilla (seed) buena, forzamos a los modelos a operar en 5 mercados distintos. Tomamos su segundo peor resultado (Percentil 25) y lo usamos como su calificación final.

---

## 2. Evolución de la Receta (Bitácora de Versiones)

El núcleo de Reinforcement Learning es el diseño de Recompensas (Reward Shaping). A continuación, el registro de lo que cambiamos, por qué lo hicimos, y cómo reaccionaron las redes neuronales.

### 🧬 V30: El Fundamento Base
*   **La Receta:** Modelo nativo PPO con `gamma=0.99`. Recompensas simétricas por ganancias. 11 Features de entrada (Velas, RSI, EMAs cortas, Z-Score de CVD).
*   **El Problema:** La IA no sabía gestionar el lateral. Entraba en operaciones y se quedaba atorada durante 20 horas esperando un milagro.
*   **Resultados:** PnL oscilaba cerca de $12 - $14, pero el Drawdown tocaba picos letales.

### 🧬 V31: Estabilización Defensiva
*   **La Receta:** Introdujimos el filtro restrictivo de muerte (`max_dd > 0.80`) y penalizaciones agresivas por quedarse quieto (`Idle Penalty`).
*   **El Problema:** El modelo se estancó en un mínimo local. Aprendió que la mejor forma de no morir era no hacer nada. Literalmente se cruzaba de brazos ("Hold Forever") para evitar castigos.
*   **Resultados (El Muro de Cristal):** El Campeón se congeló matemáticamente alrededor de los **$16.50**, incapaz de romper la barrera de los $25 porque reprimió su agresividad.

### 🧬 V32: El Protocolo Sniper y Las 15 Dimensiones
*   **La Receta (Cirugía Mayor):**
    1.  **Nuevos Ojos:** Expandimos el estado a **15 Features** (MTF EMA Slopes, Volume Z-Score, CVD Divergence).
    2.  **Inference Drift Fix (CRÍTICO):** Descubrimos que el bot en vivo descargaba solo 100 velas, por lo que la EMA 200 y 4H colapsaban por falta de "Warmup". Lo arreglamos forzando la descarga de 1000 velas en `inference_server.py`. Tu PnL real dejó de sangrar.
    3.  **RSI Exhaustion Masking (-1.5):** Castigo eléctrico letal si la IA oprimía LONG en sobrecompra extrema (RSI > 80).
    4.  **Sniper Organic Bonus (+1.0):** Recompensa gigante por cobrar entre +10% a +25% ROE antes de toparse con el Muro Lateral.
    5.  **Bleeding Penalty (-0.05):** Castigo por aguantar posiciones en rojo (Táctica Anti-Hold).
*   **El Comportamiento de la IA:** Las IAs mutantes se volvieron locas. Cayeron al "Valle de Aprendizaje", con PnLs de **$9 a $14**. El castigo de -1.5 del RSI los aniquilaba cada vez que exploraban locuras.

### 🧬 V33: Convergencia Cuantitativa (El Ajuste Fino)
*   **La Receta:**
    1.  **Gamma `0.99` → `0.95`:** Forzamos a la máquina a ignorar matemáticamente el futuro lejano. Ahora solo le importan los próximos 100 minutos. Así nace el Verdadero Scalper.
    2.  **Timesteps `8M` → `16M`:** Al darle 15 Features, la red se sobrecargó de variables. Duplicamos su tiempo de estudio a 16 millones de iteraciones (3-4 horas por epoch).
    3.  **Entry Bonus `0.1` → `0.15`:** Le dimos 3 velas de gracia a sus entradas para que el "Bleeding Penalty" no la inmovilizara.
    4.  **Clip Range `0.2` → `0.15`:** Menos bandazos violentos para el estabilizador de gradientes.
*   **El Comportamiento de la IA:**
    En las Iteraciones 1 y 2, su PnL se estabilizó bajo (**$10.85**), PERO su Drawdown (Riesgo) mejoró a un escandaloso **88.2%**. El modelo entró en el "Valle de la Cautela".

---

## 3. Diagnóstico Actual y El Siguiente Paso (V33.1)

### ¿Por qué seguimos estancados en ~$10.85 PnL?
Al combinar el `gamma=0.95` (no aguantar Trades largos) con el `RSI Exhaustion = -1.5` (electrocutarla si arriesga en FOMO), creamos una máquina **ultra-paranoica**.
La Inteligencia Artificial prefiere **No Operar** antes que comerse multas letales. Obtiene ganancias mínimas con cero riesgo.

### El Camino Hacia la Agresividad ($25.00+):
Para que nuestro "Francotirador Paranoico" vuelva a apretar el gatillo con ganas, debemos relajar el castigo letal del RSI, confiando en que el Gamma 0.95 lo defenderá de holdear estupideces.

*   **Siguiente Ajuste:** Bajar el Castigo de RSI de `-1.5` a `-0.3`.
*   **Premisa Experimental:** Permitirle comprar picos extremos (Momentum) sabiendo que si se equivoca, cortará el trade instintivamente sin morir electrocutado en la primera vela.
