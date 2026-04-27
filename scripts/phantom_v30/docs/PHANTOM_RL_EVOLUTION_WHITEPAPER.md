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

---

## 4. V34: De "Granja de Dopamina" a Maximizador Cuantitativo ($2k Goal)

### La Epifanía del Mercado Lateral y el "Bono Plano"
En versiones anteriores insertamos Bonos Planos (`+1.0` por hacer Sniper) para obligar a la IA a cobrar rápido ante un mercado altamente lateral (ETH estancado). Fue un parche necesario porque con `gamma=0.99` la máquina no sentía urgencia por el tiempo.
Sin embargo, esto creó un nuevo problema: **Farming de Dopamina**.
La IA ganaba `+0.15` por la ganancia financiera real y `+1.0` de bono artificial. La motivación real de la IA era cobrar centavos para escuchar el *Ding* del bono artificial de 1.0, limitando sus ganancias a topes absurdos ($15-$20) y perdiéndose todas las tendencias grandes.

### La Receta: Multiplicadores Relacionales
Ya que el nuevo `gamma=0.95` naturalmente introduce el sentido de urgencia por cobrar rápido (creando un Scalper nato ideal para mercado lateral), **erradicamos los bonos artificiales**.
*   **Ajuste:** El Sniper Bonus (`+1.0`) y el Execution Bonus (`+0.5`) se reemplazaron por **Multiplicadores Exponenciales** (`reward = reward * 1.5` y `profit_pct * 15.0`).
*   **Razón Matemática:** Ahora, si cierra con +1 Centavo, el bono 1.5x le paga migajas. Pero si cierra valientemente con +$5 dólares, el bono se vuelve gigantesco. Le enseñamos a maximizar el dinero (Compounding), no la cantidad mecánica de clics.

---

## 5. V35: Ojos de Segunda Derivada + Exploración Risk-Seeking

### Consultoría con Experto Externo (Lead Quant PhD RL/HFT)
Se consultó a un experto especializado en modelos cuantitativos estilo Jane Street/RenTech. De sus 3 propuestas de mutación, se aceptaron 2 después de audité contra el código real y se rechazó 1 por bugs matemáticos fatales.

### Mutación 3 (ACEPTADA): Asymmetric Momentum Accelerator
*   **Qué es:** Features de 2ª derivada (aceleración) de las slopes EMA 1H, 4H y CVD ROC, computadas en `tensor_loader.py`.
*   **Por qué funciona:** El Transformer ahora VE directamente si el momentum se está acelerando ANTES de un breakout. Ya no necesita deducirlo implícitamente de los slopes crudos. `N_FEATURES` pasó de 15 a 18.
*   **Código:** `ema_1h_accel = diff(ema_1h_slope) × 2000`, clips a [-30, 30].

### Mutación 2 (ACEPTADA, moderada): Risk-Seeking Entropy Schedule
*   **Qué es:** Un callback que decae la entropía de `0.15 → 0.04` y el clip_range de `0.20 → 0.12` durante el entrenamiento.
*   **Por qué funciona:** Al inicio el agente explora agresivamente (entropía alta) para descubrir estrategias de outlier. Al final se estabiliza (entropía baja) para no perder lo aprendido. Los valores originales del experto (0.25 / 0.28) eran demasiado agresivos para nuestro espacio discreto de 4 acciones.

### Mutación 1 del Experto (RECHAZADA): Exponential Compounding Hunter
*   **Por qué se rechazó (2 bugs fatales):**
    1.  `equity_growth = log(equity/INITIAL_BALANCE)` = 0 al inicio → el power-law da CERO cuando más se necesita la señal (gallina-y-huevo).
    2.  `peak_relative` es casi siempre ≤ 0, lo cual penaliza trades ganadores durante recuperaciones.
    3.  En la versión "Fixed", `compounding_bonus = 12.0` se inyectaba en TODOS los steps sin trades (bug de gating).
*   **Alternativa V34 mantenida:** `profit_pct × 15.0` + `organic_sniper × 1.5` funciona desde el primer centavo.
