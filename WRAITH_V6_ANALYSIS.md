# 🦅 Wraith V6: Análisis del Sistema "Time Sentinel" (BTC)

Este documento detalla la arquitectura, entrenamiento y lógica operativa del sistema **Wraith V6**, un modelo de Deep Learning especializado en operar **Shorts en Bitcoin (BTC)**.

## 1. Arquitectura del Sistema

El sistema no es un solo script, sino una orquestación de 4 componentes fundamentales.

### 📂 Archivos Involucrados

| Archivo | Función | Descripción |
| :--- | :--- | :--- |
| **`scripts/backtest_wraith_v6.py`** | **El Ejecutor** | Motor de backtest que simula la operativa, aplica los filtros de horario ("Time Sentinel") y gestiona el riesgo (SL/TP). |
| **`scripts/train_wraith_dqn.py`** | **El Entrenador** | Script que entrena la red neuronal (DQN). Aquí vive la "inteligencia" que aprende a filtrar las señales. |
| **`scripts/detect_distribution_tops.py`** | **El Ojo** | Ingeniero de características. Calcula la física del precio (velocidad, aceleración) y detecta los candidatos iniciales. |
| **`data/storage/database_manager.py`** | **La Memoria** | Gestiona la conexión con la base de datos SQLite (`data/binance_candles.db`) para extraer las velas de 5 minutos. |

---

## 2. La "Caja Negra": Detalles del Entrenamiento

El modelo **WraithNet** es un agente de Reinforcement Learning (DQN) entrenado para ser un **Francotirador de Shorts**.

*   **Activo:** Bitcoin (BTC/USDT).
*   **Timeframe:** 5 minutos (5m).
*   **Dirección:** **SOLO SHORTS** (El modelo solo tiene dos acciones: `PASS` o `SHORT`).
*   **Data de Entrenamiento:** 5,000 velas (aprox. 17 días de historia reciente para el entrenamiento).
*   **Duración del Entrenamiento:** **150 Episodios** (El agente recorrió la data 150 veces para aprender).
*   **Arquitectura Neuronal:**
    *   Entrada: 6 Neuronas (Distancia a EMA, Velocidad, Aceleración, Volatilidad Z, Distancia BB, Ratio Volumen).
    *   Capas Ocultas: 64 -> 32 neuronas.
    *   Salida: 2 Neuronas (Probabilidad de PASS vs SHORT).
*   **Función de Recompensa:** `PnL - (Drawdown * 2.0)`.
    *   *Traducción:* El modelo es castigado **el doble** por sufrir drawdown que lo que es premiado por ganar dinero. Esto lo fuerza a buscar entradas "limpias" donde el precio cae casi inmediatamente.

---

## 3. El "Time Sentinel": Lógica de Horarios

En `backtest_wraith_v6.py`, encontramos un filtro rígido llamado **Time Sentinel**. Este filtro bloquea operaciones en momentos estadísticamente perdedores para esta estrategia.

### 🚫 Días Prohibidos: Martes
```python
FORBIDDEN_DAYS = ['Tuesday']
```
**¿Por qué?** En la microestructura de BTC, los martes suelen ser días de "Trend Continuation" o reversiones sucias que atrapan a los sistemas de reversión a la media (como este). El backtest demostró que eliminar los martes aumentaba drásticamente el Profit Factor.

### 🚫 Horas Prohibidas (UTC)
```python
FORBIDDEN_HOURS = [1, 4, 5, 10, 13, 18, 19, 23]
```
Estas horas corresponden a momentos de baja liquidez o "trampas" de apertura de mercados (Asia/Londres/NY) donde la volatilidad es errática pero no direccional.

---

## 4. Resultados del Backtest (Verificación Actual)

Acabamos de re-ejecutar el backtest (`backtest_wraith_v6.py`) sobre 50,000 velas (aprox. 6 meses).

### 📊 Métricas Clave
*   **Balance Inicial:** $20.00
*   **Balance Final:** **$41.80**
*   **Retorno Total:** **+109.00%**
*   **Profit Factor:** **2.23** (Por cada $1 perdido, gana $2.23)
*   **Win Rate:** **58.97%**
*   **Drawdown Máximo:** 18.08%
*   **Total Trades:** 39 (aprox. 1.5 trades por semana - Alta selectividad).

### 🏆 Análisis de Salidas
*   **Trailing Stop (19 trades):** La mayoría de las ganancias vienen de dejar correr la ganancia y salir cuando el precio se devuelve.
*   **Time Limit (14 trades):** Salidas por tiempo (el precio no hizo nada en 4 horas).
*   **Stop Loss (6 trades):** Solo 6 pérdidas directas en 6 meses.

---

## 5. ¿Cómo funciona paso a paso?

1.  **Detección Física (`detect_distribution_tops.py`):**
    *   El sistema escanea el mercado buscando una "Firma de Colapso": Precio cerca de la EMA 200 + Desaceleración del movimiento + Volumen bajando.
    *   Esto genera una lista de "Candidatos".

2.  **El Juicio de la IA (`train_wraith_dqn.py`):**
    *   Cada candidato pasa por la red neuronal **WraithNet**.
    *   La IA analiza 6 factores de física pura (Velocidad, Aceleración, etc.).
    *   Si la IA tiene una confianza > 85% (`CONFIDENCE_THRESHOLD = 0.85`), aprueba el trade.

3.  **El Filtro Sentinel (`backtest_wraith_v6.py`):**
    *   Antes de ejecutar, el script verifica: ¿Es Martes? ¿Es una hora prohibida?
    *   En este backtest, el Sentinel bloqueó **198 operaciones** que la IA quería tomar pero que ocurrían en malos horarios.

4.  **Ejecución:**
    *   Entra en **SHORT** con apalancamiento 5x.
    *   Si el precio cae y ganamos 5%, movemos el Stop Loss a Break-Even (riesgo cero).
    *   Si sigue cayendo, un Trailing Stop persigue el precio para asegurar ganancias.

---

### Conclusión
El sistema **Wraith V6** es rentable porque **no opera casi nunca**. Combina una detección de patrones físicos (Candidatos) con un filtro de IA (DQN) y un filtro estadístico de tiempo (Sentinel). Su ventaja no es la velocidad, sino la **paciencia**.
