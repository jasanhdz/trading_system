# 🏛️ El Sistema de Trading de 4 Pilares: Whitepaper Técnico

**Versión:** 3.0 (Consejo de Sabios + Transformer)
**Fecha:** 28 de Diciembre, 2025
**Estado:** Producción

---

## 1. Resumen Ejecutivo

Este documento detalla, con precisión quirúrgica, la arquitectura completa de nuestro Sistema de Trading Algorítmico de Alta Frecuencia. El sistema opera sobre una **Arquitectura de 4 Pilares**, diseñada para maximizar el "alpha" (retorno superior al mercado) mientras gestiona estrictamente el riesgo mediante un híbrido de Machine Learning (ML) y Lógica Determinista.

**Filosofía Central:** "Entrada Agresiva, Salida Inteligente."

---

## 2. Arquitectura de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                    BINANCE FUTURES API                          │
│                 wss://fstream.binance.com                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket (Order Book, Trades)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PILAR I: COLECTOR DE DATOS (Python)                            │
│  Proceso: pm2 "02-Data-Collector"                               │
│  Archivo: scripts/next_gen/market_data_collector.py             │
│  Intervalo: Loop cada 10 segundos                               │
│  Salida: SQLite market_data_v2.db                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ INSERT INTO orderbook_metrics
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  BASE DE DATOS: data/market_data_v2.db                          │
│  ├── orderbook_metrics (timestamp, symbol, obi_5, obi_10, ...)  │
│  └── derivatives_data (timestamp, symbol, funding_rate, ...)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ SELECT últimas 60 filas
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PILAR II: SERVICIO ML V2 (Python FastAPI)                      │
│  Proceso: pm2 "03-ML-Service-V2"                                │
│  Archivo: services/ml_service_v2.py                             │
│  Puerto: 8001                                                   │
│  Modelos: models/v2_ensemble/{SYMBOL}/*.pt, *.joblib            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST /ml-v2/predict
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PILAR III: BOT DE EJECUCIÓN (TypeScript)                       │
│  Proceso: pm2 "01-Trading-Bot"                                  │
│  Archivo: binance-futures-bot-ts/src/app/strategy-runner.ts     │
│  Intervalo: tick() cada 1000ms                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ API REST (Market Order)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BINANCE FUTURES API                          │
│                 POST /fapi/v1/order                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. PILAR I: Recolección de Datos (La Verdad del Mercado)

### 3.1 Proceso de Captura

**Archivo:** `scripts/next_gen/market_data_collector.py`
**Proceso PM2:** `02-Data-Collector`
**Intervalo:** Cada 10 segundos

### 3.2 Esquema de Base de Datos

```sql
-- Tabla 1: Métricas del Libro de Órdenes
CREATE TABLE orderbook_metrics (
    timestamp INTEGER,      -- Unix timestamp en milisegundos
    symbol TEXT,            -- Ej: "ETH/USDT:USDT"
    obi_5 REAL,             -- Order Book Imbalance (5 niveles)
    obi_10 REAL,            -- Order Book Imbalance (10 niveles)
    obi_20 REAL,            -- Order Book Imbalance (20 niveles)
    spread_pct REAL,        -- (Ask - Bid) / Mid Price
    mid_price REAL,         -- (Best Bid + Best Ask) / 2
    micro_price REAL,       -- Precio ponderado por volumen del top
    bid_depth_20 REAL,      -- Suma volumen 20 mejores bids
    ask_depth_20 REAL,      -- Suma volumen 20 mejores asks
    PRIMARY KEY (timestamp, symbol)
);

-- Tabla 2: Datos de Derivados
CREATE TABLE derivatives_data (
    timestamp INTEGER,
    symbol TEXT,
    funding_rate REAL,      -- Tasa de funding actual
    open_interest REAL,     -- Contratos abiertos
    open_interest_value REAL, -- Valor en USD
    taker_buy_vol REAL,     -- Volumen agresivo de compra (última vela)
    taker_sell_vol REAL,    -- Volumen agresivo de venta
    PRIMARY KEY (timestamp, symbol)
);
```

### 3.3 Fórmulas de Cálculo

**Order Book Imbalance (OBI):**
```python
def calculate_obi(bids, asks, depth):
    bid_vol = sum(b[1] for b in bids[:depth])  # Suma volumen bids
    ask_vol = sum(a[1] for a in asks[:depth])  # Suma volumen asks
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)
```
- **Interpretación:** OBI > 0 = Presión compradora. OBI < 0 = Presión vendedora.

**Micro-Price (Precio Justo Instantáneo):**
```python
micro_price = (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)
```

---

## 4. PILAR II: El Consejo de Sabios (Ensemble de Modelos)

### 4.1 Los 4 Modelos del Consejo

| Modelo | Arquitectura | Especialidad | Framework |
|--------|--------------|--------------|-----------|
| **XGBoost** | Gradient Boosting (1000 árboles) | Datos tabulares, niveles exactos | XGBoost 2.0 |
| **LSTM** | 2 capas LSTM (64 hidden units) | Memoria temporal larga | PyTorch |
| **TCN** | Temporal Convolutional Network [32, 64] | Patrones visuales | PyTorch |
| **Transformer** | 2 capas, 4 heads, d_model=64 | Relaciones no lineales | PyTorch |

### 4.2 Las 13 Características de Entrada

```python
feature_cols = [
    # Libro de Órdenes (Microestructura)
    'bid_depth',        # Liquidez de compra
    'ask_depth',        # Liquidez de venta
    'bid_ask_spread',   # Costo de transacción implícito
    'obi_5',            # Presión de corto plazo (5 niveles)
    'obi_10',           # Presión de mediano plazo
    'obi',              # Presión de largo plazo (20 niveles)
    'micro_price',      # Precio justo instantáneo
    
    # Derivados (Sentimiento Institucional)
    'funding_rate',     # Costo de mantener posiciones largas
    'open_interest',    # Contratos abiertos totales
    
    # Volumen Agresivo (Urgencia)
    'taker_buy_vol',    # Compras agresivas (market orders)
    'taker_sell_vol',   # Ventas agresivas
    
    # Derivadas Calculadas
    'buy_sell_ratio',   # taker_buy_vol / taker_sell_vol
    'depth_imbalance'   # (bid_depth - ask_depth) / (bid_depth + ask_depth)
]
```

### 4.3 Algoritmo de Entrenamiento (train_v2_production.py)

**Archivo:** `scripts/train_v2_production.py`
**Frecuencia:** Diario (via pm2 cron o manual)

```python
# PASO 1: Cargar datos históricos
query = """
    SELECT o.*, d.funding_rate, d.open_interest, d.taker_buy_vol, d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{symbol}'
    ORDER BY o.timestamp ASC
"""

# PASO 2: Crear etiquetas (Target)
df['future_price'] = df['price'].shift(-5)  # Precio en 5 snapshots (~50 segundos)
df['return_5m'] = (df['future_price'] - df['price']) / df['price']

threshold = 0.001  # 0.1% de movimiento mínimo
# Clase 0: SHORT (return < -0.1%)
# Clase 1: NEUTRAL (|return| < 0.1%)
# Clase 2: LONG (return > 0.1%)

# PASO 3: Normalizar con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PASO 4: Crear secuencias temporales
SEQ_LEN = 12  # Últimos 12 snapshots (2 minutos)
for i in range(len(X_scaled) - SEQ_LEN):
    X_sequences.append(X_scaled[i:i+SEQ_LEN])  # Shape: (12, 13)
    y_labels.append(y[i + SEQ_LEN])

# PASO 5: Entrenar cada modelo
# LSTM: 10 epochs, Adam optimizer, lr=0.001, CrossEntropyLoss
# TCN: 10 epochs, Adam optimizer, lr=0.001
# XGBoost: 1000 trees, max_depth=6, learning_rate=0.1
# Transformer: 10 epochs, Adam optimizer, lr=0.0005
```

### 4.4 Algoritmo de Predicción (ml_service_v2.py)

**Archivo:** `services/ml_service_v2.py`
**Endpoint:** `POST /ml-v2/predict`

```python
# PASO 1: Recibir símbolo del Bot
request = {"symbol": "ETHUSDT"}

# PASO 2: Cargar últimas 60 filas de la DB
query = """
    SELECT o.*, d.funding_rate, d.open_interest, d.taker_buy_vol, d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp
    WHERE o.symbol = '{symbol}'
    ORDER BY o.timestamp DESC
    LIMIT 60
"""

# PASO 3: Calcular features derivadas
df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])

# PASO 4: Aplicar scaler guardado
X_scaled = scaler.transform(df[feature_cols].values)

# PASO 5: Crear secuencia de los últimos 12 snapshots
X_seq = X_scaled[-12:]  # Shape: (12, 13)
X_tensor = torch.FloatTensor(X_seq).unsqueeze(0)  # Shape: (1, 12, 13)

# PASO 6: Obtener predicción de cada modelo
lstm_probs = softmax(lstm_model(X_tensor)['logits'])    # [short, neutral, long]
tcn_probs = softmax(tcn_model(X_tensor)['logits'])
transformer_probs = softmax(transformer_model(X_tensor)['logits'])
xgb_probs = xgb_model.predict(X_seq[-1])  # Solo última fila para XGBoost

# PASO 7: Calcular promedio ponderado (votos iguales por ahora)
ensemble_probs = (lstm_probs + tcn_probs + transformer_probs + xgb_probs) / 4

# PASO 8: Retornar al Bot
return {
    "symbol": "ETHUSDT",
    "long_prob": ensemble_probs[2],    # Probabilidad LONG
    "short_prob": ensemble_probs[0],   # Probabilidad SHORT
    "neutral_prob": ensemble_probs[1]  # Probabilidad NEUTRAL
}
```

---

## 5. PILAR III: El Bot de Ejecución (Ninja Protocol)

### 5.1 Ciclo Principal (tick())

**Archivo:** `binance-futures-bot-ts/src/app/strategy-runner.ts`
**Intervalo:** Cada 1000ms

```typescript
async tick() {
    // 1. Obtener predicción del Servicio ML
    const prediction = await fetch('http://127.0.0.1:8001/ml-v2/predict', {
        method: 'POST',
        body: JSON.stringify({ symbol: 'ETHUSDT' })
    });
    const { long_prob, short_prob, neutral_prob } = await prediction.json();

    // 2. Leer umbral dinámico
    const threshold = MlConfigWatcher.getInstance().getThreshold('ETHUSDT');
    // Ejemplo: threshold = 0.40

    // 3. Decidir acción
    if (long_prob > threshold && !hasPosition) {
        await enterLong();
    } else if (short_prob > threshold && !hasPosition) {
        await enterShort();
    } else if (hasPosition) {
        await evaluateExit(long_prob, short_prob, neutral_prob);
    }
}
```

### 5.2 Protocolo Ninja de Salida (3 Capas)

```typescript
async evaluateExit(longProb, shortProb, neutralProb) {
    const position = getActivePosition();
    const roiPct = calculateROI(position.entryPrice, currentPrice);
    const opposingProb = position.side === 'LONG' ? shortProb : longProb;

    // CAPA 1: PÁNICO (Prioridad Máxima)
    // Si el modelo grita en contra con fuerza (>50%)
    if (opposingProb > 0.50) {
        await closeSideMarket(position);
        log('EXIT: PANIC_REVERSAL');
        return;
    }

    // CAPA 2: NEUTRALIDAD (Tomar Ganancias)
    // Si ganamos >5% y el mercado se duerme (Neutral >60%)
    if (roiPct > 5 && neutralProb > 0.60) {
        await closeSideMarket(position);
        log('EXIT: NEUTRAL_TAKE_PROFIT');
        return;
    }

    // CAPA 3: TRAILING STOP (Proteger Ganancias)
    const peak = Math.max(position.peakROI, roiPct);
    let trailDistance = 0.40;  // Default: 40%
    if (peak > 50) trailDistance = 0.10;  // Super estricto
    else if (peak > 30) trailDistance = 0.20;  // Estricto

    if (peak > 10 && roiPct < peak * (1 - trailDistance)) {
        await closeSideMarket(position);
        log('EXIT: TRAILING_STOP');
        return;
    }
}
```

---

## 6. Flujo Completo: Del Dato a la Orden

```
T=0       Binance emite actualización de Order Book para ETHUSDT
          │
T=10s     Data Collector captura snapshot, calcula OBI, inserta en DB
          │ INSERT INTO orderbook_metrics VALUES (1703800000000, 'ETH/USDT:USDT', 0.15, ...)
          ▼
T=11s     Bot hace tick(), llama POST /ml-v2/predict {"symbol": "ETHUSDT"}
          │
          ├── ML Service lee SELECT ... LIMIT 60 ORDER BY timestamp DESC
          │
          ├── Calcula features derivadas (buy_sell_ratio, depth_imbalance)
          │
          ├── Aplica scaler: X_scaled = scaler.transform(X)
          │
          ├── Crea tensor: (1, 12, 13)
          │
          ├── Pasa por 4 modelos:
          │   ├── LSTM:        [0.18, 0.42, 0.40]
          │   ├── TCN:         [0.15, 0.45, 0.40]
          │   ├── Transformer: [0.12, 0.48, 0.40]
          │   └── XGBoost:     [0.10, 0.45, 0.45]
          │
          ├── Promedia: ensemble = [0.1375, 0.45, 0.4125]
          │
          └── Retorna: {"long_prob": 0.4125, "short_prob": 0.1375, "neutral_prob": 0.45}
          │
T=11.05s  Bot recibe respuesta, compara:
          │   long_prob (0.41) > threshold (0.40) ✓
          │   No hay posición abierta ✓
          │
          └── Bot envía: POST /fapi/v1/order {"side": "BUY", "type": "MARKET", ...}
          │
T=11.1s   Binance ejecuta orden, Bot registra entrada en state
          │
T=12s     Bot hace tick(), ahora en modo VIGILANCIA
          │   Evalúa: ¿PÁNICO? ¿NEUTRAL? ¿TRAILING?
          │   Continúa monitoreando...
```

---

## 7. Archivos Clave del Sistema

| Archivo | Propósito |
|---------|-----------|
| `scripts/next_gen/market_data_collector.py` | Captura datos cada 10s |
| `services/ml_service_v2.py` | Sirve predicciones vía REST |
| `scripts/train_v2_production.py` | Entrena los 4 modelos |
| `ml/advanced_models/ensemble_manager.py` | Combina votos del Consejo |
| `binance-futures-bot-ts/src/app/strategy-runner.ts` | Ejecuta órdenes |
| `models/v2_ensemble/{SYMBOL}/` | Modelos entrenados |
| `models/advanced/thresholds_config.json` | Umbrales por símbolo |
| `data/market_data_v2.db` | Base de datos de mercado |

---

## 8. Métricas de Rendimiento

*   **Latencia Predicción:** <50ms (end-to-end)
*   **Frecuencia de Datos:** 1 snapshot cada 10 segundos
*   **Símbolos Activos:** 15 (BTC, ETH, ADA, AVAX, SOL, XRP, LINK, ATOM, BNB, DOGE, DOT, LTC, NEAR, UNI, POL)
*   **Reentrenamiento:** Diario (automático vía cron)
*   **Disponibilidad:** 99.9% (PM2 con auto-restart)

---

**Aviso de Confidencialidad:** Este documento contiene detalles arquitectónicos propietarios. Distribución restringida.
