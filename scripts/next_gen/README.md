# Next Gen Data Collector (V2)

Este directorio contiene la infraestructura para la "Refinería de Datos" del Plan Maestro Institucional.

## Objetivo
Recolectar datos de alta fidelidad que no están presentes en las velas OHLCV estándar, para alimentar los modelos de Ensemble Learning futuros sin afectar el sistema de producción actual.

## Archivos
*   `market_data_collector.py`: Script principal que conecta a Binance Futures, descarga snapshots de Order Book, Funding Rates y Open Interest, calcula métricas derivadas (OBI, Spread, Micro-price) y las guarda en una base de datos separada.

## Base de Datos
Los datos se almacenan en `data/market_data_v2.db`. Esta base de datos es **independiente** de la que usa el bot actual (`market_data.db`), garantizando cero riesgo de corrupción o bloqueos en producción.

## Métricas Recolectadas
1.  **Order Book Imbalance (OBI):** Presión de compra/venta en el libro (Top 5, 10, 20 niveles).
2.  **Spread %:** Costo de liquidez.
3.  **Funding Rate:** Sentimiento del mercado de perpetuos.
4.  **Open Interest:** Interés abierto en contratos y USD.
