# 🧠 Anatomía del Sistema de Trading

Este documento contiene el código fuente completo y la explicación de los dos subsistemas críticos del bot:
1.  **Grid Search System:** Optimización de parámetros.
2.  **ML Pipeline (Consejo de Sabios):** Desde la recolección de datos hasta la predicción en tiempo real.

---

# PARTE 1: Grid Search System (Optimización)

El sistema se compone de **3 pilares fundamentales**:

1.  **El Orquestador (`run_grid_search_sequential.sh`)**: Gestiona la ejecución segura y secuencial para proteger el hardware.
2.  **El Cerebro (`grid_search_optimizer.py`)**: Genera las combinaciones de parámetros y elige la ganadora.
3.  **El Motor (`backtest_system_v2.py`)**: Simula la realidad tick a tick con precisión quirúrgica.

## 1.1 El Orquestador: `run_grid_search_sequential.sh`

**Función:**`
Es el script de entrada. Su trabajo es iterar sobre la lista de símbolos (`SYMBOLS`) y lanzar el proceso de optimización para uno a la vez. Esto es crucial porque correr 20 optimizaciones en paralelo (cada una cargando modelos ML y DB) colapsaría la memoria RAM (8GB) y la CPU.

### 💻 Código Fuente

```bash
#!/bin/bash
# scripts/run_grid_search_sequential.sh
# Runs Grid Search Optimizer for all symbols SEQUENTIALLY to prevent system freeze.
# Uses AMD GPU 0 for acceleration but processes one symbol at a time.

# 1. Define Symbols (21 Total)
SYMBOLS=(
    "ADA/USDT:USDT"
    "AVAX/USDT:USDT"
    "BTC/USDT:USDT"
    "ETH/USDT:USDT"
    "LINK/USDT:USDT"
    "SOL/USDT:USDT"
    "XRP/USDT:USDT"
    "ATOM/USDT:USDT"
    "BNB/USDT:USDT"
    "DOGE/USDT:USDT"
    "DOT/USDT:USDT"
    "LTC/USDT:USDT"
    "NEAR/USDT:USDT"
    "UNI/USDT:USDT"
    "POL/USDT:USDT"
    "APT/USDT:USDT"
    "FET/USDT:USDT"
    "INJ/USDT:USDT"
    "SEI/USDT:USDT"
    "WLD/USDT:USDT"
    "1000PEPE/USDT:USDT"
)

# 2. Setup Environment (AMD ROCm)
source .venv_rocm62/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

echo "🚀 Starting Sequential Grid Search (Safe Mode)"
echo "   Testing 10x AND 15x Leverage."
echo "   Processing 1 symbol at a time to protect system stability."
echo "---------------------------------------------------"

# 3. Launch Sequential Loop
count=1
total=${#SYMBOLS[@]}

for symbol in "${SYMBOLS[@]}"; do
    echo "🧩 [$count/$total] Processing $symbol on GPU 0..."
    
    # Run in FOREGROUND (wait for completion)
    HIP_VISIBLE_DEVICES=0 python scripts/grid_search_optimizer.py --symbol "$symbol" --days 3 > "logs/grid_search_${symbol//\//_}.log" 2>&1
    
    echo "✅ Completed $symbol."
    echo "---------------------------------------------------"
    ((count++))
    
    # Cooldown to let system breathe
    sleep 5
done

echo "🎉 All Grid Search tasks completed successfully."
```

## 1.2 El Cerebro: `grid_search_optimizer.py`

**Función:**
Es el estratega. No simula el mercado, sino que **planifica los experimentos**.
1.  Define el "Espacio de Búsqueda":
    *   **Thresholds:** 0.30, 0.35, 0.40, 0.45, 0.50 (¿Qué tan seguro debe estar el ML para entrar?)
    *   **Hard Stops:** -5%, -10%, -15% (¿Cuánto dolor aguantamos antes de rendirnos?)
    *   **Leverage:** 10x, 15x (¿Qué tan agresivos somos?)
2.  Carga los datos históricos **una sola vez** en memoria (para eficiencia).
3.  Crea un bucle triple anidado (Leverage x Threshold x Stop) probando todas las combinaciones (30 escenarios por moneda).
4.  Para cada combinación, instancia un `NinjaBotSimulator`, le inyecta los parámetros y corre la simulación.
5.  Recopila métricas (Win Rate, Profit Factor, Retorno).
6.  Elige el ganador y guarda el reporte JSON.

### 💻 Código Fuente

```python
"""
Grid Search Optimizer for the Ninja Trading System

This script systematically tests different configurations of:
- Base Threshold (0.30 - 0.50)
- Hard Stop Loss (-5% to -15% ROE)

And ranks them by performance to find the mathematically optimal configuration.

Usage:
    python scripts/grid_search_optimizer.py --symbol BTCUSDT --days 3
"""
import os
import sys
import argparse
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backtest_system_v2 import NinjaBotSimulator

# Supress verbose ML logging during grid search
import logging
logging.getLogger("ml_service_v2").setLevel(logging.WARNING)
logging.getLogger("EnsembleManager").setLevel(logging.WARNING)

def run_grid_search(symbol: str, days: int = 3, hours: int = 0):
    """
    Prueba diferentes configuraciones de Threshold, Hard Stop y Leverage para encontrar el óptimo.
    """
    # Definir variables a probar
    base_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    hard_stop_options = [-0.05, -0.10, -0.15] # -5%, -10%, -15% ROE
    leverage_options = [10, 15] # Prueba de fuego: 10x vs 15x
    
    results = []

    print(f"\n{'='*70}")
    print(f"🚀 GRID SEARCH OPTIMIZER: {symbol}")
    print(f"   Período: {days} días, {hours} horas")
    print(f"   Configuraciones a probar: {len(base_thresholds) * len(hard_stop_options) * len(leverage_options)}")
    print(f"{'='*70}\n")
    
    # Cargar datos UNA VEZ para reutilizarlos (eficiencia)
    print("📥 Cargando datos (una sola vez para todas las pruebas)...")
    # Instancia temporal para cargar datos (leverage no importa aquí)
    temp_sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=10)
    shared_data = temp_sim.load_data(days=days, hours=hours)
    print(f"✅ Datos cargados: {len(shared_data)} registros.\n")
    
    total_combos = len(base_thresholds) * len(hard_stop_options) * len(leverage_options)
    combo_num = 0
    
    for lev in leverage_options:
        for base_thr in base_thresholds:
            for stop_pct in hard_stop_options:
                combo_num += 1
                
                # 1. Instanciar Simulador con la config específica
                sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=lev)
                
                # 2. Sobreescribir configuración manualmente
                sim.base_threshold = base_thr
                sim.hard_stop_pct = stop_pct
                
                # 3. Ejecutar Backtest con datos compartidos (sin recargar)
                sim.run(shared_data)
                
                # 4. Calcular métricas
                num_trades = len(sim.trades)
                if num_trades > 0:
                    wins = [t for t in sim.trades if t['pnl'] > 0]
                    losses = [t for t in sim.trades if t['pnl'] <= 0]
                    win_rate = len(wins) / num_trades
                    profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf')
                    avg_pnl = sum(t['pnl'] for t in sim.trades) / num_trades
                else:
                    win_rate = 0
                    profit_factor = 0
                    avg_pnl = 0
                
                return_pct = ((sim.balance - 1000)/1000)*100
                
                # 5. Guardar Resultados
                results.append({
                    'leverage': lev,
                    'base_thr': base_thr,
                    'stop_pct': stop_pct,
                    'config': f"Lev:{lev}x Thr:{base_thr:.2f} Stop:{stop_pct:.0%}",
                    'final_balance': sim.balance,
                    'return_pct': return_pct,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_pnl': avg_pnl,
                    'total_trades': num_trades
                })
                
                print(f"[{combo_num}/{total_combos}] Lev:{lev}x Thr:{base_thr:.2f} Stop:{stop_pct:.0%} -> Return: {return_pct:>+6.2f}% | Trades: {num_trades} | WR: {win_rate:.0%}")

    # 6. Ranking de Mejores Configuraciones
    print("\n" + "="*70)
    print("📊 RANKING DE CONFIGURACIONES (TOP 10)")
    print("="*70)
    
    # Ordenar por Retorno descendente
    ranked_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    
    print(f"{'Rank':<5} {'Config':<30} {'Return':<10} {'Trades':<8} {'WinRate':<10} {'PF':<8}")
    print("-" * 75)
    
    for i, r in enumerate(ranked_results[:10], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "Inf"
        print(f"{i:<5} {r['config']:<30} {r['return_pct']:>+7.2f}% {r['total_trades']:<8} {r['win_rate']:.0%}{'':<6} {pf_str:<8}")
    
    # 7. Guardar reporte JSON
    import json
    report_path = Path(REPO_ROOT) / "reports" / f"grid_search_{symbol.replace('/','')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(ranked_results, f, indent=2)
    print(f"\n💾 Reporte guardado en: {report_path}")

    # 8. Imprimir Mejor Configuración
    best = ranked_results[0]
    print("\n" + "="*70)
    print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*70)
    print(f"   Leverage:       {best['leverage']}x")
    print(f"   Threshold Base: {best['base_thr']:.2f}")
    print(f"   Hard Stop:      {best['stop_pct']:.0%} ROE")
    print(f"   Retorno:        {best['return_pct']:+.2f}%")
    print(f"   Win Rate:       {best['win_rate']:.0%}")
    print(f"   Total Trades:   {best['total_trades']}")
    print("="*70 + "\n")
    
    return ranked_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search Optimizer for Trading Bot")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=3, help="Days of historical data")
    parser.add_argument("--hours", type=int, default=0, help="Additional hours of historical data")
    args = parser.parse_args()
    
    run_grid_search(args.symbol, days=args.days, hours=args.hours)
```

## 1.3 El Motor: `backtest_system_v2.py`

**Función:**
Es el "Gemelo Digital" del bot. Contiene la lógica exacta de trading (`strategy-runner.ts`) pero traducida a Python para simulación rápida.
1.  **Carga Datos:** Lee de la base de datos `market_data_v2.db` (Order Book Metrics, Funding Rates, etc.).
2.  **Predicción ML:** Usa `ml_service_v2` para obtener probabilidades reales (Short/Neutral/Long) usando el modelo XGBoost entrenado.
3.  **Simulación Tick a Tick:** Recorre los datos históricos fila por fila.
    *   **Entrada:** Si `Prob > Threshold`, abre posición.
    *   **Gestión:** Calcula PnL, Fees, Funding.
    *   **Salida:** Aplica las reglas Ninja:
        *   **Hard Stop:** Si ROE < -10% (o lo que diga el grid search).
        *   **Panic Reversal:** Si el modelo cambia de opinión drásticamente.
        *   **Trailing Stop Logarítmico:** Protege ganancias.
        *   **Breakeven:** Asegura 0.2% si ganamos >1.5%.
4.  **Reporte:** Genera CSVs y gráficos de Equity Curve.

### 💻 Código Fuente

```python
"""
Backtester del Sistema Completo (v2.1) - "El Gemelo Digital"

Este script simula la operación EXACTA del bot de producción, utilizando:
1. Datos reales de microestructura (market_data_v2.db).
2. El servicio ML real (services.ml_service_v2) con Meta-Features y Filtro Ninja.
3. La lógica de ejecución del bot (Protocolo Ninja) replicada en Python.

Uso:
    python scripts/backtest_system_v2.py --symbol BTCUSDT --days 7
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Importar el Cerebro Real
from services.ml_service_v2 import V2ModelManager, DB_PATH

# Configuración de Logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("BacktesterV2")

class NinjaBotSimulator:
    def __init__(self, symbol: str, initial_capital: float = 1000.0, leverage: int = 10):
        self.symbol = symbol
        self.capital = initial_capital
        self.leverage = leverage
        self.balance = initial_capital
        self.position = None # None, 'LONG', 'SHORT'
        self.entry_price = 0.0
        self.entry_time = None
        self.qty = 0.0
        
        # ═════════════════════════════════════════════════════════
        # CONFIGURACIÓN: CONSERVAR + GANAR
        # ═════════════════════════════════════════════════════════
        
        # 1. Umbral Dinámico (0.35 - 0.50 dependiendo de volatilidad)
        self.base_threshold = 0.35
        self.max_threshold = 0.50
        self.current_threshold = self.base_threshold # Se actualiza en tiempo real
        
        # 2. Protección de Capital (Conservar)
        self.hard_stop_pct = -0.10 # -10% ROE (Más espacio para respirar)
        
        # 3. Protección de Ganancias (Ganar + Lock-in)
        self.breakeven_trigger_roi = 0.015 # 1.5% ROI para activar breakeven
        self.breakeven_profit_pct = 0.002 # 0.2% ganancia garantizada
        
        # 4. Tiempo Máximo en Trade (Evitar zombis)
        self.max_trade_minutes = 10 # 10 minutos máximo
        self.min_profit_for_time = 0.01 # Si no tengo 1% en 10 min, me voy

        self.commission_rate = 0.0004 
        self.peak_roi = -999.0
        self.trailing_stop_price = 0.0
        self.trailing_active = False
        
        self.trades = []
        self.equity_curve = []
        
        # ML Manager
        self.ml_manager = V2ModelManager()
        self.ml_manager.load_model_for_symbol(symbol)

    def load_data(self, days: int = 7, hours: int = 0) -> pd.DataFrame:
        """Carga datos directamente de la DB de producción."""
        LOGGER.info(f"📥 Cargando últimos {days} días y {hours} horas de datos para {self.symbol}...")
        
        # Calcular timestamp de inicio
        start_ts = int((datetime.now() - timedelta(days=days, hours=hours)).timestamp() * 1000)
        
        # Mapeo de símbolo si es necesario (simple)
        db_symbol = self.symbol
        if "USDT" in self.symbol and "/" not in self.symbol:
             # Intentar formato CCXT si no encuentra el simple
             pass 
             
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT 
            o.timestamp,
            o.mid_price as price,
            o.micro_price,
            o.bid_depth_20 as bid_depth, 
            o.ask_depth_20 as ask_depth, 
            o.spread_pct as bid_ask_spread, 
            o.obi_5,
            o.obi_10,
            o.obi_20 as obi,
            d.funding_rate, 
            d.open_interest,
            d.taker_buy_vol,
            d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{db_symbol}' OR o.symbol = '{db_symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
            
        LOGGER.info(f"✅ Cargados {len(df)} registros.")
        return df

    def run(self, df: pd.DataFrame):
        """Ejecuta el bucle de simulación tick a tick."""
        LOGGER.info("🚀 Iniciando simulación...")
        
        # Pre-calcular predicciones para velocidad (en producción es tiempo real, aquí batch es ok)
        # PERO para validar el Filtro Ninja (stateful), debemos hacerlo secuencial o simular el estado.
        # El V2ModelManager mantiene estado en self.smoothed_probs_cache.
        # Así que iteraremos fila por fila pasando un DF ventana.
        
        window_size = 60 # Necesitamos contexto para features
        
        for i in range(window_size, len(df)):
            # Contexto actual
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            price = current_row['price']
            
            # DataFrame ventana para el ML Service
            # (El servicio espera un DF con historia para calcular features)
            df_window = df.iloc[i-window_size:i+1].copy()
            
            # 1. Obtener Predicción (Cerebro)
            # Esto invoca feature engineering + ensemble + NINJA FILTER
            prediction = self.ml_manager.predict(self.symbol, df_window)
            probs = prediction['ensemble_probs'][0].tolist() # [Short, Neutral, Long]
            
            short_prob = probs[0]
            neutral_prob = probs[1]
            long_prob = probs[2]
            
            # 2. Ejecutar Lógica de Bot (Cuerpo)
            self._update_bot_logic(timestamp, price, long_prob, short_prob, neutral_prob)
            
            # Registrar equidad
            self._record_equity(price)
            
            if i % 1000 == 0:
                print(f"⏳ Procesados {i}/{len(df)} ticks...", end='\r')
                
        print("\n✅ Simulación completada.")
        self._generate_report()

    def _update_bot_logic(self, timestamp, price, long_prob, short_prob, neutral_prob):
        """Máquina de estados: Conservar Capital + Maximizar Ganancias."""
        
        # --- CALCULAR VOLATILIDAD SIMPLE (OBI) para Ajustar Threshold ---
        # Usamos el spread o la diferencia de probs como proxy de volatilidad
        # Si el mercado es confuso (neutral alto), necesitamos más señal para entrar.
        market_uncertainty = abs(long_prob - short_prob) # Bajo = Incierto
        
        # Dinámica: Si hay mucha duda (Spread de probs grande), subimos el umbral de entrada
        # market_uncertainty va de 0 (50/50) a 1 (100/0)
        # Queremos: 0.35 -> 0.50
        # NOTA: market_uncertainty es alto cuando hay convicción (e.g. 0.8 vs 0.1 -> 0.7)
        #       market_uncertainty es bajo cuando hay duda (e.g. 0.3 vs 0.3 -> 0.0)
        # Invertimos lógica: Si uncertainty es BAJA (duda), subimos threshold.
        # Si uncertainty es ALTA (convicción), bajamos threshold (base).
        # Corrección lógica del usuario: "Si el mercado está en calma [duda], entraremos al 0.35" -> Esto suena al revés.
        # Si está en calma/rango, suele haber ruido. Si hay tendencia, hay convicción.
        # Sigamos la instrucción literal del usuario pero ajustando la fórmula para que tenga sentido:
        # "Si el mercado está en calma, entraremos al 0.35. Si está loco, exigiremos 0.50"
        # Interpretación: Calma = Baja Volatilidad = Menor riesgo = Threshold bajo?
        # O Calma = Rango = Mayor riesgo de chop = Threshold alto?
        # El usuario dice: "Si el mercado está en calma, entraremos al 0.35".
        # Asumiremos que "calma" se refiere a baja volatilidad de precios, pero aquí solo tenemos probs.
        # Usaremos la instrucción literal: Threshold base + factor.
        
        # Vamos a simplificar: Usar la incertidumbre de probs.
        # Si |L-S| es pequeño, hay incertidumbre.
        # El usuario dice: "Si hay mucha duda (Spread de probs grande), subimos el umbral".
        # Espera, si el spread es grande (ej 0.8 vs 0.1), NO hay duda. Hay claridad.
        # Si el spread es pequeño (0.3 vs 0.3), HAY duda.
        # La instrucción del usuario en el código propuesto dice:
        # "Dinámica: Si hay mucha duda (Spread de probs grande), subimos el umbral de entrada"
        # Esto es contradictorio. Spread grande = Poca duda.
        # Asumiré que quiso decir: "Si hay mucha duda (Spread de probs PEQUEÑO)..."
        # PERO, el código que me dio hace: self.current_threshold = self.base_threshold + (market_uncertainty * 0.15)
        # Si market_uncertainty (abs diff) es 1.0 (máxima diferencia), threshold sube a 0.35 + 0.15 = 0.50.
        # O sea, exige MÁS certeza cuando el modelo YA tiene mucha diferencia?
        # Eso haría que en tendencias fuertes (donde el modelo grita 0.8), le pidamos 0.50. (Facil de cumplir).
        # Y en rangos (donde el modelo dice 0.35 vs 0.30, diff 0.05), le pedimos 0.35 + 0.0075 = 0.3575.
        # Esto parece al revés de "Filtrar rangos".
        # PERO, voy a implementar el código LITERAL que me dio el usuario para no desviarme de su deseo explícito.
        
        self.current_threshold = self.base_threshold + (market_uncertainty * 0.15)
        
        # --- SI NO HAY POSICIÓN ---
        if self.position is None:
            # Entrada Dinámica
            if long_prob > self.current_threshold:
                self._open_position("LONG", price, timestamp, f"ML_LONG (Thr:{self.current_threshold:.2f})")
            elif short_prob > self.current_threshold:
                self._open_position("SHORT", price, timestamp, f"ML_SHORT (Thr:{self.current_threshold:.2f})")
                
        # --- SI HAY POSICIÓN ---
        else:
            # Calcular ROIs
            roi_raw = (price - self.entry_price) / self.entry_price if self.position == "LONG" else (self.entry_price - price) / self.entry_price
            roi_lev = roi_raw * self.leverage
            roi_lev_pct = roi_lev * 100.0
            
            # Actualizar Peak
            if roi_lev_pct > self.peak_roi:
                self.peak_roi = roi_lev_pct
                if roi_lev > self.breakeven_trigger_roi:
                    self._update_trailing_stop(price, roi_lev_pct)
            
            # 1. CHECK DE TIEMPO -> ELIMINADO (Oportunidad vs Paciencia)
            # Ya no salimos por reloj. Dejamos que el Hard Stop o el Pánico nos protejan.
            # time_in_min = (timestamp - self.entry_time) / 60000
            # if time_in_min > self.max_trade_minutes:
            #     if roi_lev < self.min_profit_for_time:
            #         self._close_position(price, timestamp, "TIME_DECAY_LOW_PROFIT")
            #         return

            # 2. CHECK DE BREAKEVEN AGRESIVO (Consevar Capital)
            # Si tenemos al menos 1.5% de ganancia, movemos el stop a +0.2%
            # Esto asegura que NUNCA perdamos este trade.
            if roi_lev > self.breakeven_trigger_roi:
                if self.position == "LONG":
                    # Stop a precio de entrada + 0.2%
                    be_stop = self.entry_price * (1 + self.breakeven_profit_pct)
                    if self.trailing_stop_price < be_stop:
                        self.trailing_stop_price = be_stop
                        self.trailing_active = True # Bloquear trailing normal
                else: # SHORT
                    be_stop = self.entry_price * (1 - self.breakeven_profit_pct)
                    if self.trailing_stop_price > be_stop or self.trailing_stop_price == 0:
                        self.trailing_stop_price = be_stop
                        self.trailing_active = True

            # 3. CHECK DE HARD STOP LOSS (Muro de contención)
            if roi_lev < self.hard_stop_pct:
                self._close_position(price, timestamp, "HARD_STOP_LOSS")
                return

            # 4. CHECK DE PÁNICO (Sistema inmune)
            if self.position == "LONG" and short_prob > 0.55: # Umbral de salida más alto
                self._close_position(price, timestamp, "PANIC_REVERSAL")
                return
            if self.position == "SHORT" and long_prob > 0.55:
                self._close_position(price, timestamp, "PANIC_REVERSAL")
                return
                
            # 5. CHECK DE NEUTRALIDAD (Tomar ganancias medias)
            if roi_lev > 0.03 and neutral_prob > 0.60:
                self._close_position(price, timestamp, "NEUTRALITY_EXIT")
                return
                
            # 6. CHECK DE TRAILING STOP (Solo si Breakeven NO se activó)
            if not self.trailing_active and roi_lev > 0.02:
                 self._update_trailing_stop(price, roi_lev_pct)
            
            # 7. EJECUCIÓN DE TRAILING STOP
            if self.position == "LONG" and price < self.trailing_stop_price:
                self._close_position(price, timestamp, "TRAILING_STOP")
                return
            if self.position == "SHORT" and price > self.trailing_stop_price:
                self._close_position(price, timestamp, "TRAILING_STOP")
                return

    def _update_trailing_stop(self, current_price, roi_lev):
        """Lógica LOGARÍTMICA con Mínimo Amplio."""
        # Convertir Peak ROI a porcentaje para la fórmula
        peak_pct_val = max(5, self.peak_roi * 100.0) 
        
        base_trail_pct = 30 - (22 * math.log10(peak_pct_val / 5))
        
        # ⚠️ FIX: Forzar un mínimo de 1.5% de distancia para evitar cortes por ruido
        # (Calculamos la distancia en % del PEAK, no del precio directo)
        trail_distance_pct = max(15, min(30, base_trail_pct)) 
        
        # Calcular nivel de ROE del stop
        # Usamos self.peak_roi (decimal) para el cálculo final
        stop_roi_level = (self.peak_roi * 100.0) * (1 - (trail_distance_pct / 100.0))
        
        # Convertir ROE a Precio
        if self.position == "LONG":
            stop_price = self.entry_price * (1 + (stop_roi_level / 100.0 / self.leverage))
            if stop_price > self.trailing_stop_price:
                self.trailing_stop_price = stop_price
        else: # SHORT
            stop_price = self.entry_price * (1 - (stop_roi_level / 100.0 / self.leverage))
            if self.trailing_stop_price == 0 or stop_price < self.trailing_stop_price:
                self.trailing_stop_price = stop_price

    def _open_position(self, side, price, timestamp, reason):
        self.position = side
        self.entry_price = price
        self.entry_time = timestamp
        self.qty = (self.balance * self.leverage) / price # Full margin simulation
        self.peak_roi = -999.0
        self.trailing_stop_price = price * (0.97 if side == "LONG" else 1.03) # Initial wide stop
        self.trailing_active = False
        
        # Fee de entrada
        fee = (self.qty * price) * self.commission_rate
        self.balance -= fee
        
        # LOGGER.info(f"🟢 OPEN {side} at {price:.2f} | {reason}")

    def _close_position(self, price, timestamp, reason):
        # Calcular PnL
        if self.position == "LONG":
            pnl = (price - self.entry_price) * self.qty
        else:
            pnl = (self.entry_price - price) * self.qty
            
        # Fee de salida
        fee = (self.qty * price) * self.commission_rate
        pnl -= fee
        
        self.balance += pnl
        roi_pct = (pnl / (self.capital if len(self.trades)==0 else self.trades[-1]['balance'])) * 100
        
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'side': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'reason': reason,
            'pnl': pnl,
            'roi': roi_pct,
            'balance': self.balance
        })
        
        # LOGGER.info(f"🔴 CLOSE {self.position} at {price:.2f} | PnL: ${pnl:.2f} ({roi_pct:.2f}%) | {reason}")
        
        self.position = None
        self.qty = 0

    def _record_equity(self, current_price):
        # Valor de la cuenta = Balance (cerrado) + PnL no realizado
        unrealized_pnl = 0
        if self.position:
            if self.position == "LONG":
                unrealized_pnl = (current_price - self.entry_price) * self.qty
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.qty
            
            # Restar fee estimado de salida
            unrealized_pnl -= (self.qty * current_price) * self.commission_rate
            
        self.equity_curve.append(self.balance + unrealized_pnl)

    def _generate_report(self):
        if not self.trades:
            LOGGER.warning("⚠️ No trades executed.")
            return
            
        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        print("\n" + "="*50)
        print(f"📊 REPORTE FINAL: {self.symbol}")
        print("="*50)
        print(f"Capital Inicial: ${self.capital:.2f}")
        print(f"Capital Final:   ${self.balance:.2f}")
        print(f"Retorno Total:   {((self.balance - self.capital)/self.capital)*100:.2f}%")
        print(f"Total Trades:    {len(df_trades)}")
        print(f"Win Rate:        {len(wins)/len(df_trades)*100:.2f}%")
        print(f"Profit Factor:   {wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf'):.2f}")
        print("-" * 30)
        print("Motivos de Salida:")
        print(df_trades['reason'].value_counts())
        print("="*50)
        
        # Guardar CSV
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        csv_path = f"backtest_trades_{safe_symbol}_v2.csv"
        df_trades.to_csv(csv_path, index=False)
        LOGGER.info(f"📝 Trades guardados en {csv_path}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title(f"Equity Curve - {self.symbol} (Ninja Protocol)")
        plt.xlabel("Trades")
        plt.ylabel("Balance (USDT)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"backtest_equity_{safe_symbol}_v2.png")
        plt.close()
        LOGGER.info(f"📈 Gráfico guardado en backtest_equity_{safe_symbol}_v2.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--hours", type=int, default=0)
    args = parser.parse_args()
    
    sim = NinjaBotSimulator(symbol=args.symbol)
    try:
        data = sim.load_data(days=args.days, hours=args.hours)
        sim.run(data)
    except Exception as e:
        LOGGER.error(f"Error en backtest: {e}")
```

---

# PARTE 2: ML Pipeline (Consejo de Sabios)

Este pipeline es el corazón de la inteligencia del bot. Transforma datos crudos del mercado en probabilidades de trading accionables.

**Flujo de Datos:**
1.  **Recolección:** `market_data_collector.py` captura Order Book y Derivados cada 10s -> `market_data_v2.db`.
2.  **Entrenamiento:** `train_v2_production.py` entrena el "Consejo de Sabios" (Ensemble de 4 modelos) usando datos históricos.
3.  **Inferencia:** `ml_service_v2.py` expone una API REST que carga los modelos y predice en tiempo real.
4.  **Consumo:** `ml_probability_service.ts` (en el bot) consulta la API y entrega las probabilidades a la estrategia.

---

## 2.1 Recolección de Datos: `market_data_collector.py`

**Función:**
Es un proceso demonio (`pm2 start 02-Data-Collector`) que monitorea 21 pares de criptomonedas.
*   Captura snapshots del Order Book (Profundidad, OBI, Spread).
*   Captura datos de derivados (Funding Rate, Open Interest, Taker Volume).
*   Almacena todo en SQLite (`market_data_v2.db`) optimizado para escritura rápida.

### 💻 Código Fuente

```python
#!/usr/bin/env python3
import os
import sys
import time
import sqlite3
import logging
import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuración
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "market_data_v2.db"
LOG_DIR = ROOT_DIR / "logs"
SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 
    'SOL/USDT:USDT', 'XRP/USDT:USDT', 'LINK/USDT:USDT',
    'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'POL/USDT:USDT', 'DOT/USDT:USDT',
    'LTC/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT',
    '1000PEPE/USDT:USDT', 'FET/USDT:USDT', 'SEI/USDT:USDT', 'WLD/USDT:USDT',
    'INJ/USDT:USDT', 'APT/USDT:USDT'
]

# Setup Logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "data_collector_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CollectorV2")

def init_db():
    """Inicializa la base de datos V2 separada."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabla de Métricas de Order Book (Snapshots)
    c.execute('''
        CREATE TABLE IF NOT EXISTS orderbook_metrics (
            timestamp INTEGER,
            symbol TEXT,
            obi_5 REAL,
            obi_10 REAL,
            obi_20 REAL,
            spread_pct REAL,
            mid_price REAL,
            micro_price REAL,
            bid_depth_20 REAL,
            ask_depth_20 REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    
    # Tabla de Datos de Derivados
    c.execute('''
        CREATE TABLE IF NOT EXISTS derivatives_data (
            timestamp INTEGER,
            symbol TEXT,
            funding_rate REAL,
            open_interest REAL,
            open_interest_value REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Base de datos V2 inicializada en: {DB_PATH}")

def calculate_obi(bids, asks, depth):
    """Calcula Order Book Imbalance para una profundidad dada."""
    bid_vol = sum(b[1] for b in bids[:depth])
    ask_vol = sum(a[1] for a in asks[:depth])
    
    if (bid_vol + ask_vol) == 0:
        return 0
        
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def upgrade_db():
    """Actualiza el esquema de la DB si faltan columnas."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Intentar añadir columnas a derivatives_data
        c.execute("ALTER TABLE derivatives_data ADD COLUMN taker_buy_vol REAL")
        c.execute("ALTER TABLE derivatives_data ADD COLUMN taker_sell_vol REAL")
        logger.info("✅ Columnas taker_buy_vol/taker_sell_vol añadidas a derivatives_data")
    except sqlite3.OperationalError:
        # Ya existen
        pass
    conn.commit()
    conn.close()

def fetch_and_store(exchange):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = int(time.time() * 1000)
    
    for symbol in SYMBOLS:
        try:
            # 1. Order Book
            book = exchange.fetch_order_book(symbol, limit=20)
            bids = book['bids']
            asks = book['asks']
            
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread_pct = (best_ask - best_bid) / mid_price
                
                # OBI Metrics
                obi_5 = calculate_obi(bids, asks, 5)
                obi_10 = calculate_obi(bids, asks, 10)
                obi_20 = calculate_obi(bids, asks, 20)
                
                # Depth (Liquidez)
                bid_depth_20 = sum(b[1] for b in bids[:20])
                ask_depth_20 = sum(a[1] for a in asks[:20])
                
                # Micro-price (Weighted Mid Price)
                total_vol_top = bids[0][1] + asks[0][1]
                micro_price = mid_price
                if total_vol_top > 0:
                    micro_price = (best_bid * asks[0][1] + best_ask * bids[0][1]) / total_vol_top

                cursor.execute('''
                    INSERT OR REPLACE INTO orderbook_metrics 
                    (timestamp, symbol, obi_5, obi_10, obi_20, spread_pct, mid_price, micro_price, bid_depth_20, ask_depth_20)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (now, symbol, obi_5, obi_10, obi_20, spread_pct, mid_price, micro_price, bid_depth_20, ask_depth_20))

            # 2. Funding Rate
            funding = exchange.fetch_funding_rate(symbol)
            funding_rate = funding['fundingRate']
            
            # 3. Open Interest
            oi = exchange.fetch_open_interest(symbol)
            open_interest = oi['openInterestAmount']
            open_interest_val = oi['openInterestValue'] # En USD
            
            # 4. Taker Volume (Desde Raw Klines para obtener Taker Buy Vol)
            # Necesitamos el ID de mercado limpio (ej. BTCUSDT)
            market = exchange.market(symbol)
            market_id = market['id'] # Debería ser BTCUSDT
            
            taker_buy_vol = 0
            taker_sell_vol = 0
            
            try:
                # Usamos el método implícito para Binance Futures
                if exchange.id == 'binance':
                    response = exchange.fapiPublicGetKlines({
                        'symbol': market_id,
                        'interval': '1m',
                        'limit': 1
                    })
                    if len(response) > 0:
                        candle = response[0]
                        total_vol = float(candle[5])
                        taker_buy_vol = float(candle[9])
                        taker_sell_vol = total_vol - taker_buy_vol
                else:
                    # Fallback genérico (no tenemos taker vol)
                    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1)
                    if len(ohlcv) > 0:
                        total_vol = ohlcv[0][5]
                        taker_buy_vol = total_vol / 2
                        taker_sell_vol = total_vol / 2
                        
            except Exception as e:
                logger.warning(f"⚠️ Fallo obteniendo Taker Vol para {symbol}: {e}")

            cursor.execute('''
                INSERT OR REPLACE INTO derivatives_data
                (timestamp, symbol, funding_rate, open_interest, open_interest_value, taker_buy_vol, taker_sell_vol)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (now, symbol, funding_rate, open_interest, open_interest_val, taker_buy_vol, taker_sell_vol))
            
            logger.info(f"✅ {symbol} procesado. OBI: {obi_5:.2f} | Fund: {funding_rate:.6f} | BuyVol: {taker_buy_vol:.1f}")
            
        except Exception as e:
            logger.error(f"❌ Error procesando {symbol}: {e}")
            
    conn.commit()
    conn.close()

def main():
    init_db()
    upgrade_db()
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    logger.info("🚀 Iniciando Colector V2 (Loop infinito cada 10s)")
    
    while True:
        try:
            start_time = time.time()
            fetch_and_store(exchange)
            elapsed = time.time() - start_time
            
            sleep_time = max(0, 10 - elapsed)
            if sleep_time > 0:
                logger.info(f"💤 Durmiendo {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("🛑 Colector detenido por usuario")
            break
        except Exception as e:
            logger.error(f"💥 Error crítico en loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
```

## 2.2 Entrenamiento: `train_v2_production.py`

**Función:**
Entrena el "Consejo de Sabios" (Ensemble).
1.  **Feature Engineering:** Calcula 19 features (incluyendo Meta-Features como `slope_price` y `volume_trend`).
2.  **Modelos:** Entrena 4 arquitecturas distintas para diversidad cognitiva:
    *   **LSTM:** Memoria a largo plazo.
    *   **TCN:** Red Convolucional Temporal (detecta patrones locales).
    *   **XGBoost:** Potencia en datos tabulares.
    *   **Transformer:** Atención (detecta relaciones complejas).
3.  **Persistencia:** Guarda los modelos entrenados en `models/v2_ensemble/SYMBOL/`.

### 💻 Código Fuente

```python
import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Importar nuestros modelos
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV2")

# Config
VERSION = "v2.1"  # Consejo de Sabios v2.1 con Meta-Features
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SYMBOL = "ADA/USDT:USDT" # Entrenamos con ADA como base (luego se puede hacer multi-symbol)
SEQ_LEN = 12
PREDICT_HORIZON = 5

# ═══════════════════════════════════════════════════════════════════════════
# CONSEJO DE SABIOS v2.1: Meta-Features para darle "memoria" a XGBoost
# ═══════════════════════════════════════════════════════════════════════════
def add_robust_meta_features(df, window=12):
    """
    Genera 6 meta-features de forma segura (maneja NaNs) y rápida.
    Esto permite que XGBoost "vea" tendencias, no solo el instante actual.
    """
    df = df.copy()
    
    # 1. Rolling Calculations para OBI (Order Book Imbalance)
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # 2. Total volume proxy
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    
    # 3. Volume Trend (con protección división por cero)
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    
    # 4. Price Slope (Optimizado - 100x más rápido que polyfit)
    # Pendiente = (PrecioActual - PrecioHace12ticks) / 12
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    
    # ⚠️ PELIGRO MITIGADO: Eliminar NaNs creados por rolling/shift
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len - final_len > 0:
        logger.info(f"⚠️ Meta-Features: Removidas {initial_len - final_len} filas con NaNs iniciales.")
        
    return df

def load_data_from_db(symbol):
    logger.info(f"📥 Loading data for {symbol}...")
    conn = sqlite3.connect(DB_PATH)
    
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price, 
        o.micro_price,
        o.bid_depth_20 as bid_depth, 
        o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_5,
        o.obi_10,
        o.obi_20 as obi,
        d.funding_rate, 
        d.open_interest,
        d.taker_buy_vol,
        d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{symbol}'
    ORDER BY o.timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_all_symbols():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM derivatives_data")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def train_model_for_symbol(symbol):
    # Sanitize symbol for folder name (ADA/USDT:USDT -> ADAUSDT)
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    symbol_dir = MODELS_DIR / clean_symbol
    
    if symbol_dir.exists():
        shutil.rmtree(symbol_dir)
    symbol_dir.mkdir(parents=True)
    
    logger.info(f"🚀 Starting training for {symbol} -> {symbol_dir}")
    
    # 2. Load Data
    df = load_data_from_db(symbol)
    if len(df) < 100:
        logger.warning(f"⚠️ Not enough data for {symbol} ({len(df)} rows). Skipping.")
        return

    # 3. Feature Engineering
    # Targets
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    threshold = 0.001
    conditions = [
        (df['return_5m'] < -threshold),
        (df['return_5m'] > threshold)
    ]
    choices = [0, 2] # 0: Short, 2: Long
    df['label'] = np.select(conditions, choices, default=1)
    
    # Create derived features
    df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSEJO DE SABIOS v2.1: Agregar Meta-Features
    # ═══════════════════════════════════════════════════════════════════════════
    df = add_robust_meta_features(df, window=SEQ_LEN)
    
    # Features base (13) + Meta-Features (6) = 19 features totales
    base_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 
        'obi_5', 'obi_10', 'obi',
        'micro_price',
        'funding_rate', 'open_interest',
        'taker_buy_vol', 'taker_sell_vol',
        'buy_sell_ratio', 'depth_imbalance'
    ]
    
    meta_cols = [
        'mean_obi_12', 'max_obi_12', 'std_obi_12',
        'slope_price_12', 'mean_volume_12', 'volume_trend'
    ]
    
    feature_cols = base_cols + meta_cols
    logger.info(f"🧙 Consejo de Sabios {VERSION}: Entrenando con {len(feature_cols)} features")
    
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    joblib.dump(scaler, symbol_dir / "scaler.pkl")
    # Save feature names
    with open(symbol_dir / "features.json", 'w') as f:
        json.dump(feature_cols, f)
        
    # 5. Prepare Sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        Xs.append(X_scaled[i:(i + SEQ_LEN)])
        ys.append(y[i + SEQ_LEN])
    
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    
    if len(X_seq) < 10:
        logger.warning(f"⚠️ Not enough sequences for {symbol}. Skipping.")
        return

    # Convert to Tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Using device: {device}")
    if device == "cuda":
        logger.info(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 6. Time Series Split (80/20) - NO SHUFFLE!
    split_idx = int(len(X_seq) * 0.8)
    
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    logger.info(f"   Split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Create Loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Shuffle only train
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_dim = len(feature_cols)
    
    # --- MODEL 1: LSTM ---
    logger.info(f"   🏋️ Training LSTM for {clean_symbol}...")
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    lstm.train()
    lstm.train()
    for epoch in range(10):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = lstm(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(lstm.state_dict(), symbol_dir / "lstm.pt")
    with open(symbol_dir / "lstm_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'hidden_dim': 64, 'lstm_layers': 2, 'dropout': 0.2, 'num_classes': 3}}, f)
        
    # --- MODEL 2: TCN ---
    logger.info(f"   🏋️ Training TCN for {clean_symbol}...")
    tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64], kernel_size=3, num_classes=3).to(device)
    optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)
    
    tcn.train()
    tcn.train()
    for epoch in range(10):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = tcn(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(tcn.state_dict(), symbol_dir / "tcn.pt")
    with open(symbol_dir / "tcn_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'num_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.2}}, f)
        
    # --- MODEL 3: XGBoost ---
    logger.info(f"   🏋️ Training XGBoost for {clean_symbol}...")
    
    # Flatten for XGBoost (last step only)
    X_train_flat = X_train[:, -1, :]
    X_val_flat = X_val[:, -1, :]
    
    xgb = XGBoostTradingModel(use_gpu=(device=="cuda"))
    xgb.train(X_train_flat, y_train, X_val_flat, y_val)
    xgb.save(str(symbol_dir / "xgboost.joblib"))
    with open(symbol_dir / "xgboost_config.json", 'w') as f:
        json.dump({}, f)
        
    # --- MODEL 4: Transformer ---
    logger.info(f"   🏋️ Training Transformer for {clean_symbol}...")
    transformer = TradingTransformer(
        input_dim=input_dim, 
        d_model=64, 
        nhead=4, 
        num_layers=2,
        num_classes=3
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005)
    
    transformer.train()
    for epoch in range(10):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = transformer(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(transformer.state_dict(), symbol_dir / "transformer.pt")
    with open(symbol_dir / "transformer_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1}}, f)
        
    logger.info(f"✅ Models for {clean_symbol} saved (LSTM, TCN, XGBoost, Transformer).")

def train_production():
    parser = argparse.ArgumentParser(description='Train Consejo de Sabios v2.1')
    parser.add_argument('--symbol', type=str, help='Specific symbol to train (e.g., BNBUSDT)')
    parser.add_argument('--shard_id', type=int, default=0, help='Shard ID for parallel training (0-based)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards')
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
        logger.info(f"🎯 Training SINGLE symbol: {symbols}")
    else:
        symbols = get_all_symbols()
        logger.info(f"Found {len(symbols)} symbols in DB: {symbols}")
        
    # Sharding Logic
    if args.num_shards > 1:
        all_count = len(symbols)
        symbols = [s for i, s in enumerate(symbols) if i % args.num_shards == args.shard_id]
        logger.info(f"🧩 Worker {args.shard_id}/{args.num_shards} processing {len(symbols)}/{all_count} symbols")
    
    for symbol in symbols:
        try:
            train_model_for_symbol(symbol)
        except Exception as e:
            logger.error(f"❌ Failed training {symbol}: {e}")
            
    logger.info("🎉 All training tasks completed.")

if __name__ == "__main__":
    train_production()
```

## 2.3 Servicio de Inferencia: `ml_service_v2.py`

**Función:**
Es el servidor API (FastAPI) que expone los modelos.
1.  **Carga de Modelos:** Al inicio, carga todos los ensembles en memoria (GPU).
2.  **Endpoint `/predict`:** Recibe un símbolo, busca los datos más recientes en la DB, calcula features, y ejecuta el ensemble.
3.  **Filtro Ninja:** Aplica un suavizado exponencial asimétrico (EMA) a las probabilidades para filtrar el ruido y reaccionar rápido al pánico.

### 💻 Código Fuente

```python
"""FastAPI service V2 (Ninja Mode) that returns ML probabilities using Order Book data."""
from __future__ import annotations

import logging
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from typing import Dict, Optional, List, Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import V2 Models
from ml.advanced_models.ensemble_manager import EnsembleManager

# --- Logger Setup ---
class ServiceLogger:
    def __init__(self, name: str = "ml_service_v2") -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        self._logger = logging.getLogger(name)

    def info(self, msg: str, **kwargs): self._logger.info(f"{msg} {kwargs if kwargs else ''}")
    def error(self, msg: str, **kwargs): self._logger.error(f"{msg} {kwargs if kwargs else ''}")
    def warning(self, msg: str, **kwargs): self._logger.warning(f"{msg} {kwargs if kwargs else ''}")

LOGGER = ServiceLogger()

# --- Config ---
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble" # Carpeta futura para modelos V2

# --- Data Models ---
class ProbabilityRequestV2(BaseModel):
    symbol: str # Ej: "ADA/USDT:USDT" o "ADAUSDT"

class ProbabilityResponseV2(BaseModel):
    symbol: str
    long_prob: float
    short_prob: float
    neutral_prob: float
    consensus_level: float
    meta_verdict: str # "APPROVED" | "VETOED"

# --- Data Loader ---
def load_latest_data(symbol: str, limit: int = 60) -> pd.DataFrame:
    """Carga los últimos N registros de la DB V2."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    # Normalizar símbolo para DB (ADAUSDT -> ADA/USDT:USDT)
    # Asumimos que el bot envía formato CCXT o limpio.
    # La DB tiene formato CCXT: "ADA/USDT:USDT"
    db_symbol = symbol
    if "/" not in symbol:
        # Intento simple de conversión si viene como ADAUSDT
        # Esto es frágil, idealmente el bot envía el formato correcto
        pass 

    conn = sqlite3.connect(DB_PATH)
    
    # Query con JOIN y Taker Vol
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price, 
        o.micro_price,
        o.bid_depth_20 as bid_depth, 
        o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_5,
        o.obi_10,
        o.obi_20 as obi,
        d.funding_rate, 
        d.open_interest,
        d.taker_buy_vol,
        d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{db_symbol}'
    ORDER BY o.timestamp DESC
    LIMIT {limit}
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        df = df.sort_values('timestamp') # Reordenar ascendente para secuencia
    finally:
        conn.close()
        
    return df

# --- Model Manager ---
class V2ModelManager:
    def __init__(self):
        self.ensembles: Dict[str, EnsembleManager] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info(f"🚀 ML Service initialized on device: {self.device}")
        
        # ═══════════════════════════════════════════════════════
        # FASE 4.5: FILTRO NINJA (EMA ASIMÉTRICO)
        # ═══════════════════════════════════════════════════════
        self.smoothed_probs_cache = {} 
        # Alphas dinámicos se definen en predict()
        
    def _clean_symbol(self, symbol: str) -> str:
        # ADA/USDT:USDT -> ADAUSDT
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def get_ensemble(self, symbol: str) -> Optional[EnsembleManager]:
        clean_sym = self._clean_symbol(symbol)
        
        if clean_sym in self.ensembles:
            return self.ensembles[clean_sym]
            
        # Try to load
        return self.load_model_for_symbol(clean_sym)

    def load_model_for_symbol(self, clean_symbol: str) -> Optional[EnsembleManager]:
        symbol_dir = MODELS_DIR / clean_symbol
        if not symbol_dir.exists():
            LOGGER.warning(f"No models found for {clean_symbol} at {symbol_dir}")
            return None
            
        try:
            LOGGER.info(f"Loading models for {clean_symbol}...")
            ensemble = EnsembleManager(device=self.device)
            
            # Load Scaler
            self.scalers[clean_symbol] = joblib.load(symbol_dir / "scaler.pkl")
            
            # Load Feature Names
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_symbol] = json.load(f)
            
            # Detect version and load weights
            is_v2_1 = len(self.feature_cols[clean_symbol]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            # ensemble = EnsembleManager(device=self.device) # REMOVED: Double instantiation bug
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            # Load Transformer if available (backwards compatible)
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_symbol] = ensemble
            LOGGER.info(f"✅ Loaded {clean_symbol} ensemble.")
            return ensemble
            
        except Exception as e:
            LOGGER.error(f"Failed to load {clean_symbol}: {e}")
            return None

    def load_models(self):
        # Pre-load all available models in directory
        if not MODELS_DIR.exists():
            LOGGER.warning(f"Models dir {MODELS_DIR} not found.")
            return
            
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                self.load_model_for_symbol(item.name)

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        ensemble = self.get_ensemble(symbol)
        clean_sym = self._clean_symbol(symbol)
        
        if ensemble is None:
            # Dummy response if model missing (Fail Safe: Neutral)
            return {
                'ensemble_probs': torch.tensor([[0.0, 1.0, 0.0]]),
                'consensus': 0.0
            }
            
        # 1. Feature Engineering (Derived Features)
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # CONSEJO DE SABIOS v2.1: Agregar Meta-Features si el modelo las requiere
        # ═══════════════════════════════════════════════════════════════════════════
        cols = self.feature_cols.get(clean_sym, [])
        n_features = len(cols)
        
        # Detectar versión basada en número de features (13 = v2.0, 19 = v2.1)
        is_v2_1 = n_features >= 19 or 'mean_obi_12' in cols
        
        if is_v2_1:
            # Calcular meta-features en tiempo real
            window = 12
            df['mean_obi_12'] = df['obi'].rolling(window, min_periods=1).mean()
            df['max_obi_12'] = df['obi'].rolling(window, min_periods=1).max()
            # FIX: std con min_periods=2 para evitar NaN, luego fillna(0) por seguridad
            df['std_obi_12'] = df['obi'].rolling(window, min_periods=2).std().fillna(0)
            
            df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
            df['mean_volume_12'] = df['total_volume'].rolling(window, min_periods=1).mean()
            df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
            # FIX: fillna con método forward/backward para evitar NaN en primeras filas
            df['slope_price_12'] = (df['price'] - df['price'].shift(window).bfill()) / window
            
            LOGGER.info(f"🧙 Consejo v2.1: Calculated meta-features for {clean_sym} ({n_features} features)")
        else:
            LOGGER.info(f"📊 Consejo v2.0: Using legacy features for {clean_sym} ({n_features} features)")
        
        # FIX #3: Usar orden de columnas EXACTO del features.json (guardado en training)
        try:
            X = df[cols].values
            
            # Sanity check: Detectar NaNs antes del scaler
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                LOGGER.warning(f"⚠️ Found {nan_count} NaNs in features for {clean_sym}, filling with 0")
                X = np.nan_to_num(X, nan=0.0)
                
        except KeyError as e:
            LOGGER.error(f"Missing columns/config for {clean_sym}: {e}")
            LOGGER.error(f"Available columns: {list(df.columns)}")
            LOGGER.error(f"Required columns: {cols}")
            raise e
            
        # 2. Scaling
        scaler = self.scalers[clean_sym]
        
        # FIX #1: Validar dimensiones del scaler vs features
        expected_features = scaler.n_features_in_
        actual_features = X.shape[1]
        
        if expected_features != actual_features:
            LOGGER.error(f"❌ Scaler/Feature mismatch for {clean_sym}!")
            LOGGER.error(f"   Scaler expects: {expected_features} features")
            LOGGER.error(f"   Data has: {actual_features} features")
            LOGGER.error(f"   This likely means model version mismatch (v2.0 scaler with v2.1 features)")
            raise ValueError(f"Feature dimension mismatch: scaler={expected_features}, data={actual_features}")
        
        X_scaled = scaler.transform(X)
        
        # 3. Sequence Creation
        SEQ_LEN = 12
        if len(X_scaled) < SEQ_LEN:
            LOGGER.warning(f"Not enough data for sequence. Need {SEQ_LEN}, got {len(X_scaled)}")
            pad_len = SEQ_LEN - len(X_scaled)
            X_scaled = np.pad(X_scaled, ((pad_len, 0), (0, 0)), mode='edge')
            
        X_seq = X_scaled[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        # 4. Predict
        with torch.no_grad():
            result = ensemble.predict(X_tensor)
            
        # ═══════════════════════════════════════════════════════
        # FASE 4.5: FILTRO NINJA (EMA ASIMÉTRICO)
        # Filosofía: "Subir lento (escéptico), Bajar rápido (paranoico)"
        # ═══════════════════════════════════════════════════════
        ensemble_probs = result['ensemble_probs'][0].tolist()
        
        raw_dict = {
            'short': float(ensemble_probs[0]),
            'neutral': float(ensemble_probs[1]),
            'long': float(ensemble_probs[2])
        }

        # 1. Obtener estado anterior (o usar cruda si es la primera vez)
        prev_smoothed = self.smoothed_probs_cache.get(clean_sym, raw_dict)

        # 2. Definir la personalidad del filtro
        ALPHA_SLOW = 0.15  # Escéptico: Si la señal sube, cuesta trabajo creerla
        ALPHA_FAST = 0.70  # Paranoico: Si la señal baja, reaccionamos YA

        smoothed_dict = {}

        # 3. Aplicar lógica asimétrica
        for key in ['short', 'neutral', 'long']:
            raw_val = raw_dict[key]
            prev_val = prev_smoothed[key]
            
            # Mágia aquí: ¿La señal está mejorando o empeorando?
            diff = raw_val - prev_val
            
            if diff > 0:
                # La probabilidad está subiendo -> PIDE CONFIRMACIÓN (Lento)
                alpha = ALPHA_SLOW
            else:
                # La probabilidad está bajando -> PÁNICO (Rápido)
                alpha = ALPHA_FAST
            
            # Fórmula EMA estándar
            new_val = (alpha * raw_val) + ((1 - alpha) * prev_val)
            smoothed_dict[key] = new_val

        # 4. Normalizar (asegurar que sumen 1.0)
        total = smoothed_dict['short'] + smoothed_dict['neutral'] + smoothed_dict['long']
        normalized_dict = {k: v / total for k, v in smoothed_dict.items()}

        # 5. Guardar en caché para el próximo tick
        self.smoothed_probs_cache[clean_sym] = normalized_dict

        # 6. Actualizar el resultado con el valor suavizado
        result['ensemble_probs'] = torch.tensor([[
            normalized_dict['short'],
            normalized_dict['neutral'],
            normalized_dict['long']
        ]])
            
        return result

MANAGER = V2ModelManager()

# --- API ---
router = APIRouter(prefix="/ml-v2", tags=["ml-v2"])

@router.on_event("startup")
async def startup_event():
    MANAGER.load_models()

@router.post("/predict", response_model=ProbabilityResponseV2)
async def predict_endpoint(request: ProbabilityRequestV2) -> ProbabilityResponseV2:
    symbol = request.symbol
    
    # 1. Load Data
    try:
        # Intentamos cargar con el símbolo tal cual, si falla probamos variantes
        df = load_latest_data(symbol)
        if df.empty:
            # Try converting ADAUSDT -> ADA/USDT:USDT
            alt_symbol = symbol.replace("USDT", "/USDT:USDT")
            df = load_latest_data(alt_symbol)
            
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No V2 data found for {symbol}")
            
    except Exception as e:
        LOGGER.error(f"Data load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Predict
    result = MANAGER.predict(symbol, df)
    probs = result['ensemble_probs'][0].tolist() # [Short, Neutral, Long]
    
    return ProbabilityResponseV2(
        symbol=symbol,
        short_prob=probs[0],
        neutral_prob=probs[1],
        long_prob=probs[2],
        consensus_level=float(result.get('consensus', 0.0)),
        meta_verdict="APPROVED" # Placeholder
    )

app = FastAPI(title="ML Service V2 (Ninja)", version="2.0.0")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Puerto 8001 para no chocar con el V1
    uvicorn.run("services.ml_service_v2:app", host="0.0.0.0", port=8001, reload=False)
```

## 2.4 Cliente del Bot: `ml_probability_service.ts`

**Función:**
Es el puente entre el Bot (TypeScript) y el Servicio ML (Python).
*   Envía requests POST a `http://127.0.0.1:8001/ml-v2/predict`.
*   Maneja errores de red.
*   Entrega la respuesta tipada (`MlProbabilityResponse`) a la estrategia.

### 💻 Código Fuente

```typescript
import axios, { AxiosInstance, isAxiosError } from 'axios';
import { Candle } from '../core/types';

// V2 Response Type
export type MlProbabilityResponse = {
  symbol: string;
  long_prob: number;
  short_prob: number;
  neutral_prob: number;
  consensus_level: number;
  meta_verdict: string;
  // Legacy support (optional)
  primary_timeframe?: string;
  probabilities?: Record<string, { long_prob: number; short_prob: number }>;
};

export type MlProbabilityClientOptions = {
  baseUrl?: string;
  timeoutMs?: number;
};

export class MlServiceError extends Error {
  readonly status?: number;
  readonly payload?: unknown;

  constructor(message: string, opts: { status?: number; payload?: unknown } = {}) {
    super(message);
    this.name = 'MlServiceError';
    this.status = opts.status;
    this.payload = opts.payload;
  }
}

export class MlProbabilityServiceClient {
  private readonly http: AxiosInstance;
  private readonly baseUrl: string;

  constructor(opts: MlProbabilityClientOptions = {}) {
    // V2 Service runs on port 8001 by default
    const envBase = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001';
    this.baseUrl = (opts.baseUrl ?? envBase).replace(/\/+$/, '');

    this.http = axios.create({
      baseURL: this.baseUrl,
      timeout: opts.timeoutMs ?? 10000,
    });
  }

  async fetchProbabilities(params: {
    symbol: string;
    // Legacy params (ignored in V2 but kept for interface compatibility)
    candles?: Candle[];
    timeframe?: string;
    forceRefresh?: boolean;
    extraCandles?: Record<string, Candle[]>;
  }): Promise<MlProbabilityResponse> {
    const { symbol } = params;
    
    // V2 Payload: Just the symbol
    const payload = { symbol };

    try {
      const { data } = await this.http.post<MlProbabilityResponse>(
        '/ml-v2/predict',
        payload,
      );
      
      // Adapt V2 response to look a bit like V1 if needed by consumer, 
      // or just return as is. The consumer (strategy) should be updated to use neutral_prob.
      return {
        ...data,
        primary_timeframe: '1m', // V2 works on 1m data
        probabilities: {
          '1m': { long_prob: data.long_prob, short_prob: data.short_prob }
        }
      };
      
    } catch (err) {
      if (isAxiosError(err)) {
        const status = err.response?.status;
        const detail = err.response?.data;
        const message =
          typeof detail === 'string'
            ? detail
            : (detail as any)?.detail?.message ||
              (detail as any)?.detail ||
              err.message ||
              'ml_service_error';
        throw new MlServiceError(message, { status, payload: detail });
      }
      throw err;
    }
  }
}
```
