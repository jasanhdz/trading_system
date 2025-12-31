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
        print(f"Profit Factor:   {wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 'Inf':.2f}")
        print("-" * 30)
        print("Motivos de Salida:")
        print(df_trades['reason'].value_counts())
        print("="*50)
        
        # Guardar CSV
        csv_path = f"backtest_trades_{self.symbol}_v2.csv"
        df_trades.to_csv(csv_path, index=False)
        LOGGER.info(f"📝 Trades guardados en {csv_path}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title(f"Equity Curve - {self.symbol} (Ninja Protocol)")
        plt.xlabel("Ticks")
        plt.ylabel("Equity ($)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"backtest_equity_{self.symbol}_v2.png")
        LOGGER.info(f"📈 Gráfico guardado en backtest_equity_{self.symbol}_v2.png")

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
