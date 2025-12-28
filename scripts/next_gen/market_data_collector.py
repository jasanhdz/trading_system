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
    'LTC/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT'
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
