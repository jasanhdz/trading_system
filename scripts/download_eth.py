import ccxt
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time

# Add root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.storage.database_manager import db_manager

def download_and_save(symbol, timeframe, days=1000):
    print(f"Downloading {symbol} {timeframe} ({days} days)...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"  Fetched {len(ohlcv)} candles, last: {datetime.fromtimestamp(ohlcv[-1][0]/1000)}")
            
            if len(all_ohlcv) >= 5000:
                 _save_batch(symbol, timeframe, all_ohlcv)
                 all_ohlcv = []
                 
            time.sleep(0.5)
                 
        except Exception as e:
            print(f"Error: {e}")
            break
            
    if all_ohlcv:
        _save_batch(symbol, timeframe, all_ohlcv)

def _save_batch(symbol, timeframe, ohlcv_list):
    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Use the symbol format expected by the DB
    db_symbol = symbol.replace("USDT", "/USDT") if "USDT" in symbol and "/" not in symbol else symbol
    
    print(f"  Saving {len(df)} rows to DB as {db_symbol}...")
    try:
        db_manager.insert_ohlcv_data(df, db_symbol, timeframe)
        print("  Saved.")
    except Exception as e:
        print(f"  Save failed: {e}")

if __name__ == "__main__":
    # Download ETHUSDT for 15m and 1h
    download_and_save("ETHUSDT", "15m", 1000)
    download_and_save("ETHUSDT", "1h", 1000)
