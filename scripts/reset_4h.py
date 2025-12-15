import sys
from pathlib import Path
import ccxt
import pandas as pd
import time
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.storage.database_manager import db_manager
from data.storage.models import OHLCVData

def reset_4h():
    print("Deleting existing 4h data...")
    with db_manager.get_session() as session:
        # Try both symbol formats just in case
        session.query(OHLCVData).filter_by(symbol="BTC/USDT", timeframe="4h").delete()
        session.query(OHLCVData).filter_by(symbol="BTCUSDT", timeframe="4h").delete()
        print("Deleted.")

    print("Downloading fresh 4h data (1000 days)...")
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = "BTCUSDT"
    timeframe = "4h"
    days = 1000
    
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
            time.sleep(0.5)
                 
        except Exception as e:
            print(f"Error: {e}")
            break
            
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        db_symbol = "BTC/USDT"
        
        print(f"  Saving {len(df)} rows to DB as {db_symbol}...")
        try:
            db_manager.insert_ohlcv_data(df, db_symbol, timeframe)
            print("  Saved.")
        except Exception as e:
            print(f"  Save failed: {e}")

if __name__ == "__main__":
    reset_4h()
