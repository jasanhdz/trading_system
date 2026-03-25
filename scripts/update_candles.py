import sys
import os
import pandas as pd
from pathlib import Path
import ccxt

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from data.collectors.binance_collector import BinanceDataCollector
from data.storage.database_manager import DatabaseManager

def update_db():
    print("🔄 Updating Market Data...")
    collector = BinanceDataCollector()
    
    # We need to update the main source used by load_hybrid_data
    # load_hybrid_data reads from database_manager or CSV?
    # Let's check data_loader.py logic.
    # Assuming standard DB management.
    
    db_url = settings.DATABASE_URL
    db = DatabaseManager(db_url)
    db.create_tables() # Ensure tables exist
    
    symbol = settings.SYMBOL # Target
    
    # Fetch last 1000 candles (covers ~3.5 days of 5m data)
    df = collector.get_ohlcv(symbol, "5m", limit=1000)
    
    if not df.empty:
        print(f"   Fetched {len(df)} candles via API.")
        
        # Filter duplicates
        last_ts = db.get_latest_timestamp(symbol, "5m")
        if last_ts:
            # Ensure timezone awareness matches
            if df['timestamp'].dt.tz is None and last_ts.tzinfo:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            elif df['timestamp'].dt.tz and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=pd.Timestamp(df['timestamp'].dt.tz).tz)
                
            new_df = df[df['timestamp'] > last_ts]
        else:
            new_df = df
            
        if not new_df.empty:
            count = db.insert_ohlcv_data(new_df, symbol, "5m")
            print(f"✅ inserted {count} new candles.")
        else:
            print("✨ Database up to date (No new candles).")
    else:
        print("⚠️ No data fetched.")

if __name__ == "__main__":
    update_db()
