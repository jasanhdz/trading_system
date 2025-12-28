import sqlite3
import pandas as pd
import time
from datetime import datetime

DB_PATH = "data/market_data_v2.db"
SYMBOL = "ETHUSDT"
ENTRY_TIME_MS = 1766934733824 # 15:05 aprox
WINDOW_MS = 20 * 60 * 1000 # 20 min

def check_entry():
    conn = sqlite3.connect(DB_PATH)
    
    # Query around entry time
    start = ENTRY_TIME_MS - WINDOW_MS
    end = ENTRY_TIME_MS
    
    query = f"""
    SELECT timestamp, mid_price, obi_20, taker_buy_vol, taker_sell_vol
    FROM market_metrics
    WHERE symbol = '{SYMBOL}'
    AND timestamp BETWEEN {start} AND {end}
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data found")
        return

    print("--- ETH Metrics around Entry (Last 20m) ---")
    print(df.tail(20))
    
    # Check Sage Logic
    last_row = df.iloc[-1]
    obi = last_row['obi_20']
    buy_vol = last_row['taker_buy_vol']
    sell_vol = last_row['taker_sell_vol']
    
    print("\n--- Latest Metrics ---")
    print(f"Time: {last_row['timestamp']}")
    print(f"Price: {last_row['mid_price']}")
    print(f"OBI: {obi:.4f}")
    print(f"Buy Vol: {buy_vol:.0f}")
    print(f"Sell Vol: {sell_vol:.0f}")
    
    print("\n--- SageGuard Simulation (SHORT) ---")
    # Rule 1: OBI Check
    # Block SHORT if OBI > 0.2 (Bullish Wall)
    if obi > 0.2:
        print(f"❌ BLOCKED by OBI ({obi:.2f} > 0.2)")
    else:
        print(f"✅ ALLOWED by OBI ({obi:.2f} <= 0.2)")
        
    # Rule 2: Volume Check
    # Block SHORT if Buy Vol > Sell Vol * 3
    if buy_vol > sell_vol * 3 and buy_vol > 100:
         print(f"❌ BLOCKED by Volume (Buy {buy_vol} >> Sell {sell_vol})")
    else:
         print("✅ ALLOWED by Volume")

if __name__ == "__main__":
    check_entry()
