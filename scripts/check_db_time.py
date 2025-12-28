import sqlite3
import datetime

DB_PATH = '/home/jasan/Develop/trading_system/data/market_data_v2.db'

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM orderbook_metrics")
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0]:
        ts = result[0]
        dt = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc)
        print(f"Latest DB Timestamp: {ts}")
        print(f"Latest DB DateTime: {dt}")
    else:
        print("No data found in DB.")
except Exception as e:
    print(f"Error: {e}")
