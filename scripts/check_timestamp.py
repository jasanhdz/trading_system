import sqlite3

conn = sqlite3.connect("data/market_data_v2.db")
cursor = conn.cursor()
cursor.execute("SELECT timestamp FROM orderbook_metrics ORDER BY timestamp DESC LIMIT 1")
print(cursor.fetchone())
conn.close()
