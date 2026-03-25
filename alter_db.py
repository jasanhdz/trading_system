import sqlite3
try:
    conn = sqlite3.connect('/home/jasan/Develop/trading_system/data/binance_candles.db')
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE ohlcv_data ADD COLUMN buy_volume FLOAT;")
    conn.commit()
    print("Column added.")
except Exception as e:
    print(e)
finally:
    conn.close()
