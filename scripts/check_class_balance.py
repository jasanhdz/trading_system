import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "data/market_data_v2.db"
SYMBOL = "AVAX/USDT:USDT"
PREDICT_HORIZON = 5

def check_balance():
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT mid_price as price
    FROM orderbook_metrics
    WHERE symbol = '{SYMBOL}'
    ORDER BY timestamp DESC
    LIMIT 100000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Recreate labels
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    threshold = 0.001 # 0.1%
    conditions = [
        (df['return_5m'] < -threshold),
        (df['return_5m'] > threshold)
    ]
    choices = [0, 2] # 0: Short, 2: Long
    df['label'] = np.select(conditions, choices, default=1)
    
    # Drop NaNs
    df = df.dropna()
    
    print(f"--- Class Balance for {SYMBOL} ---")
    counts = df['label'].value_counts(normalize=True).sort_index()
    print(counts)
    
    print("\n--- Raw Counts ---")
    print(df['label'].value_counts().sort_index())

if __name__ == "__main__":
    check_balance()
