import sqlite3
import pandas as pd
import os

db_path = 'data/binance_futures_data.db'
csv_path = 'ETH_CLEAN_FOR_THESIS.csv'

print(f"Checking DB: {db_path}")
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {tables}")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT count(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Table '{table_name}' has {count} rows.")
            # Show first row
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1", conn)
            print(f"Columns in {table_name}: {df.columns.tolist()}")
        conn.close()
    except Exception as e:
        print(f"Error reading DB: {e}")
else:
    print("DB file not found.")

print(f"\nChecking CSV: {csv_path}")
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path, nrows=5)
        print(f"CSV Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print("CSV file not found.")
