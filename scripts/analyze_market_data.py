import sqlite3
import time
from datetime import datetime, timedelta
import math

DB_PATH = '/home/jasan/Develop/trading_system/data/market_data_v2.db'

def calculate_std_dev(data):
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def analyze_market():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Get Schema
        print("--- Schema ---")
        cursor.execute("PRAGMA table_info(orderbook_metrics)")
        columns_info = cursor.fetchall()
        columns = [info[1] for info in columns_info]
        print(f"Columns: {columns}")
        
        # Identify column indices
        try:
            idx_symbol = columns.index('symbol')
            idx_ts = columns.index('timestamp')
            # Look for 'obi' or similar
            obi_col = next((c for c in columns if 'obi' in c), None)
            price_col = next((c for c in columns if 'price' in c or 'mid' in c), None)
            spread_col = next((c for c in columns if 'spread' in c), None)
            
            print(f"Using columns: Symbol={idx_symbol}, Time={idx_ts}, OBI={obi_col}, Price={price_col}, Spread={spread_col}")
            
            if not all([obi_col, price_col]):
                print("Critical columns missing. Aborting analysis.")
                return

        except ValueError:
            print("Could not find required columns.")
            return

        # 2. Query Data (Last 6 Hours)
        six_hours_ago = int((datetime.now() - timedelta(hours=6)).timestamp() * 1000)
        print(f"\nQuerying data since: {datetime.fromtimestamp(six_hours_ago/1000)}")
        
        query = f"SELECT * FROM orderbook_metrics WHERE timestamp > {six_hours_ago}"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("No data found for the last 6 hours.")
            return

        print(f"\nData Points Found: {len(rows)}")
        
        # 3. Aggregate Data
        data_by_symbol = {}
        
        idx_obi = columns.index(obi_col)
        idx_price = columns.index(price_col)
        idx_spread = columns.index(spread_col) if spread_col else -1
        
        for row in rows:
            sym = row[idx_symbol]
            obi = row[idx_obi]
            price = row[idx_price]
            spread = row[idx_spread] if idx_spread != -1 else 0
            
            if sym not in data_by_symbol:
                data_by_symbol[sym] = {'obi': [], 'price': [], 'spread': []}
            
            if obi is not None: data_by_symbol[sym]['obi'].append(float(obi))
            if price is not None: data_by_symbol[sym]['price'].append(float(price))
            if spread is not None: data_by_symbol[sym]['spread'].append(float(spread))

        # 4. Calculate Stats
        print(f"\n{'Symbol':<10} | {'Price Change %':<15} | {'Avg OBI':<10} | {'OBI StdDev':<10} | {'Count':<5}")
        print("-" * 65)
        
        total_volatility = 0
        count_symbols = 0
        
        for sym, values in data_by_symbol.items():
            prices = values['price']
            obis = values['obi']
            
            if not prices: continue
            
            min_p = min(prices)
            max_p = max(prices)
            price_change_pct = ((max_p - min_p) / min_p) * 100 if min_p > 0 else 0
            
            avg_obi = sum(obis) / len(obis) if obis else 0
            std_obi = calculate_std_dev(obis)
            
            print(f"{sym:<10} | {price_change_pct:<15.2f} | {avg_obi:<10.2f} | {std_obi:<10.2f} | {len(prices):<5}")
            
            total_volatility += price_change_pct
            count_symbols += 1
            
        avg_vol = total_volatility / count_symbols if count_symbols > 0 else 0
        print(f"\nAverage Market Volatility (Price Fluctuation): {avg_vol:.2f}%")
        
        if avg_vol < 1.0:
            print("CONCLUSION: Market is LATERAL (Low Volatility).")
        else:
            print("CONCLUSION: Market shows activity.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    analyze_market()
