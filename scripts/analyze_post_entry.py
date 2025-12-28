import sqlite3
import pandas as pd
import time

DB_PATH = "data/market_data_v2.db"
SYMBOL = "ETHUSDT"

def analyze():
    conn = sqlite3.connect(DB_PATH)
    
    # Last 10 minutes
    now = int(time.time() * 1000)
    start = now - (10 * 60 * 1000)
    
    query = f"""
    SELECT timestamp, mid_price, obi_20, taker_buy_vol, taker_sell_vol
    FROM market_metrics
    WHERE symbol = '{SYMBOL}'
    AND timestamp > {start}
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data found")
        return

    print(f"--- {SYMBOL} Post-Entry Analysis (Last 10m) ---")
    print(df.tail(20))
    
    # Check for Reversal Signals
    # 1. OBI Flip (Negative -> Positive)
    # 2. Volume Spike (Buy Vol >> Sell Vol)
    
    print("\n--- Reversal Check ---")
    last_obi = df['obi_20'].iloc[-1]
    if last_obi > 0.3:
        print(f"⚠️ OBI FLIP DETECTED: {last_obi:.4f} (Bullish Pressure)")
    else:
        print(f"✅ OBI Stable: {last_obi:.4f}")
        
    last_buy = df['taker_buy_vol'].iloc[-1]
    last_sell = df['taker_sell_vol'].iloc[-1]
    
    if last_buy > last_sell * 2:
        print(f"⚠️ BUY VOLUME SPIKE: {last_buy:.0f} vs {last_sell:.0f}")
    else:
         print(f"✅ Volume Normal: Buy {last_buy:.0f} / Sell {last_sell:.0f}")

if __name__ == "__main__":
    analyze()
