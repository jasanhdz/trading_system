import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "data/market_data_v2.db"
SYMBOL = "AVAX/USDT:USDT"
ENTRY_PRICE = 12.66
LOOKBACK_HOURS = 24

def analyze_entry():
    conn = sqlite3.connect(DB_PATH)
    
    # Get recent data
    import time
    now_ms = int(time.time() * 1000)
    lookback_ms = now_ms - (LOOKBACK_HOURS * 3600 * 1000)
    
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price,
        o.bid_depth_20,
        o.ask_depth_20,
        o.obi_20,
        d.taker_buy_vol,
        d.taker_sell_vol,
        d.open_interest
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{SYMBOL}'
    AND o.timestamp > {lookback_ms}
    ORDER BY o.timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data found for the last 24 hours.")
        return

    # Find timestamp closest to entry price
    # We look for when price crossed 12.66. 
    # Since it's a SHORT, we likely entered when price was dropping through 12.66 or bouncing off it.
    # Let's find exact matches or closest points.
    
    df['price_diff'] = abs(df['mid_price'] - ENTRY_PRICE)
    
    # Sort by difference to find closest points
    closest = df.sort_values('price_diff').head(5)
    
    print(f"--- Potential Entry Points (Price ~ {ENTRY_PRICE}) ---")
    print(closest[['timestamp', 'mid_price', 'price_diff']])
    
    # Let's assume the earliest "closest" point in the last few hours is the entry
    # Or better, look for the moment it crossed.
    
    # Let's pick the best candidate (closest price)
    best_match = closest.iloc[0]
    entry_time = best_match['timestamp']
    
    print(f"\nAnalyzing Entry Context around: {entry_time}")
    
    # Get 10 minutes before and after
    idx = df[df['timestamp'] == entry_time].index[0]
    start_idx = max(0, idx - 10) # Assuming 1 min candles? No, timestamp is usually ms or s. 
    # If rows are frequent, 10 rows might be seconds.
    # Let's check time diff between rows.
    
    # Just take a window of rows around the match
    window = df.iloc[max(0, idx-20) : min(len(df), idx+20)]
    
    print("\n--- Market Metrics (Window) ---")
    print(window[['timestamp', 'mid_price', 'obi_20', 'taker_buy_vol', 'taker_sell_vol', 'open_interest']].to_string())
    
    # Analysis
    avg_buy_vol = window['taker_buy_vol'].mean()
    avg_sell_vol = window['taker_sell_vol'].mean()
    
    print("\n--- Analysis Report ---")
    print(f"Entry Price: {ENTRY_PRICE}")
    print(f"Detected Price: {best_match['mid_price']} at {entry_time}")
    
    # Volume Analysis
    print(f"Avg Buy Vol: {avg_buy_vol:.2f}")
    print(f"Avg Sell Vol: {avg_sell_vol:.2f}")
    
    if avg_sell_vol > avg_buy_vol * 1.5:
        print("✅ Strong Sell Volume detected (Bearish Pressure).")
    elif avg_buy_vol > avg_sell_vol * 1.5:
        print("❌ Strong Buy Volume detected (Bullish Pressure) - Risky for Short.")
    else:
        print("⚠️ Volume is neutral.")

    # OBI Analysis
    obi = best_match['obi_20']
    print(f"Order Book Imbalance (OBI): {obi:.4f}")
    if obi < -0.2:
        print("✅ Order Book skewed to Ask side (Selling Pressure).")
    elif obi > 0.2:
        print("❌ Order Book skewed to Bid side (Buying Support) - Risky for Short.")
    else:
        print("⚠️ Order Book is balanced.")
        
    # Whale Detection (Spike in volume)
    max_vol = window[['taker_buy_vol', 'taker_sell_vol']].max().max()
    if max_vol > 50000: # Arbitrary large number for AVAX
        print(f"🐋 Whale Activity Detected! Max Vol Spike: {max_vol:.2f}")
    else:
        print(f"No massive whale spikes detected (Max: {max_vol:.2f})")

if __name__ == "__main__":
    analyze_entry()
