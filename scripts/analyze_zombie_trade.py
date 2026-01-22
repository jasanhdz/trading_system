import pandas as pd
import numpy as np

# Config
FEATURES_PATH = "data/phantom_features.csv"
START_TIME = "2025-06-09 14:15:00"
END_TIME = "2025-11-01 04:15:00"

# Exit Params (from backtest_phantom_v8.py)
SL_PCT = 0.015       # 1.5% SL
TP_PCT = 0.06        # 6% TP
TRAILING_DEV = 0.015 # 1.5% trailing
BE_TRIGGER_ROE = 0.10  # 10% ROE before BE
LEVERAGE = 5

def analyze_trade():
    print(f"🔍 Analyzing Zombie Trade: {START_TIME} -> {END_TIME}")
    
    df = pd.read_csv(FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Find start index
    start_row = df[df['timestamp'] == START_TIME]
    if start_row.empty:
        print("Start time not found")
        return
    
    start_idx = start_row.index[0]
    entry_price = start_row.iloc[0]['close_eth']
    
    print(f"📉 Entry Price: {entry_price:.2f}")
    
    # Initial State
    sl_price = entry_price * (1 + SL_PCT)
    tp_price = entry_price * (1 - TP_PCT)
    be_price = entry_price * (1 - 0.003)
    peak_price = entry_price
    is_breakeven = False
    
    print(f"   Initial SL: {sl_price:.2f} (+1.5%)")
    print(f"   TP Target: {tp_price:.2f} (-6.0%)")
    
    # Simulate
    subset = df.iloc[start_idx+1:]
    
    days_passed = 0
    last_day = subset.iloc[0]['timestamp'].day
    
    for i, row in subset.iterrows():
        # Update Peak (Lowest Low for SHORT)
        if row['low_eth'] < peak_price:
            peak_price = row['low_eth']
        
        # Check BE Trigger
        current_roe = (entry_price - row['low_eth']) / entry_price * LEVERAGE
        if current_roe >= BE_TRIGGER_ROE and not is_breakeven:
            sl_price = be_price
            is_breakeven = True
            print(f"   ✨ BE Triggered at {row['timestamp']} (Low: {row['low_eth']:.2f})")
            print(f"      New SL: {sl_price:.2f}")
        
        # Check Trailing
        trailing_sl = peak_price * (1 + TRAILING_DEV)
        
        # Check Exits
        hit_sl = row['high_eth'] >= sl_price
        hit_trailing = is_breakeven and row['high_eth'] >= trailing_sl
        hit_tp = row['low_eth'] <= tp_price
        
        # Daily Status Update
        if row['timestamp'].day != last_day:
            days_passed += 1
            last_day = row['timestamp'].day
            if days_passed % 10 == 0: # Every 10 days
                pnl_pct = (entry_price - row['close_eth']) / entry_price * 100
                print(f"   📅 Day {days_passed}: Close {row['close_eth']:.2f} ({pnl_pct:+.2f}%) | Peak {peak_price:.2f} | TrailingSL {trailing_sl:.2f}")
        
        if hit_sl:
            print(f"❌ STOP LOSS HIT at {row['timestamp']} | High: {row['high_eth']:.2f} >= SL {sl_price:.2f}")
            break
            
        if hit_trailing:
            print(f"💰 TRAILING STOP HIT at {row['timestamp']} | High: {row['high_eth']:.2f} >= Trailing {trailing_sl:.2f}")
            break
            
        if hit_tp:
            print(f"🎯 TAKE PROFIT HIT at {row['timestamp']} | Low: {row['low_eth']:.2f} <= TP {tp_price:.2f}")
            break
            
        if row['timestamp'] >= pd.to_datetime(END_TIME):
            print(f"⏳ End of Analysis Period (Trade still open)")
            break

if __name__ == "__main__":
    analyze_trade()
