import json

# Load JSON
json_path = '/home/jasan/Develop/trading_system/data/phantom_backtest_candles.json'
with open(json_path, 'r') as f:
    candles = json.load(f)

# Trade 4 Params
entry_price = 3318.59
entry_ts = 1737182400000 # 06:40
trailing_dev = 0.015
be_trigger_roe = 0.10
leverage = 5

print(f"Analyzing Trade 4: Entry {entry_price} at {entry_ts}")

peak_price = entry_price
is_breakeven = False
be_price = entry_price * (1 - 0.003)

for c in candles:
    if c['timestamp'] < entry_ts:
        continue
    
    # Update Peak (Lowest Low for Short)
    if c['low'] < peak_price:
        peak_price = c['low']
        print(f"New Peak: {peak_price} at {c['timestamp']}")
    
    print(f"Candle {c['timestamp']}: L={c['low']}, H={c['high']}, Peak={peak_price}")

    # Check BE Trigger
    current_roe = (entry_price - c['low']) / entry_price * leverage
    if current_roe >= be_trigger_roe and not is_breakeven:
        is_breakeven = True
        print(f"BE Triggered at {c['timestamp']} (ROE: {current_roe*100:.2f}%)")
    
    # Check Trailing Exit
    if is_breakeven:
        trailing_sl = peak_price * (1 + trailing_dev)
        if c['high'] >= trailing_sl:
            print(f"EXIT TRIGGERED at {c['timestamp']}")
            print(f"  High: {c['high']}")
            print(f"  Trailing SL: {trailing_sl:.2f} (Peak: {peak_price})")
            break
