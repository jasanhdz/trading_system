import json
import pandas as pd
import os

# Load JSON
json_path = '/home/jasan/Develop/trading_system/data/phantom_backtest_candles.json'
with open(json_path, 'r') as f:
    candles = json.load(f)

# Dump candles for Trade 4 analysis
start_ts = 1737190800000 # 09:00
end_ts = 1737190800000   # 09:00

print(f"Dumping candles from {start_ts} to {end_ts}")
for c in candles:
    if start_ts <= c['timestamp'] <= end_ts:
        print(f"TS: {c['timestamp']} | O: {c['open']} | H: {c['high']} | L: {c['low']} | C: {c['close']}")
