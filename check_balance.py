import ccxt
import os
import json
from dotenv import load_dotenv

load_dotenv('binance-futures-bot-ts/.env')
exchange = ccxt.binanceusdm({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
})

try:
    balance = exchange.fetch_balance()
    usdt = balance.get('USDT', {})
    print(f"USDT Total: {usdt.get('total', 0)}")
    print(f"USDT Free: {usdt.get('free', 0)}")
    print(f"USDT Used: {usdt.get('used', 0)}")
    
    # Get ETHUSDT position for extra info
    positions = balance['info'].get('positions', [])
    for pos in positions:
        if pos['symbol'] == 'ETHUSDT' and float(pos['positionAmt']) != 0:
            print(f"Active Position: {pos['positionAmt']} ETH at {pos['entryPrice']}")
except Exception as e:
    print(f"Error: {e}")
