import ccxt
import os
from dotenv import load_dotenv

load_dotenv('binance-futures-bot-ts/.env')

exchange = ccxt.binanceusdm({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
})

trades = exchange.fetch_my_trades('ETH/USDT:USDT', limit=20)
print("--- LAST 20 TRADES ---")
for t in reversed(trades):
    date = t['datetime']
    side = t['side'].upper()
    amount = t['amount']
    price = t['price']
    pnl = t['info'].get('realizedPnl', '0')
    print(f"[{date}] {side} {amount} ETH @ ${price} | PnL: {pnl}")
