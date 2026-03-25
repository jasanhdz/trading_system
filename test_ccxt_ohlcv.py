import ccxt
exchange = ccxt.binanceusdm()
exchange.load_markets()
bars = exchange.fapiPublicGetKlines({'symbol': 'ETHUSDT', 'interval': '5m', 'limit': 1, 'startTime': 1700000000000})
print(bars[0])
print("Length:", len(bars[0]))
