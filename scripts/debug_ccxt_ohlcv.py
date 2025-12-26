import ccxt
import json

def debug_ohlcv():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    symbol = 'BTC/USDT:USDT'
    print(f"Fetching OHLCV for {symbol}...")
    
    # Pedimos la última vela
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1)
    
    if len(ohlcv) > 0:
        candle = ohlcv[0]
        print(f"\nRaw Candle Data ({len(candle)} elements):")
        print(candle)
        
        print("\nInterpretation:")
        print(f"0. Timestamp: {candle[0]}")
        print(f"1. Open: {candle[1]}")
        print(f"2. High: {candle[2]}")
        print(f"3. Low: {candle[3]}")
        print(f"4. Close: {candle[4]}")
        print(f"5. Volume: {candle[5]}")
        
        if len(candle) > 5:
            for i in range(6, len(candle)):
                print(f"{i}. Extra: {candle[i]}")
    else:
        print("No data returned.")

if __name__ == "__main__":
    debug_ohlcv()
