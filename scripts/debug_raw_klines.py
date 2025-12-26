import ccxt
import json

def debug_raw_klines():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    symbol = 'BTCUSDT' # Para endpoints raw usamos el ID de mercado, no el símbolo unificado
    print(f"Fetching RAW Klines for {symbol}...")
    
    # Llamada directa a la API de Binance Futures (fapi)
    # Endpoint: GET /fapi/v1/klines
    try:
        response = exchange.fapiPublicGetKlines({
            'symbol': symbol,
            'interval': '1m',
            'limit': 1
        })
        
        if len(response) > 0:
            candle = response[0]
            print(f"\nRaw API Response ({len(candle)} elements):")
            print(candle)
            
            # Estructura Binance Futures:
            # 0: Open time
            # 1: Open
            # 2: High
            # 3: Low
            # 4: Close
            # 5: Volume
            # 6: Close time
            # 7: Quote asset volume
            # 8: Number of trades
            # 9: Taker buy base asset volume  <-- ESTE QUEREMOS
            # 10: Taker buy quote asset volume
            # 11: Ignore
            
            print(f"\nTarget Data (Index 9): {candle[9]}")
        else:
            print("No data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_raw_klines()
