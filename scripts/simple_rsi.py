
import urllib.request
import json

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
LIMIT = 200

try:
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
    with urllib.request.urlopen(url, timeout=5) as response:
       data = json.loads(response.read().decode())

    closes = [float(x[4]) for x in data]
    current_price = closes[-1]

    # Calculate RSI (Wilder's Smoothing)
    period = 14
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    
    avg_gain = 0
    avg_loss = 0
    
    # First average
    for i in range(period):
        if deltas[i] > 0: avg_gain += deltas[i]
        else: avg_loss += abs(deltas[i])
    
    avg_gain /= period
    avg_loss /= period
    
    # Subsequent (Smoothing)
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta > 0 else 0
        loss = abs(delta) if delta < 0 else 0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    print(f"Price: {current_price}")
    print(f"RSI: {rsi:.2f}")

except Exception as e:
    print(f"Error: {e}")
