
import requests
import pandas as pd
import numpy as np

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
LIMIT = 200

# Scalper Config (Matches Live YAML)
RSI_PERIOD = 14
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25

def get_binance_data():
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
    resp = requests.get(url)
    data = resp.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'trades', 't_base', 't_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['close_time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    
    return df

def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma = df['close'].rolling(window=BB_PERIOD).mean()
    std = df['close'].rolling(window=BB_PERIOD).std()
    df['bb_upper'] = sma + (std * BB_STD)
    df['bb_lower'] = sma - (std * BB_STD)
    
    # ADX (Simplified True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    up = df['high'] - df['high'].shift()
    down = df['low'].shift() - df['low']
    pos_dm = ((up > down) & (up > 0)) * up
    neg_dm = ((down > up) & (down > 0)) * down
    
    # Smoothed
    tr_s = tr.rolling(window=ADX_PERIOD).mean()
    pos_dm_s = pos_dm.rolling(window=ADX_PERIOD).mean()
    neg_dm_s = neg_dm.rolling(window=ADX_PERIOD).mean()
    
    pos_di = 100 * (pos_dm_s / tr_s)
    neg_di = 100 * (neg_dm_s / tr_s)
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    df['adx'] = dx.rolling(window=ADX_PERIOD).mean()
    
    return df

def analyze():
    print(f"Fetching Live Data for {SYMBOL}...")
    df = get_binance_data()
    df = calculate_indicators(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    print(f"\n📊 MARKET SNAPSHOT ({last['timestamp']})")
    print(f"Price: ${last['close']:.2f}")
    
    # RSI
    rsi_status = "NEUTRAL"
    if last['rsi'] < RSI_OVERSOLD: rsi_status = "OVERSOLD (BUY ZONE)"
    elif last['rsi'] > RSI_OVERBOUGHT: rsi_status = "OVERBOUGHT (SELL ZONE)"
    
    print(f"RSI (14): {last['rsi']:.2f} [{rsi_status}]")
    print(f"   Target Buy < {RSI_OVERSOLD} | Target Sell > {RSI_OVERBOUGHT}")
    
    # Bollinger
    bb_status = "INSIDE BANDS"
    if last['close'] < last['bb_lower']: bb_status = "BELOW LOWER BAND (BUY SIGNAL)"
    elif last['close'] > last['bb_upper']: bb_status = "ABOVE UPPER BAND (SELL SIGNAL)"
    
    print(f"Bollinger Bands: Lower ${last['bb_lower']:.2f} | Upper ${last['bb_upper']:.2f}")
    print(f"   Position: {bb_status}")
    
    # ADX
    adx_status = "TRENDING (NO SCALPING)"
    if last['adx'] < ADX_THRESHOLD: adx_status = "RANGING (SCALPING OK)"
    
    print(f"ADX (14): {last['adx']:.2f} [{adx_status}]")
    print(f"   Threshold < {ADX_THRESHOLD}")

    # Conclusion
    print("\n🔮 PREDICTION:")
    
    can_scalp = last['adx'] < ADX_THRESHOLD
    
    if not can_scalp:
        print("❌ ADX too high. Market is trending. Scalper waits.")
    else:
        if last['rsi'] < RSI_OVERSOLD and last['close'] < last['bb_lower']:
            print("🚀 IMMEDIATE LONG ENTRY LIKELY! (RSI Oversold + Below BB)")
        elif last['rsi'] > RSI_OVERBOUGHT and last['close'] > last['bb_upper']:
            print("🔻 IMMEDIATE SHORT ENTRY LIKELY! (RSI Overbought + Above BB)")
        elif last['rsi'] < RSI_OVERSOLD + 5:
            print("⚠️ Approaching LONG Entry (RSI close to oversold)")
        elif last['rsi'] > RSI_OVERBOUGHT - 5:
            print("⚠️ Approaching SHORT Entry (RSI close to overbought)")
        else:
            print("💤 Market is quiet. No immediate entry.")

if __name__ == "__main__":
    analyze()
