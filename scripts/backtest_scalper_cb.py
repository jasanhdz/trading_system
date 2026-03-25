
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager

# --- CONFIGURATION ---
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
INITIAL_BALANCE = 100.0
LEVERAGE = 5.0 
COST_PER_TRADE = 0.0006 * 2 # 0.06% taker fee x 2 (entry/exit)

# Scalper Params (Aggressive Tuning from User Request)
SCALPER_ENABLED = True
RSI_PERIOD = 14
RSI_OVERSOLD = 40  # More aggressive than 30
RSI_OVERBOUGHT = 60 # More aggressive than 70
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Risk Params
SL_PCT = 0.015  # 1.5% price movement = 7.5% equity loss at 5x
TP_PCT = 0.015  # 1:1 Risk/Reward
CIRCUIT_BREAKER_HOURS = 24 # If loss, stop scalping for 24h as requested

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
    
    # ADX (Simplified)
    # True Range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up = df['high'] - df['high'].shift()
    down = df['low'].shift() - df['low']
    pos_dm = ((up > down) & (up > 0)) * up
    neg_dm = ((down > up) & (down > 0)) * down
    
    # Smoothed
    tr_s = tr.rolling(window=ADX_PERIOD).mean() # Using SMA for simplicity
    pos_dm_s = pos_dm.rolling(window=ADX_PERIOD).mean()
    neg_dm_s = neg_dm.rolling(window=ADX_PERIOD).mean()
    
    pos_di = 100 * (pos_dm_s / tr_s)
    neg_di = 100 * (neg_dm_s / tr_s)
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    df['adx'] = dx.rolling(window=ADX_PERIOD).mean()
    
    return df

def run_backtest(df, cb_hours):
    balance = INITIAL_BALANCE
    trades = []
    cooldown_until = None
    position = None 
    
    print(f"\n⚡ Running with Circuit Breaker: {cb_hours}h")
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        ts = row['timestamp']
        
        # 1. Manage Position
        if position:
            pnl_pct = 0
            exit_reason = None
            
            if position['type'] == 'LONG':
                if row['low'] <= position['sl']:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                elif row['high'] >= position['tp']:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
                
            elif position['type'] == 'SHORT':
                if row['high'] >= position['sl']:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                elif row['low'] <= position['tp']:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
            
            if exit_reason:
                pnl_dollars = balance * pnl_pct * LEVERAGE
                fee = balance * LEVERAGE * COST_PER_TRADE
                net_pnl = pnl_dollars - fee
                balance += net_pnl
                trades.append({'time': ts, 'pnl': net_pnl, 'reason': exit_reason, 'balance': balance})
                position = None
                
                # Check bankruptcy
                if balance <= 0: break

                # CB
                if net_pnl < 0 and cb_hours > 0:
                    cooldown_until = ts + pd.Timedelta(hours=cb_hours)
            
            continue

        # 2. Cooldown
        if cooldown_until:
             if ts < cooldown_until: continue
             else: cooldown_until = None
            
        # 3. Entry Logic
        if row['adx'] > ADX_THRESHOLD: continue
        
        if row['rsi'] < RSI_OVERSOLD and row['close'] < row['bb_lower']:
            position = {
                'type': 'LONG',
                'entry_price': row['close'],
                'sl': row['close'] * (1 - SL_PCT),
                'tp': row['close'] * (1 + TP_PCT),
                'time': ts
            }
        elif row['rsi'] > RSI_OVERBOUGHT and row['close'] > row['bb_upper']:
            position = {
                'type': 'SHORT',
                'entry_price': row['close'],
                'sl': row['close'] * (1 + SL_PCT),
                'tp': row['close'] * (1 - TP_PCT),
                'time': ts
            }

    return trades, balance

def main():
    try:
        db = DatabaseManager(DB_URL)
        print("Fetching data from DB...")
        df = db.get_ohlcv_data(SYMBOL, '5m', limit=50000) 
        if 'timestamp' not in df.columns: df = df.reset_index()
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("Calculating indicators...")
        df = calculate_indicators(df)
        df.dropna(inplace=True)
        
        # Run 1: No CB
        t1, b1 = run_backtest(df, cb_hours=0)
        print(f"Outcome (0h CB - Always Trade): ${b1:.2f} | Trades: {len(t1)}")
        
        # Run 2: 24h CB
        t2, b2 = run_backtest(df, cb_hours=24)
        print(f"Outcome (24h CB - Safe Exit): ${b2:.2f} | Trades: {len(t2)}")
        
        # Run 3: 48h CB
        t3, b3 = run_backtest(df, cb_hours=48)
        print(f"Outcome (48h CB - Very Safe): ${b3:.2f} | Trades: {len(t3)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
