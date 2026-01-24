#!/usr/bin/env python3
"""
Phantom V9: ETH Setup Detector
Uses Phantom DNA Features to filter noise before DQN.
Target: High Momentum Decay / Staleness
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix path to include project root (scripts/phantom_v9 -> scripts -> root)
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.storage.database_manager import DatabaseManager

# Config
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m" # Usaremos 5m como Wraith, más estable que 1m para entrenar
DB_URL = "sqlite:///data/binance_candles.db" 

def calculate_phantom_dna(df):
    """Calculates the 12 Phantom Features"""
    df = df.copy()
    
    # --- Features Base (Cinética) ---
    df['velocity'] = df['close'].diff(periods=5)
    df['acceleration'] = df['velocity'].diff(periods=5)

    # --- CVD Proxy ---
    if 'cvd' not in df.columns:
        price_diff = df['close'].diff()
        direction = np.sign(price_diff).replace(0, np.nan).ffill().fillna(0)
        df['cvd'] = (direction * df['volume']).cumsum()

    df['cvd_slope'] = df['cvd'].diff(periods=10)
    df['price_slope'] = df['close'].diff(periods=10)
    df['bear_trap'] = ((df['price_slope'] > 0) & (df['cvd_slope'] < 0)).astype(float)

    # --- Volatilidad ---
    rolling_std_20 = df['close'].rolling(20).std()
    rolling_std_200 = df['close'].rolling(200).std()
    df['vol_z'] = (rolling_std_20 - rolling_std_200) / (rolling_std_200 + 1e-8)
    
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_ma'] + 1e-8)

    # --- EMA & Staleness ---
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean() # Usamos EMA 50 para ETH (rápido)
    
    df['dist_ema_20'] = (df['close'] - ema_20) / df['close']
    df['dist_ema_200'] = (df['close'] - ema_50) / df['close'] # Hack: usamos slot 200 para 50

    # Staleness Logic
    df['is_high'] = df['close'] == df['close'].rolling(20).max()
    s = ~df['is_high']
    df['staleness'] = s.groupby((s != s.shift()).cumsum()).cumsum()

    # --- Weakness & Fakeout ---
    df['returns'] = df['close'].pct_change()
    df['weakness_score'] = (df['returns'].rolling(20).sum() / (df['vol_z'] + 1e-8)).clip(-5, 5)
    
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['is_fakeout'] = ((df['high'] > df['open'] * 1.005) & (df['range'] > df['body'] * 2)).astype(float)

    df['reserved'] = 0.0
    
    return df

def detect_eth_setups(df):
    """
    Filtro para ETH: Busca agotamiento y rechazo técnico.
    ETH es volátil, necesitamos que esté "cansado" (Staleness) para atacar.
    """
    candidates = []
    
    # Necesitamos al menos 200 velas para EMAs
    for i in range(200, len(df)):
        row = df.iloc[i]
        
        # 1. Proximidad a Resistencia (EMA 20/50)
        near_resistance = abs(row['dist_ema_20']) < 0.005 
        
        # 2. Agotamiento (Staleness)
        is_tired = row['staleness'] > 15
        
        # 3. Volatilidad (Entropía)
        # Relaxed threshold: > 0.2 (1.2x normal volatility)
        is_volatile = abs(row['vol_z']) > 0.2
        
        # 4. Acción de Precio (Trigger)
        is_rejection = (row['close'] < row['open']) or (row['is_fakeout'] == 1)
        
        if near_resistance and is_tired and is_volatile and is_rejection:
            candidates.append(i)
            
    return df.iloc[candidates]

def main():
    print("Scanning ETH for Phantom V9 Setups...")
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, TIMEFRAME, limit=10000) # Buscamos más data
    
    if df.empty:
        print("No data found.")
        return

    if 'timestamp' not in df.columns:
        df = df.reset_index()

    df = calculate_phantom_dna(df)
    candidates = detect_eth_setups(df)
    
    print(f"Found {len(candidates)} Phantom V9 Candidates.")
    return df, candidates

if __name__ == "__main__":
    main()
