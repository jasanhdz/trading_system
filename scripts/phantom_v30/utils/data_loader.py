import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager

from config.settings import settings

DB_URL = settings.DATABASE_URL
SYMBOL = settings.SYMBOL
TIMEFRAME = "5m"

def load_hybrid_data():
    """
    Loads data from different market regimes (2021-2025) and combines them.
    Currently just loads all available data from DB.
    """
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, TIMEFRAME)
    
    if df.empty:
        raise ValueError("No data found in database.")
    
    # Feature Engineering (Minimal for RL Input)
    # We rely on the Agent's Transformer to learn patterns from raw OHLCV + Vol + Funding.
    # But usually Log Returns or Z-Scores are better inputs.
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_norm'] = df['volume'] / df['volume'].rolling(24).mean()
    
    # Drop NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    
    if df.empty:
         raise ValueError("Data empty after cleaning.")

    return df

def augment_mirror(df):
    """
    Creates a 'Mirror Universe' where Up is Down.
    Price' = 1 / Price
    Volume = Volume (Same activity)
    """
    mirror_df = df.copy()
    mirror_df['open'] = 1 / df['open']
    mirror_df['high'] = 1 / df['low'] # High becomes Low
    mirror_df['low'] = 1 / df['high'] # Low becomes High
    mirror_df['close'] = 1 / df['close']
    
    # Recalculate Returns for consistency
    mirror_df['log_ret'] = np.log(mirror_df['close'] / mirror_df['close'].shift(1))
    
    return mirror_df

if __name__ == "__main__":
    df = load_hybrid_data()
    print(f"Loaded {len(df)} candles.")
    
    aug_df = augment_mirror(df)
    print(f"Augmented {len(aug_df)} mirror candles.")
