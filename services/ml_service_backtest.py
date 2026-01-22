import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
# import torch  <-- REMOVED TO PREVENT CRASH
# import torch.nn as nn <-- REMOVED
import requests
from typing import Dict, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Config
# MODEL_PATH = "models/phantom_eth/phantom_net_best.pth" <-- Not needed for Mock
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("ml_service_backtest.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("phantom_backtest")

app = FastAPI(title="Phantom V8 Backtest Service", version="8.0.0-BACKTEST")

# --- DATA STRUCTURES ---
class PredictRequest(BaseModel):
    symbol: str
    custom_candles: Optional[List[Dict[str, float]]] = None
    custom_btc_candles: Optional[List[Dict[str, float]]] = None
    precalculated_features: Optional[Dict[str, float]] = None  # NEW: For direct feature injection

class PhantomResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    long_prob: float
    short_prob: float
    neutral_prob: float
    model_version: str = "phantom_v8_mock"
    features: Optional[Dict[str, float]] = None

# --- FEATURE ENGINEERING ---
class PhantomFeatureEngine:
    @staticmethod
    def calculate_cvd_proxy(df: pd.DataFrame) -> pd.DataFrame:
        df['direction'] = np.where(df['close_eth'] > df['open_eth'], 1, -1)
        df['volume_delta'] = df['direction'] * df['volume_eth']
        df['cvd_20'] = df['volume_delta'].rolling(20).sum()
        df['cvd_slope'] = df['cvd_20'].diff(5)
        rolling_mean = df['cvd_20'].rolling(50).mean()
        rolling_std = df['cvd_20'].rolling(50).std()
        df['cvd_z'] = (df['cvd_20'] - rolling_mean) / (rolling_std + 1e-8)
        return df

    @staticmethod
    def calculate_features(df_eth: pd.DataFrame, df_btc: pd.DataFrame) -> np.ndarray:
        df = pd.merge(
            df_eth, 
            df_btc[['timestamp', 'close']], 
            on='timestamp', 
            how='inner', 
            suffixes=('_eth', '_btc')
        )
        df = df.rename(columns={
            'open': 'open_eth', 'high': 'high_eth', 'low': 'low_eth', 'volume': 'volume_eth'
        })
        
        df = PhantomFeatureEngine.calculate_cvd_proxy(df)
        
        df['eth_btc_ratio'] = df['close_eth'] / df['close_btc']
        df['eth_btc_ema'] = df['eth_btc_ratio'].ewm(span=20).mean()
        df['weakness_score'] = (df['eth_btc_ema'] - df['eth_btc_ratio']) / (df['eth_btc_ema'] + 1e-8) * 100
        
        df['returns'] = df['close_eth'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_z'] = (df['volatility'] - df['volatility'].expanding().mean()) / (df['volatility'].expanding().std() + 1e-8)
        
        df['body'] = abs(df['open_eth'] - df['close_eth'])
        df['upper_wick'] = df['high_eth'] - df[['open_eth', 'close_eth']].max(axis=1)
        df['is_fakeout'] = (df['upper_wick'] > (df['body'] * 1.5)).astype(float)
        
        df['vol_sma'] = df['volume_eth'].rolling(20).mean()
        df['vol_ratio'] = df['volume_eth'] / (df['vol_sma'] + 1e-8)
        
        df['is_doji'] = df['body'] < (df['high_eth'] - df['low_eth']) * 0.1
        df['staleness'] = df['is_doji'].rolling(10).sum()
        
        df['velocity'] = df['close_eth'].diff()
        df['acceleration'] = df['velocity'].diff()
        df['velocity_sm'] = df['velocity'].ewm(span=5).mean()
        df['acceleration_sm'] = df['acceleration'].ewm(span=5).mean()
        
        df['ema_20'] = df['close_eth'].ewm(span=20).mean()
        df['ema_200'] = df['close_eth'].ewm(span=200).mean()
        df['dist_ema20'] = (df['close_eth'] - df['ema_20']) / df['close_eth']
        df['dist_ema200'] = (df['close_eth'] - df['ema_200']) / df['close_eth']
        
        row = df.iloc[-1]
        
        state = np.array([
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] if not pd.isna(row['cvd_slope']) else 0, # This is raw cvd_slope
            row['weakness_score'] if not pd.isna(row['weakness_score']) else 0,
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            row['is_fakeout'],
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            row['velocity_sm'] / row['close_eth'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['close_eth'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            0.0
        ], dtype=np.float32)
        
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

# MOCK MODEL (Bypass Torch)
class PhantomManager:
    def __init__(self):
        self.device = "cpu"
        LOGGER.info("✅ Phantom V8 Model MOCKED (Rule-Based)")
        
    def load_model(self):
        # No model to load for rule-based system
        LOGGER.info("✅ Phantom V8 Model MOCKED (Rule-Based) - No model file loaded.")

    def predict(self, symbol: str, custom_candles: Optional[List[Dict[str, float]]] = None, custom_btc_candles: Optional[List[Dict[str, float]]] = None, precalculated_features: Optional[Dict[str, float]] = None) -> PhantomResponse:
        
        # 1. Direct Feature Injection (Fast Path for Backtest Client)
        if precalculated_features:
            LOGGER.info(f"[DEBUG] Using precalculated_features: slope={precalculated_features.get('cvd_slope')}, z={precalculated_features.get('cvd_z')}")
            try:
                raw_cvd_slope = precalculated_features.get('cvd_slope', 0.0)
                cvd_z = precalculated_features.get('cvd_z', 0.0)
                weakness_score = precalculated_features.get('weakness_score', 0.0)
                # We need close/open for bearish check. 
                # If not provided, assume bearish if not specified? 
                # Better to pass is_bearish flag or open/close.
                # Let's assume precalculated_features has 'is_bearish' or we pass open/close.
                
                is_bearish = False
                if 'is_bearish' in precalculated_features:
                    is_bearish = bool(precalculated_features['is_bearish'])
                elif 'close' in precalculated_features and 'open' in precalculated_features:
                    is_bearish = precalculated_features['close'] < precalculated_features['open']
                
                # TRIGGER LOGIC
                if raw_cvd_slope < 0 and cvd_z <= 0.5 and is_bearish and weakness_score >= 0:
                    action = "SHORT"
                    short_prob = 0.95
                else:
                    action = "PASS"
                    short_prob = 0.10
                
                return PhantomResponse(
                    symbol=symbol,
                    action=action,
                    confidence=short_prob,
                    long_prob=0.0,
                    short_prob=short_prob,
                    neutral_prob=1.0 - short_prob,
                    features=precalculated_features
                )
            except Exception as e:
                LOGGER.error(f"Precalculated feature error: {e}")
                return PhantomResponse(symbol=symbol, action="ERROR", confidence=0.0, long_prob=0.0, short_prob=0.0, neutral_prob=1.0)

        # 2. Standard Path (Calculate from Candles)
        if not custom_candles:
            return PhantomResponse(symbol=symbol, action="ERROR", confidence=0.0, long_prob=0.0, short_prob=0.0, neutral_prob=1.0)
            
        df_eth = pd.DataFrame(custom_candles)
        for col in ["open", "high", "low", "close", "volume"]:
            df_eth[col] = df_eth[col].astype(float)
        df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
        
        # Handle BTC Context
        if custom_btc_candles and len(custom_btc_candles) > 0:
            df_btc = pd.DataFrame(custom_btc_candles)
            for col in ["open", "high", "low", "close", "volume"]:
                df_btc[col] = df_btc[col].astype(float)
            df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
        else:
            df_btc = df_eth.copy() 
        
        try:
            # Calculate features (Keep this!)
            state = PhantomFeatureEngine.calculate_features(df_eth, df_btc)
            
            # TRIGGER LOGIC (MATCHING PYTHON backtest_phantom_v8.py)
            # Note: state[1] is RAW cvd_slope from calculate_features
            raw_cvd_slope = float(state[1])
            cvd_z = float(state[0])
            weakness_score = float(state[2])
            is_bearish = df_eth.iloc[-1]['close'] < df_eth.iloc[-1]['open']

            # DEBUG LOGGING
            import random
            if random.random() < 0.01: 
                 ts_str = str(df_eth.iloc[-1]['timestamp'])
                 LOGGER.info(f"DEBUG: TS={ts_str}, Slope={raw_cvd_slope:.2f}, Bearish={is_bearish}")

            # 1. CVD Slope negative (distribution)
            # 2. CVD Z-Score <= 0.5 (selling pressure)
            # 3. Bearish candle
            # 4. ETH weaker than BTC (weakness_score >= 0)
            
            if raw_cvd_slope < 0 and cvd_z <= 0.5 and is_bearish and weakness_score >= 0:
                # Mock Model Approval
                action = "SHORT"
                short_prob = 0.95 # High confidence
            else:
                action = "PASS"
                short_prob = 0.10
            
            return PhantomResponse(
                symbol=symbol,
                action=action,
                confidence=short_prob,
                long_prob=0.0,
                short_prob=short_prob,
                neutral_prob=1.0 - short_prob,
                features={
                    "cvd_z": float(state[0]),
                    "cvd_slope": float(state[1]),
                    "weakness": float(state[2]),
                    "volatility_z": float(state[3])
                }
            )
            
        except Exception as e:
            import traceback
            LOGGER.error(f"Prediction error: {e}\n{traceback.format_exc()}")
            return PhantomResponse(
                symbol=symbol, action="ERROR", confidence=0.0, 
                long_prob=0.0, short_prob=0.0, neutral_prob=1.0
            )

manager = PhantomManager()

@app.post("/ml-v2/backtest_predict", response_model=PhantomResponse)
async def predict(req: PredictRequest):
    return manager.predict(req.symbol, req.custom_candles, req.custom_btc_candles, req.precalculated_features)

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Phantom V8 Backtest"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
