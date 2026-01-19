import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
from typing import Dict, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Config
MODEL_PATH = "models/phantom_eth/phantom_net_best.pth"
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("ml_service_phantom.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("phantom_service")

app = FastAPI(title="Phantom V8 AI Service", version="8.0.0")

# --- DATA STRUCTURES ---
class PredictRequest(BaseModel):
    symbol: str

class PhantomResponse(BaseModel):
    symbol: str
    action: str          # "SHORT" | "PASS"
    confidence: float    # 0.0 to 1.0
    long_prob: float     # 0.0 (Phantom is short-only)
    short_prob: float    # Confidence
    neutral_prob: float  # 1 - Confidence
    model_version: str = "phantom_v8"
    features: Optional[Dict[str, float]] = None

# --- PHANTOM NET ---
class PhantomNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# --- FEATURE ENGINEERING ---
class PhantomFeatureEngine:
    @staticmethod
    def calculate_cvd_proxy(df: pd.DataFrame) -> pd.DataFrame:
        # Direction: 1 if green, -1 if red
        df['direction'] = np.where(df['close'] > df['open'], 1, -1)
        df['volume_delta'] = df['direction'] * df['volume']
        
        # CVD rolling
        df['cvd_20'] = df['volume_delta'].rolling(20).sum()
        df['cvd_slope'] = df['cvd_20'].diff(5)
        
        # Normalized CVD
        rolling_mean = df['cvd_20'].rolling(50).mean()
        rolling_std = df['cvd_20'].rolling(50).std()
        df['cvd_z'] = (df['cvd_20'] - rolling_mean) / (rolling_std + 1e-8)
        
        return df

    @staticmethod
    def calculate_features(df_eth: pd.DataFrame, df_btc: pd.DataFrame) -> np.ndarray:
        """
        Calculate the 12 features required by PhantomNet.
        Returns the state vector for the last candle.
        """
        # Merge BTC data for weakness calculation
        df = pd.merge(
            df_eth, 
            df_btc[['timestamp', 'close']], 
            on='timestamp', 
            how='inner', 
            suffixes=('_eth', '_btc')
        )
        
        # Rename for consistency with generator logic
        df = df.rename(columns={
            'open': 'open_eth', 'high': 'high_eth', 'low': 'low_eth', 'volume': 'volume_eth'
        })
        
        # 1. CVD Features
        df = PhantomFeatureEngine.calculate_cvd_proxy(df)
        
        # 2. Weakness Score
        df['eth_btc_ratio'] = df['close_eth'] / df['close_btc']
        df['eth_btc_ema'] = df['eth_btc_ratio'].ewm(span=20).mean()
        df['weakness_score'] = (df['eth_btc_ema'] - df['eth_btc_ratio']) / (df['eth_btc_ema'] + 1e-8) * 100
        
        # 3. Volatility Z
        df['returns'] = df['close_eth'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_z'] = (df['volatility'] - df['volatility'].expanding().mean()) / (df['volatility'].expanding().std() + 1e-8)
        
        # 4. Fakeout
        df['body'] = abs(df['open_eth'] - df['close_eth'])
        df['upper_wick'] = df['high_eth'] - df[['open_eth', 'close_eth']].max(axis=1)
        df['is_fakeout'] = (df['upper_wick'] > (df['body'] * 1.5)).astype(float)
        
        # 5. Volume Ratio
        df['vol_sma'] = df['volume_eth'].rolling(20).mean()
        df['vol_ratio'] = df['volume_eth'] / (df['vol_sma'] + 1e-8)
        
        # 6. Staleness
        df['is_doji'] = df['body'] < (df['high_eth'] - df['low_eth']) * 0.1
        df['staleness'] = df['is_doji'].rolling(10).sum()
        
        # 7. Momentum
        df['velocity'] = df['close_eth'].diff()
        df['acceleration'] = df['velocity'].diff()
        df['velocity_sm'] = df['velocity'].ewm(span=5).mean()
        df['acceleration_sm'] = df['acceleration'].ewm(span=5).mean()
        
        # 8. Price Position
        df['ema_20'] = df['close_eth'].ewm(span=20).mean()
        df['ema_200'] = df['close_eth'].ewm(span=200).mean()
        df['dist_ema20'] = (df['close_eth'] - df['ema_20']) / df['close_eth']
        df['dist_ema200'] = (df['close_eth'] - df['ema_200']) / df['close_eth']
        
        # Extract last row
        row = df.iloc[-1]
        
        # Construct 12-feature vector
        state = np.array([
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] if not pd.isna(row['cvd_slope']) else 0,
            row['weakness_score'] if not pd.isna(row['weakness_score']) else 0,
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            row['is_fakeout'],
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            row['velocity_sm'] / row['close_eth'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['close_eth'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            0.0 # Reserved
        ], dtype=np.float32)
        
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

# --- MANAGER ---
class PhantomManager:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = PhantomNet(input_dim=12, output_dim=2).to(self.device)
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            LOGGER.error(f"Model not found at {MODEL_PATH}")
            return
            
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            LOGGER.info("✅ Phantom V8 Model Loaded")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")

    def fetch_candles(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch live candles from Binance Futures"""
        try:
            params = {
                "symbol": symbol,
                "interval": "5m",
                "limit": limit
            }
            resp = requests.get(BINANCE_API_URL, params=params, timeout=5)
            data = resp.json()
            
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.astype(float)
            return df
        except Exception as e:
            LOGGER.error(f"Binance API error: {e}")
            return pd.DataFrame()

    def predict(self, symbol: str) -> PhantomResponse:
        # 1. Fetch Data (ETH + BTC for context)
        df_eth = self.fetch_candles(symbol, limit=200) # Need 200 for EMA200
        df_btc = self.fetch_candles("BTCUSDT", limit=200)
        
        if df_eth.empty or df_btc.empty:
            return PhantomResponse(
                symbol=symbol, action="PASS", confidence=0.0, 
                long_prob=0.0, short_prob=0.0, neutral_prob=1.0
            )
            
        # 2. Calculate Features
        try:
            state = PhantomFeatureEngine.calculate_features(df_eth, df_btc)
            
            # 3. Inference
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_t)
                # Softmax for probabilities
                probs = torch.softmax(q_values, dim=1)
                short_prob = probs[0][1].item()
                
            # 4. Decision
            # Enforce Weakness Rule (ETH must be weaker than BTC) - Matches Backtest V8
            weakness_score = float(state[2])
            
            if weakness_score > 0 and short_prob > 0.50:
                action = "SHORT"
            else:
                action = "PASS"
            
            return PhantomResponse(
                symbol=symbol,
                action=action,
                confidence=short_prob,
                long_prob=0.0,
                short_prob=short_prob,
                neutral_prob=1.0 - short_prob,
                features={
                    "cvd_z": float(state[0]),
                    "weakness": float(state[2]),
                    "volatility_z": float(state[3])
                }
            )
            
        except Exception as e:
            LOGGER.error(f"Prediction error: {e}")
            return PhantomResponse(
                symbol=symbol, action="ERROR", confidence=0.0, 
                long_prob=0.0, short_prob=0.0, neutral_prob=1.0
            )

# --- GLOBAL INSTANCE ---
manager = PhantomManager()

@app.post("/ml-v2/predict", response_model=PhantomResponse)
async def predict(req: PredictRequest):
    return manager.predict(req.symbol)

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Phantom V8"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
