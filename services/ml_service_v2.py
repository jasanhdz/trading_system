import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import Dict, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Importaciones del proyecto (Minimal)
from data.storage.database_manager import DatabaseManager # Changed from db_manager instance to class

# Configurar nueva DB de velas
CANDLES_DB_URL = "sqlite:///data/binance_candles.db"
candles_db = DatabaseManager(CANDLES_DB_URL)

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("ml_service_v2.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("ml_service_v2")

app = FastAPI(title="Berzerker Pure Mode AI", version="3.0.0")

# --- DATA STRUCTURES ---
class PredictRequestV2(BaseModel):
    symbol: str
    timeframe: str = "5m" # Default to 5m for Berzerker
    limit: int = 100

class ProbabilityResponseV2(BaseModel):
    symbol: str
    long_prob: float
    short_prob: float
    neutral_prob: float
    consensus_level: float
    meta_verdict: str # "APPROVED" | "VETOED" | "NEUTRAL"
    berzerker_score: float # 0.0 to 1.0 (Wave Intensity)

# --- BERZERKER NET ---
class BerzerkerNet(nn.Module):
    def __init__(self, input_size=8):
        super(BerzerkerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.layer3 = nn.Linear(32, 16)
        
        self.output = nn.Linear(16, 1)
        # self.sigmoid = nn.Sigmoid() # Not needed for forward if returning logits, but useful for inference if not handled outside
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

# --- MODEL MANAGER (PURE BERZERKER) ---
class BerzerkerManager:
    def __init__(self):
        # Force CPU for stability (ROCm gfx1032 issues)
        self.device = torch.device("cpu")
        LOGGER.info(f"Using device: {self.device}")
        
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Carga el Escuadrón Berzerker Completo."""
        model_map = {
            'BTCUSDT':  'models/berzerker_BTC_USDT',
            'DOGEUSDT': 'models/berzerker_DOGE_USDT',
            'SOLUSDT':  'models/berzerker_SOL_USDT',
            'XRPUSDT':  'models/berzerker_v1', # Mantener v1 para XRP como base
            'AVAXUSDT': 'models/berzerker_AVAX_USDT',
            'NEARUSDT': 'models/berzerker_NEAR_USDT',
            'FETUSDT':  'models/berzerker_FET_USDT',
            'SEIUSDT':  'models/berzerker_SEI_USDT',
            '1000PEPEUSDT': 'models/berzerker_1000PEPE_USDT'
        }
        
        # Mapeo de alias para compatibilidad con el bot
        alias_map = {
            'BTC/USDT': 'BTCUSDT',
            'DOGE/USDT': 'DOGEUSDT',
            'SOL/USDT': 'SOLUSDT',
            'XRP/USDT': 'XRPUSDT',
            'AVAX/USDT': 'AVAXUSDT',
            'NEAR/USDT': 'NEARUSDT',
            'FET/USDT': 'FETUSDT',
            'SEI/USDT': 'SEIUSDT',
            '1000PEPE/USDT': '1000PEPEUSDT'
        }

        for symbol_key, path_str in model_map.items():
            try:
                path = Path(path_str)
                if not path.exists():
                    LOGGER.warning(f"Model path not found for {symbol_key}: {path}")
                    continue
                    
                # Cargar Scaler
                scaler_path = path / "scaler.pkl"
                if not scaler_path.exists():
                    LOGGER.warning(f"Scaler not found for {symbol_key}")
                    continue
                    
                self.scalers[symbol_key] = joblib.load(scaler_path)
                
                # Cargar Modelo
                model = BerzerkerNet(input_size=8).to(self.device)
                
                # Safe load with weights_only=True if possible, or suppress warning
                # For now using standard load as we trust our own models
                state_dict = torch.load(path / "berzerker_net.pth", map_location=self.device)
                model.load_state_dict(state_dict)
                
                model.eval()
                self.models[symbol_key] = model
                LOGGER.info(f"Loaded Berzerker model for {symbol_key}")
                
                # Registrar alias
                for alias, target in alias_map.items():
                    if target == symbol_key:
                        self.models[alias] = model
                        self.scalers[alias] = self.scalers[symbol_key]
                        LOGGER.info(f"Loaded Berzerker model for {alias}")
                        
            except Exception as e:
                LOGGER.error(f"Failed to load model for {symbol_key}: {e}")

    def calculate_score(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calculate Berzerker Score using:
        1. Massive Body Filter (> 0.45%)
        2. Volume Factor (> 2.0)
        3. Neural Network Inference
        """
        try:
            if len(df) < 20:
                return 0.0
                
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # 1. Massive Body Filter
            body_size = abs(last_candle['close'] - last_candle['open'])
            body_pct = body_size / last_candle['open']
            
            if body_pct < 0.0045: # 0.45%
                return 0.0 # VETO
                
            # 2. Volume Factor
            vol_ma_20 = df['volume'].tail(20).mean()
            vol_factor = last_candle['volume'] / (vol_ma_20 + 1e-8)
            
            # 3. Structure (2 Green Candles)
            is_green = last_candle['close'] > last_candle['open']
            prev_green = prev_candle['close'] > prev_candle['open']
            
            if is_green and prev_green:
                # AI Inference
                if symbol in self.models:
                    try:
                        # Feature Extraction
                        # [vol_factor, body_pct, upper_wick, lower_wick, rsi, prev_vol, prev_body, mom_5m]
                        
                        # RSI
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / (loss + 1e-8)
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        
                        # Momentum 5m (assuming df is 5m)
                        mom_5m = (last_candle['close'] - df.iloc[-6]['close']) / df.iloc[-6]['close'] if len(df) > 6 else 0
                        
                        upper_wick = (last_candle['high'] - max(last_candle['open'], last_candle['close'])) / last_candle['open']
                        lower_wick = (min(last_candle['open'], last_candle['close']) - last_candle['low']) / last_candle['open']
                        
                        prev_vol_factor = prev_candle['volume'] / (vol_ma_20 + 1e-8)
                        prev_body_pct = abs(prev_candle['close'] - prev_candle['open']) / prev_candle['open']
                        
                        features = np.array([[
                            vol_factor, body_pct, upper_wick, lower_wick, 
                            current_rsi / 100.0, prev_vol_factor, prev_body_pct, mom_5m
                        ]])
                        
                        # Scale & Predict
                        scaler = self.scalers[symbol]
                        feat_scaled = scaler.transform(features)
                        feat_tensor = torch.FloatTensor(feat_scaled).to(self.device)
                        
                        with torch.no_grad():
                            logits = self.models[symbol](feat_tensor)
                            prob = torch.sigmoid(logits).item()
                            
                        return prob if prob > 0.5 else 0.0
                        
                    except Exception as e:
                        LOGGER.error(f"Inference error for {symbol}: {e}")
                        
                # Fallback Heuristic
                return 0.85 + min((vol_factor - 2.0) * 0.05, 0.14)
                
            return 0.0
            
        except Exception as e:
            LOGGER.error(f"Score calculation error: {e}")
            return 0.0

# --- GLOBAL INSTANCE ---
manager = BerzerkerManager()

@app.post("/ml-v2/predict", response_model=ProbabilityResponseV2)
async def predict_v2(request: PredictRequestV2):
    """
    Pure Berzerker Endpoint.
    Returns DUMMY neutral probabilities.
    Only 'berzerker_score' is real.
    """
    try:
        # 1. Fetch Data (Force 5m if possible, or resample)
        symbol = request.symbol
        # Normalize symbol
        if "USDT" in symbol and "/" not in symbol:
            db_symbol = symbol.replace("USDT", "/USDT")
        else:
            db_symbol = symbol
            
        # Fetch 5m data
        df = candles_db.get_ohlcv_data(db_symbol, "5m", limit=100)
        
        # 1m Fallback removed as we now enforce 5m data in binance_candles.db
        
        score = 0.0
        if not df.empty:
            score = manager.calculate_score(df, symbol) # Pass raw symbol (e.g. XRPUSDT)
            
        return ProbabilityResponseV2(
            symbol=symbol,
            long_prob=0.0,
            short_prob=0.0,
            neutral_prob=1.0, # DUMMY NEUTRAL
            consensus_level=0.0,
            meta_verdict="NEUTRAL", # Bot will ignore this and look at berzerker_score
            berzerker_score=score
        )
        
    except Exception as e:
        LOGGER.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "BERZERKER_PURE"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
