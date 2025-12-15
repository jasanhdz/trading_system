"""FastAPI service that returns ML model probabilities for supplied candles."""
from __future__ import annotations

import logging
import sys
import os
import talib # Force load talib early
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import features first to avoid torch/talib deadlock
from ml.nn_pattern import features
from ml.advanced_models.predictor import AdvancedPredictor
from binance_futures_bot_py.src.core.types import Candle

# --- Logger Setup ---
class ServiceLogger:
    def __init__(self, name: str = "ml_probability_service") -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        self._logger = logging.getLogger(name)

    def info(self, msg: str, **kwargs): self._logger.info(f"{msg} {kwargs if kwargs else ''}")
    def error(self, msg: str, **kwargs): self._logger.error(f"{msg} {kwargs if kwargs else ''}")
    def debug(self, msg: str, **kwargs): self._logger.debug(f"{msg} {kwargs if kwargs else ''}")
    def warning(self, msg: str, **kwargs): self._logger.warning(f"{msg} {kwargs if kwargs else ''}")

LOGGER = ServiceLogger()

# --- Data Models ---
class CandlePayload(BaseModel):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    def to_candle(self) -> Candle:
        return Candle(
            open_time=int(self.open_time),
            open=float(self.open),
            high=float(self.high),
            low=float(self.low),
            close=float(self.close),
            volume=float(self.volume),
            close_time=int(self.close_time),
        )

class ProbabilityRequest(BaseModel):
    symbol: str
    candles: List[CandlePayload]
    timeframe: Optional[str] = None
    extra_candles: Dict[str, List[CandlePayload]] = {}

class TimeframeProbability(BaseModel):
    long_prob: float
    short_prob: float

class ProbabilityResponse(BaseModel):
    symbol: str
    primary_timeframe: str
    long_prob: float
    short_prob: float
    probabilities: Dict[str, TimeframeProbability]

# --- Model Registry ---
class ModelRegistry:
    def __init__(self):
        self._predictors: Dict[str, AdvancedPredictor] = {}
        self._models_root = REPO_ROOT / "models" / "advanced"

    def get_predictor(self, symbol: str, timeframe: str) -> AdvancedPredictor:
        # Normalize symbol: "BTC/USDT:USDT" -> "BTCUSDT"
        clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").upper()
        # Handle cases where symbol might have suffix
        if "_" in clean_symbol:
            clean_symbol = clean_symbol.split("_")[0]
            
        key = f"{clean_symbol}_{timeframe}"
        
        if key not in self._predictors:
            self._load_predictor(clean_symbol, timeframe, key)
            
        return self._predictors[key]

    def _load_predictor(self, symbol: str, timeframe: str, key: str):
        model_dir = self._models_root / symbol / timeframe
        
        if not model_dir.exists():
            # Try finding folder that starts with symbol (e.g. BTCUSDT_...)
            candidates = list(self._models_root.glob(f"{symbol}*"))
            if candidates:
                # Check inside candidate for timeframe
                candidate_tf = candidates[0] / timeframe
                if candidate_tf.exists():
                    model_dir = candidate_tf
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model artifact not found for {symbol} {timeframe} at {model_dir}")

        LOGGER.info(f"Loading model for {key} from {model_dir}")
        
        try:
            # Force CPU to avoid HIP error: invalid device function
            DEVICE = "cpu"
            # DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # AdvancedPredictor handles fallback
            predictor = AdvancedPredictor(
                model_path=model_dir,
                scaler_path=model_dir / "scaler.pkl",
                meta_path=model_dir / "meta.json",
                device=DEVICE
            )
            self._predictors[key] = predictor
            LOGGER.info(f"Successfully loaded model {key}")
        except Exception as e:
            LOGGER.error(f"Failed to load model {key}: {e}")
            raise e

REGISTRY = ModelRegistry()

# --- Logic ---

def normalize_symbol(raw: str) -> str:
    return raw.replace("/", "").replace(":", "").replace("-", "").upper()

def payloads_to_sorted_candles(payloads: Iterable[CandlePayload]) -> List[Candle]:
    return sorted((payload.to_candle() for payload in payloads), key=lambda c: c.close_time)

def compute_probabilities(symbol: str, timeframe: str, candles: List[Candle]) -> Tuple[float, float]:
    try:
        predictor = REGISTRY.get_predictor(symbol, timeframe)
    except FileNotFoundError:
        # If model not found, return neutral (0.0, 0.0) or raise error?
        # Raising error allows bot to handle "model missing" logic
        raise HTTPException(status_code=404, detail=f"Model not found for {symbol} {timeframe}")
    except Exception as e:
        LOGGER.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Convert candles to raw OHLCV DataFrame (predictor will compute features internally)
    try:
        import pandas as pd
        
        # Create OHLCV DataFrame (predictor expects this format)
        data = {
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
            'open_time': [c.open_time for c in candles],
        }
        frame = pd.DataFrame(data)
        
        # Set datetime index (required by predictor's feature engineering)
        frame['open_time'] = pd.to_datetime(frame['open_time'], unit='ms', utc=True)
        frame.set_index('open_time', inplace=True)
        frame.sort_index(inplace=True)
        
        # Predictor will handle feature engineering internally
        probs = predictor.predict(frame)
        return float(probs["long"]), float(probs["short"])
    except Exception as e:
        LOGGER.error(f"Prediction failed for {symbol} {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- API ---

router = APIRouter(prefix="/ml", tags=["ml"])

@router.post("/probabilities", response_model=ProbabilityResponse)
async def probability_endpoint(request: ProbabilityRequest) -> ProbabilityResponse:
    symbol = normalize_symbol(request.symbol)
    primary_tf = request.timeframe or "1h" # Default to 1h if not provided
    
    # Organize candles
    candles_map = {}
    candles_map[primary_tf] = payloads_to_sorted_candles(request.candles)
    
    for tf, payloads in request.extra_candles.items():
        candles_map[tf] = payloads_to_sorted_candles(payloads)
        
    results = {}
    
    # Compute for all provided timeframes
    for tf, candles in candles_map.items():
        try:
            long_p, short_p = compute_probabilities(symbol, tf, candles)
            results[tf] = TimeframeProbability(long_prob=long_p, short_prob=short_p)
        except HTTPException as e:
            if e.status_code == 404:
                LOGGER.warning(f"Skipping {tf} for {symbol}: Model not found")
                continue
            raise e
            
    if primary_tf not in results:
        # If primary failed, try to use another available one as primary?
        # Or fail? Let's fail for now.
        if not results:
             raise HTTPException(status_code=404, detail=f"No models found for {symbol}")
        
        # Fallback: use first available result
        primary_tf = list(results.keys())[0]
        
    primary_res = results[primary_tf]
    
    return ProbabilityResponse(
        symbol=symbol,
        primary_timeframe=primary_tf,
        long_prob=primary_res.long_prob,
        short_prob=primary_res.short_prob,
        probabilities=results
    )

app = FastAPI(title="ML Probability Service", version="2.0.0")
app.include_router(router)

@app.get("/health")
async def healthcheck():
    return {"status": "ok", "models_loaded": list(REGISTRY._predictors.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.ml_probability_service:app", host="0.0.0.0", port=8000, reload=False)
