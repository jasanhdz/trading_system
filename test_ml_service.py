#!/usr/bin/env python3
"""Test the ML probability service directly."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import os
os.environ["ML_MODELS_ROOT"] = str(REPO_ROOT / "models" / "advanced")

from binance_futures_bot_py.src.core.types import Candle
from binance_futures_bot_py.src.infra.config import Config
from binance_futures_bot_py.src.strategies.ml_probability import (
    MLProbabilityStrategy,
    _candles_to_frame,
)
from services.ml_probability_service import ServiceLogger, compute_single_timeframe_probabilities

# Create fake candles for testing
def generate_fake_candles(count: int, symbol: str = "ETHUSDT"):
    """Generate fake candles for testing."""
    candles = []
    base_time = int(datetime.now().timestamp() * 1000)
    
    for i in range(count):
        open_time = base_time - (count - i) * 15 * 60 * 1000  # 15m candles
        close_time = open_time + 15 * 60 * 1000 - 1
        
        # Random walk
        base_price = 2000 + np.random.randn() * 10
        candle = Candle(
            open_time=open_time,
            open=base_price,
            high=base_price + abs(np.random.randn() * 5),
            low=base_price - abs(np.random.randn() * 5),
            close=base_price + np.random.randn() * 3,
            volume=1000000 + np.random.randn() * 100000,
            close_time=close_time,
        )
        candles.append(candle)
    
    return candles

# Test configuration
symbol = "ETHUSDT"
timeframe = "15m"
config = Config()
logger = ServiceLogger()
strategy = MLProbabilityStrategy(history_bars=512)

print(f"Testing ML service for {symbol}/{timeframe}")
print(f"Config ML_MODELS_ROOT: {os.environ.get('ML_MODELS_ROOT')}")
print()

try:
    # Generate candles
    candles = generate_fake_candles(512)
    print(f"✅ Generated {len(candles)} fake candles")
    
    # Test prediction
    print(f"Loading predictor and making prediction...")
    long_prob, short_prob = compute_single_timeframe_probabilities(
        strategy=strategy,
        config=config,
        logger=logger,
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
    )
    
    print(f"✅ Prediction successful!")
    print(f"   Long probability: {long_prob:.4f}")
    print(f"   Short probability: {short_prob:.4f}")
    
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
