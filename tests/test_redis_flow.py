import sys
import time
from pathlib import Path
import pandas as pd
import json

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from data.collectors.binance_collector import BinanceDataCollector
from services.ml_service_v2 import load_latest_data

def test_redis_flow():
    print("🚀 Starting Redis Integration Test...")
    
    # 1. Setup Collector
    collector = BinanceDataCollector()
    symbol = "TEST/USDT:USDT"
    
    # 2. Generate Dummy Data
    dummy_tick = {
        "timestamp": int(time.time() * 1000),
        "mid_price": 100.0,
        "obi_20": 0.5,
        "funding_rate": 0.0001,
        "spread_pct": 0.001,
        "taker_buy_vol": 10.0,
        "taker_sell_vol": 5.0,
        "price": 100.0,
        "obi": 0.5
    }
    
    print(f"📝 Writing tick to Redis for {symbol}...")
    collector.save_tick(symbol, dummy_tick)
    
    # 3. Read from ML Service
    print(f"📖 Reading data from Redis via ML Service...")
    # Note: load_latest_data expects the symbol format used in Redis key
    # collector.save_tick uses "market:{symbol}"
    # ml_service uses "market:{symbol}"
    
    # Give Redis a moment (though it should be instant)
    time.sleep(0.1)
    
    df = load_latest_data(symbol, limit=10)
    
    # 4. Verify
    if df.empty:
        print("❌ Test Failed: DataFrame is empty!")
        sys.exit(1)
        
    print("✅ Data received:")
    print(df.head())
    
    # Check values
    last_row = df.iloc[-1]
    assert last_row['mid_price'] == 100.0
    assert last_row['obi_20'] == 0.5
    
    print("\n🎉 SUCCESS: Redis Hot Path is working!")

if __name__ == "__main__":
    test_redis_flow()
