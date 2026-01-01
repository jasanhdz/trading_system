#!/bin/bash
# scripts/run_grid_search_sequential.sh
# Runs Grid Search Optimizer for all symbols SEQUENTIALLY to prevent system freeze.
# Uses AMD GPU 0 for acceleration but processes one symbol at a time.

# 1. Define Symbols (21 Total)
SYMBOLS=(
    "ADA/USDT:USDT"
    "AVAX/USDT:USDT"
    "BTC/USDT:USDT"
    "ETH/USDT:USDT"
    "LINK/USDT:USDT"
    "SOL/USDT:USDT"
    "XRP/USDT:USDT"
    "ATOM/USDT:USDT"
    "BNB/USDT:USDT"
    "DOGE/USDT:USDT"
    "DOT/USDT:USDT"
    "LTC/USDT:USDT"
    "NEAR/USDT:USDT"
    "UNI/USDT:USDT"
    "POL/USDT:USDT"
    "APT/USDT:USDT"
    "FET/USDT:USDT"
    "INJ/USDT:USDT"
    "SEI/USDT:USDT"
    "WLD/USDT:USDT"
    "1000PEPE/USDT:USDT"
)

# 2. Setup Environment (AMD ROCm)
source .venv_rocm62/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

echo "🚀 Starting Sequential Grid Search (Safe Mode)"
echo "   Testing 10x AND 15x Leverage."
echo "   Processing 1 symbol at a time to protect system stability."
echo "---------------------------------------------------"

# 3. Launch Sequential Loop
count=1
total=${#SYMBOLS[@]}

for symbol in "${SYMBOLS[@]}"; do
    echo "🧩 [$count/$total] Processing $symbol on GPU 0..."
    
    # Run in FOREGROUND (wait for completion)
    HIP_VISIBLE_DEVICES=0 python scripts/grid_search_optimizer.py --symbol "$symbol" --days 3 > "logs/grid_search_${symbol//\//_}.log" 2>&1
    
    echo "✅ Completed $symbol."
    echo "---------------------------------------------------"
    ((count++))
    
    # Cooldown to let system breathe
    sleep 5
done

echo "🎉 All Grid Search tasks completed successfully."
