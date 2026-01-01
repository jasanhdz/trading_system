#!/bin/bash
# scripts/run_full_grid_search.sh
# Runs Grid Search Optimizer for all symbols in parallel using both AMD GPUs.

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

# 2. Setup Environment
source .venv_rocm62/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

echo "🚀 Starting Full Grid Search (AMD Parallel Mode)"
echo "   Testing 10x AND 15x Leverage for all symbols."
echo "   Total Symbols: ${#SYMBOLS[@]}"
echo "---------------------------------------------------"

# 3. Launch Parallel Workers
count=0
for symbol in "${SYMBOLS[@]}"; do
    # Determine GPU ID (0 or 1) based on index parity
    if (( count % 2 == 0 )); then
        gpu_id=0
    else
        gpu_id=1
    fi
    
    echo "🧩 [GPU $gpu_id] Launching Grid Search for $symbol..."
    
    # Run in background
    HIP_VISIBLE_DEVICES=$gpu_id python scripts/grid_search_optimizer.py --symbol "$symbol" --days 3 > "logs/grid_search_${symbol//\//_}.log" 2>&1 &
    
    ((count++))
    
    # Optional: Small delay to stagger launches
    sleep 2
done

echo "---------------------------------------------------"
echo "⏳ All 21 processes launched in background."
echo "   Monitor progress with: tail -f logs/grid_search_*.log"
echo "   Waiting for completion..."

# 4. Wait for all background jobs to finish
wait

echo "✅ Full Grid Search Completed!"
