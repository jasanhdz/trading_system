#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/daily_retrain.log"
DATE=$(date)

echo "[$DATE] 🚀 Starting Daily Retraining (Priority Mode)..." >> $LOG_FILE

# 1. Go to project dir
cd $PROJECT_DIR

# 2. Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY TRAINING: Production symbols first (9 active), then others (12)
# ═══════════════════════════════════════════════════════════════════════════

# PHASE 1: Production Symbols (HIGH PRIORITY)
# These are the 9 symbols actively trading - train them FIRST
# Source: binance-futures-bot-ts/.env SYMBOLS variable
# Format: Must match DB format (ADA/USDT:USDT, not ADAUSDT)
PRIORITY_SYMBOLS=(
    "DOGE/USDT:USDT"
    "LINK/USDT:USDT"
    "AVAX/USDT:USDT"
    "POL/USDT:USDT"
    "ETH/USDT:USDT"
    "XRP/USDT:USDT"
    "SOL/USDT:USDT"
    "ADA/USDT:USDT"
    "BTC/USDT:USDT"
)

echo "[$DATE] 🎯 PHASE 1: Training PRIORITY symbols (${#PRIORITY_SYMBOLS[@]} production symbols)..." >> $LOG_FILE

for symbol in "${PRIORITY_SYMBOLS[@]}"; do
    echo "[$DATE]   Training $symbol (Priority)..." >> $LOG_FILE
    
    # Alternate between GPU 0 and GPU 1 for load balancing
    export HIP_VISIBLE_DEVICES=$((RANDOM % 2))
    $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol" >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$DATE]   ✅ $symbol complete" >> $LOG_FILE
    else
        echo "[$DATE]   ❌ $symbol failed" >> $LOG_FILE
    fi
done

echo "[$DATE] ✅ PHASE 1 Complete: Priority symbols trained." >> $LOG_FILE

# PHASE 2: Secondary Symbols (LOWER PRIORITY)
# These are not actively trading but we keep models fresh
# Format: Must match DB format (ADA/USDT:USDT, not ADAUSDT)
SECONDARY_SYMBOLS=(
    "BNB/USDT:USDT"
    "DOT/USDT:USDT"
    "LTC/USDT:USDT"
    "UNI/USDT:USDT"
    "ATOM/USDT:USDT"
    "NEAR/USDT:USDT"
    "1000PEPE/USDT:USDT"
    "FET/USDT:USDT"
    "SEI/USDT:USDT"
    "WLD/USDT:USDT"
    "INJ/USDT:USDT"
    "APT/USDT:USDT"
)

echo "[$DATE] 📦 PHASE 2: Training SECONDARY symbols (${#SECONDARY_SYMBOLS[@]} backup symbols)..." >> $LOG_FILE

# ═══════════════════════════════════════════════════════════════════════════
# FIX: Train 2 symbols at a time (1 per GPU), wait for both before next pair
# This prevents GPU memory exhaustion on RX 6600
# ═══════════════════════════════════════════════════════════════════════════
TOTAL=${#SECONDARY_SYMBOLS[@]}
for ((i=0; i<TOTAL; i+=2)); do
    # Symbol for GPU 0 (even index)
    symbol_0="${SECONDARY_SYMBOLS[$i]}"
    
    # Symbol for GPU 1 (odd index, if exists)
    symbol_1=""
    if [ $((i+1)) -lt $TOTAL ]; then
        symbol_1="${SECONDARY_SYMBOLS[$((i+1))]}"
    fi
    
    echo "[$DATE]   Training pair: $symbol_0 (GPU 0) + $symbol_1 (GPU 1)..." >> $LOG_FILE
    
    # Launch GPU 0 job
    HIP_VISIBLE_DEVICES=0 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_0" >> $LOG_FILE 2>&1 &
    PID_0=$!
    
    # Launch GPU 1 job (if symbol exists)
    if [ -n "$symbol_1" ]; then
        HIP_VISIBLE_DEVICES=1 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_1" >> $LOG_FILE 2>&1 &
        PID_1=$!
        wait $PID_0 $PID_1
    else
        wait $PID_0
    fi
    
    echo "[$DATE]   ✅ Pair complete: $symbol_0 + $symbol_1" >> $LOG_FILE
done

echo "[$DATE] ✅ PHASE 2 Complete: Secondary symbols trained." >> $LOG_FILE

# 3. Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
echo "[$DATE] 🔄 ML Service reloaded with new models." >> $LOG_FILE

echo "[$DATE] 🎉 Daily Retraining Complete!" >> $LOG_FILE
