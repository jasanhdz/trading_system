#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/retrain_priority.log"
DATE=$(date)

echo "[$DATE] 🦁 Starting PRIORITY Retraining (Alpha Batch - 12h Cycle)..." >> $LOG_FILE

# 1. Go to project dir
cd $PROJECT_DIR

# 2. Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# 🦁 BATALLÓN ALPHA (Prioridad - 9 Símbolos)
# Frecuencia: Cada 12 Horas
# ═══════════════════════════════════════════════════════════════════════════
PRIORITY_SYMBOLS=(
    "WLD/USDT:USDT"
    "1000PEPE/USDT:USDT"
    "SEI/USDT:USDT"
    "AVAX/USDT:USDT"
    "BTC/USDT:USDT"
    "SOL/USDT:USDT"
    "ETH/USDT:USDT"
    "BNB/USDT:USDT"
    "LTC/USDT:USDT"
)

echo "[$DATE] 🎯 Training ${#PRIORITY_SYMBOLS[@]} ALPHA symbols..." >> $LOG_FILE

for symbol in "${PRIORITY_SYMBOLS[@]}"; do
    echo "[$DATE]   🚀 Training $symbol..." >> $LOG_FILE
    
    # Load balancing: Alternate GPUs
    export HIP_VISIBLE_DEVICES=$((RANDOM % 2))
    $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol" >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$DATE]   ✅ $symbol complete" >> $LOG_FILE
    else
        echo "[$DATE]   ❌ $symbol failed" >> $LOG_FILE
    fi
    
    # Pause to let GPU cool down and GC run
    sleep 5
done

# Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
echo "[$DATE] 🔄 ML Service reloaded." >> $LOG_FILE
echo "[$DATE] 🎉 Alpha Batch Complete!" >> $LOG_FILE
