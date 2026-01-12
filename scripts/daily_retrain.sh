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
# ═══════════════════════════════════════════════════════════════════════════
# 🛡️ BRAVE BATTALION - Entrenamiento Diario (24h)
# ═══════════════════════════════════════════════════════════════════════════
# Símbolos secundarios y de reserva. Se entrenan una vez al día.

# BRAVE SYMBOLS (The Rest)
BRAVE_SYMBOLS=(
    # ═══════════════════════════════════════════════════════════════════════
    # 🛡️ BRAVE BATTALION (The Remaining 12 Symbols)
    # ═══════════════════════════════════════════════════════════════════════
    # --- Beta-1 (Momentum) ---
    "LINK/USDT:USDT"
    "POL/USDT:USDT"
    "ADA/USDT:USDT"
    "XRP/USDT:USDT"
    
    # --- Beta-2 (Meme) ---
    "DOGE/USDT:USDT"
    
    # --- Gamma-1 (Stable) ---
    "DOT/USDT:USDT"
    "UNI/USDT:USDT"
    "ATOM/USDT:USDT"
    
    # --- Gamma-2 (Runners) ---
    "NEAR/USDT:USDT"
    "APT/USDT:USDT"
    "INJ/USDT:USDT"
    
    # --- Delta (High Risk) ---
    "FET/USDT:USDT"
)

echo "[$DATE] 🛡️ Training BRAVE symbols (${#BRAVE_SYMBOLS[@]} symbols)..." >> $LOG_FILE

# ═══════════════════════════════════════════════════════════════════════════
# BUCLE PARALELO (2 GPUs a la vez)
# ═══════════════════════════════════════════════════════════════════════════
TOTAL=${#BRAVE_SYMBOLS[@]}

for ((i=0; i<TOTAL; i+=2)); do
    # Symbol for GPU 0 (even index)
    symbol_0="${BRAVE_SYMBOLS[$i]}"
    
    # Symbol for GPU 1 (odd index, if exists)
    symbol_1=""
    if [ $((i+1)) -lt $TOTAL ]; then
        symbol_1="${BRAVE_SYMBOLS[$((i+1))]}"
    fi
    
    echo "[$DATE]   Training pair: $symbol_0 (GPU 0) + ${symbol_1:-None} (GPU 1)..." >> $LOG_FILE
    
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
    
    echo "[$DATE]   ✅ Pair complete." >> $LOG_FILE
done


# 3. Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
echo "[$DATE] 🔄 ML Service reloaded with new models." >> $LOG_FILE

echo "[$DATE] 🎉 Daily Retraining Complete!" >> $LOG_FILE
