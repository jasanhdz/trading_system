#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 🦁 BATALLÓN ALPHA - Entrenamiento PRIORITARIO (Cada 12 Horas)
# ═══════════════════════════════════════════════════════════════════════════
# Símbolos de alta prioridad que mueven el mercado.
# Schedule: 00:00 y 12:00 UTC (06:00 AM y 06:00 PM CDMX)
# Duración estimada: ~2.5 horas

PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/train_alpha.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] 🦁 Iniciando Entrenamiento PRIORITARIO (Alpha Batch)..." >> $LOG_FILE

# Go to project dir
cd $PROJECT_DIR

# Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# 🦁 ALPHA SYMBOLS (9 Production Symbols)
# Los que operan en binance-futures-bot-ts/.env
# ═══════════════════════════════════════════════════════════════════════════
ALPHA_SYMBOLS=(
    "BTC/USDT:USDT"
    "ETH/USDT:USDT"
    "SOL/USDT:USDT"
    "XRP/USDT:USDT"
    "ADA/USDT:USDT"
    "DOGE/USDT:USDT"
    "LINK/USDT:USDT"
    "AVAX/USDT:USDT"
    "POL/USDT:USDT"
)

echo "[$DATE] 🎯 Training ${#ALPHA_SYMBOLS[@]} ALPHA symbols..." >> $LOG_FILE

START_TIME=$(date +%s)

for symbol in "${ALPHA_SYMBOLS[@]}"; do
    SYMBOL_START=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   🚀 Training $symbol..." >> $LOG_FILE
    
    # Load balancing: Alternate GPUs
    export HIP_VISIBLE_DEVICES=$((RANDOM % 2))
    $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol" >> $LOG_FILE 2>&1
    
    SYMBOL_END=$(date +%s)
    SYMBOL_DURATION=$((SYMBOL_END - SYMBOL_START))
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   ✅ $symbol complete (${SYMBOL_DURATION}s)" >> $LOG_FILE
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   ❌ $symbol failed" >> $LOG_FILE
    fi
    
    # Pause to let GPU breathe
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))

# Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 ML Service reloaded." >> $LOG_FILE
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🦁 Alpha Batch Complete! Total: ${TOTAL_MINUTES} minutes." >> $LOG_FILE
