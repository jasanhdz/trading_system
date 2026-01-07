#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 🦊 BATALLÓN BRAVO - Entrenamiento SECUNDARIO (Cada 24 Horas)
# ═══════════════════════════════════════════════════════════════════════════
# Símbolos secundarios que no están en producción activa pero se mantienen frescos.
# Schedule: 08:00 UTC (02:00 AM CDMX) - Una vez al día
# Duración estimada: ~3.5 horas (entrenan en pares para ahorrar tiempo)

PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/train_bravo.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] 🦊 Iniciando Entrenamiento SECUNDARIO (Bravo Batch)..." >> $LOG_FILE

# Go to project dir
cd $PROJECT_DIR

# Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# 🦊 BRAVO SYMBOLS (12 Secondary Symbols)
# Símbolos con datos pero no en producción activa
# ═══════════════════════════════════════════════════════════════════════════
BRAVO_SYMBOLS=(
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

echo "[$DATE] 📦 Training ${#BRAVO_SYMBOLS[@]} BRAVO symbols..." >> $LOG_FILE

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════════
# Train 2 symbols at a time (1 per GPU) to save time
# This prevents GPU memory exhaustion on RX 6600
# ═══════════════════════════════════════════════════════════════════════════
TOTAL=${#BRAVO_SYMBOLS[@]}
for ((i=0; i<TOTAL; i+=2)); do
    # Symbol for GPU 0 (even index)
    symbol_0="${BRAVO_SYMBOLS[$i]}"
    
    # Symbol for GPU 1 (odd index, if exists)
    symbol_1=""
    if [ $((i+1)) -lt $TOTAL ]; then
        symbol_1="${BRAVO_SYMBOLS[$((i+1))]}"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   🚀 Training pair: $symbol_0 (GPU 0) + $symbol_1 (GPU 1)..." >> $LOG_FILE
    
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
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   ✅ Pair complete: $symbol_0 + $symbol_1" >> $LOG_FILE
    
    # Pause between pairs
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))

# Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 ML Service reloaded." >> $LOG_FILE
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🦊 Bravo Batch Complete! Total: ${TOTAL_MINUTES} minutes." >> $LOG_FILE
