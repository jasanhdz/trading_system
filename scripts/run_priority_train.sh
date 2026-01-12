#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 🦁 BATALLÓN ALPHA - Entrenamiento PRIORITARIO (PARALELO)
# ═══════════════════════════════════════════════════════════════════════════
# Símbolos de alta prioridad que mueven el mercado.
# Optimización: Uso simultáneo de GPU 0 y GPU 1.
# Duración estimada: ~1.2 horas (vs 2.5h antes)

PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/train_alpha.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] 🦁 Iniciando Entrenamiento PRIORITARIO (Alpha Batch - Paralelo)..." >> $LOG_FILE

# Go to project dir
cd $PROJECT_DIR

# Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# 🦁 ALPHA SYMBOLS (9 Símbolos)
# ═══════════════════════════════════════════════════════════════════════════
ALPHA_SYMBOLS=(
    # ═══════════════════════════════════════════════════════════════════════
    # 🗡️ OPERACIÓN PUNTA DE LANZA (The Spearhead Squad - 9 Symbols)
    # ═══════════════════════════════════════════════════════════════════════
    "WLD/USDT:USDT"       # Monster (Acc 89.8%)
    "1000PEPE/USDT:USDT"  # Monster (Acc 88.6%)
    "SEI/USDT:USDT"       # Monster (Acc 87.0%)
    "AVAX/USDT:USDT"      # Leader (Acc 86.7%)
    "BTC/USDT:USDT"       # Leader (The King)
    "ETH/USDT:USDT"       # Leader (The Queen)
    "SOL/USDT:USDT"       # Leader (Volume)
    "BNB/USDT:USDT"       # Sniper (Win Rate 83%)
    "LTC/USDT:USDT"       # Sniper (Win Rate 77%)
)

echo "[$DATE] 🎯 Training ${#ALPHA_SYMBOLS[@]} ALPHA symbols in Parallel..." >> $LOG_FILE

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════════
# BUCLE PARALELO (2 GPUs a la vez)
# ═══════════════════════════════════════════════════════════════════════════
TOTAL=${#ALPHA_SYMBOLS[@]}

for ((i=0; i<TOTAL; i+=2)); do
    # Symbol for GPU 0 (índice par)
    symbol_0="${ALPHA_SYMBOLS[$i]}"
    
    # Symbol for GPU 1 (índice impar, si existe)
    symbol_1=""
    if [ $((i+1)) -lt $TOTAL ]; then
        symbol_1="${ALPHA_SYMBOLS[$((i+1))]}"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]    🚀 Launching pair: $symbol_0 (GPU 0) + ${symbol_1:-None} (GPU 1)..." >> $LOG_FILE
    
    # --- Lanzar Trabajo GPU 0 en Background (&) ---
    HIP_VISIBLE_DEVICES=0 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_0" >> $LOG_FILE 2>&1 &
    PID_0=$!
    
    # --- Lanzar Trabajo GPU 1 en Background (&) ---
    if [ -n "$symbol_1" ]; then
        HIP_VISIBLE_DEVICES=1 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_1" >> $LOG_FILE 2>&1 &
        PID_1=$!
        
        # Esperar a que AMBOS terminen antes de seguir
        wait $PID_0 $PID_1
    else
        # Si es el último y está solo (impar), esperar solo a él
        wait $PID_0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]    ✅ Pair complete." >> $LOG_FILE
    
    # Pausa técnica para enfriar VRAM y Garbage Collector
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))

# Reload ML Service
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 ML Service reloaded." >> $LOG_FILE
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🦁 Alpha Batch Complete! Total: ${TOTAL_MINUTES} minutes." >> $LOG_FILE
