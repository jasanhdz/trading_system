#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/daily_retrain.log"
DATE=$(date)

echo "[$DATE] 🚀 Starting Daily Retraining..." >> $LOG_FILE

# 1. Go to project dir
cd $PROJECT_DIR

# 2. Run Training
# Full AMD ROCm Environment (per ROCM_AMD_SETUP.md)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0

$PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "[$DATE] ✅ Training completed successfully." >> $LOG_FILE
    
    # 3. Reload ML Service to pick up new models
    pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
    echo "[$DATE] 🔄 ML Service reloaded." >> $LOG_FILE
else
    echo "[$DATE] ❌ Training failed. Check logs." >> $LOG_FILE
fi
