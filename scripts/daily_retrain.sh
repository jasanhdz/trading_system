#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/daily_retrain.log"
DATE=$(date)

echo "[$DATE] 🚀 Starting Daily Retraining..." >> $LOG_FILE

# 1. Go to project dir
cd $PROJECT_DIR

# 2. Run Training
# Force GPU usage
export HSA_OVERRIDE_GFX_VERSION=10.3.0 
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
