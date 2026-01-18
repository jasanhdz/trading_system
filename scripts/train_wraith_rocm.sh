#!/bin/bash
#
# Wraith Symbol Training with AMD ROCm
# Usage: ./train_wraith_rocm.sh SYMBOL [GPU_ID]
# Example: ./train_wraith_rocm.sh ETH/USDT 0
#

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
SYMBOL=${1:-"SOL/USDT"}
GPU_ID=${2:-0}

echo "🦅 Wraith Training: $SYMBOL on GPU $GPU_ID"

# Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

cd $PROJECT_DIR

# Run training on specified GPU
HIP_VISIBLE_DEVICES=$GPU_ID $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_wraith_symbol.py "$SYMBOL"

echo "✅ Training complete for $SYMBOL"
