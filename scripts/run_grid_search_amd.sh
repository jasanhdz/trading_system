#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"

# AMD ROCm Environment Setup
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0,1 # Use both AMD GPUs

echo "🚀 Starting Grid Search on AMD GPUs..."
echo "   Devices: $HIP_VISIBLE_DEVICES"

# Run Optimizer
$PROJECT_DIR/.venv_rocm62/bin/python3 scripts/grid_search_optimizer.py "$@"
