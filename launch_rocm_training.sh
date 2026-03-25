#!/bin/bash
# Phantom V30: AMD ROCm Training Launcher
# Targeted for RX 6600 (gfx1032)

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH

echo "🚀 Launching Phantom V30 on AMD ROCm (RX 6600)..."

# Ensure venv dependencies are installed (first run only check)
if [ ! -d ".venv_rocm62/lib/python3.12/site-packages/stable_baselines3" ]; then
    echo "📦 Installing missing dependencies..."
    .venv_rocm62/bin/pip install gymnasium stable-baselines3[extra] sb3-contrib pandas numpy
fi

# Continuous Loop
while true; do
    echo "🔄 Fetching Recent Market Data..."
    .venv_rocm62/bin/python scripts/update_candles.py
    
    echo "💪 Training Round Start..."
    .venv_rocm62/bin/python scripts/phantom_v30/train_v30_mlp.py
    
    echo "💤 Resting 10s before next round..."
    sleep 10
done
