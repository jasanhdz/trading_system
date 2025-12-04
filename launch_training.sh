#!/bin/bash
# Script para lanzar el entrenamiento multi-GPU después de verificar que las AMD funcionan

cd /home/jasan/Develop/trading_system

echo "🚀 Lanzando entrenamiento en 4 GPUs (3 AMD + 1 NVIDIA)..."
echo ""

# Entorno ROCm para AMD
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH

# Matar cualquier proceso previo
pkill -9 -f "python.*train" 2>/dev/null
pkill -9 -f "python.*dispatch" 2>/dev/null

# Limpiar logs antiguos (el dispatcher lo hace automáticamente)

# Lanzar dispatcher
nohup .venv_rocm62/bin/python -u scripts/dispatch_training.py \
  --symbols BTCUSDT,SOLUSDT,LINKUSDT,XRPUSDT,ADAUSDT,ETHUSDT \
  --timeframes 15m \
  --target-return 0.005 \
  --prediction-horizon 6 \
  > logs/training_post_reboot_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DISPATCHER_PID=$!

echo "✅ Dispatcher iniciado con PID: $DISPATCHER_PID"
echo ""
echo "Para monitorear:"
echo "  - Ver procesos: ps aux | grep train"
echo "  - Ver logs individuales: tail -f logs/multi_gpu/*.log"
echo "  - Ver GPUs AMD: rocm-smi"
echo "  - Ver GPU NVIDIA: nvidia-smi"
echo ""
