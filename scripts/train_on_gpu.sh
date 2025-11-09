#!/bin/bash
# Script optimizado para entrenar en GPU AMD RX 6600 con ROCm
# Ubuntu Server

set -e

echo "=================================="
echo "GPU AMD TRAINING OPTIMIZER"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check ROCm installation
echo -e "${YELLOW}Checking ROCm installation...${NC}"
if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✓ ROCm found${NC}"
    rocm-smi --showproductname
    echo ""
else
    echo -e "${RED}✗ ROCm not found. Install ROCm first:${NC}"
    echo "  wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*.deb"
    echo "  sudo dpkg -i amdgpu-install_*.deb"
    echo "  sudo amdgpu-install --usecase=rocm"
    exit 1
fi

# Check PyTorch ROCm
echo -e "${YELLOW}Checking PyTorch ROCm support...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || {
    echo -e "${RED}✗ PyTorch with ROCm not found${NC}"
    echo -e "${YELLOW}Installing PyTorch for ROCm...${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
}

echo ""
echo -e "${GREEN}✓ System ready for GPU training${NC}"
echo ""

# Training configurations
SYMBOL="${1:-ETHUSDT}"
TIMEFRAME="${2:-15m}"

echo "=================================="
echo "TRAINING CONFIGURATION"
echo "=================================="
echo "Symbol: $SYMBOL"
echo "Timeframe: $TIMEFRAME"
echo ""

# Optimized parameters for RX 6600 (8GB VRAM)
EPOCHS=150
BATCH_SIZE=256
LR=0.0005
HIDDEN_DIM=128
LSTM_LAYERS=2
DROPOUT=0.25

echo "Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  LSTM layers: $LSTM_LAYERS"
echo "  Dropout: $DROPOUT"
echo ""

# Set ROCm environment variables for optimal performance
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6600
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MIOPEN_DEBUG=0

# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo -e "${YELLOW}Starting training...${NC}"
echo ""

# Run training
python3 scripts/train_improved_gpu.py \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --hidden-dim $HIDDEN_DIM \
    --lstm-layers $LSTM_LAYERS \
    --dropout $DROPOUT \
    --device cuda

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================="
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    echo "==================================${NC}"
    echo ""
    echo "Model saved to: models/advanced/$SYMBOL/$TIMEFRAME/"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate: python scripts/analyze_btc_models.py"
    echo "  2. Backtest: python scripts/backtest_strategy.py"
    echo "  3. Paper trade: python scripts/paper_trading.py"
else
    echo ""
    echo -e "${RED}=================================="
    echo "✗ TRAINING FAILED"
    echo "==================================${NC}"
    echo ""
    echo "Check logs above for errors"
    exit $EXIT_CODE
fi
