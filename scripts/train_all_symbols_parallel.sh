#!/bin/bash
# Train all symbols in parallel across 3 GPUs
# This script optimally distributes training jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment
source .venv/bin/activate

echo "════════════════════════════════════════════════════════════════════════"
echo "  Multi-GPU Parallel Training - All Symbols"
echo "  Expected time: ~4-6 hours (vs ~12-18 hours single GPU)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Tier 1 symbols (high priority)
TIER1_SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"

# Tier 2 symbols (medium priority)
TIER2_SYMBOLS="XRPUSDT,ADAUSDT,DOGEUSDT,DOTUSDT,AVAXUSDT"

# Configuration
TIMEFRAMES="5m,15m"
EPOCHS=200
BATCH_SIZE=256
HIDDEN_DIM=192
LSTM_LAYERS=3
DROPOUT=0.3

echo "Training Configuration:"
echo "  Tier 1 Symbols: $TIER1_SYMBOLS"
echo "  Tier 2 Symbols: $TIER2_SYMBOLS"
echo "  Timeframes: $TIMEFRAMES"
echo "  Epochs: $EPOCHS"
echo "  GPUs: 0, 1, 2 (3 GPUs in parallel)"
echo ""
echo "Total jobs: $(echo $TIER1_SYMBOLS,$TIER2_SYMBOLS | tr ',' '\n' | wc -l) symbols × 2 timeframes = $(($(echo $TIER1_SYMBOLS,$TIER2_SYMBOLS | tr ',' ' ' | wc -w) * 2)) jobs"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting parallel training..."
echo ""

# Train Tier 1 first
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --mode single \
    --symbols "$TIER1_SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --lstm-layers $LSTM_LAYERS \
    --dropout $DROPOUT

echo ""
echo "✅ Tier 1 complete!"
echo ""
echo "Starting Tier 2..."
echo ""

# Train Tier 2
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --mode single \
    --symbols "$TIER2_SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --lstm-layers $LSTM_LAYERS \
    --dropout $DROPOUT

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Check trained models in: models/advanced/"
echo "Check logs in: logs/multi_gpu/"
echo ""
