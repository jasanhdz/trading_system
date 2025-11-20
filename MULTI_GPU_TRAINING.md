# Multi-GPU Training Guide

## 🎮 Your Hardware Setup

You have **3x AMD RX 6600** GPUs (8GB VRAM each) available for training.

**Current Status:**
- ✅ GPU 0: AMD RX 6600 (8GB) - 6.2 TFLOPS
- ✅ GPU 1: AMD RX 6600 (8GB) - 75 TFLOPS
- ✅ GPU 2: AMD RX 6600 (8GB) - 74.6 TFLOPS
- ❌ NVIDIA GTX 1660 - Not available (PyTorch compiled with ROCm only)

**Expected Speedup:** ~3x faster training (3 models simultaneously)

---

## 🚀 Quick Start

### Option 1: Train BTC + ETH + SOL in Parallel (Fastest)

Train 3 different symbols simultaneously (one per GPU):

```bash
# Activate environment
source .venv/bin/activate

# Train 3 symbols in parallel (5m and 15m timeframes)
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --timeframes 5m,15m \
    --epochs 200
```

**Time:** ~3-4 hours (vs ~9-12 hours single GPU)

---

### Option 2: Train All Symbols (Batch Mode)

Train all Tier 1 + Tier 2 symbols automatically:

```bash
# One command to train everything
./scripts/train_all_symbols_parallel.sh
```

**Symbols trained:**
- Tier 1: BTC, ETH, BNB, SOL (4 symbols)
- Tier 2: XRP, ADA, DOGE, DOT, AVAX (5 symbols)
- Timeframes: 5m, 15m
- **Total:** 18 models

**Time:** ~6-8 hours (vs ~20-24 hours single GPU)

---

### Option 3: Manual Control (Advanced)

Train specific combinations manually:

```bash
# Terminal 1 - GPU 0: BTC 5m
CUDA_VISIBLE_DEVICES=0 python scripts/train_production_ready.py \
    --symbol BTCUSDT --timeframe 5m --device cuda:0 --epochs 200

# Terminal 2 - GPU 1: BTC 15m
CUDA_VISIBLE_DEVICES=1 python scripts/train_production_ready.py \
    --symbol BTCUSDT --timeframe 15m --device cuda:0 --epochs 200

# Terminal 3 - GPU 2: ETH 5m
CUDA_VISIBLE_DEVICES=2 python scripts/train_production_ready.py \
    --symbol ETHUSDT --timeframe 5m --device cuda:0 --epochs 200
```

**Note:** Always use `--device cuda:0` with `CUDA_VISIBLE_DEVICES` to isolate GPUs.

---

## 📊 Monitor Training Progress

### Check GPU Usage

```bash
# Check GPU utilization
rocm-smi

# Watch in real-time (updates every 2 seconds)
watch -n 2 rocm-smi
```

**Expected output during training:**
```
GPU  Temp   Power   GPU%   Memory
0    70°C   120W    95%    6.2/8 GB
1    68°C   118W    98%    6.5/8 GB
2    69°C   119W    97%    6.3/8 GB
```

### Check Training Logs

All training logs are saved to `logs/multi_gpu/`:

```bash
# List all training logs
ls -lth logs/multi_gpu/

# Tail live training log
tail -f logs/multi_gpu/BTCUSDT_5m_gpu0_*.log
```

### Check Process Status

```bash
# List running training processes
ps aux | grep train_production_ready

# Kill all training if needed
pkill -f train_production_ready
```

---

## 🎯 Recommended Training Strategies

### Strategy 1: Prioritize Best Performers (Recommended)

Train BTC and ETH first (highest liquidity), evaluate results, then expand:

```bash
# Phase 1: Train BTC + ETH (both timeframes)
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT,ETHUSDT \
    --timeframes 5m,15m \
    --epochs 200

# Check results, if good (F1 > 0.45), proceed to Phase 2

# Phase 2: Train SOL + BNB
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols SOLUSDT,BNBUSDT \
    --timeframes 5m,15m \
    --epochs 200
```

**Time:** 2 phases × 2-3 hours = 4-6 hours total

---

### Strategy 2: Explore Best Timeframe First

If unsure which timeframe works best, train 3 symbols on one timeframe first:

```bash
# Test 5m timeframe with BTC, ETH, SOL
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --timeframes 5m \
    --epochs 200

# If 5m is good, train 15m for same symbols
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --timeframes 15m \
    --epochs 200
```

**Time:** 2 batches × 3 hours = 6 hours total

---

### Strategy 3: Train Ensembles (Best Quality)

Train ensemble models for maximum accuracy (slower but better results):

```bash
# Train ensemble of 3 models (one per GPU)
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --mode ensemble \
    --symbols BTCUSDT \
    --timeframes 5m \
    --epochs 200

# This trains 3 independent models and ensembles them
# Better accuracy but 3x slower than single model
```

**Time:** ~9-12 hours for full ensemble

---

## ⚙️ Advanced Configuration

### Adjust Hyperparameters

```bash
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --timeframes 5m,15m \
    --epochs 200 \
    --batch-size 256 \
    --lr 5e-4 \
    --hidden-dim 192 \
    --lstm-layers 3 \
    --dropout 0.3 \
    --sequence-length 48 \
    --prediction-horizon 6 \
    --target-return 0.005
```

### Reduce Training Time (Faster Iteration)

For quick experiments:

```bash
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT \
    --timeframes 5m \
    --epochs 50 \           # Reduce from 200
    --batch-size 512 \      # Increase (faster but more memory)
    --sequence-length 24    # Reduce from 48 (less context)
```

**Time:** ~30-60 minutes per model

### Increase Model Capacity (Better Accuracy)

For production models:

```bash
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols BTCUSDT \
    --timeframes 5m,15m \
    --epochs 300 \          # More training
    --hidden-dim 256 \      # Larger model
    --lstm-layers 4 \       # Deeper network
    --dropout 0.25          # Less dropout (more capacity)
```

**Time:** ~6-8 hours per model

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error

**Symptoms:** Training crashes with "HIP out of memory"

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch-size 128  # Instead of 256
   ```

2. Reduce model size:
   ```bash
   --hidden-dim 128 --lstm-layers 2
   ```

3. Use gradient accumulation (edit training script):
   ```python
   accumulation_steps = 2  # Effective batch size = 128 × 2 = 256
   ```

---

### Issue: GPU Throttling / Overheating

**Symptoms:** Training slower than expected, GPU temp > 85°C

**Solutions:**
1. Improve cooling:
   ```bash
   # Check GPU temperature
   watch -n 1 rocm-smi

   # If temp > 80°C, consider:
   # - Improve case airflow
   # - Lower room temperature
   # - Reduce power limit
   ```

2. Use 2 GPUs instead of 3:
   ```bash
   --gpus 0,2  # Skip GPU 1 to reduce heat
   ```

---

### Issue: Training Process Hangs

**Symptoms:** No progress in logs for > 10 minutes

**Solutions:**
1. Kill and restart:
   ```bash
   # Find hanging process
   ps aux | grep train_production

   # Kill by PID
   kill -9 <PID>

   # Restart training
   ```

2. Check GPU status:
   ```bash
   rocm-smi

   # If GPU shows 0% utilization, restart
   sudo systemctl restart rocm
   ```

---

### Issue: NVIDIA GPU Not Detected

**Symptoms:** Only 3 AMD GPUs detected, NVIDIA missing

**Cause:** PyTorch compiled with ROCm only (no CUDA support)

**Solution (Optional):**

To use NVIDIA GPU, install CUDA-enabled PyTorch in separate environment:

```bash
# Create new environment for CUDA
python -m venv .venv_cuda
source .venv_cuda/bin/activate

# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Now you can use:
# --gpus 0  (NVIDIA GPU in CUDA environment)
```

**Note:** You cannot use ROCm + CUDA in same PyTorch installation. Keep separate environments.

---

## 📈 Performance Benchmarks

### Training Time Comparison

| Configuration | Single GPU | 3 GPUs Parallel | Speedup |
|--------------|-----------|----------------|---------|
| 1 model (200 epochs) | ~3 hours | ~3 hours | 1x |
| 3 models (BTC/ETH/SOL) | ~9 hours | ~3 hours | 3x |
| 9 models (Tier 1, 5m+15m) | ~27 hours | ~9 hours | 3x |
| 18 models (All symbols) | ~54 hours | ~18 hours | 3x |

### Memory Usage per Model

| Model Configuration | VRAM Usage | Fits in 8GB? |
|--------------------|-----------|--------------|
| hidden_dim=128, layers=2, batch=256 | ~4.5 GB | ✅ Yes |
| hidden_dim=192, layers=3, batch=256 | ~6.2 GB | ✅ Yes |
| hidden_dim=256, layers=4, batch=256 | ~8.5 GB | ❌ No (OOM) |
| hidden_dim=256, layers=4, batch=128 | ~5.8 GB | ✅ Yes |

**Recommendation:** Stick with default (192, 3, 256) for 8GB GPUs.

---

## 🎯 Optimal Training Plan

Here's the recommended training sequence for best results:

### Phase 1: Proof of Concept (Day 1)
```bash
# Train BTC on both timeframes
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1 \
    --symbols BTCUSDT \
    --timeframes 5m,15m \
    --epochs 200
```
**Time:** ~3 hours
**Goal:** Verify F1 > 0.45 on at least one timeframe

---

### Phase 2: Expand Top Performers (Day 2)
```bash
# If BTC looks good, train ETH + SOL
python scripts/train_parallel_multi_gpu.py \
    --gpus 0,1,2 \
    --symbols ETHUSDT,SOLUSDT,BNBUSDT \
    --timeframes <best_timeframe_from_phase1> \
    --epochs 200
```
**Time:** ~3 hours
**Goal:** Build portfolio of 4 strong models

---

### Phase 3: Production Models (Day 3-4)
```bash
# Train ensembles for best 2 symbols
python scripts/train_ensemble.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --n-models 5 \
    --epochs 200

python scripts/train_ensemble.py \
    --symbol ETHUSDT \
    --timeframe 5m \
    --n-models 5 \
    --epochs 200
```
**Time:** ~12 hours (overnight)
**Goal:** Production-ready ensemble models

---

### Phase 4: Full Deployment (Day 5-6)
```bash
# Train all Tier 1 + Tier 2 symbols
./scripts/train_all_symbols_parallel.sh
```
**Time:** ~6-8 hours
**Goal:** Complete model library for live trading

---

## 📝 Next Steps After Training

1. **Evaluate Results:**
   ```bash
   python scripts/evaluate_all_models.py
   ```

2. **Select Best Models:**
   - F1 > 0.45: Production ready
   - F1 > 0.40: Paper trading first
   - F1 < 0.40: Retrain with different config

3. **Paper Trading:**
   - Test for 2 weeks minimum
   - Monitor Sharpe ratio > 1.5
   - Check drawdown < 15%

4. **Production Deployment:**
   - Update bot config with new models
   - Start with small capital (2-5%)
   - Gradually increase position sizes

---

## 🔗 Related Documentation

- `GUIA_MEJORAS_MODELO.md` - Model improvements guide
- `CLAUDE.md` - Full system documentation
- `scripts/train_production_ready.py` - Single model training
- `scripts/train_ensemble.py` - Ensemble training

---

## ❓ Questions?

Run diagnostics anytime:
```bash
python scripts/diagnose_multi_gpu.py
```

This will show:
- ✅ GPU detection status
- 💾 Memory availability
- ⚡ Compute performance
- 📋 Training recommendations
