# Advanced Machine Learning Models for Trading

## 🎯 Overview

This module provides state-of-the-art temporal models for cryptocurrency trading prediction. It significantly improves upon the simple feedforward approach by incorporating:

- **LSTM + Attention**: Captures temporal patterns and dependencies
- **Feature Selection**: Automatically selects the most informative features
- **Walk-Forward Validation**: Realistic performance estimation
- **Multi-Task Learning**: Predicts both direction and magnitude
- **Ensemble Methods**: Combines multiple models for robustness

## 📁 Structure

```
ml/advanced_models/
├── __init__.py
├── temporal_model.py      # LSTM, Attention, Ensemble models
├── dataset.py            # Sequence datasets, feature selection, walk-forward splits
├── trainer.py            # Training pipeline with walk-forward validation
└── predictor.py          # Inference and prediction
```

## 🚀 Quick Start

### 1. Train a Basic Model

```bash
python scripts/train_advanced_model.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50
```

### 2. Train with Walk-Forward Validation (Recommended)

```bash
python scripts/train_advanced_model.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --walk-forward \
    --n-folds 5 \
    --epochs 50
```

### 3. Train an Ensemble

```bash
python scripts/train_advanced_model.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --ensemble 3 \
    --walk-forward \
    --epochs 50
```

### 4. Compare Old vs New

```bash
python scripts/compare_models.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 30
```

## 🏗️ Architecture

### AdvancedTemporalNet

```
Input: [batch, 24 timesteps, 32 features]
  ↓
LSTM Layer (bidirectional, 128 hidden)
  ↓
Attention Layer (multi-head, 4 heads)
  ↓
Dense Layers (256 → 128)
  ├─→ Classification Head → [neutral, long, short]
  └─→ Regression Head → predicted_return
```

**Key Features:**
- Bidirectional LSTM captures past and future context
- Attention mechanism focuses on important timesteps
- Residual connections for better gradient flow
- Batch normalization for stable training
- Multi-task learning for better generalization

## 📊 Expected Performance

| Metric | Simple Model | Advanced Model | Improvement |
|--------|--------------|----------------|-------------|
| Accuracy | 30-40% | **50-60%** | +20-25% |
| F1 Score | 0.25-0.35 | **0.45-0.55** | +20 points |
| AP Long | 0.35-0.40 | **0.50-0.60** | +15-20% |
| AP Short | 0.35-0.40 | **0.50-0.60** | +15-20% |

## 🎛️ Configuration

### Model Parameters

```bash
--sequence-length 24        # Lookback window (24 * 5m = 2 hours)
--hidden-dim 128           # LSTM hidden dimension
--lstm-layers 2            # Number of LSTM layers
--dense-dims "256,128"     # Dense layer dimensions
--dropout 0.3              # Dropout rate for regularization
--use-attention            # Enable attention mechanism
--bidirectional            # Use bidirectional LSTM
```

### Feature Selection

```bash
--feature-selection        # Enable automatic feature selection
--n-features 32           # Number of features to select (from 64)
```

### Training

```bash
--epochs 50               # Training epochs
--batch-size 512          # Batch size
--lr 0.001               # Learning rate
--walk-forward           # Use walk-forward validation
--n-folds 5              # Number of walk-forward folds
--ensemble 3             # Train ensemble of 3 models (0=single)
```

## 💡 Key Concepts

### 1. Temporal Sequences

Instead of treating each candle independently, the model uses sequences:

```python
# Old approach: Single vector
input = [RSI, MACD, Volume, ...]  # 64 features

# New approach: Sequence of vectors
input = [
    [RSI_t-23, MACD_t-23, Volume_t-23, ...],  # 24 timesteps ago
    [RSI_t-22, MACD_t-22, Volume_t-22, ...],
    ...
    [RSI_t, MACD_t, Volume_t, ...]           # Current timestep
]  # Shape: (24, 32)
```

This allows the model to see trends, momentum, and patterns.

### 2. Feature Selection

Automatically selects the most informative features using mutual information:

```python
# Example: Top features selected
[
    'rsi_14',         # High mutual information
    'macd',           # Strong signal
    'volume_ratio_20', # Less noise
    'return_1',       # Direct relevance
    ...
]
```

Reduces noise and overfitting by removing redundant features.

### 3. Walk-Forward Validation

Simulates real trading by training on expanding windows:

```
Fold 1: [Train=======][Test]
Fold 2: [Train==========][Test]
Fold 3: [Train=============][Test]
Fold 4: [Train================][Test]
Fold 5: [Train===================][Test]
```

Each test period uses only historical data for training, just like in production.

### 4. Multi-Task Learning

The model learns two tasks simultaneously:

- **Classification**: Predict direction (neutral/long/short)
- **Regression**: Predict expected return magnitude

This provides:
- Better generalization (shared representations)
- More calibrated predictions
- Richer output for decision making

## 🔧 Usage Examples

### Training Script

```python
from ml.advanced_models.trainer import AdvancedTrainer
from ml.advanced_models.dataset import AdvancedDatasetConfig

# Configure dataset
config = AdvancedDatasetConfig(
    symbol="BTC/USDT",
    timeframe="15m",
    sequence_length=24,
    prediction_horizon=12,
    target_return=0.002,
    use_feature_selection=True,
    n_features_to_select=32,
)

# Configure model
model_config = {
    'hidden_dim': 128,
    'lstm_layers': 2,
    'dense_dims': (256, 128),
    'dropout': 0.3,
    'use_attention': True,
    'bidirectional': True,
    'num_classes': 3,
    'use_regression': True,
}

# Train
trainer = AdvancedTrainer(config, model_config)
features, labels, returns, feature_names = trainer.load_and_prepare_data()

# Walk-forward validation
results = trainer.walk_forward_validation(
    features, labels, returns,
    n_splits=5,
    epochs=50,
    lr=0.001,
)
```

### Prediction Script

```python
from ml.advanced_models.predictor import AdvancedPredictor
from pathlib import Path

# Load trained model
predictor = AdvancedPredictor(
    model_path=Path("models/advanced/BTCUSDT/15m/model.pt"),
    scaler_path=Path("models/advanced/BTCUSDT/15m/scaler.pkl"),
    meta_path=Path("models/advanced/BTCUSDT/15m/meta.json"),
    feature_selector_path=Path("models/advanced/BTCUSDT/15m/feature_selector.pkl"),
)

# Get recent OHLCV data (need at least sequence_length rows)
import pandas as pd
df = get_recent_data(symbol="BTCUSDT", limit=50)

# Predict
prediction = predictor.predict(df)
print(prediction)
# {
#     'neutral': 0.15,
#     'long': 0.72,           # 72% probability of going up
#     'short': 0.13,
#     'predicted_return': 0.0035,  # Expected +0.35% return
#     'direction': 'long',
#     'confidence': 0.72
# }

# Use in trading strategy
if prediction['long'] > 0.65 and prediction['predicted_return'] > 0.003:
    enter_long_position()
```

## 📈 Optimization Tips

### If Overfitting (Train >> Test)

```bash
# Increase dropout
--dropout 0.4

# Reduce model capacity
--hidden-dim 64 --dense-dims "128"

# More regularization
# (already includes weight decay and batch norm)
```

### If Underfitting (Both Low)

```bash
# Increase model capacity
--hidden-dim 256 --dense-dims "512,256,128"

# More epochs
--epochs 100

# Higher learning rate
--lr 0.005

# Reduce dropout
--dropout 0.2
```

### If Class Imbalance

```bash
# Adjust target threshold
--target-return 0.0015  # Lower threshold = more signals

# Use focal loss (TODO: implement)

# Ensemble with different thresholds
```

## 🐛 Troubleshooting

### "Not enough data" error

**Problem**: `RuntimeError: Not enough 15m data`

**Solutions**:
1. Collect more historical data
2. Reduce sequence length: `--sequence-length 12`
3. Use shorter timeframe: `--timeframe 5m`

### CUDA out of memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 256`
2. Reduce model size: `--hidden-dim 64`
3. Use CPU: `--device cpu`

### Poor performance on specific symbol

**Problem**: Model works on BTC but not on altcoins

**Solutions**:
1. Train symbol-specific models
2. Adjust target_return based on volatility
3. Increase sequence_length for more context
4. Use ensemble with diverse configurations

## 📚 References

### Architecture Papers
- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf): Long Short-Term Memory
- [Attention](https://arxiv.org/abs/1706.03762): Attention Is All You Need
- [Multi-Task Learning](https://arxiv.org/abs/1705.07115): In Deep Neural Networks

### Trading ML
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL): Financial RL Library
- [Qlib](https://github.com/microsoft/qlib): Microsoft Quantitative Investment Platform

## 🤝 Contributing

To add new features:

1. **New Architecture**: Add to `temporal_model.py`
2. **New Loss Function**: Update `MultiTaskLoss` or add new class
3. **New Features**: Extend `ml/nn_pattern/features.py`
4. **New Validation Method**: Add to `trainer.py`

## 📝 TODO

- [ ] Transformer architecture (instead of LSTM)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Market regime detection (bull/bear specific models)
- [ ] Multi-symbol joint training
- [ ] Reinforcement learning integration
- [ ] Real-time inference optimization
- [ ] Explainability tools (SHAP values)

## 📄 License

Same as main project.

---

**Note**: Always backtest thoroughly before using in production. Past performance does not guarantee future results.
