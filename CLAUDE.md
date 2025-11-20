# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **cryptocurrency futures trading system** for Binance Futures (USDT perpetual contracts) that combines ML-driven signal generation with automated risk management. The system uses LSTM+Attention neural networks to predict market direction on 5m and 15m timeframes across 19 cryptocurrency pairs.

**Technology Stack**: Python 3.12, PyTorch (ROCm/CUDA), SQLite/SQLAlchemy, CCXT, python-binance

## Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Collection
```bash
# Collect historical data for all symbols
python scripts/collect_historical_data.py

# Update ML training candles
python scripts/update_ml_candles.py
```

### Model Training

**Single Symbol Training** (GPU-optimized):
```bash
# Train with GPU (AMD ROCm or NVIDIA CUDA)
python scripts/train_improved_gpu.py \
    --symbol BTCUSDT \
    --timeframe 5m \
    --epochs 150 \
    --batch-size 256 \
    --lr 5e-4 \
    --hidden-dim 128 \
    --lstm-layers 2 \
    --dropout 0.25 \
    --device cuda
```

**Batch Training** (all symbols):
```bash
# Train all 19 symbols with optimized hyperparameters
python scripts/train_all_symbols_optimized.py
```

**Hyperparameter Search**:
```bash
# Run Optuna-based hyperparameter optimization
python scripts/hyperparameter_search.py --symbol BTCUSDT
```

**GPU Training Script** (AMD ROCm):
```bash
# Wrapper script with ROCm environment variables
./scripts/train_on_gpu.sh ETHUSDT 15m
```

### Analysis & Testing
```bash
# Exploratory data analysis
python scripts/run_eda.py

# Statistical tests and feature analysis
python scripts/run_formal_statistics.py

# Regime detection (bull/bear/range identification)
python scripts/run_regime_analysis.py

# Market microstructure analysis
python scripts/run_microstructure_analysis.py

# Test trained models
python scripts/test_advanced_models.py
```

### Running the Trading Bot
```bash
# Production mode (requires .env with API keys)
cd binance_futures_bot_py
python main.py

# Testnet mode
IS_TESTNET=1 python main.py
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_models.py -v

# Coverage report
pytest --cov=ml --cov=data --cov-report=html
```

## Code Architecture

### High-Level Structure

The codebase follows a **clean architecture** pattern with clear separation of concerns:

```
trading_system/
├── ml/                     # Machine learning models and training
├── data/                   # Data collection and storage layer
├── binance_futures_bot_py/ # Production trading bot
├── analysis/               # Research and feature engineering
├── backtesting/           # Strategy backtesting framework
├── scripts/               # Training and utility scripts
├── models/                # Trained model artifacts
└── config/                # Global configuration
```

### ML Pipeline Architecture

**Data Flow**:
```
Binance API (CCXT)
    ↓
SQLite Database (xrp_trading.db)
    ↓
Feature Engineering (64 technical indicators)
    ↓
Feature Selection (Mutual Information → 32 features)
    ↓
Sequence Creation (24 timesteps, sliding window)
    ↓
Walk-Forward Splits (5 folds, 70/30 train/test)
    ↓
Model Training (LSTM+Attention)
    ↓
Validation & Checkpointing
    ↓
Model Artifacts (model.pt, scaler.pkl, meta.json)
    ↓
Production Inference (services/predictor.py)
    ↓
Trading Bot (binance_futures_bot_py/)
```

### Neural Network Architecture

**Model**: `ml/advanced_models/temporal_model.py::AdvancedTemporalNet`

Architecture components:
```
Input: [batch, 24 timesteps, 32 features]
    ↓
TemporalEncoder (LSTM + Attention):
  - Bidirectional LSTM (128 hidden, 2 layers)
  - Multi-head Attention (4 heads)
  - Layer normalization
    ↓
Dense Backbone:
  - FC layers: 256 → 128
  - BatchNorm + ReLU + Dropout
  - Residual blocks (when dims match)
    ↓
Multi-Task Heads:
  ├─→ Classifier → [neutral, long, short] probabilities
  └─→ Regressor → predicted return (continuous)
```

**Key Features**:
- **Multi-task learning**: Joint classification + regression improves feature learning
- **Attention mechanism**: Captures important timesteps dynamically
- **Bidirectional LSTM**: Processes sequences forward and backward
- **Residual connections**: Improves gradient flow for deep networks
- **ROCm compatibility**: Automatic detection and workaround for AMD GPU dropout issues

### Training Strategy

**Walk-Forward Validation** (`ml/advanced_models/trainer.py`):
- 5-fold expanding window splits (70/30 train/test)
- Gap of `prediction_horizon` bars between train/test to prevent lookahead bias
- Class weighting based on inverse frequency to handle imbalance
- Cosine annealing with warmup for learning rate scheduling
- Early stopping on macro F1 score (patience: 25 epochs)
- Mixed precision training (FP16) for GPU efficiency

**Data Augmentation**:
- Gaussian noise injection (σ=0.01) on features
- Applied only to training set
- Improves model robustness to market noise

### Trading Bot Architecture

**Location**: `binance_futures_bot_py/`

**Core Components**:

1. **Strategy Layer** (`src/strategies/ml_probability.py`):
   - Loads ML models for primary + confirmation timeframes
   - Generates signals when probability > threshold
   - Multi-timeframe confirmation (5m + 15m)
   - Anti-loss filter prevents entries against strong adverse signals

2. **Guard System** (`src/app/guards/`):
   - `profit_guard.py`: Breakeven locks, giveback protection
   - `intelligent_take_profit.py`: Dynamic TP based on trend strength
   - `ensure_brackets.py`: SL/TP order management
   - `pyramid_guard.py`: Scale-in logic for winning positions

3. **Risk Management** (`src/core/risk_manager.py`):
   - ATR-based position sizing
   - Leverage-aware stop loss placement
   - Portfolio heat limits
   - Max drawdown protection

4. **Infrastructure** (`src/infra/`):
   - `binance/exchange.py`: Futures API wrapper
   - `config.py`: Comprehensive configuration (600+ lines)
   - `logger.py`: Structured logging with context

**Multi-Symbol Management**:
- Each symbol runs as independent task (asyncio)
- Staggered startup (5-second intervals) to avoid API rate limits
- Shared ML predictor service for efficiency

### Feature Engineering

**Location**: `ml/nn_pattern/features.py`, `analysis/features/`

**64 Technical Indicators** (before selection):

- **Trend**: SMA(10,20,50,100,200), EMA(10,20,50), MACD, ADX, Aroon, SAR
- **Momentum**: RSI(14), Stochastic(14,3), Williams %R, ROC, CCI, Momentum
- **Volatility**: Bollinger Bands, Keltner Channels, ATR(14,20), NATR, Historical Volatility
- **Volume**: Volume ratios, OBV, A/D Line, CMF, MFI, VPT
- **Custom**: Log returns, rolling volatility, volume z-scores, return features (1,3,6,12 periods)

**Feature Selection** (`ml/advanced_models/dataset.py::FeatureSelector`):
- Method: Mutual Information (MI between features and target labels)
- Selects top 32 features from 64
- Reduces noise and overfitting
- Improves training speed

### Database Schema

**Location**: `data/storage/models.py`

**SQLite Database**: `data/xrp_trading.db` (2.3GB)

**Tables**:
```sql
-- OHLCV candlestick data
ohlcv_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(symbol, timeframe, timestamp)
)
CREATE INDEX idx_ohlcv_lookup ON ohlcv_data(symbol, timeframe, timestamp);

-- Market data (funding, open interest)
market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    funding_rate REAL,
    open_interest REAL,
    UNIQUE(symbol, timestamp)
)

-- Trading signals (historical)
trading_signals (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    timeframe TEXT,
    timestamp INTEGER,
    signal_type TEXT,  -- 'long', 'short', 'neutral'
    probability REAL,
    price REAL,
    metadata TEXT      -- JSON string with extra info
)
```

**Data Collection** (`data/collectors/binance_collector.py`):
- Uses CCXT library for exchange abstraction
- Rate limiting: 0.1s delay between requests
- Chunked downloads: 1000 candles per request
- Automatic deduplication and gap detection
- Timezone-aware (UTC)

## Model Configuration Details

### Saved Model Artifacts

**Location**: `models/advanced/{SYMBOL}/{TIMEFRAME}/`

**Files**:
- `model.pt`: PyTorch state dict (~4MB)
- `scaler.pkl`: StandardScaler fitted on training data
- `feature_selector.pkl`: Selected feature indices
- `meta.json`: Model config, hyperparameters, and metrics
- `walk_forward_results.json`: Per-fold validation metrics

### Critical Configuration Options

**Dataset Config** (`ml/advanced_models/dataset.py::AdvancedDatasetConfig`):
```python
sequence_length = 24          # Lookback window (24 bars = 2 hours on 5m)
prediction_horizon = 12       # Forecast ahead (12 bars = 1 hour on 5m)
target_return = 0.002         # 0.2% threshold for long/short labels
max_history_days = 365        # Training data duration
use_feature_selection = True  # Enable MI feature selection
n_features_to_select = 32     # Reduce from 64 to 32 features
```

**Model Config** (in `meta.json`):
```python
hidden_dim = 128              # LSTM hidden units
lstm_layers = 2               # Number of stacked LSTM layers
dense_dims = [256, 128]       # Dense layer dimensions
dropout = 0.3                 # Dropout probability
use_attention = True          # Enable attention mechanism
bidirectional = True          # Bidirectional LSTM
```

**Trading Bot Config** (`binance_futures_bot_py/src/infra/config.py`):
```python
# Risk parameters
LEVERAGE = 50                 # Leverage multiplier (WARNING: High risk)
CAPITAL_USAGE_PCT = 0.85     # Use 85% of available capital per trade
TP_ROE = 1.0                 # Take profit at 100% ROE
SL_TICKS_ABOVE_LIQ_DEFAULT = 69  # Stop loss safety margin

# ML thresholds
ML_THRESHOLD_LONG = 0.65     # Long entry threshold (0-1)
ML_THRESHOLD_SHORT = 0.70    # Short entry threshold (0-1)
ML_MARGIN = 0.15             # Required probability gap neutral vs signal
ML_REQUIRE_CONFIRMATION = True  # Multi-timeframe confirmation

# Filters
ML_MAX_RSI = 68              # Block longs above this RSI
ML_MIN_RSI = 32              # Block shorts below this RSI
ML_MAX_BODY_ATR = 2.5        # Block on large candles (> 2.5 ATR)
ML_MAX_EXT_PCT = 0.015       # Max 1.5% extension from EMA
```

## Important Patterns and Conventions

### Model Training Pattern
When training models, always follow this sequence:
1. Load data with proper configuration
2. Apply feature selection (if enabled)
3. Scale features with StandardScaler
4. Create walk-forward splits (time-based, no shuffling!)
5. Train with class weights for imbalance
6. Save model + scaler + selector as a set
7. Validate with out-of-fold predictions

**Never shuffle time series data** - this causes lookahead bias!

### Model Loading Pattern
```python
# Load model artifacts together
model = AdvancedTemporalNet(input_dim=32, **model_config)
model.load_state_dict(torch.load("model.pt"))
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Inference pipeline
raw_features = engineer_features(ohlcv_data)  # 64 features
selected_features = selector.transform(raw_features)  # 32 features
scaled_features = scaler.transform(selected_features)
sequences = create_sequences(scaled_features, seq_len=24)
predictions = model(sequences)
```

### ROCm GPU Training
For AMD GPUs with ROCm, set these environment variables:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6600
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MIOPEN_DEBUG=0

# Disable in-LSTM dropout (ROCm compatibility issue)
# The code automatically detects ROCm and disables it
# To force enable: export FORCE_LSTM_DROPOUT=1
```

### Configuration Management
- **Bot configuration**: Use `.env` file in `binance_futures_bot_py/`
- **Training configuration**: Pass via CLI arguments to training scripts
- **Never hardcode API keys** - always use environment variables
- **Symbol format**: Use "BTCUSDT" (no slash) for bot, "BTC/USDT:USDT" for CCXT

### Error Handling Philosophy
- **Data collection**: Skip problematic chunks, log warnings, continue
- **Trading bot**: Retry with exponential backoff (max 3 attempts)
- **Model inference**: Fallback to neutral signal on errors
- **Exchange errors**: Distinguish between recoverable (retry) vs fatal (stop)

## Known Limitations and Gotchas

### Model Performance
- Current models show **weak predictive power** (36-44% accuracy vs 33% baseline)
- Short predictions are particularly weak (F1 ~13-30%)
- High variance across walk-forward folds indicates regime sensitivity
- Models trained on 1 year of data may not capture all market regimes

### Risk Parameters
- **Default 50x leverage is extremely high** - consider 10-20x for production
- **85% capital usage leaves little buffer** - consider 50-70%
- No portfolio-level risk limits (e.g., max total exposure)
- No correlation analysis between symbol positions

### Data Quality
- Single data source (Binance) - no cross-validation with other exchanges
- No outlier detection or data quality checks
- Missing data gaps not handled systematically
- Funding rate and OI data collected but not used in models

### Infrastructure Gaps
- **SQLite not suitable for production** - migrate to PostgreSQL for concurrent writes
- No monitoring/alerting system (Grafana, Prometheus)
- No trade journaling database
- No paper trading mode for safe validation
- No model versioning (consider MLflow or DVC)
- No CI/CD pipeline or automated testing

### Scalability Constraints
- LSTM models are slow for inference (not suitable for < 1m timeframes)
- Feature engineering requires full window of historical data
- Model retraining is manual (no automatic retraining pipeline)

## Critical Files to Understand

### Model Architecture
- `ml/advanced_models/temporal_model.py` - Neural network definitions
- `ml/advanced_models/trainer.py` - Training loop with walk-forward validation
- `ml/advanced_models/dataset.py` - Data loading and sequence creation

### Feature Engineering
- `ml/nn_pattern/features.py` - Technical indicator calculations
- `analysis/features/ta_features.py` - TA-Lib based features
- `analysis/features/pattern_features.py` - Pattern detection

### Trading Bot
- `binance_futures_bot_py/main.py` - Bot entry point and multi-symbol orchestration
- `binance_futures_bot_py/src/strategies/ml_probability.py` - ML signal generation
- `binance_futures_bot_py/src/app/guards/profit_guard.py` - Core risk management
- `binance_futures_bot_py/src/infra/config.py` - Configuration system

### Data Pipeline
- `data/collectors/binance_collector.py` - Historical data collection
- `data/storage/db_manager.py` - Database operations
- `data/storage/models.py` - SQLAlchemy models

## Production Readiness Checklist

Before deploying to production with real capital:

**Model Quality** (Critical):
- [ ] Achieve > 55% accuracy with F1 > 0.45 on walk-forward validation
- [ ] Verify model performance on held-out test set (last 3 months)
- [ ] Paper trade for minimum 2 weeks with acceptable Sharpe ratio
- [ ] Implement ensemble of multiple models for robustness

**Risk Management** (Critical):
- [ ] Reduce leverage to 10-20x maximum
- [ ] Implement portfolio-level risk limits (max 100% notional exposure)
- [ ] Add circuit breakers (halt on 5% daily drawdown)
- [ ] Implement correlation matrix to avoid redundant positions
- [ ] Add per-trade risk limit (max 2% account per trade)

**Infrastructure** (High Priority):
- [ ] Migrate from SQLite to PostgreSQL
- [ ] Setup monitoring (Grafana + Prometheus)
- [ ] Implement alerting for critical events
- [ ] Create trade journaling database
- [ ] Setup automated backups
- [ ] Dockerize application
- [ ] Add health check endpoints

**Testing** (High Priority):
- [ ] Achieve 80%+ test coverage for trading logic
- [ ] Add integration tests for exchange API
- [ ] Setup CI/CD pipeline (GitHub Actions)
- [ ] Add pre-commit hooks for code quality

**Documentation** (Medium Priority):
- [ ] Create API documentation
- [ ] Write deployment guide
- [ ] Document disaster recovery procedures
- [ ] Create runbook for common issues

## Symbol-Specific Notes

### Tier 1 (BTC, ETH, BNB, SOL)
- Highest liquidity and tightest spreads
- Use larger `hidden_dim=128`, lower `dropout=0.25`
- Target return: `0.0015` (0.15%)
- More reliable for 5m timeframe

### Tier 2 (XRP, ADA, DOGE, DOT, AVAX, LINK, LTC)
- Good liquidity but more volatile
- Use `hidden_dim=96`, `dropout=0.30`
- Target return: `0.002` (0.2%)
- Consider 15m timeframe for better signals

### Tier 3 (BCH, UNI, TRX, ETC, XLM, XMR, RUNE, ARB)
- Lower liquidity, higher slippage risk
- Use `hidden_dim=64`, `dropout=0.35`
- Target return: `0.0025` (0.25%)
- Prefer 15m or 1h timeframes
- Reduce position size by 50%

## Additional Context

- The repository uses structured logging with context (structlog)
- All timestamps are stored in UTC
- The bot supports both testnet and mainnet Binance Futures
- Recent commits focused on GPU training optimization and hyperparameter search
- Code is compatible with Python 3.12+
- AMD GPU support via ROCm 5.7+ (tested on RX 6600)
