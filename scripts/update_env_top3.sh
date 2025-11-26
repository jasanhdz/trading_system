#!/bin/bash
# Script para actualizar .env con solo modelos Top 3

ENV_FILE="binance-futures-bot-ts/.env"

echo "📝 Actualizando $ENV_FILE con modelos Top 3..."

# Crear backup
cp "$ENV_FILE" "${ENV_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Crear archivo temporal con configuración actualizada
cat > "${ENV_FILE}.tmp" << 'EOF'
APP_ENV=prod

BINANCE_API_KEY=uUdF1I8QFPT5i3FxdVenULlplu3fugSnWmg8GxKZwCF51kEuGqQ19HSr1ilO3zs0
BINANCE_API_SECRET=GILTICkBtaj97esvdsikGRZkpkz8lw9S3QR2LgsaQOBlrA4Cf9CX8LMbS7fKmzXu
LEVERAGE=100
TP_ROE=2.0

STRATEGY=ml_probability

# ============================================================================
# SYMBOLS - Solo modelos ML activos (Top 3: ETH, XRP, LTC 15m)
# ============================================================================
SYMBOLS="
ETHUSDT:40:0.8,
XRPUSDT:50:0.8,
LTCUSDT:20:0.8,
"

# ============================================================================
# SÍMBOLOS COMENTADOS (Esperando re-entrenamiento)
# ============================================================================
# BTCUSDT:20:0.8,
# BNBUSDT:20:0.8,
# BCHUSDT:20:0.8,
# TRXUSDT:30:0.8,
# ETCUSDT:20:0.8,
# SOLUSDT:50:0.8,
# LINKUSDT:50:0.8,
# XLMUSDT:30:0.2,
# Otros símbolos comentados...

BOT_INTERVAL_SEC=10
INT_TP_SUP_INTERVAL_MS=3000
INT_TP_SUP_CONCURRENCY=6
LOG_PRETTY=1
LOG_LEVEL="info"
LOG_TO_FILE=1
INT_TP_MIN_ROE=0.25
TIME_STOP_MIN_ROE=0.25

BINANCE_TESTNET_API_KEY=fQ411ZK6UxwFpBjVp1KNFJ5wPv0gsYfgyO8DoI6TJ9OJvdNgrFOVwDCKinnQTk2N
BINANCE_TESTNET_API_SECRET=egnT7CWQaJyO2YGlnl2zCWaWPVHY9UFZxI0p9kDkVGmTzQQ0DYQa8p6hULtmuRvY

# ============================================================================
# ML MODEL CONFIGURATION
# ============================================================================
ML_MODELS_ROOT=./models
ML_DATA_DIR=./data/ml/raw

# Solo usar modelos 15m
ML_USE_15M_ONLY=true
ML_DEFAULT_TIMEFRAME=15m
ENTRY_TIMEFRAME=15m
ML_EXTRA_TIMEFRAMES=

ML_HISTORY_DAYS=180
ML_TRAIN_HORIZON=12
ML_TRAIN_TARGET_RETURN=0.002

# ML Filters
ML_MIN_ATR=5
ML_MAX_EXT_PCT=0.015
ML_MAX_RSI=70
ML_MIN_RSI=30
EOF

# Reemplazar archivo original
mv "${ENV_FILE}.tmp" "$ENV_FILE"

echo "✅ .env actualizado con:"
echo "   - SYMBOLS: ETHUSDT, XRPUSDT, LTCUSDT (solo estos 3)"
echo "   - ML_USE_15M_ONLY=true"
echo "   - Todos los demás símbolos comentados"
echo ""
echo "📁 Backup creado: ${ENV_FILE}.backup_*"
