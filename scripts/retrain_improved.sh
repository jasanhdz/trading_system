#!/bin/bash
# Script para re-entrenar modelos con nuevas mejoras

echo "🚀 Iniciando re-entrenamiento con nuevas mejoras"
echo "   - Nuevas features: atr_pct, volume_flow, price_location"
echo "   - Focal Loss activado"
echo "   - Target return: 0.5% (más realista)"
echo "   - Prediction horizon: 6 barras"
echo ""

# Verificar que dispatch_training.py existe
if [ ! -f "scripts/dispatch_training.py" ]; then
    echo "❌ Error: scripts/dispatch_training.py no encontrado"
    exit 1
fi

# Símbolos a re-entrenar (todos excepto los TOP que ya están bien)
SYMBOLS="BTCUSDT,BNBUSDT,BCHUSDT,SOLUSDT,TRXUSDT,LINKUSDT,ADAUSDT,DOGEUSDT"

# Timeframes
TIMEFRAMES="5m,15m"

echo "📊 Configuración:"
echo "   Símbolos: $SYMBOLS"
echo "   Timeframes: $TIMEFRAMES"
echo "   Target Return: 0.005 (0.5%)"
echo "   Prediction Horizon: 6"
echo ""

read -p "¿Continuar con el re-entrenamiento? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelado por el usuario"
    exit 1
fi

echo "🔄 Ejecutando dispatcher..."
echo ""

.venv/bin/python scripts/dispatch_training.py \
  --symbols $SYMBOLS \
  --timeframes $TIMEFRAMES \
  --target-return 0.005 \
  --prediction-horizon 6

echo ""
echo "✅ Re-entrenamiento completado!"
echo ""
echo "📊 Para analizar resultados:"
echo "   .venv/bin/python scripts/analyze_models.py"
