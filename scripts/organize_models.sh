#!/bin/bash
# Script para organizar modelos: mantener TOP 3, deprecar el resto

set -e  # Salir si hay errores

echo "🗂️  Organizando modelos para re-entrenamiento..."
echo ""

# Crear carpeta deprecated
DEPRECATED_DIR="models/deprecated_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPRECATED_DIR"

echo "📁 Carpeta de respaldo creada: $DEPRECATED_DIR"
echo ""

# Modelos TOP que se quedan (no tocar)
declare -a TOP_MODELS=(
    "ETHUSDT/15m"
    "XRPUSDT/15m"
    "LTCUSDT/15m"
)

echo "✅ MODELOS TOP (No se mueven):"
for model in "${TOP_MODELS[@]}"; do
    echo "   - $model"
done
echo ""

# Mover todos los demás modelos a deprecated
echo "🔄 Moviendo modelos a deprecated..."
echo ""

cd models/advanced

for symbol_dir in */; do
    symbol="${symbol_dir%/}"
    
    for timeframe_dir in "$symbol_dir"*/; do
        timeframe="${timeframe_dir%/}"
        timeframe="${timeframe##*/}"
        
        model_path="$symbol/$timeframe"
        
        # Verificar si es un modelo TOP
        is_top=false
        for top in "${TOP_MODELS[@]}"; do
            if [ "$model_path" == "$top" ]; then
                is_top=true
                break
            fi
        done
        
        if [ "$is_top" == false ]; then
            if [ -d "$symbol/$timeframe" ]; then
                echo "   📦 Moviendo: $model_path"
                mkdir -p "../../$DEPRECATED_DIR/$symbol"
                mv "$symbol/$timeframe" "../../$DEPRECATED_DIR/$symbol/"
            fi
        fi
    done
    
    # Eliminar directorio del símbolo si quedó vacío
    if [ -d "$symbol" ] && [ -z "$(ls -A "$symbol")" ]; then
        rmdir "$symbol"
    fi
done

cd ../..

echo ""
echo "✅ Reorganización completada!"
echo ""
echo "📊 RESUMEN:"
echo "   - Modelos TOP activos: ${#TOP_MODELS[@]}"
echo "   - Modelos deprecated: $DEPRECATED_DIR"
echo ""
echo "🚀 PRÓXIMO PASO:"
echo "   Ejecuta el orquestador para re-entrenar con nuevos parámetros:"
echo ""
echo "   .venv/bin/python scripts/dispatch_training.py \\"
echo "     --symbols BTCUSDT,BNBUSDT,BCHUSDT,SOLUSDT,TRXUSDT \\"
echo "     --timeframes 5m,15m \\"
echo "     --target-return 0.005 \\"
echo "     --prediction-horizon 6"
echo ""
