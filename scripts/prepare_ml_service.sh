#!/bin/bash
# Script para preparar modelos para el servicio Flask ML

echo "🔗 Preparando modelos para servicio ML..."
echo ""

# Función para preparar un modelo
prepare_model() {
    local symbol=$1
    local timeframe=$2
    
    echo "📦 Preparando $symbol $timeframe..."
    
    BASE_DIR="models/advanced/$symbol/$timeframe"
    
    if [ ! -d "$BASE_DIR" ]; then
        echo "   ❌ Directorio no existe: $BASE_DIR"
        return 1
    fi
    
    # 1. Link/Copy mejor modelo como model.pt
    # Usar fold 5 (el más reciente, generalmente el mejor)
    if [ -f "$BASE_DIR/best_model_fold5.pt" ]; then
        echo "   🔗 Creando model.pt desde best_model_fold5.pt"
        ln -sf best_model_fold5.pt "$BASE_DIR/model.pt"
    elif [ -f "$BASE_DIR/best_model_fold1.pt" ]; then
        echo "   🔗 Creando model.pt desde best_model_fold1.pt"
        ln -sf best_model_fold1.pt "$BASE_DIR/model.pt"
    else
        echo "   ❌ No se encontró ningún modelo (.pt)"
        return 1
    fi
    
    # 2. Verificar scaler.pkl existe
    if [ ! -f "$BASE_DIR/scaler.pkl" ]; then
        echo "   ❌ Falta scaler.pkl"
        return 1
    else
        echo "   ✅ scaler.pkl encontrado"
    fi
    
    # 3. Crear meta.json desde production_training_results.json
    if [ -f "$BASE_DIR/production_training_results.json" ]; then
        echo "   🔗 Creando meta.json desde production_training_results.json"
        ln -sf production_training_results.json "$BASE_DIR/meta.json"
    else
        echo "   ⚠️  No se encontró production_training_results.json, creando meta.json básico"
        cat > "$BASE_DIR/meta.json" << EOF
{
    "symbol": "$symbol",
    "timeframe": "$timeframe",
    "model_type": "advanced",
    "features": 101,
    "created_from": "legacy_training"
}
EOF
    fi
    
    echo "   ✅ $symbol $timeframe listo"
    echo ""
}

# Preparar modelos Top 3
prepare_model "ETHUSDT" "15m"
prepare_model "XRPUSDT" "15m"
prepare_model "LTCUSDT" "15m"

echo "✅ Modelos preparados para servicio ML"
echo ""
echo "Archivos creados:"
echo "   - model.pt (symlink a best_model_fold5.pt)"
echo "   - meta.json (symlink a production_training_results.json)"
echo "   - scaler.pkl (ya existía)"
echo ""
echo "🔄 Reinicia el servicio Flask:"
echo "   pkill -f 'ml_probability_service' && sleep 2 && python binance_futures_bot_py/src/services/ml_probability_service.py &"
