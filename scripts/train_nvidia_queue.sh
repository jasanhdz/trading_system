#!/bin/bash

# Lista de modelos a entrenar en NVIDIA (secuencial)
MODELS=(
    "BTCUSDT 15m"
    "SOLUSDT 5m"
    "SOLUSDT 15m"
    "ETHUSDT 5m"
    "BNBUSDT 5m"
    "BNBUSDT 15m"
)

LOG_DIR="logs/nvidia_queue"
mkdir -p "$LOG_DIR"

echo "🚀 Iniciando cola de entrenamiento NVIDIA..."

for model in "${MODELS[@]}"; do
    read -r symbol timeframe <<< "$model"
    
    echo "----------------------------------------------------------------"
    echo "🕒 $(date): Iniciando $symbol $timeframe"
    
    # Verificar si ya existe
    if [ -f "models/advanced/$symbol/$timeframe/production_training_results.json" ]; then
        echo "⏭️  Ya existe, saltando..."
        continue
    fi
    
    # Ejecutar entrenamiento
    .venv_cuda/bin/python scripts/train_production_ready.py \
        --symbol "$symbol" \
        --timeframe "$timeframe" \
        --epochs 200 \
        --batch-size 128 \
        --target-return 0.005 \
        --prediction-horizon 6 \
        --device cuda:0 \
        > "$LOG_DIR/${symbol}_${timeframe}.log" 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ Completado con éxito"
    else
        echo "❌ Falló (ver log en $LOG_DIR/${symbol}_${timeframe}.log)"
    fi
    
    # Pequeña pausa para enfriar GPU
    sleep 5
done

echo "🎉 Cola NVIDIA completada!"
