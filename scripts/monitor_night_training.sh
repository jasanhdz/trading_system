#!/bin/bash
# Script para monitorear el progreso del entrenamiento nocturno

echo "=" | python3 -c "print('='*80)"
echo "🌙 MONITOREO DE ENTRENAMIENTOS NOCTURNOS"
echo "=" | python3 -c "print('='*80)"
echo ""

# Verificar dispatcher
DISPATCH_PID=$(ps aux | grep "dispatch_training.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$DISPATCH_PID" ]; then
    echo "❌ Dispatcher no está corriendo"
else
    echo "✅ Dispatcher corriendo (PID: $DISPATCH_PID)"
fi

echo ""

# Contar trabajos de entrenamiento activos
TRAINER_COUNT=$(ps aux | grep "trainer.py" | grep -v grep | wc -l)
echo "🏃 Entrenamientos activos: $TRAINER_COUNT"

echo ""

# Mostrar últimas líneas del log principal
LATEST_LOG=$(ls -t logs/training_night_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "📄 Últimos mensajes del dispatcher ($LATEST_LOG):"
    echo "---"
    tail -20 "$LATEST_LOG" | grep -E "Iniciando|Completado|Error|GPU" || echo "(Sin mensajes recientes)"
    echo ""
fi

# Verificar logs individuales de entrenamiento
echo "📊 Modelos en progreso:"
echo "---"
for log in logs/training_*_*.log 2>/dev/null; do
    if [ -f "$log" ]; then
        # Extraer símbolo y timeframe del nombre del archivo
        basename "$log" | grep -oP "training_\K[^_]+_[0-9]+m" | head -5
    fi
done | sort -u | while read model; do
    echo "  ✓ $model"
done

echo ""

# Verificar uso de GPU
echo "🎮 Estado de GPUs:"
echo "---"
echo "AMD GPUs:"
rocm-smi --showuse 2>/dev/null | grep -A 1 "GPU" | head -12 || echo "  (rocm-smi no disponible)"

echo ""
echo "NVIDIA GPUs:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi no disponible)"

echo ""
echo "=" | python3 -c "print('='*80)"
echo "💡 Para ver logs en tiempo real:"
echo "   tail -f $(ls -t logs/training_night_*.log | head -1)"
echo ""
echo "💡 Para ver log de un modelo específico:"
echo "   tail -f logs/training_SYMBOL_TIMEFRAME.log"
echo "=" | python3 -c "print('='*80)"
