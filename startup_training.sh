#!/bin/bash
# Script para iniciar entrenamiento automáticamente al arrancar

LOG_FILE="/home/jasan/Develop/trading_system/logs/startup_training.log"
cd /home/jasan/Develop/trading_system

echo "🚀 Inicio de sistema detectado: $(date)" >> $LOG_FILE

# Esperar a que el sistema cargue drivers
sleep 60

# Cargar módulo amdgpu explícitamente por si acaso
echo "hasanazael" | sudo -S modprobe amdgpu
sleep 10

# Verificar GPUs
echo "🔍 GPUs detectadas:" >> $LOG_FILE
rocm-smi >> $LOG_FILE 2>&1
nvidia-smi >> $LOG_FILE 2>&1

# Lanzar entrenamiento
echo "🚀 Lanzando dispatcher..." >> $LOG_FILE
./launch_training.sh >> $LOG_FILE 2>&1

echo "✅ Dispatcher lanzado." >> $LOG_FILE
