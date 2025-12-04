#!/bin/bash

LOG_FILE="/home/jasan/Develop/trading_system/logs/auto_reboot.log"

echo "🔄 Iniciando monitor de entrenamiento..." >> $LOG_FILE
echo "📅 Fecha: $(date)" >> $LOG_FILE

# Esperar a que terminen los entrenamientos
while pgrep -f "scripts/train_production_ready.py" > /dev/null; do
    echo "⏳ Entrenamiento en curso... $(date)" >> $LOG_FILE
    sleep 300 # Verificar cada 5 minutos
done

echo "✅ Entrenamiento finalizado. Reiniciando sistema en 1 minuto..." >> $LOG_FILE
echo "📅 Fecha: $(date)" >> $LOG_FILE

# Esperar un poco por si acaso
sleep 60

# Reiniciar
echo "hasanazael" | sudo -S reboot
