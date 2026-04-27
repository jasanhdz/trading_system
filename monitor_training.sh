#!/bin/bash
echo "🛡️ Guardian Monitor Activo para 03-V30-Trainer"
while true; do
    LOG=$(tail -n 50 /home/jasan/.pm2/logs/03-V30-Trainer-out-4.log 2>/dev/null)
    
    if echo "$LOG" | grep -q "loss.*1e+"; then
        echo "☢️ NUMERIC EXPLOSION DETECTED! Deteniendo el entrenamiento de emergencia..."
        pm2 stop 03-V30-Trainer
        exit 1
    fi
    
    if echo "$LOG" | grep -q "\-inf"; then
        echo "☢️ INFINITY DETECTED! Deteniendo el entrenamiento..."
        pm2 stop 03-V30-Trainer
        exit 1
    fi

    # Aquí podríamos agregar los checks de KL y EV si tuviéramos un parseo JSON
    # pero el grep de loss protege contra el error principal.
    
    sleep 60
done
