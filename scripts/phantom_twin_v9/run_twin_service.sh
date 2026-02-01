#!/bin/bash
# PHANTOM TWIN V11: SERVICE LAUNCHER
# Switches from Production V9 to Sandbox V11 Twin Service.
# WARNING: This temporarily stops the V9 service on Port 8001.

PORT=8001

echo "🛑 Stopping any service on Port $PORT..."
fuser -k $PORT/tcp

echo "⏳ Waiting for port release..."
sleep 2

echo "🚀 Launching Phantom V11 (Twin Sniper)..."
source /home/jasan/Develop/trading_system/.venv_rocm62/bin/activate
python3 /home/jasan/Develop/trading_system/scripts/phantom_twin_v9/smart_money_markdown/ml_service_v11.py
