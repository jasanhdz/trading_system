#!/bin/bash
# Deploy Phantom V9
# 1. Builds TS Bot
# 2. Reconfigures PM2 for Phantom V9 Service and Collector

echo "🚀 Deploying Phantom V9..."

# 1. Build TS Bot
echo "🔨 Building TypeScript Bot..."
cd binance-futures-bot-ts
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
cd ..

# 2. Configure PM2
echo "⚙️ Configuring PM2..."

# Stop everything
pm2 stop all

# Delete old processes (ignore errors)
pm2 delete 02-Data-Collector 2>/dev/null
pm2 delete 03-ML-Service-V2 2>/dev/null
pm2 delete 02-Phantom-Collector 2>/dev/null
pm2 delete 03-Phantom-Service 2>/dev/null

# Start New Collector
echo "Starting Phantom Collector..."
pm2 start scripts/phantom_v9/phantom_collector.py --name "02-Phantom-Collector" --interpreter /home/jasan/Develop/trading_system/.venv_cuda/bin/python3

# Start New Service
echo "Starting Phantom Service..."
pm2 start scripts/phantom_v9/phantom_v9_service.py --name "03-Phantom-Service" --interpreter /home/jasan/Develop/trading_system/.venv_cuda/bin/python3

# Restart Trading Bot (ensure it picks up new build)
echo "Restarting Trading Bot..."
pm2 restart 01-Trading-Bot

# Save PM2 list
pm2 save

echo "✅ Deployment Complete!"
pm2 list
