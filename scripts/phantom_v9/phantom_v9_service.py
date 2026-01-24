#!/usr/bin/env python3
"""
Phantom V9 Production Inference Service
Exposes the Phantom V9 model via HTTP for the TypeScript Bot.
"""
import pandas as pd
import numpy as np
import torch
import sys
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna
from scripts.phantom_v9.train_phantom_dqn import PhantomNet

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
MODEL_PATH = "models/phantom_v9/phantom_v9_best.pth"
PORT = 5000

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/phantom_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhantomService")

# Global State
model = None
device = None

def initialize():
    global model, device
    logger.info("👻 Initializing Phantom V9 Service...")
    
    # Load Model
    try:
        device = torch.device("cpu")
        model = PhantomNet(input_dim=12, output_dim=2).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                
                # Expecting OHLCV data for the last N candles to calculate features
                # Or just the features directly?
                # The TS bot sends a request. In the backtest we sent 'timestamp' and looked up data.
                # In production, the bot should probably send the features OR the service fetches live data.
                # Given the architecture, the service should probably fetch the latest data from DB or API.
                # BUT, the bot is running live.
                
                # Let's assume the bot sends the symbol and we fetch the latest data from the DB 
                # (which should be updated by the bot or another process).
                # OR, simpler: The bot sends the latest candle? No, we need history for features.
                
                # DECISION: The service will fetch the latest 500 candles from the DB to calculate features.
                # This assumes the DB is being updated by the bot (BinanceAdapter).
                
                symbol = data.get('symbol', SYMBOL)
                
                # 1. Fetch Data
                db = DatabaseManager(DB_URL)
                df = db.get_ohlcv_data(symbol, '5m', limit=1000)
                
                if df.empty:
                    raise ValueError("No data found in DB")
                    
                if 'timestamp' not in df.columns: df = df.reset_index()
                
                # 2. Calculate DNA
                df = calculate_phantom_dna(df)
                row = df.iloc[-1] # Latest candle
                
                # 3. Check Setup (Filter)
                # Match logic from detect_phantom_tops.py
                near_resistance = abs(row['dist_ema_20']) < 0.005 
                is_tired = row['staleness'] > 15
                is_volatile = abs(row['vol_z']) > 0.2
                is_rejection = (row['close'] < row['open']) or (row['is_fakeout'] == 1)
                
                is_setup = near_resistance and is_tired and is_volatile and is_rejection
                
                response = {}
                
                if not is_setup:
                    response = {
                        'action': 0, 
                        'confidence': 0.0, 
                        'reason': 'Not a setup',
                        'features': {
                            'dist_ema_20': float(row['dist_ema_20']),
                            'staleness': float(row['staleness']),
                            'vol_z': float(row['vol_z'])
                        }
                    }
                else:
                    # 4. Inference
                    state = np.array([
                        row['velocity'] / row['close'] * 10000,
                        row['acceleration'] / row['close'] * 10000,
                        row['cvd_slope'] / 1e6,
                        row['bear_trap'],
                        row['vol_z'],
                        row['volume_ratio'],
                        row['dist_ema_20'] * 100,
                        row['dist_ema_200'] * 100,
                        row['staleness'] / 50.0,
                        row['weakness_score'],
                        row['is_fakeout'],
                        0.0 # reserved
                    ], dtype=np.float32)
                    
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = model(state_t)
                        action = torch.argmax(q_values).item()
                        confidence = torch.softmax(q_values, dim=1)[0][1].item()
                    
                    response = {
                        'action': action,
                        'confidence': confidence,
                        'close': float(row['close']),
                        'timestamp': int(row['timestamp']),
                        'features': {
                            'cvd_slope': float(row['cvd_slope']),
                            'cvd_z': float(row['vol_z']), # Using vol_z as proxy for now
                            'weakness': float(row['weakness_score'])
                        }
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run():
    initialize()
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    logger.info(f"🚀 Phantom V9 Service running on port {PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
