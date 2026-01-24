#!/usr/bin/env python3
"""
Phantom V9 Inference Server
Exposes the Phantom V9 model via HTTP for TypeScript backtesting.
"""
import pandas as pd
import numpy as np
import torch
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna, detect_eth_setups
from scripts.phantom_v9.train_phantom_dqn import PhantomNet

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
MODEL_PATH = "models/phantom_v9/phantom_v9_best.pth"
PORT = 5000

# Global State
df = None
model = None
device = None

def initialize():
    global df, model, device
    print("👻 Initializing Phantom V9 Inference Server...")
    
    # 1. Load Data
    db = DatabaseManager(DB_URL)
    print("Loading data...")
    raw_df = db.get_ohlcv_data(SYMBOL, '5m', limit=50000)
    if 'timestamp' not in raw_df.columns: raw_df = raw_df.reset_index()
    
    # Convert timestamps to int (milliseconds) to match TS input
    raw_df['timestamp'] = raw_df['timestamp'].astype(np.int64) // 10**6
    
    # 2. Calculate DNA
    print("Calculating DNA...")
    df = calculate_phantom_dna(raw_df)
    
    # 3. Load Model
    print("Loading Model...")
    device = torch.device("cpu")
    model = PhantomNet(input_dim=12, output_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("✅ Server Ready!")

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            timestamp = data.get('timestamp')
            
            if not timestamp:
                self.send_response(400)
                self.end_headers()
                return

            # Find row
            row = df[df['timestamp'] == timestamp]
            
            if row.empty:
                response = {'action': 0, 'confidence': 0.0, 'error': 'Timestamp not found'}
            else:
                row = row.iloc[0]
                
                # Check setup conditions (detect_eth_setups logic)
                # We need to replicate the filtering logic or just return 0 if not a candidate
                # But wait, the TS runner iterates ALL candles.
                # The Python backtest filters by `detect_eth_setups`.
                # So we should check if it's a valid setup here.
                
                # Match logic from detect_phantom_tops.py
                near_resistance = abs(row['dist_ema_20']) < 0.005 
                is_tired = row['staleness'] > 15
                is_volatile = abs(row['vol_z']) > 0.2
                is_rejection = (row['close'] < row['open']) or (row['is_fakeout'] == 1)
                
                is_setup = near_resistance and is_tired and is_volatile and is_rejection
                
                if not is_setup:
                     response = {'action': 0, 'confidence': 0.0, 'reason': 'Not a setup'}
                else:
                    # Create State
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
                        'close': float(row['close'])
                    }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run():
    initialize()
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Starting httpd on port {PORT}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
