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
PORT = 8001

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
        if self.path == '/ml-v2/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                
                symbol = data.get('symbol', SYMBOL)
                
                # 1. Fetch Data (Real-Time from Binance)
                # We need enough history for indicators (e.g. EMA 200). 1000 candles is safe.
                import ccxt
                exchange = ccxt.binance({'enableRateLimit': True})
                
                # Map symbol to CCXT format (ETHUSDT -> ETH/USDT)
                ccxt_symbol = symbol
                if "/" not in ccxt_symbol and ccxt_symbol.endswith("USDT"):
                    ccxt_symbol = f"{ccxt_symbol[:-4]}/USDT"
                
                # Fetch OHLCV
                ohlcv = exchange.fetch_ohlcv(ccxt_symbol, '5m', limit=1000)
                
                if not ohlcv:
                    raise ValueError("No data received from Binance")
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # 2. Calculate DNA
                df = calculate_phantom_dna(df)
                row = df.iloc[-1] # Latest candle
                
                # 3. Check Setup (Filter)
                near_resistance = abs(row['dist_ema_20']) < 0.005 
                is_tired = row['staleness'] > 15
                is_volatile = abs(row['vol_z']) > 0.2
                is_rejection = (row['close'] < row['open']) or (row['is_fakeout'] == 1)
                
                is_setup = near_resistance and is_tired and is_volatile and is_rejection
                
                # 4. Inference (Always run to get probability)
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
                    confidence = torch.softmax(q_values, dim=1)[0][1].item()
                    short_prob = confidence

                # DEBUG: Log setup conditions and probability
                # logger.info(f"SETUP CHECK: setup={is_setup} prob={short_prob:.4f} | res={near_resistance} tired={is_tired} vol={is_volatile} rej={is_rejection}")

                features = {
                    'cvd_slope': float(row['cvd_slope']),
                    'cvd_z': float(row['vol_z']),
                    'weakness': float(row['weakness_score']),
                    'volatility_z': float(row['vol_z']),
                    # Add setup flags for TS logging/filtering
                    'is_setup': 1.0 if is_setup else 0.0,
                    'near_resistance': 1.0 if near_resistance else 0.0,
                    'is_tired': 1.0 if is_tired else 0.0,
                    'is_volatile': 1.0 if is_volatile else 0.0,
                    'is_rejection': 1.0 if is_rejection else 0.0
                }

                # Construct V2-compatible response
                response = {
                    'symbol': symbol,
                    'long_prob': 0.0, # Phantom V9 is Short-Only
                    'short_prob': short_prob,
                    'neutral_prob': 1.0 - short_prob,
                    'consensus_level': 0, # Legacy
                    'meta_verdict': 'SHORT' if (short_prob > 0.5 and is_setup) else 'NEUTRAL', # Only verdict SHORT if setup is good
                    'features': features
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
