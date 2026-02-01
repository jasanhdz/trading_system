#!/usr/bin/env python3
"""
Phantom V11 Inference Service (Twin Specialist)
Exposes the "Sniper" model via HTTP for the TypeScript Bot.
Logic: 
1. Fetch 5m candles from Binance.
2. Calculate partial DNA + Regime Metrics (Hurst, Slope, RSI).
3. Check Regime Router V2 (Gatekeeper).
4. If ALLOW -> Run Inference -> Return Probability.
5. If DENY -> Return 0.0 Probability (Idle).
"""
import pandas as pd
import numpy as np
import torch
import sys
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from scipy.stats import linregress

# Fix path to include project root
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import Project Modules
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna
from scripts.phantom_twin_v9.smart_money_markdown.train_specialist import PhantomNet
from scripts.phantom_twin_v9.smart_money_markdown.regime_router_v2 import check_regime

# Config
SYMBOL = "ETH/USDT"
MODEL_PATH = ROOT_DIR / "models/phantom_v11_twin/phantom_v11_final.pth"
PORT = 8001 

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhantomV11Service")

# Global State
model = None
device = None

# --- HELPER FUNCTIONS (For Regime Router Inputs) ---
def calculate_slope(series, window=90):
    if len(series) < window: return 0.0
    y = series.values[-window:]
    x = np.arange(window)
    slope, _, _, _, _ = linregress(x, y)
    return slope

def calculate_hurst(series, window=100):
    """
    Robust Hurst Calculation (R/S Analysis)
    """
    if len(series) < window: return 0.5
    
    # Use log-log plot of R/S vs n
    # Simplified R/S for speed (Scalar implementation)
    # We want a sliding window Hurst? Or single Hurst of last N?
    # Router expects 'hurst' in the row. Let's calculate for the last window.
    
    X = series.values[-window:]
    
    # Calculate R/S for this window
    # To get a valid Hurst we need multiple sub-windows.
    # Approximation: standard RS
    mean = np.mean(X)
    Y = X - mean
    Z = np.cumsum(Y)
    R = np.max(Z) - np.min(Z)
    S = np.std(X)
    if S == 0: return 0.5
    
    # H = log(R/S) / log(N) (Basic estimation)
    H = np.log(R/S) / np.log(window/2) # Scaling factor adjustments usually needed
    
    # Cap between 0 and 1
    return max(0.0, min(1.0, H))

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    """
    Combines Standard Phantom DNA with Twin/Router Metrics.
    """
    # 1. Standard DNA (Vol Z, Weakness, etc.)
    df = calculate_phantom_dna(df)
    
    # 2. Router Metrics (Hurst, RSI, Macro Slope)
    # Slope (Macro - 90 candles ~ 7.5 hours)
    slope_window = 90
    df['slope'] = df['close'].rolling(window=slope_window).apply(lambda x: calculate_slope(x, slope_window), raw=False)
    
    # Hurst (Macro - 100 candles)
    # Note: Rolling Hurst is expensive. We only need the LAST row for inference.
    # We will compute it just for the latest state to save time.
    
    # RSI (Safety)
    df['rsi'] = calculate_rsi(df['close'])
    
    return df

def initialize():
    global model, device
    logger.info("👻 Initializing Phantom V11 (Sniper) Service...")
    
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
                import ccxt
                exchange = ccxt.binance({'enableRateLimit': True})
                
                # Fetch more data for Hurst/Slope logic (need ~200-300 candles)
                ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
                
                try:
                    ohlcv = exchange.fetch_ohlcv(ccxt_symbol, '5m', limit=500)
                except Exception as e:
                    # Fallback to public API if API keys missing/error (safe for read-only)
                    exchange = ccxt.binance()
                    ohlcv = exchange.fetch_ohlcv(ccxt_symbol, '5m', limit=500)
                
                if not ohlcv:
                    raise ValueError("No data received from Binance")
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # 2. Prepare Features
                df = prepare_features(df)
                
                # Get Latest Row
                current_row = df.iloc[-1].copy()
                
                # Compute Hurst manual for the last window (faster than rolling)
                h_val = calculate_hurst(df['close'], window=100)
                current_row['hurst'] = h_val
                
                # 3. ASK THE ROUTER
                active, reason = check_regime(current_row)
                
                logger.info(f"ROUTER CHECK: {reason} | Active={active}")
                
                short_prob = 0.0
                meta_verdict = "NEUTRAL"
                
                if active:
                    # 4. RUN INFERENCE (Only if Active)
                    state = np.array([
                        current_row.get('velocity', 0) / current_row['close'] * 10000,
                        current_row.get('acceleration', 0) / current_row['close'] * 10000,
                        current_row.get('cvd_slope', 0) / 1e6,
                        current_row.get('bear_trap', 0),
                        current_row.get('vol_z', 0),
                        current_row.get('volume_ratio', 1),
                        current_row.get('dist_ema_20', 0) * 100,
                        current_row.get('dist_ema_200', 0) * 100,
                        current_row.get('staleness', 0) / 50.0,
                        current_row.get('weakness_score', 0),
                        current_row.get('is_fakeout', 0),
                        0.0 # reserved
                    ], dtype=np.float32)
                    
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = model(state_t)
                        confidence = torch.softmax(q_values, dim=1)[0][1].item()
                        short_prob = confidence
                        
                        # Inference Decision
                        action = torch.argmax(q_values).item()
                        if action == 1 and short_prob > 0.50:
                            meta_verdict = "SHORT"
                            logger.info(f"🔥 FIRE SIGNAL! Prob: {short_prob:.2f}")
                        else:
                            meta_verdict = "NEUTRAL"
                            logger.info(f"OBSERVATION: Model sees nothing. Prob: {short_prob:.2f}")
                
                # DEBUG Feature payload for TS
                # Note: TS Bot expects specific keys. We can just pass standard ones.
                features_payload = {
                    'cvd_slope': float(current_row.get('cvd_slope', 0)),
                    'volatility_z': float(current_row.get('vol_z', 0)),
                    'active': 1.0 if active else 0.0,
                    'router_reason': str(reason)
                }

                # Construct Response
                response = {
                    'symbol': symbol,
                    'long_prob': 0.0,
                    'short_prob': short_prob,
                    'neutral_prob': 1.0 - short_prob,
                    'consensus_level': 0, 
                    'meta_verdict': meta_verdict,
                    'features': features_payload
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Error: {e}")
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
    logger.info(f"🚀 Phantom V11 (Sniper) Service running on port {PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
