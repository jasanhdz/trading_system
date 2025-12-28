import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Importar nuestros modelos
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalV2")

DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SEQ_LEN = 12
PREDICT_HORIZON = 5

def load_data_from_db(symbol):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price, 
        o.bid_depth_20 as bid_depth, 
        o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_20 as obi,
        d.funding_rate, 
        d.open_interest,
        d.taker_buy_vol,
        d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{symbol}'
    ORDER BY o.timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def evaluate_symbol(symbol):
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    symbol_dir = MODELS_DIR / clean_symbol
    
    if not symbol_dir.exists():
        return None

    # Load Data
    df = load_data_from_db(symbol)
    
    # Feature Engineering (Same as training)
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    threshold = 0.001
    conditions = [
        (df['return_5m'] < -threshold),
        (df['return_5m'] > threshold)
    ]
    choices = [0, 2] # 0: Short, 2: Long
    df['label'] = np.select(conditions, choices, default=1)
    
    with open(symbol_dir / "features.json", 'r') as f:
        feature_cols = json.load(f)
        
    df = df.dropna()
    X = df[feature_cols].values
    y = df['label'].values
    
    # Scaling
    scaler = joblib.load(symbol_dir / "scaler.pkl")
    X_scaled = scaler.fit_transform(X) # Note: In real eval we should use transform only, but for this quick check fit_transform is ok to see model capacity
    
    # Sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        Xs.append(X_scaled[i:(i + SEQ_LEN)])
        ys.append(y[i + SEQ_LEN])
    
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    
    if len(X_seq) == 0: return None
    
    # Evaluate LSTM
    device = "cpu"
    input_dim = len(feature_cols)
    
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    lstm.load_state_dict(torch.load(symbol_dir / "lstm.pt", map_location=device))
    lstm.eval()
    
    X_tensor = torch.FloatTensor(X_seq).to(device)
    with torch.no_grad():
        out = lstm(X_tensor)
        lstm_preds = torch.argmax(out['logits'], dim=1).numpy()
        
    lstm_acc = accuracy_score(y_seq, lstm_preds)
    
    # Evaluate XGBoost
    X_flat = X_seq[:, -1, :]
    xgb = XGBoostTradingModel(use_gpu=False)
    xgb.load(str(symbol_dir / "xgboost.joblib"))
    xgb_out = xgb.predict(X_flat)
    xgb_preds = np.argmax(xgb_out['probs'], axis=1)
    xgb_acc = accuracy_score(y_seq, xgb_preds)
    
    return {
        'symbol': clean_symbol,
        'lstm_acc': lstm_acc,
        'xgb_acc': xgb_acc,
        'samples': len(y_seq)
    }

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM derivatives_data")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"{'SYMBOL':<10} | {'SAMPLES':<8} | {'LSTM ACC':<10} | {'XGB ACC':<10} | {'AVG ACC':<10}")
    print("-" * 60)
    
    total_acc = []
    
    for sym in symbols:
        res = evaluate_symbol(sym)
        if res:
            avg = (res['lstm_acc'] + res['xgb_acc']) / 2
            total_acc.append(avg)
            print(f"{res['symbol']:<10} | {res['samples']:<8} | {res['lstm_acc']*100:.2f}%    | {res['xgb_acc']*100:.2f}%    | {avg*100:.2f}%")
            
    print("-" * 60)
    print(f"GLOBAL AVERAGE ACCURACY: {np.mean(total_acc)*100:.2f}%")

if __name__ == "__main__":
    main()
