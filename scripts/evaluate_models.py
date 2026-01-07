import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.tabular_model import XGBoostTradingModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Eval")

DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SEQ_LEN = 60
PREDICT_HORIZON = 60

def add_robust_meta_features(df, window=12):
    df = df.copy()
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window).sum()
    df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)
    df['std_price_12'] = df['price'].rolling(window).std()
    df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
    return df.dropna()

def load_data_from_db(symbol):
    conn = sqlite3.connect(DB_PATH)
    db_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT:USDT")
    query = f"""
    SELECT 
        o.timestamp, o.mid_price as price, o.micro_price,
        o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread,
        o.obi_5, o.obi_10, o.obi_20 as obi,
        d.funding_rate, d.open_interest,
        d.taker_buy_vol, d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{db_symbol}'
    ORDER BY o.timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def evaluate_symbol(symbol):
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    symbol_dir = MODELS_DIR / clean_symbol
    
    if not symbol_dir.exists():
        logger.error(f"❌ Model dir not found for {symbol}")
        return

    logger.info(f"\n🔍 Evaluating {symbol}...")

    # 1. Load Data
    df = load_data_from_db(symbol)
    if len(df) < 500:
        logger.warning("Not enough data")
        return

    # Targets
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return'] = (df['future_price'] - df['price']) / df['price']
    
    if "BTC" in symbol or "ETH" in symbol:
        THRESHOLD = 0.0015
    else:
        THRESHOLD = 0.0030
        
    conditions = [(df['return'] < -THRESHOLD), (df['return'] > THRESHOLD)]
    choices = [0, 2]
    df['label'] = np.select(conditions, choices, default=1)
    
    df = add_robust_meta_features(df, window=12)
    
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 'obi_5', 'obi_10', 'obi',
        'micro_price', 'funding_rate', 'open_interest', 'taker_buy_vol', 'taker_sell_vol',
        'mean_obi_12', 'max_obi_12', 'std_obi_12', 'slope_price_12', 
        'mean_volume_12', 'volume_trend', 'cvd_12', 'cvd_norm_12', 
        'std_price_12', 'volatility_ratio'
    ]
    
    df = df.dropna()
    
    # Use last 20% for validation (same as training split)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()
    
    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df['label'].values
    
    # Load Scaler
    scaler = joblib.load(symbol_dir / "scaler.pkl")
    X_val_scaled = scaler.transform(X_val_raw)
    
    # Sequences
    def to_sequences(data, labels, seq_len):
        Xs, ys = [], []
        for i in range(len(data) - seq_len):
            Xs.append(data[i:(i + seq_len)])
            ys.append(labels[i + seq_len])
        return np.array(Xs), np.array(ys)
        
    X_val, y_val = to_sequences(X_val_scaled, y_val_raw, SEQ_LEN)
    
    # --- EVALUATE TCN ---
    device = "cpu" # Evaluate on CPU for simplicity
    input_dim = len(feature_cols)
    
    tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64, 128], kernel_size=3, num_classes=3)
    tcn.load_state_dict(torch.load(symbol_dir / "tcn.pt", map_location=device))
    tcn.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val)
        out = tcn(X_tensor)
        probs = torch.softmax(out['logits'], dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
        
    acc_tcn = accuracy_score(y_val, preds)
    f1_tcn = f1_score(y_val, preds, average='weighted')
    
    logger.info(f"🧠 TCN Accuracy: {acc_tcn:.2%} | F1: {f1_tcn:.2f}")
    
    # --- EVALUATE XGBOOST ---
    xgb = XGBoostTradingModel(use_gpu=False)
    xgb.load(str(symbol_dir / "xgboost.joblib"))
    
    X_val_flat = X_val[:, -1, :]
    xgb_output = xgb.predict(X_val_flat)
    preds_xgb = np.argmax(xgb_output['probs'], axis=1)
    
    acc_xgb = accuracy_score(y_val, preds_xgb)
    f1_xgb = f1_score(y_val, preds_xgb, average='weighted')
    
    logger.info(f"🌲 XGB Accuracy: {acc_xgb:.2%} | F1: {f1_xgb:.2f}")
    
    # Ensemble (Simple Average)
    # XGB Probs? XGBoostTradingModel might not expose predict_proba easily in wrapper, 
    # but let's assume we just average predictions for now or just report individual.
    # Actually, let's just report individual for comparison.

if __name__ == "__main__":
    evaluate_symbol("DOGE/USDT:USDT")
    evaluate_symbol("LINK/USDT:USDT")
