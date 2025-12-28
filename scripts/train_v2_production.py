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
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Importar nuestros modelos
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV2")

# Config
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SYMBOL = "ADA/USDT:USDT" # Entrenamos con ADA como base (luego se puede hacer multi-symbol)
SEQ_LEN = 12
PREDICT_HORIZON = 5

def load_data_from_db(symbol):
    logger.info(f"📥 Loading data for {symbol}...")
    conn = sqlite3.connect(DB_PATH)
    
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price, 
        o.micro_price,
        o.bid_depth_20 as bid_depth, 
        o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_5,
        o.obi_10,
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

def get_all_symbols():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM derivatives_data")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def train_model_for_symbol(symbol):
    # Sanitize symbol for folder name (ADA/USDT:USDT -> ADAUSDT)
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    symbol_dir = MODELS_DIR / clean_symbol
    
    if symbol_dir.exists():
        shutil.rmtree(symbol_dir)
    symbol_dir.mkdir(parents=True)
    
    logger.info(f"🚀 Starting training for {symbol} -> {symbol_dir}")
    
    # 2. Load Data
    df = load_data_from_db(symbol)
    if len(df) < 100:
        logger.warning(f"⚠️ Not enough data for {symbol} ({len(df)} rows). Skipping.")
        return

    # 3. Feature Engineering
    # Targets
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    threshold = 0.001
    conditions = [
        (df['return_5m'] < -threshold),
        (df['return_5m'] > threshold)
    ]
    choices = [0, 2] # 0: Short, 2: Long
    df['label'] = np.select(conditions, choices, default=1)
    
    # Create derived features
    df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
    
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 
        'obi_5', 'obi_10', 'obi',
        'micro_price',
        'funding_rate', 'open_interest',
        'taker_buy_vol', 'taker_sell_vol',
        'buy_sell_ratio', 'depth_imbalance'
    ]
    
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    joblib.dump(scaler, symbol_dir / "scaler.pkl")
    # Save feature names
    with open(symbol_dir / "features.json", 'w') as f:
        json.dump(feature_cols, f)
        
    # 5. Prepare Sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        Xs.append(X_scaled[i:(i + SEQ_LEN)])
        ys.append(y[i + SEQ_LEN])
    
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    
    if len(X_seq) < 10:
        logger.warning(f"⚠️ Not enough sequences for {symbol}. Skipping.")
        return

    # Convert to Tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_seq))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    input_dim = len(feature_cols)
    
    # --- MODEL 1: LSTM ---
    logger.info(f"   🏋️ Training LSTM for {clean_symbol}...")
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    lstm.train()
    for epoch in range(10):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = lstm(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(lstm.state_dict(), symbol_dir / "lstm.pt")
    with open(symbol_dir / "lstm_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'hidden_dim': 64, 'lstm_layers': 2, 'dropout': 0.2, 'num_classes': 3}}, f)
        
    # --- MODEL 2: TCN ---
    logger.info(f"   🏋️ Training TCN for {clean_symbol}...")
    tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64], kernel_size=3, num_classes=3).to(device)
    optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)
    
    tcn.train()
    for epoch in range(10):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = tcn(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(tcn.state_dict(), symbol_dir / "tcn.pt")
    with open(symbol_dir / "tcn_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'num_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.2}}, f)
        
    # --- MODEL 3: XGBoost ---
    logger.info(f"   🏋️ Training XGBoost for {clean_symbol}...")
    X_flat = X_seq[:, -1, :] # Last step
    xgb = XGBoostTradingModel(use_gpu=(device=="cuda"))
    xgb.train(X_flat, y_seq, X_flat, y_seq)
    xgb.save(str(symbol_dir / "xgboost.joblib"))
    with open(symbol_dir / "xgboost_config.json", 'w') as f:
        json.dump({}, f)
        
    # --- MODEL 4: Transformer ---
    logger.info(f"   🏋️ Training Transformer for {clean_symbol}...")
    transformer = TradingTransformer(
        input_dim=input_dim, 
        d_model=64, 
        nhead=4, 
        num_layers=2,
        num_classes=3
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005)
    
    transformer.train()
    for epoch in range(10):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = transformer(X_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            
    torch.save(transformer.state_dict(), symbol_dir / "transformer.pt")
    with open(symbol_dir / "transformer_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1}}, f)
        
    logger.info(f"✅ Models for {clean_symbol} saved (LSTM, TCN, XGBoost, Transformer).")

def train_production():
    symbols = get_all_symbols()
    logger.info(f"Found {len(symbols)} symbols in DB: {symbols}")
    
    for symbol in symbols:
        try:
            train_model_for_symbol(symbol)
        except Exception as e:
            logger.error(f"❌ Failed training {symbol}: {e}")
            
    logger.info("🎉 All training tasks completed.")

if __name__ == "__main__":
    train_production()
