import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import sqlite3
import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importar nuestros modelos
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.ensemble_manager import EnsembleManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PilotV2")

DB_PATH = "data/market_data_v2.db"
SYMBOL = "ADA/USDT:USDT" # Probaremos con ADA
SEQ_LEN = 12 # 12 minutos de historia para predecir
PREDICT_HORIZON = 5 # Predecir precio a 5 minutos

def load_data_from_db(symbol):
    logger.info(f"📥 Loading data for {symbol} from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query con JOIN
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
    
    logger.info(f"✅ Loaded {len(df)} records.")
    return df

def prepare_features_and_targets(df):
    logger.info("🛠️ Engineering features...")
    
    # 1. Targets (Futuro)
    # ¿Subirá el precio en 5 minutos?
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    # Label: 0=Baja, 1=Neutro, 2=Sube (Umbral 0.1%)
    threshold = 0.001
    conditions = [
        (df['return_5m'] < -threshold),
        (df['return_5m'] > threshold)
    ]
    choices = [0, 2] # 0: Short, 2: Long
    df['label'] = np.select(conditions, choices, default=1) # 1: Neutral
    
    # 2. Features (Normalización)
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 'obi',
        'funding_rate', 'open_interest',
        'taker_buy_vol', 'taker_sell_vol'
    ]
    
    # Drop NaNs (por el shift del target)
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['label'].values
    returns = df['return_5m'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, returns, feature_cols

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

def train_pilot():
    # 1. Load & Prep
    df = load_data_from_db(SYMBOL)
    if len(df) < 100:
        logger.error("Not enough data to train.")
        return

    X_scaled, y_raw, returns, feature_names = prepare_features_and_targets(df)
    
    # Create Sequences for LSTM
    X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LEN)
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    
    logger.info(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
    
    # Convert to Tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # 2. Train LSTM (The Historian)
    input_dim = X_train.shape[2]
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.info("🏋️ Training LSTM Pilot...")
    lstm.train()
    for epoch in range(5): # 5 Epochs rapidas
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = lstm(X_batch)
            loss = criterion(out['logits'], y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    # 3. Train XGBoost (The Accountant)
    logger.info("🏋️ Training XGBoost Pilot...")
    # XGBoost usa datos 2D (último paso de tiempo)
    X_train_flat = X_train[:, -1, :]
    X_test_flat = X_test[:, -1, :]
    
    xgb_model = XGBoostTradingModel(use_gpu=(device=="cuda"))
    xgb_model.train(X_train_flat, y_train, X_test_flat, y_test)
    
    # 4. Evaluate
    logger.info("\n📊 PILOT RESULTS:")
    
    # LSTM Eval
    lstm.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(device)
        lstm_out = lstm(test_tensor)
        lstm_preds = torch.argmax(lstm_out['logits'], dim=1).cpu().numpy()
        lstm_acc = np.mean(lstm_preds == y_test)
        logger.info(f"   LSTM Accuracy: {lstm_acc*100:.2f}%")
        
    # XGBoost Eval
    xgb_out = xgb_model.predict(X_test_flat)
    xgb_preds = np.argmax(xgb_out['probs'], axis=1)
    xgb_acc = np.mean(xgb_preds == y_test)
    logger.info(f"   XGBoost Accuracy: {xgb_acc*100:.2f}%")
    
    # Feature Importance (XGBoost)
    # Simple hack to get importance from the booster
    importance = xgb_model.model.get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("\n🔑 Key Features (What matters most?):")
    for i, (k, v) in enumerate(sorted_imp[:5]):
        # Map f0, f1... to names
        idx = int(k.replace('f', ''))
        name = feature_names[idx] if idx < len(feature_names) else k
        logger.info(f"   {i+1}. {name}: {v:.2f}")

if __name__ == "__main__":
    train_pilot()
