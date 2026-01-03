import argparse
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
from sklearn.preprocessing import RobustScaler  # CAMBIO v3.0: RobustScaler es más resistente a outliers de crypto

# Importar nuestros modelos
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV2")

# Config
VERSION = "v2.1"  # Consejo de Sabios v2.1 con Meta-Features
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SYMBOL = "ADA/USDT:USDT" # Entrenamos con ADA como base (luego se puede hacer multi-symbol)
SEQ_LEN = 12
PREDICT_HORIZON = 15  # CAMBIO v3.0: 5 -> 15 minutos (menos ruido, más moonbag-friendly)

# ═══════════════════════════════════════════════════════════════════════════
# CONSEJO DE SABIOS v2.1: Meta-Features para darle "memoria" a XGBoost
# ═══════════════════════════════════════════════════════════════════════════
def add_robust_meta_features(df, window=12):
    """
    Genera 6 meta-features de forma segura (maneja NaNs) y rápida.
    Esto permite que XGBoost "vea" tendencias, no solo el instante actual.
    """
    df = df.copy()
    
    # 1. Rolling Calculations para OBI (Order Book Imbalance)
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # 2. Total volume proxy
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    
    # 3. Volume Trend (con protección división por cero)
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    
    # 4. Price Slope (Optimizado - 100x más rápido que polyfit)
    # Pendiente = (PrecioActual - PrecioHace12ticks) / 12
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NUEVO v2.2: CVD (Cumulative Volume Delta) - El "Medidor de Fuerza"
    # ═══════════════════════════════════════════════════════════════════════════
    # Positivo = Dominio Comprador, Negativo = Dominio Vendedor
    df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window).sum()
    df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # NUEVO v2.2: Volatilidad del Precio - El "Termómetro de Histeria"
    # ═══════════════════════════════════════════════════════════════════════════
    df['std_price_12'] = df['price'].rolling(window).std()
    df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
    
    # ⚠️ PELIGRO MITIGADO: Eliminar NaNs creados por rolling/shift
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len - final_len > 0:
        logger.info(f"⚠️ Meta-Features: Removidas {initial_len - final_len} filas con NaNs iniciales.")
        
    return df

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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSEJO DE SABIOS v2.1: Agregar Meta-Features
    # ═══════════════════════════════════════════════════════════════════════════
    df = add_robust_meta_features(df, window=SEQ_LEN)
    
    # Features base (13) + Meta-Features (10) = 23 features totales
    base_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 
        'obi_5', 'obi_10', 'obi',
        'micro_price',
        'funding_rate', 'open_interest',
        'taker_buy_vol', 'taker_sell_vol',
        'buy_sell_ratio', 'depth_imbalance'
    ]
    
    meta_cols = [
        'mean_obi_12', 'max_obi_12', 'std_obi_12',
        'slope_price_12', 'mean_volume_12', 'volume_trend',
        'cvd_12', 'cvd_norm_12',           # NUEVO: CVD
        'std_price_12', 'volatility_ratio'  # NUEVO: Volatilidad
    ]
    
    feature_cols = base_cols + meta_cols
    logger.info(f"🧙 Consejo de Sabios {VERSION}: Entrenando con {len(feature_cols)} features (v2.2 +CVD +Vol)")
    
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # 4. Scaling (v3.0: RobustScaler para crypto outliers)
    scaler = RobustScaler()
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
    logger.info(f"   Using device: {device}")
    if device == "cuda":
        logger.info(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 6. Time Series Split (80/20) - NO SHUFFLE!
    split_idx = int(len(X_seq) * 0.8)
    
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    logger.info(f"   Split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Create Loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Shuffle only train
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_dim = len(feature_cols)
    
    # --- MODEL 1: LSTM ---
    logger.info(f"   🏋️ Training LSTM for {clean_symbol}...")
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    lstm.train()
    for epoch in range(50):  # CAMBIO v3.0: 10 -> 50 épocas para convergencia real
        for X_b, y_b in train_loader:
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
    for epoch in range(50):  # CAMBIO v3.0: 10 -> 50 épocas para convergencia real
        for X_b, y_b in train_loader:
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
    
    # Flatten for XGBoost (last step only)
    X_train_flat = X_train[:, -1, :]
    X_val_flat = X_val[:, -1, :]
    
    xgb = XGBoostTradingModel(use_gpu=(device=="cuda"))
    xgb.train(X_train_flat, y_train, X_val_flat, y_val)
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
    for epoch in range(50):  # CAMBIO v3.0: 10 -> 50 épocas para convergencia real
        for X_b, y_b in train_loader:
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
    parser = argparse.ArgumentParser(description='Train Consejo de Sabios v2.1')
    parser.add_argument('--symbol', type=str, help='Specific symbol to train (e.g., BNBUSDT)')
    parser.add_argument('--shard_id', type=int, default=0, help='Shard ID for parallel training (0-based)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards')
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
        logger.info(f"🎯 Training SINGLE symbol: {symbols}")
    else:
        symbols = get_all_symbols()
        logger.info(f"Found {len(symbols)} symbols in DB: {symbols}")
        
    # Sharding Logic
    if args.num_shards > 1:
        all_count = len(symbols)
        symbols = [s for i, s in enumerate(symbols) if i % args.num_shards == args.shard_id]
        logger.info(f"🧩 Worker {args.shard_id}/{args.num_shards} processing {len(symbols)}/{all_count} symbols")
    
    for symbol in symbols:
        try:
            train_model_for_symbol(symbol)
        except Exception as e:
            logger.error(f"❌ Failed training {symbol}: {e}")
            
    logger.info("🎉 All training tasks completed.")

if __name__ == "__main__":
    train_production()
