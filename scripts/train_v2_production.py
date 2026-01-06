import argparse
import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Importar modelos (asumiendo que las rutas existen)
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV2_AuditFix")

# Configuración Institucional
VERSION = "v2.3_HorizonFix"
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SEQ_LEN = 60         # 60 ticks * 10s = 10 Minutos de Contexto (Input)
PREDICT_HORIZON = 60 # 60 ticks * 10s = 10 Minutos futuro (Output)

def add_robust_meta_features(df, window=12):
    """
    Feature Engineering Optimizado
    """
    df = df.copy()
    
    # Features básicas de volatilidad y tendencia
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # Volumen
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    
    # Pendiente
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    
    # CVD (Cumulative Volume Delta)
    df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window).sum()
    df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)

    # Volatilidad
    df['std_price_12'] = df['price'].rolling(window).std()
    df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
    
    return df.dropna()

def load_data_from_db(symbol):
    conn = sqlite3.connect(DB_PATH)
    # Query normalizada
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

def get_all_symbols():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM derivatives_data")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def create_sequences(data, seq_len):
    """Crea secuencias X y objetivos y para series temporales"""
    Xs, ys = [], []
    for i in range(len(data) - seq_len):
        Xs.append(data[i:(i + seq_len)])
        # El label ya está alineado en el dataframe original, tomamos el del último paso
        # Nota: 'label' debe estar en una columna separada o pasarse aparte.
        # Aquí asumimos que data contiene SOLO features y pasamos labels aparte.
        pass
    return np.array(Xs)

def train_model_for_symbol(symbol):
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    
    # 1. DEFINIR RUTAS (PRODUCCIÓN Y TEMPORAL)
    prod_dir = MODELS_DIR / clean_symbol
    temp_dir = MODELS_DIR / f"{clean_symbol}_temp"
    
    # Limpiar temp si existe de un run fallido anterior
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    logger.info(f"🛡️ Atomic Training started for {symbol} (Building in {temp_dir})...")
    
    try:
        # --- CARGA DE DATOS Y PREPARACIÓN (Igual que antes) ---
        df = load_data_from_db(symbol)
        if len(df) < 500:
            logger.warning(f"⚠️ Insufficient data ({len(df)}). Skipping.")
            shutil.rmtree(temp_dir) # Limpiar basura
            return

        # Targets (Horizonte 60 ticks = 10 min)
        df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
        df['return'] = (df['future_price'] - df['price']) / df['price']
        
        # ══════════════════════════════════════════════════════════════════
        # LÓGICA DE THRESHOLD ADAPTATIVO (NINJA v6.1)
        # ══════════════════════════════════════════════════════════════════
        if "BTC" in symbol or "ETH" in symbol:
            THRESHOLD = 0.0015
            logger.info(f"⚖️ Adaptive Threshold for Major ({symbol}): {THRESHOLD*100}%")
        else:
            THRESHOLD = 0.0030
            logger.info(f"🌪️ Adaptive Threshold for Altcoin ({symbol}): {THRESHOLD*100}%")
            
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
        
        # Split y Scaling
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df['label'].values
        X_val_raw = val_df[feature_cols].values
        y_val_raw = val_df['label'].values
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # --- GUARDAR ARTEFACTOS EN CARPETA TEMPORAL ---
        joblib.dump(scaler, temp_dir / "scaler.pkl")
        with open(temp_dir / "features.json", 'w') as f:
            json.dump(feature_cols, f)
            
        # ... (Creación de secuencias y DataLoaders igual que antes) ...
        def to_sequences(data, labels, seq_len):
            Xs, ys = [], []
            for i in range(len(data) - seq_len):
                Xs.append(data[i:(i + seq_len)])
                ys.append(labels[i + seq_len])
            return np.array(Xs), np.array(ys)
            
        X_train, y_train = to_sequences(X_train_scaled, y_train_raw, SEQ_LEN)
        X_val, y_val = to_sequences(X_val_scaled, y_val_raw, SEQ_LEN)
        
        if len(X_train) < 100:
             logger.warning("Not enough sequences.")
             shutil.rmtree(temp_dir)
             return

        # Tensores
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        input_dim = len(feature_cols)

        # --- ENTRENAMIENTO (Guardando en temp_dir) ---
        
        # 1. TCN
        logger.info(f"Training TCN (Temp)...")
        tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64, 128], kernel_size=3, num_classes=3).to(device)
        optim = torch.optim.Adam(tcn.parameters(), lr=0.001)
        crit = torch.nn.CrossEntropyLoss()
        
        tcn.train()
        for ep in range(25):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optim.zero_grad()
                out = tcn(Xb)
                loss = crit(out['logits'], yb)
                loss.backward()
                optim.step()
        
        torch.save(tcn.state_dict(), temp_dir / "tcn.pt")
        with open(temp_dir / "tcn_config.json", 'w') as f:
            json.dump({'model_config': {'input_dim': input_dim, 'num_channels': [32, 64, 128], 'kernel_size': 3, 'dropout': 0.2}}, f)

        # 2. XGBoost
        logger.info("Training XGBoost (Temp)...")
        X_train_flat = X_train[:, -1, :]
        X_val_flat = X_val[:, -1, :]
        
        xgb = XGBoostTradingModel(use_gpu=(device=="cuda"))
        xgb.train(X_train_flat, y_train, X_val_flat, y_val)
        xgb.save(str(temp_dir / "xgboost.joblib"))
        with open(temp_dir / "xgboost_config.json", 'w') as f:
            json.dump({}, f)
            
        logger.info(f"✅ Training Complete in TEMP. Swapping to PRODUCTION...")

        # ══════════════════════════════════════════════════════════════════
        # 🔄 EL CAMBIO ATÓMICO (SWAP)
        # ══════════════════════════════════════════════════════════════════
        # 1. Si existe la carpeta vieja, la movemos a backup (o borramos)
        # 2. Renombramos la temporal a la oficial
        
        if prod_dir.exists():
            shutil.rmtree(prod_dir) # Borramos la vieja solo ahora que tenemos la nueva lista
        
        # Renombrar Temp -> Prod (Operación instantánea en el mismo disco)
        temp_dir.rename(prod_dir)
        
        logger.info(f"✨ Atomic Swap Successful for {clean_symbol}")

        # 5. HOT RELOAD
        try:
            import requests
            response = requests.post("http://localhost:8001/ml-v2/reload", json={"symbol": symbol}, timeout=2)
            if response.status_code == 200:
                logger.info(f"🔄 Hot Reload requested for {symbol}: OK")
            else:
                logger.warning(f"⚠️ Hot Reload failed: {response.text}")
        except Exception as e:
            logger.warning(f"⚠️ Hot Reload connection failed: {e}")

    except Exception as e:
        logger.error(f"❌ Critical Training Failure for {symbol}: {e}")
        # Limpieza en caso de error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("🧹 Cleaned up temp directory.")

def train_production():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str)
    args = parser.parse_args()
    
    if args.symbol:
        train_model_for_symbol(args.symbol)
    else:
        symbols = get_all_symbols()
        for s in symbols:
            try:
                train_model_for_symbol(s)
            except Exception as e:
                logger.error(f"Fail {s}: {e}")

if __name__ == "__main__":
    train_production()
