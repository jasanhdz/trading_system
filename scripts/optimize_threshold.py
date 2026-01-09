import argparse
import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThresholdOptimizer")

DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
CONFIG_PATH = REPO_ROOT / "config" / "threshold_config.json"
PREDICT_HORIZON = 60 # 10 minutes

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

def add_robust_meta_features(df, window=12):
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

def update_config(symbol, threshold):
    # Load existing or create new
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            try:
                config = json.load(f)
            except:
                config = {}
    else:
        config = {}
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize symbol key (e.g. ETHUSDT)
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "")
    config[clean_symbol] = threshold
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Saved to {CONFIG_PATH}")

def optimize_symbol(symbol):
    logger.info(f"🧪 Starting Threshold Optimization for {symbol}...")
    
    # Load Data
    df = load_data_from_db(symbol)
    if len(df) < 1000:
        logger.error(f"❌ Not enough data for {symbol} ({len(df)} rows).")
        return

    # Prepare Base Features
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return'] = (df['future_price'] - df['price']) / df['price']
    df = add_robust_meta_features(df)
    
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 'obi_5', 'obi_10', 'obi',
        'micro_price', 'funding_rate', 'open_interest', 'taker_buy_vol', 'taker_sell_vol',
        'mean_obi_12', 'max_obi_12', 'std_obi_12', 'slope_price_12', 
        'mean_volume_12', 'volume_trend', 'cvd_12', 'cvd_norm_12', 
        'std_price_12', 'volatility_ratio'
    ]
    
    df = df.dropna()
    
    # Grid Search Range (0.10% to 1.00%)
    thresholds = [
        0.0010, 0.0012, 0.0015, 0.0018, 0.0020, 
        0.0022, 0.0025, 0.0030, 0.0035, 0.0040, 
        0.0050, 0.0060, 0.0070, 0.0080, 0.0090, 0.0100
    ]
    
    results = []
    
    logger.info(f"📊 Testing {len(thresholds)} thresholds on {len(df)} samples...")
    
    for th in thresholds:
        # Labeling
        conditions = [(df['return'] < -th), (df['return'] > th)]
        choices = [0, 2]
        df['label'] = np.select(conditions, choices, default=1)
        
        # Check Class Balance
        counts = df['label'].value_counts(normalize=True)
        neutral_pct = counts.get(1, 0)
        
        if neutral_pct > 0.90:
            logger.warning(f"   ⚠️ Threshold {th*100:.2f}% too high (Neutral > 90%). Skipping.")
            continue
            
        # Split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train Fast XGBoost
        # Try to use GPU if available, else CPU
        try:
            model = XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                n_jobs=-1,
                eval_metric='mlogloss',
                device='cuda',
                tree_method='hist'
            )
            model.fit(X_train_scaled, y_train)
        except Exception:
             # Fallback to CPU
             model = XGBClassifier(
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=6, 
                 n_jobs=-1, 
                 eval_metric='mlogloss'
             )
             model.fit(X_train_scaled, y_train)

        preds = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        
        logger.info(f"   Threshold {th*100:.2f}% -> Acc: {acc:.2%} | F1: {f1:.4f} | Neutral: {neutral_pct:.1%}")
        results.append({'threshold': th, 'accuracy': acc, 'f1': f1})

    if not results:
        logger.error("❌ No valid thresholds found.")
        return

    # Find Best
    best = max(results, key=lambda x: x['accuracy'])
    logger.info(f"🏆 WINNER for {symbol}: {best['threshold']*100:.2f}% (Acc: {best['accuracy']:.2%})")
    
    # Save to Config
    update_config(symbol, best['threshold'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()
    optimize_symbol(args.symbol)
