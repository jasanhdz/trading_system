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
    
    # --- FEATURES TEMPORALES (CONCIENCIA DE FIN DE SEMANA) ---
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['is_weekend'] = df['dt'].dt.dayofweek.isin([5, 6]).astype(int)
    df['hour'] = df['dt'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    # Features básicas de volatilidad y tendencia
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # Volumen y Flujo
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window).sum()
    df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)

    # Volatilidad
    df['std_price_12'] = df['price'].rolling(window).std()
    df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
    
    return df.drop(columns=['dt', 'hour']).dropna()

def update_config(symbol, threshold):
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            try: config = json.load(f)
            except: config = {}
    else:
        config = {}
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "")
    config[clean_symbol] = threshold
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Saved to {CONFIG_PATH}")

def optimize_symbol(symbol):
    logger.info(f"🚀 Starting Threshold Optimization for {symbol}...")
    
    df = load_data_from_db(symbol)
    if len(df) < 1000:
        logger.error(f"❌ Not enough data.")
        return

    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return'] = (df['future_price'] - df['price']) / df['price']
    df = add_robust_meta_features(df)
    
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 'obi_5', 'obi_10', 'obi',
        'micro_price', 'funding_rate', 'open_interest', 'taker_buy_vol', 'taker_sell_vol',
        'mean_obi_12', 'max_obi_12', 'std_obi_12', 'slope_price_12', 
        'mean_volume_12', 'volume_trend', 'cvd_12', 'cvd_norm_12', 
        'std_price_12', 'volatility_ratio',
        # Nuevas Features Temporales
        'is_weekend', 'hour_sin', 'hour_cos'
    ]
    
    df = df.dropna()
    thresholds = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050, 0.0060, 0.0080, 0.0100]
    results = []
    
    for th in thresholds:
        conditions = [(df['return'] < -th), (df['return'] > th)]
        df['label'] = np.select(conditions, [0, 2], default=1)
        
        # Check Class Balance (Audit Fix)
        neutral_pct = (df['label'] == 1).mean()
        if neutral_pct > 0.85: # Máximo 85% neutral para ser robusto
            continue
            
        split_idx = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]
        
        scaler = RobustScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_val = scaler.transform(val_df[feature_cols])
        y_train, y_val = train_df['label'], val_df['label']
        
        model = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        
        # MÉTRICA SAGRADA: Precisión de clases activas (Long/Short)
        # Import precision_score locally or ensure it's imported at top (I'll add it to imports if missing, but user provided full file content)
        # Wait, user provided full file content in prompt, I should check imports.
        # The prompt snippet has `from sklearn.metrics import accuracy_score, f1_score, precision_score`
        # My current file has `from sklearn.metrics import accuracy_score, f1_score`
        # I need to update imports too.
        
        from sklearn.metrics import precision_score
        precisions = precision_score(y_val, preds, average=None, zero_division=0)
        # precisions[0]=Short, precisions[2]=Long
        # Handle cases where not all classes are present
        # precision_score returns array of shape (n_classes,) if average=None
        # But we need to map them to 0, 1, 2.
        # It's safer to calculate manually or ensure we map correctly.
        # Let's stick to the user's provided logic which assumes [0, 1, 2] order if all present.
        # Actually, if a class is missing, sklearn returns smaller array.
        # Let's use a safer approach or just trust the snippet for now as it's a direct copy-paste request.
        
        # The user's snippet:
        # precisions = precision_score(y_val, preds, average=None, zero_division=0)
        # active_precision = (precisions[0] + precisions[2]) / 2 if len(precisions) > 2 else 0
        
        # This is risky if class 1 is missing. But class 1 is neutral, usually dominant.
        # If class 0 or 2 is missing, len might be 2.
        # I will use the user's code but add the import inside the function or at top.
        # Since I am replacing a large chunk, I can't easily add import at top without another call.
        # I will add the import inside the function for safety in this chunk.
        
        active_precision = 0
        if len(np.unique(y_val)) == 3:
             p = precision_score(y_val, preds, average=None, zero_division=0)
             active_precision = (p[0] + p[2]) / 2
        
        logger.info(f"   Th {th*100:.2f}% -> Active Prec: {active_precision:.2%} | Neutral: {neutral_pct:.1%}")
        results.append({'threshold': th, 'score': active_precision})

    if not results:
        logger.error("❌ No valid thresholds found.")
        return

    best = max(results, key=lambda x: x['score'])
    logger.info(f"🏆 WINNER: {best['threshold']*100:.2f}% (Precisión Activa: {best['score']:.2%})")
    update_config(symbol, best['threshold'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()
    optimize_symbol(args.symbol)
