import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Import models
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel

# Config
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
CONFIG_OUTPUT = REPO_ROOT / "models" / "advanced" / "thresholds_config.json"
SEQ_LEN = 12
PREDICT_HORIZON = 5
DEVICE = "cpu" # Force CPU to avoid HIP errors during inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptimizeV2")

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

def prepare_features(df):
    # Targets (for simulation)
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    feature_cols = [
        'bid_depth', 'ask_depth', 'bid_ask_spread', 'obi',
        'funding_rate', 'open_interest', 'taker_buy_vol', 'taker_sell_vol'
    ]
    
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    
    return df, feature_cols

def simulate_trades(predictions, returns, threshold, min_confidence_diff=0.1):
    trades = []
    equity = 1.0
    
    for i in range(len(predictions)):
        probs = predictions[i]
        long_prob = probs[2] # Class 2 is Long
        short_prob = probs[0] # Class 0 is Short
        # Class 1 is Neutral
        
        diff = abs(long_prob - short_prob)
        
        if long_prob > threshold and diff > min_confidence_diff:
            ret = returns[i]
            equity *= (1 + ret)
            trades.append(ret)
        elif short_prob > threshold and diff > min_confidence_diff:
            ret = -returns[i]
            equity *= (1 + ret)
            trades.append(ret)
            
    if not trades:
        return {'n_trades': 0, 'pnl': 0.0, 'sharpe': 0.0}
        
    trade_returns = np.array(trades)
    pnl = (equity - 1.0) * 100
    sharpe = (trade_returns.mean() / (trade_returns.std() + 1e-8)) * np.sqrt(252*288) # Approx annualized
    
    return {
        'n_trades': len(trades),
        'pnl': pnl,
        'sharpe': sharpe
    }

def optimize_symbol(symbol_dir):
    symbol_name = symbol_dir.name # e.g. ADAUSDT
    # Convert to DB format: ADAUSDT -> ADA/USDT:USDT
    # Assuming standard format. If not, we might need a map.
    # Try to guess: 
    base = symbol_name.replace("USDT", "")
    db_symbol = f"{base}/USDT:USDT"
    
    logger.info(f"🔍 Optimizing {symbol_name} ({db_symbol})...")
    
    # Load Data
    df = load_data_from_db(db_symbol)
    if len(df) < 200:
        logger.warning(f"Not enough data for {symbol_name}")
        return None
        
    df, feature_cols = prepare_features(df)
    
    # Load Scaler
    scaler_path = symbol_dir / "scaler.pkl"
    if not scaler_path.exists():
        logger.error(f"Scaler not found for {symbol_name}")
        return None
    scaler = joblib.load(scaler_path)
    
    # Prepare Test Data (Last 15%)
    test_size = int(len(df) * 0.15)
    test_df = df.iloc[-test_size:].reset_index(drop=True)
    
    X_raw = test_df[feature_cols].values
    X_scaled = scaler.transform(X_raw)
    
    # Create Sequences
    X_seq = []
    y_returns = []
    
    for i in range(SEQ_LEN, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQ_LEN:i])
        y_returns.append(test_df['return_5m'].iloc[i]) # Return at this step (looking forward)
        
    X_seq = np.array(X_seq, dtype=np.float32)
    y_returns = np.array(y_returns)
    
    if len(X_seq) == 0:
        return None

    # Load Models
    preds_xgb = None
    preds_lstm = None
    preds_tcn = None
    
    # XGBoost
    xgb_path = symbol_dir / "xgboost.joblib"
    if xgb_path.exists():
        import xgboost as xgb
        xgb_model = joblib.load(xgb_path)
        # XGB expects 2D input (flattened last step)
        X_flat = X_seq[:, -1, :]
        dtest = xgb.DMatrix(X_flat)
        preds_xgb = xgb_model.predict(dtest) # (N, 3)
        
    # LSTM
    lstm_path = symbol_dir / "lstm.pt"
    if lstm_path.exists():
        checkpoint = torch.load(lstm_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load Config
        config_path = symbol_dir / "lstm_config.json"
        model_kwargs = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                if 'model_config' in loaded_config:
                    model_kwargs = loaded_config['model_config']
                else:
                    model_kwargs = loaded_config
        
        # Infer input dim from scaler
        input_dim = X_seq.shape[2]
        
        # Filter kwargs that are not for constructor (if any)
        # DeepTemporalNet takes: input_dim, sequence_length, hidden_dim, num_layers, dropout, num_classes
        # We should pass them.
        if 'input_dim' in model_kwargs: del model_kwargs['input_dim']
        if 'sequence_length' in model_kwargs: del model_kwargs['sequence_length']
        
        model = DeepTemporalNet(input_dim=input_dim, sequence_length=SEQ_LEN, **model_kwargs)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            preds_lstm = torch.softmax(logits, dim=-1).cpu().numpy()

    # TCN
    tcn_path = symbol_dir / "tcn.pt"
    if tcn_path.exists():
        checkpoint = torch.load(tcn_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load Config
        config_path = symbol_dir / "tcn_config.json"
        model_kwargs = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                if 'model_config' in loaded_config:
                    model_kwargs = loaded_config['model_config']
                else:
                    model_kwargs = loaded_config

        input_dim = X_seq.shape[2]
        if 'input_dim' in model_kwargs: del model_kwargs['input_dim']
        model = TCNTradingModel(input_dim=input_dim, **model_kwargs)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            preds_tcn = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
            
    # Ensemble Average
    valid_preds = [p for p in [preds_xgb, preds_lstm, preds_tcn] if p is not None]
    if not valid_preds:
        logger.error(f"No valid models loaded for {symbol_name}")
        return None
        
    avg_preds = np.mean(valid_preds, axis=0)
    
    # Optimize Threshold
    best_res = None
    best_score = -999
    
    thresholds = np.arange(0.35, 0.85, 0.01)
    
    for thr in thresholds:
        res = simulate_trades(avg_preds, y_returns, thr)
        if res['n_trades'] >= 10: # Min trades constraint
            # Score: Mix of Sharpe and PnL
            score = res['sharpe']
            if score > best_score:
                best_score = score
                best_res = {
                    'threshold': float(thr),
                    'metrics': res
                }
    
    # Fallback if no trades met constraint
    if best_res is None:
        # Try to find ANY threshold with trades
        for thr in np.arange(0.30, 0.60, 0.01):
             res = simulate_trades(avg_preds, y_returns, thr)
             if res['n_trades'] > 0:
                 best_res = {'threshold': float(thr), 'metrics': res}
                 break
        if best_res is None:
            best_res = {'threshold': 0.50, 'metrics': {'n_trades': 0}} # Default fallback
            
    logger.info(f"✅ {symbol_name}: Best Threshold = {best_res['threshold']:.2f} (Trades: {best_res['metrics']['n_trades']}, Sharpe: {best_res['metrics'].get('sharpe',0):.2f})")
    
    return best_res['threshold']

def main():
    if not MODELS_DIR.exists():
        logger.error(f"Models dir not found: {MODELS_DIR}")
        return

    global_config = {}
    
    # Load existing config if available to preserve other settings
    if CONFIG_OUTPUT.exists():
        try:
            with open(CONFIG_OUTPUT, 'r') as f:
                global_config = json.load(f)
        except:
            pass

    # Process ALL symbols in v2_ensemble
    # TARGET_SYMBOLS removed to allow full optimization
    
    for symbol_dir in MODELS_DIR.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            # if symbol not in TARGET_SYMBOLS: continue # Commented out to run all
                
            print(f"Processing {symbol}...", flush=True)
            threshold = optimize_symbol(symbol_dir)
            
            if threshold:
                # Update config structure
                # Format: SYMBOL -> 1h -> threshold
                if symbol not in global_config:
                    global_config[symbol] = {}
                if '1h' not in global_config[symbol]:
                    global_config[symbol]['1h'] = {}
                    
                global_config[symbol]['1h']['threshold'] = threshold
                global_config[symbol]['1h']['leverage'] = 15 # High leverage for these top tiers
                
    # Save Config
    CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_OUTPUT, 'w') as f:
        json.dump(global_config, f, indent=2)
        
    logger.info(f"🎉 Config updated at {CONFIG_OUTPUT}")

if __name__ == "__main__":
    main()
