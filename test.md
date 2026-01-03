# 🧠 Anatomía del Sistema de Trading NINJA v4.1

Este documento contiene el código fuente COMPLETO del sistema de trading para contexto de agentes AI.

**Sistema:** NINJA Trading Bot v4.1
**Stack:** Python (ML/Backend) + TypeScript (Bot Execution)
**ML Features:** 23 dimensiones (v2.2 con CVD + Volatilidad)

---

# ÍNDICE

1. Data Collection: `scripts/next_gen/market_data_collector.py`
2. Training: `scripts/train_v2_production.py`
3. ML Service: `services/ml_service_v2.py`
4. Backtesting: `scripts/backtest_system_v2.py`
5. Grid Search: `scripts/grid_search_optimizer.py`
6. Daily Retrain: `scripts/daily_retrain.sh`

---

# 1. DATA COLLECTION

**Archivo:** `scripts/next_gen/market_data_collector.py`
**Función:** Demonio que captura Order Book + Derivados cada 10s para 21 símbolos.

```python
#!/usr/bin/env python3
import os
import sys
import time
import sqlite3
import logging
import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuración
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "market_data_v2.db"
LOG_DIR = ROOT_DIR / "logs"
SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 
    'SOL/USDT:USDT', 'XRP/USDT:USDT', 'LINK/USDT:USDT',
    'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'POL/USDT:USDT', 'DOT/USDT:USDT',
    'LTC/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT',
    '1000PEPE/USDT:USDT', 'FET/USDT:USDT', 'SEI/USDT:USDT', 'WLD/USDT:USDT',
    'INJ/USDT:USDT', 'APT/USDT:USDT'
]

# Setup Logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "data_collector_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CollectorV2")

def init_db():
    """Inicializa la base de datos V2 con modo WAL para concurrencia."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # CRITICAL FIX: Habilitar WAL para permitir lecturas concurrentes
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('PRAGMA synchronous=NORMAL;')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS orderbook_metrics (
            timestamp INTEGER,
            symbol TEXT,
            obi_5 REAL, obi_10 REAL, obi_20 REAL,
            spread_pct REAL,
            mid_price REAL, micro_price REAL,
            bid_depth_20 REAL, ask_depth_20 REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS derivatives_data (
            timestamp INTEGER,
            symbol TEXT,
            funding_rate REAL,
            open_interest REAL, open_interest_value REAL,
            taker_buy_vol REAL, taker_sell_vol REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"✅ DB inicializada en WAL mode: {DB_PATH}")

def calculate_obi(bids, asks, depth):
    """Calcula Order Book Imbalance para una profundidad dada."""
    bid_vol = sum(b[1] for b in bids[:depth])
    ask_vol = sum(a[1] for a in asks[:depth])
    if (bid_vol + ask_vol) == 0:
        return 0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def fetch_and_store(exchange):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = int(time.time() * 1000)
    
    for symbol in SYMBOLS:
        try:
            # 1. Order Book
            book = exchange.fetch_order_book(symbol, limit=20)
            bids, asks = book['bids'], book['asks']
            
            if bids and asks:
                best_bid, best_ask = bids[0][0], asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread_pct = (best_ask - best_bid) / mid_price
                
                obi_5 = calculate_obi(bids, asks, 5)
                obi_10 = calculate_obi(bids, asks, 10)
                obi_20 = calculate_obi(bids, asks, 20)
                
                bid_depth_20 = sum(b[1] for b in bids[:20])
                ask_depth_20 = sum(a[1] for a in asks[:20])
                
                total_vol_top = bids[0][1] + asks[0][1]
                micro_price = mid_price
                if total_vol_top > 0:
                    micro_price = (best_bid * asks[0][1] + best_ask * bids[0][1]) / total_vol_top

                cursor.execute('''
                    INSERT OR REPLACE INTO orderbook_metrics 
                    (timestamp, symbol, obi_5, obi_10, obi_20, spread_pct, mid_price, micro_price, bid_depth_20, ask_depth_20)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (now, symbol, obi_5, obi_10, obi_20, spread_pct, mid_price, micro_price, bid_depth_20, ask_depth_20))

            # 2. Funding Rate
            funding = exchange.fetch_funding_rate(symbol)
            funding_rate = funding['fundingRate']
            
            # 3. Open Interest
            oi = exchange.fetch_open_interest(symbol)
            open_interest = oi['openInterestAmount']
            open_interest_val = oi['openInterestValue']
            
            # 4. Taker Volume
            market = exchange.market(symbol)
            market_id = market['id']
            taker_buy_vol, taker_sell_vol = 0, 0
            
            try:
                if exchange.id == 'binance':
                    response = exchange.fapiPublicGetKlines({
                        'symbol': market_id, 'interval': '1m', 'limit': 1
                    })
                    if len(response) > 0:
                        candle = response[0]
                        total_vol = float(candle[5])
                        taker_buy_vol = float(candle[9])
                        taker_sell_vol = total_vol - taker_buy_vol
            except Exception as e:
                logger.warning(f"⚠️ Fallo obteniendo Taker Vol para {symbol}: {e}")

            cursor.execute('''
                INSERT OR REPLACE INTO derivatives_data
                (timestamp, symbol, funding_rate, open_interest, open_interest_value, taker_buy_vol, taker_sell_vol)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (now, symbol, funding_rate, open_interest, open_interest_val, taker_buy_vol, taker_sell_vol))
            
            logger.info(f"✅ {symbol} procesado. OBI: {obi_5:.2f} | Fund: {funding_rate:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Error procesando {symbol}: {e}")
            
    conn.commit()
    conn.close()

def main():
    init_db()
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    logger.info("🚀 Iniciando Colector V2 (Loop infinito cada 10s)")
    
    while True:
        try:
            start_time = time.time()
            fetch_and_store(exchange)
            elapsed = time.time() - start_time
            sleep_time = max(0, 10 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("🛑 Colector detenido por usuario")
            break
        except Exception as e:
            logger.error(f"💥 Error crítico en loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
```

---

# 2. TRAINING

**Archivo:** `scripts/train_v2_production.py`
**Función:** Entrena el "Consejo de Sabios" (4 modelos: LSTM, TCN, XGBoost, Transformer)

```python
import argparse
import sys
from pathlib import Path

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
from sklearn.preprocessing import RobustScaler

from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV2")

VERSION = "v2.1"
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"
SEQ_LEN = 12
PREDICT_HORIZON = 15

def add_robust_meta_features(df, window=12):
    """
    Genera meta-features v2.2 (23 total).
    Incluye CVD y Volatilidad para capturar 'intención del mercado'.
    """
    df = df.copy()
    
    # Rolling OBI
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # Volume
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    
    # Price Slope
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NUEVO v2.2: CVD (Cumulative Volume Delta)
    # ═══════════════════════════════════════════════════════════════════════════
    df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window).sum()
    df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # NUEVO v2.2: Volatilidad del Precio
    # ═══════════════════════════════════════════════════════════════════════════
    df['std_price_12'] = df['price'].rolling(window).std()
    df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
    
    initial_len = len(df)
    df = df.dropna()
    if initial_len - len(df) > 0:
        logger.info(f"⚠️ Meta-Features: Removidas {initial_len - len(df)} filas con NaNs iniciales.")
    return df

def load_data_from_db(symbol):
    logger.info(f"📥 Loading data for {symbol}...")
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price, o.micro_price,
        o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_5, o.obi_10, o.obi_20 as obi,
        d.funding_rate, d.open_interest,
        d.taker_buy_vol, d.taker_sell_vol
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
    clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
    symbol_dir = MODELS_DIR / clean_symbol
    
    if symbol_dir.exists():
        shutil.rmtree(symbol_dir)
    symbol_dir.mkdir(parents=True)
    
    logger.info(f"🚀 Starting training for {symbol} -> {symbol_dir}")
    
    df = load_data_from_db(symbol)
    if len(df) < 100:
        logger.warning(f"⚠️ Not enough data for {symbol} ({len(df)} rows). Skipping.")
        return

    # Feature Engineering
    df['future_price'] = df['price'].shift(-PREDICT_HORIZON)
    df['return_5m'] = (df['future_price'] - df['price']) / df['price']
    
    threshold = 0.001
    conditions = [(df['return_5m'] < -threshold), (df['return_5m'] > threshold)]
    choices = [0, 2]
    df['label'] = np.select(conditions, choices, default=1)
    
    df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
    
    df = add_robust_meta_features(df, window=SEQ_LEN)
    
    # 23 Features (v2.2)
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
        'cvd_12', 'cvd_norm_12',
        'std_price_12', 'volatility_ratio'
    ]
    feature_cols = base_cols + meta_cols
    logger.info(f"🧙 Consejo de Sabios {VERSION}: Entrenando con {len(feature_cols)} features (v2.2 +CVD +Vol)")
    
    df = df.dropna()
    X = df[feature_cols].values
    y = df['label'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, symbol_dir / "scaler.pkl")
    with open(symbol_dir / "features.json", 'w') as f:
        json.dump(feature_cols, f)
        
    # Sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        Xs.append(X_scaled[i:(i + SEQ_LEN)])
        ys.append(y[i + SEQ_LEN])
    X_seq, y_seq = np.array(Xs), np.array(ys)
    
    if len(X_seq) < 10:
        logger.warning(f"⚠️ Not enough sequences for {symbol}. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Using device: {device}")
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
    input_dim = len(feature_cols)
    criterion = torch.nn.CrossEntropyLoss()
    
    # MODEL 1: LSTM
    logger.info(f"   🏋️ Training LSTM...")
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    lstm.train()
    for epoch in range(50):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(lstm(X_b)['logits'], y_b)
            loss.backward()
            optimizer.step()
    torch.save(lstm.state_dict(), symbol_dir / "lstm.pt")
    with open(symbol_dir / "lstm_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'hidden_dim': 64, 'lstm_layers': 2, 'dropout': 0.2, 'num_classes': 3}}, f)
        
    # MODEL 2: TCN
    logger.info(f"   🏋️ Training TCN...")
    tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64], kernel_size=3, num_classes=3).to(device)
    optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)
    tcn.train()
    for epoch in range(50):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(tcn(X_b)['logits'], y_b)
            loss.backward()
            optimizer.step()
    torch.save(tcn.state_dict(), symbol_dir / "tcn.pt")
    with open(symbol_dir / "tcn_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'num_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.2}}, f)
        
    # MODEL 3: XGBoost
    logger.info(f"   🏋️ Training XGBoost...")
    X_train_flat = X_train[:, -1, :]
    X_val_flat = X_val[:, -1, :]
    xgb = XGBoostTradingModel(use_gpu=(device=="cuda"))
    xgb.train(X_train_flat, y_train, X_val_flat, y_val)
    xgb.save(str(symbol_dir / "xgboost.joblib"))
    with open(symbol_dir / "xgboost_config.json", 'w') as f:
        json.dump({}, f)
        
    # MODEL 4: Transformer
    logger.info(f"   🏋️ Training Transformer...")
    transformer = TradingTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005)
    transformer.train()
    for epoch in range(50):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(transformer(X_b)['logits'], y_b)
            loss.backward()
            optimizer.step()
    torch.save(transformer.state_dict(), symbol_dir / "transformer.pt")
    with open(symbol_dir / "transformer_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1}}, f)
        
    logger.info(f"✅ Models for {clean_symbol} saved.")

def train_production():
    parser = argparse.ArgumentParser(description='Train Consejo de Sabios v2.2')
    parser.add_argument('--symbol', type=str, help='Specific symbol to train')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
        logger.info(f"🎯 Training SINGLE symbol: {symbols}")
    else:
        symbols = get_all_symbols()
        logger.info(f"Found {len(symbols)} symbols in DB: {symbols}")
        
    if args.num_shards > 1:
        symbols = [s for i, s in enumerate(symbols) if i % args.num_shards == args.shard_id]
    
    for symbol in symbols:
        try:
            train_model_for_symbol(symbol)
        except Exception as e:
            logger.error(f"❌ Failed training {symbol}: {e}")
            
    logger.info("🎉 All training tasks completed.")

if __name__ == "__main__":
    train_production()
```

---

# CONTINUACIÓN EN SIGUIENTE BLOQUE...

El archivo continúa con:
- ML Service (services/ml_service_v2.py)
- Backtesting (scripts/backtest_system_v2.py)
- Grid Search (scripts/grid_search_optimizer.py)
- Daily Retrain (scripts/daily_retrain.sh)

---

# 3. ML SERVICE (Inference API)

**Archivo:** `services/ml_service_v2.py`
**Función:** API FastAPI que carga modelos on-demand y predice probabilidades.
**Novedades v4.1:** LRU Eviction (max 12 modelos en VRAM para GTX 1660 6GB)

```python
"""FastAPI service V2 (Ninja Mode) that returns ML probabilities using Order Book data."""
from __future__ import annotations

import logging
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from typing import Dict, Optional, List, Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.advanced_models.ensemble_manager import EnsembleManager

class ServiceLogger:
    def __init__(self, name: str = "ml_service_v2") -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
        self._logger = logging.getLogger(name)
    def info(self, msg, **kw): self._logger.info(f"{msg} {kw if kw else ''}")
    def error(self, msg, **kw): self._logger.error(f"{msg} {kw if kw else ''}")
    def warning(self, msg, **kw): self._logger.warning(f"{msg} {kw if kw else ''}")

LOGGER = ServiceLogger()
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"

class ProbabilityRequestV2(BaseModel):
    symbol: str

class ProbabilityResponseV2(BaseModel):
    symbol: str
    long_prob: float
    short_prob: float
    neutral_prob: float
    consensus_level: float
    meta_verdict: str

def load_latest_data(symbol: str, limit: int = 60) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    db_symbol = symbol
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT o.timestamp, o.mid_price as price, o.micro_price,
           o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth, 
           o.spread_pct as bid_ask_spread, o.obi_5, o.obi_10, o.obi_20 as obi,
           d.funding_rate, d.open_interest, d.taker_buy_vol, d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE o.symbol = '{db_symbol}'
    ORDER BY o.timestamp DESC LIMIT {limit}
    """
    try:
        df = pd.read_sql_query(query, conn)
        df = df.sort_values('timestamp')
    finally:
        conn.close()
    return df

class V2ModelManager:
    def __init__(self):
        self.ensembles: Dict[str, EnsembleManager] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LRU Tracking para evicción de VRAM
        self.last_accessed: Dict[str, float] = {}
        self.MAX_MODELS_IN_VRAM = 12  # GTX 1660 6GB: 12 * 250MB ≈ 3GB
        
        LOGGER.info(f"🚀 ML Service initialized on device: {self.device} | Max Models: {self.MAX_MODELS_IN_VRAM}")
        self.smoothed_probs_cache = {}
        
    def _clean_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def get_ensemble(self, symbol: str) -> Optional[EnsembleManager]:
        clean_sym = self._clean_symbol(symbol)
        import time
        self.last_accessed[clean_sym] = time.time()
        
        if clean_sym in self.ensembles:
            return self.ensembles[clean_sym]
        
        # LRU Eviction check
        if len(self.ensembles) >= self.MAX_MODELS_IN_VRAM:
            self._evict_least_recently_used()
            
        return self.load_model_for_symbol(clean_sym)

    def _evict_least_recently_used(self):
        if not self.last_accessed:
            return
        oldest = min(self.last_accessed, key=self.last_accessed.get)
        LOGGER.info(f"🗑️ Evicting {oldest} from VRAM (LRU)")
        
        if oldest in self.ensembles: del self.ensembles[oldest]
        if oldest in self.scalers: del self.scalers[oldest]
        if oldest in self.feature_cols: del self.feature_cols[oldest]
        if oldest in self.last_accessed: del self.last_accessed[oldest]
        if self.device == "cuda": torch.cuda.empty_cache()

    def load_model_for_symbol(self, clean_symbol: str) -> Optional[EnsembleManager]:
        symbol_dir = MODELS_DIR / clean_symbol
        if not symbol_dir.exists():
            LOGGER.warning(f"No models found for {clean_symbol}")
            return None
            
        try:
            LOGGER.info(f"Loading models for {clean_symbol}...")
            ensemble = EnsembleManager(device=self.device)
            self.scalers[clean_symbol] = joblib.load(symbol_dir / "scaler.pkl")
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_symbol] = json.load(f)
            
            is_v2_1 = len(self.feature_cols[clean_symbol]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_symbol] = ensemble
            LOGGER.info(f"✅ Loaded {clean_symbol} ensemble.")
            return ensemble
        except Exception as e:
            LOGGER.error(f"Failed to load {clean_symbol}: {e}")
            return None

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        ensemble = self.get_ensemble(symbol)
        clean_sym = self._clean_symbol(symbol)
        
        if ensemble is None:
            return {'ensemble_probs': torch.tensor([[0.0, 1.0, 0.0]]), 'consensus': 0.0}
            
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
        
        cols = self.feature_cols.get(clean_sym, [])
        n_features = len(cols)
        is_v2_1 = n_features >= 19 or 'mean_obi_12' in cols
        
        if is_v2_1:
            window = 12
            df['mean_obi_12'] = df['obi'].rolling(window, min_periods=1).mean()
            df['max_obi_12'] = df['obi'].rolling(window, min_periods=1).max()
            df['std_obi_12'] = df['obi'].rolling(window, min_periods=2).std().fillna(0)
            df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
            df['mean_volume_12'] = df['total_volume'].rolling(window, min_periods=1).mean()
            df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
            df['slope_price_12'] = (df['price'] - df['price'].shift(window).bfill()) / window
            
            # v2.2: CVD y Volatilidad
            df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window, min_periods=1).sum()
            df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)
            df['std_price_12'] = df['price'].rolling(window, min_periods=2).std().fillna(0)
            df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
        
        X = df[cols].values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            X = np.nan_to_num(X, nan=0.0)
            
        scaler = self.scalers[clean_sym]
        X_scaled = scaler.transform(X)
        
        SEQ_LEN = 12
        if len(X_scaled) < SEQ_LEN:
            pad_len = SEQ_LEN - len(X_scaled)
            X_scaled = np.pad(X_scaled, ((pad_len, 0), (0, 0)), mode='edge')
            
        X_seq = X_scaled[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = ensemble.predict(X_tensor)
            
        # FILTRO NINJA (EMA Asimétrico)
        ensemble_probs = result['ensemble_probs'][0].tolist()
        raw_dict = {'short': float(ensemble_probs[0]), 'neutral': float(ensemble_probs[1]), 'long': float(ensemble_probs[2])}
        prev_smoothed = self.smoothed_probs_cache.get(clean_sym, raw_dict)
        
        ALPHA_SLOW = 0.15
        ALPHA_FAST = 0.70
        smoothed_dict = {}
        
        for key in ['short', 'neutral', 'long']:
            raw_val = raw_dict[key]
            prev_val = prev_smoothed[key]
            diff = raw_val - prev_val
            alpha = ALPHA_SLOW if diff > 0 else ALPHA_FAST
            smoothed_dict[key] = (alpha * raw_val) + ((1 - alpha) * prev_val)
        
        total = sum(smoothed_dict.values())
        normalized_dict = {k: v / total for k, v in smoothed_dict.items()}
        self.smoothed_probs_cache[clean_sym] = normalized_dict
        
        result['ensemble_probs'] = torch.tensor([[normalized_dict['short'], normalized_dict['neutral'], normalized_dict['long']]])
        return result

MANAGER = V2ModelManager()
router = APIRouter(prefix="/ml-v2", tags=["ml-v2"])

@router.on_event("startup")
async def startup_event():
    LOGGER.info(f"🚀 ML Service V2 ready (Lazy Loading, Max VRAM: {MANAGER.MAX_MODELS_IN_VRAM})")

@router.post("/predict", response_model=ProbabilityResponseV2)
async def predict_endpoint(request: ProbabilityRequestV2) -> ProbabilityResponseV2:
    symbol = request.symbol
    try:
        df = load_latest_data(symbol)
        if df.empty:
            alt_symbol = symbol.replace("USDT", "/USDT:USDT")
            df = load_latest_data(alt_symbol)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    except Exception as e:
        LOGGER.error(f"Data load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    result = MANAGER.predict(symbol, df)
    probs = result['ensemble_probs'][0].tolist()
    
    return ProbabilityResponseV2(
        symbol=symbol,
        short_prob=probs[0], neutral_prob=probs[1], long_prob=probs[2],
        consensus_level=float(result.get('consensus', 0.0)),
        meta_verdict="APPROVED"
    )

app = FastAPI(title="ML Service V2 (Ninja)", version="2.0.0")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.ml_service_v2:app", host="0.0.0.0", port=8001, reload=False)
```

---

# 4. BACKTESTING

**Archivo:** `scripts/backtest_system_v2.py`
**Función:** "Gemelo Digital" - Simula el bot de producción tick a tick.

```python
"""
Backtester del Sistema Completo (v2.1) - "El Gemelo Digital"
Uso: python scripts/backtest_system_v2.py --symbol BTCUSDT --days 7
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from services.ml_service_v2 import V2ModelManager, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("BacktesterV2")

class NinjaBotSimulator:
    def __init__(self, symbol: str, initial_capital: float = 1000.0, leverage: int = 10):
        self.symbol = symbol
        self.capital = initial_capital
        self.leverage = leverage
        self.balance = initial_capital
        self.position = None
        self.entry_price = 0.0
        self.entry_time = None
        self.qty = 0.0
        
        # Configuración
        self.base_threshold = 0.35
        self.max_threshold = 0.50
        self.current_threshold = self.base_threshold
        self.hard_stop_pct = -0.10
        self.breakeven_trigger_roi = 0.015
        self.breakeven_profit_pct = 0.002
        self.commission_rate = 0.0004
        self.peak_roi = -999.0
        self.trailing_stop_price = 0.0
        self.trailing_active = False
        
        self.trades = []
        self.equity_curve = []
        
        self.ml_manager = V2ModelManager()
        self.ml_manager.load_model_for_symbol(symbol)

    def load_data(self, days: int = 7, hours: int = 0) -> pd.DataFrame:
        LOGGER.info(f"📥 Cargando últimos {days} días para {self.symbol}...")
        start_ts = int((datetime.now() - timedelta(days=days, hours=hours)).timestamp() * 1000)
        db_symbol = self.symbol
        
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT o.timestamp, o.mid_price as price, o.micro_price,
               o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth, 
               o.spread_pct as bid_ask_spread, o.obi_5, o.obi_10, o.obi_20 as obi,
               d.funding_rate, d.open_interest, d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{db_symbol}' OR o.symbol = '{db_symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        LOGGER.info(f"✅ Cargados {len(df)} registros.")
        return df

    def run(self, df: pd.DataFrame):
        LOGGER.info("🚀 Iniciando simulación...")
        window_size = 60
        
        for i in range(window_size, len(df)):
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            price = current_row['price']
            
            df_window = df.iloc[i-window_size:i+1].copy()
            prediction = self.ml_manager.predict(self.symbol, df_window)
            probs = prediction['ensemble_probs'][0].tolist()
            
            short_prob, neutral_prob, long_prob = probs[0], probs[1], probs[2]
            self._update_bot_logic(timestamp, price, long_prob, short_prob, neutral_prob)
            self._record_equity(price)
            
            if i % 1000 == 0:
                print(f"⏳ Procesados {i}/{len(df)} ticks...", end='\r')
                
        print("\n✅ Simulación completada.")
        self._generate_report()

    def _update_bot_logic(self, timestamp, price, long_prob, short_prob, neutral_prob):
        market_uncertainty = abs(long_prob - short_prob)
        self.current_threshold = self.base_threshold + (market_uncertainty * 0.15)
        
        if self.position is None:
            if long_prob > self.current_threshold:
                self._open_position("LONG", price, timestamp, f"ML_LONG")
            elif short_prob > self.current_threshold:
                self._open_position("SHORT", price, timestamp, f"ML_SHORT")
        else:
            roi_raw = (price - self.entry_price) / self.entry_price if self.position == "LONG" else (self.entry_price - price) / self.entry_price
            roi_lev = roi_raw * self.leverage
            
            if roi_lev * 100 > self.peak_roi:
                self.peak_roi = roi_lev * 100
                if roi_lev > self.breakeven_trigger_roi:
                    self._update_trailing_stop(price, roi_lev * 100)
            
            # Breakeven
            if roi_lev > self.breakeven_trigger_roi:
                if self.position == "LONG":
                    be_stop = self.entry_price * (1 + self.breakeven_profit_pct)
                    if self.trailing_stop_price < be_stop:
                        self.trailing_stop_price = be_stop
                        self.trailing_active = True
                else:
                    be_stop = self.entry_price * (1 - self.breakeven_profit_pct)
                    if self.trailing_stop_price > be_stop or self.trailing_stop_price == 0:
                        self.trailing_stop_price = be_stop
                        self.trailing_active = True

            # Hard Stop
            if roi_lev < self.hard_stop_pct:
                self._close_position(price, timestamp, "HARD_STOP_LOSS")
                return
            
            # Panic Reversal
            if self.position == "LONG" and short_prob > 0.55:
                self._close_position(price, timestamp, "PANIC_REVERSAL")
                return
            if self.position == "SHORT" and long_prob > 0.55:
                self._close_position(price, timestamp, "PANIC_REVERSAL")
                return
            
            # Neutrality Exit
            if roi_lev > 0.03 and neutral_prob > 0.60:
                self._close_position(price, timestamp, "NEUTRALITY_EXIT")
                return
            
            # Trailing Stop Execution
            if self.position == "LONG" and price < self.trailing_stop_price:
                self._close_position(price, timestamp, "TRAILING_STOP")
                return
            if self.position == "SHORT" and price > self.trailing_stop_price:
                self._close_position(price, timestamp, "TRAILING_STOP")
                return

    def _update_trailing_stop(self, current_price, roi_lev):
        peak_pct_val = max(5, self.peak_roi * 100.0)
        base_trail_pct = 30 - (22 * math.log10(peak_pct_val / 5))
        trail_distance_pct = max(15, min(30, base_trail_pct))
        stop_roi_level = (self.peak_roi * 100.0) * (1 - (trail_distance_pct / 100.0))
        
        if self.position == "LONG":
            stop_price = self.entry_price * (1 + (stop_roi_level / 100.0 / self.leverage))
            if stop_price > self.trailing_stop_price:
                self.trailing_stop_price = stop_price
        else:
            stop_price = self.entry_price * (1 - (stop_roi_level / 100.0 / self.leverage))
            if self.trailing_stop_price == 0 or stop_price < self.trailing_stop_price:
                self.trailing_stop_price = stop_price

    def _open_position(self, side, price, timestamp, reason):
        self.position = side
        self.entry_price = price
        self.entry_time = timestamp
        self.qty = (self.balance * self.leverage) / price
        self.peak_roi = -999.0
        self.trailing_stop_price = price * (0.97 if side == "LONG" else 1.03)
        self.trailing_active = False
        fee = (self.qty * price) * self.commission_rate
        self.balance -= fee

    def _close_position(self, price, timestamp, reason):
        if self.position == "LONG":
            pnl = (price - self.entry_price) * self.qty
        else:
            pnl = (self.entry_price - price) * self.qty
        fee = (self.qty * price) * self.commission_rate
        pnl -= fee
        self.balance += pnl
        
        self.trades.append({
            'entry_time': self.entry_time, 'exit_time': timestamp,
            'side': self.position, 'entry_price': self.entry_price,
            'exit_price': price, 'reason': reason,
            'pnl': pnl, 'balance': self.balance
        })
        self.position = None
        self.qty = 0

    def _record_equity(self, current_price):
        unrealized_pnl = 0
        if self.position:
            if self.position == "LONG":
                unrealized_pnl = (current_price - self.entry_price) * self.qty
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.qty
            unrealized_pnl -= (self.qty * current_price) * self.commission_rate
        self.equity_curve.append(self.balance + unrealized_pnl)

    def _generate_report(self):
        if not self.trades:
            LOGGER.warning("⚠️ No trades executed.")
            return
        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        print("\n" + "="*50)
        print(f"📊 REPORTE FINAL: {self.symbol}")
        print("="*50)
        print(f"Capital Inicial: ${self.capital:.2f}")
        print(f"Capital Final:   ${self.balance:.2f}")
        print(f"Retorno Total:   {((self.balance - self.capital)/self.capital)*100:.2f}%")
        print(f"Total Trades:    {len(df_trades)}")
        print(f"Win Rate:        {len(wins)/len(df_trades)*100:.2f}%")
        print(f"Profit Factor:   {wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf'):.2f}")
        print("Motivos de Salida:")
        print(df_trades['reason'].value_counts())
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--hours", type=int, default=0)
    args = parser.parse_args()
    
    sim = NinjaBotSimulator(symbol=args.symbol)
    try:
        data = sim.load_data(days=args.days, hours=args.hours)
        sim.run(data)
    except Exception as e:
        LOGGER.error(f"Error en backtest: {e}")
```

---

# 5. GRID SEARCH

**Archivo:** `scripts/grid_search_optimizer.py`
**Función:** Optimiza parámetros por régimen (default, whale, monk, bloodbath)

```python
"""
Grid Search Optimizer v4.1
Uso:
    python scripts/grid_search_optimizer.py --symbol BTCUSDT --days 7
    python scripts/grid_search_optimizer.py --symbol ETHUSDT --days 14 --mode whale
"""
import os
import sys
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backtest_system_v2 import NinjaBotSimulator

import logging
logging.getLogger("ml_service_v2").setLevel(logging.WARNING)
logging.getLogger("EnsembleManager").setLevel(logging.WARNING)

GRID_CONFIGS = {
    'default': {
        'base_thresholds': [0.30, 0.35, 0.40, 0.45, 0.50],
        'hard_stop_options': [-0.05, -0.10, -0.15],
        'leverage_options': [10, 15],
        'trailing_activation': [0.02, 0.03, 0.05]
    },
    'whale': {
        'base_thresholds': [0.45, 0.50, 0.55, 0.60],
        'hard_stop_options': [-0.15, -0.20, -0.25],
        'leverage_options': [3, 5, 7],
        'trailing_activation': [0.03, 0.05, 0.08]
    },
    'monk': {
        'base_thresholds': [0.35, 0.40, 0.45],
        'hard_stop_options': [-0.03, -0.05, -0.07],
        'leverage_options': [10, 15],
        'trailing_activation': [0.01, 0.02]
    },
    'bloodbath': {
        'base_thresholds': [0.25, 0.30, 0.35],
        'hard_stop_options': [-0.015, -0.02, -0.025],
        'leverage_options': [15, 20],
        'trailing_activation': [0.005, 0.01]
    }
}

def run_grid_search(symbol: str, days: int = 3, hours: int = 0, mode: str = 'default'):
    config = GRID_CONFIGS.get(mode, GRID_CONFIGS['default'])
    base_thresholds = config['base_thresholds']
    hard_stop_options = config['hard_stop_options']
    leverage_options = config['leverage_options']
    trailing_options = config['trailing_activation']
    
    results = []

    print(f"\n{'='*70}")
    print(f"🚀 GRID SEARCH OPTIMIZER v4.1: {symbol}")
    print(f"   Mode: {mode.upper()}")
    print(f"   Período: {days} días, {hours} horas")
    total_combos = len(base_thresholds) * len(hard_stop_options) * len(leverage_options)
    print(f"   Configuraciones a probar: {total_combos}")
    print(f"{'='*70}\n")
    
    print("📥 Cargando datos...")
    temp_sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=10)
    shared_data = temp_sim.load_data(days=days, hours=hours)
    print(f"✅ Datos cargados: {len(shared_data)} registros.\n")
    
    combo_num = 0
    for lev in leverage_options:
        for base_thr in base_thresholds:
            for stop_pct in hard_stop_options:
                combo_num += 1
                sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=lev)
                sim.base_threshold = base_thr
                sim.hard_stop_pct = stop_pct
                sim.run(shared_data)
                
                num_trades = len(sim.trades)
                if num_trades > 0:
                    wins = [t for t in sim.trades if t['pnl'] > 0]
                    losses = [t for t in sim.trades if t['pnl'] <= 0]
                    win_rate = len(wins) / num_trades
                    profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf')
                else:
                    win_rate, profit_factor = 0, 0
                
                return_pct = ((sim.balance - 1000)/1000)*100
                results.append({
                    'leverage': lev, 'base_thr': base_thr, 'stop_pct': stop_pct,
                    'config': f"Lev:{lev}x Thr:{base_thr:.2f} Stop:{stop_pct:.0%}",
                    'return_pct': return_pct, 'win_rate': win_rate,
                    'profit_factor': profit_factor, 'total_trades': num_trades
                })
                print(f"[{combo_num}/{total_combos}] {results[-1]['config']} -> Return: {return_pct:>+6.2f}%")

    print("\n" + "="*70)
    print("📊 RANKING (TOP 10)")
    print("="*70)
    ranked = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    for i, r in enumerate(ranked[:10], 1):
        print(f"{i}. {r['config']} -> {r['return_pct']:>+7.2f}% | WR: {r['win_rate']:.0%}")
    
    report_path = Path(REPO_ROOT) / "reports" / f"grid_search_{symbol.replace('/','')}.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(ranked, f, indent=2)
    print(f"\n💾 Reporte: {report_path}")
    
    return ranked

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search v4.1")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--hours", type=int, default=0)
    parser.add_argument("--mode", type=str, default="default", choices=['default', 'whale', 'monk', 'bloodbath'])
    args = parser.parse_args()
    run_grid_search(args.symbol, days=args.days, hours=args.hours, mode=args.mode)
```

---

# 6. DAILY RETRAIN SCRIPT

**Archivo:** `scripts/daily_retrain.sh`
**Función:** Entrenamiento diario con priorización (9 prod symbols primero)

```bash
#!/bin/bash

PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/daily_retrain.log"
DATE=$(date)

echo "[$DATE] 🚀 Starting Daily Retraining (Priority Mode)..." >> $LOG_FILE
cd $PROJECT_DIR

# AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# PHASE 1: Production Symbols (HIGH PRIORITY)
PRIORITY_SYMBOLS=(
    "DOGE/USDT:USDT" "LINK/USDT:USDT" "AVAX/USDT:USDT" "POL/USDT:USDT"
    "ETH/USDT:USDT" "XRP/USDT:USDT" "SOL/USDT:USDT" "ADA/USDT:USDT" "BTC/USDT:USDT"
)

echo "[$DATE] 🎯 PHASE 1: Training ${#PRIORITY_SYMBOLS[@]} production symbols..." >> $LOG_FILE
for symbol in "${PRIORITY_SYMBOLS[@]}"; do
    echo "[$DATE]   Training $symbol..." >> $LOG_FILE
    export HIP_VISIBLE_DEVICES=$((RANDOM % 2))
    $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol" >> $LOG_FILE 2>&1
done
echo "[$DATE] ✅ PHASE 1 Complete" >> $LOG_FILE

# PHASE 2: Secondary Symbols
SECONDARY_SYMBOLS=(
    "BNB/USDT:USDT" "DOT/USDT:USDT" "LTC/USDT:USDT" "UNI/USDT:USDT"
    "ATOM/USDT:USDT" "NEAR/USDT:USDT" "1000PEPE/USDT:USDT" "FET/USDT:USDT"
    "SEI/USDT:USDT" "WLD/USDT:USDT" "INJ/USDT:USDT" "APT/USDT:USDT"
)

echo "[$DATE] 📦 PHASE 2: Training ${#SECONDARY_SYMBOLS[@]} secondary symbols (parallel)..." >> $LOG_FILE
# Parallel execution omitted for brevity - see full script

pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
echo "[$DATE] 🎉 Daily Retraining Complete!" >> $LOG_FILE
```

---

# RESUMEN DE ARQUITECTURA

| Componente | Archivo | Función |
|------------|---------|---------|
| Data Collector | `scripts/next_gen/market_data_collector.py` | Captura Order Book + Derivados cada 10s |
| Training | `scripts/train_v2_production.py` | Entrena 4 modelos (LSTM, TCN, XGBoost, Transformer) |
| ML Service | `services/ml_service_v2.py` | API FastAPI con LRU eviction y filtro Ninja |
| Backtester | `scripts/backtest_system_v2.py` | Simulación tick-a-tick del bot |
| Grid Search | `scripts/grid_search_optimizer.py` | Optimización por régimen |
| Daily Retrain | `scripts/daily_retrain.sh` | Cron job con priorización |

---

**Versión:** NINJA v4.1
**Features:** 23 dimensiones (v2.2 con CVD + Volatilidad)
**Hardware:** 2x AMD RX 6600 (training), GTX 1660 6GB (inference)
