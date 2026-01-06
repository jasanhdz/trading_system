# 🥷 NINJA Trading System - Technical Documentation

> Complete codebase documentation for expert trading agent review.
> Last Updated: 2026-01-05

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Data Collection](#2-data-collection)
3. [ML Training Pipeline](#3-ml-training-pipeline)
4. [Ensemble Manager (Council of Sages)](#4-ensemble-manager)
5. [ML Inference Service](#5-ml-inference-service)
6. [Daily Retrain Script](#6-daily-retrain)
7. [Backtest Engine](#7-backtest-engine)
8. [Grid Search Optimizer](#8-grid-search-optimizer)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NINJA TRADING SYSTEM v6.0                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │   DATA      │   │   ML        │   │   TRADING   │   │   INFRA     │ │
│  │  PIPELINE   │   │  MODELS     │   │    BOT      │   │             │ │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤   ├─────────────┤ │
│  │ Collector   │-->│ TCN         │-->│ TS Bot      │   │ PM2         │ │
│  │ (Python)    │   │ XGBoost     │   │ (TypeScript)│   │ ROCm 6.2    │ │
│  │ SQLite DB   │   │ Transformer │   │ Binance API │   │ 2x RX 6600  │ │
│  └─────────────┘   │ Ensemble    │   └─────────────┘   └─────────────┘ │
│                    └─────────────┘                                      │
│  Data Flow:                                                             │
│  Binance -> Collector -> SQLite -> ML Training -> ML Service -> Bot    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Services (PM2 Managed):**
- `01-Trading-Bot` - TypeScript live trading bot
- `02-Data-Collector` - Python order book data collector
- `03-ML-Service-V2` - FastAPI ML inference service

---

## 2. Data Collection

### File: `data/collectors/binance_collector.py`

Collects order book data, funding rates, and taker volume from Binance Futures.

```python
# =============================================================================
# DATA/COLLECTORS/BINANCE_COLLECTOR.PY - Versión Corregida
# =============================================================================
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import time
import logging

from data.collectors.base_collector import BaseDataCollector
from config.settings import settings
from utils.exceptions import DataCollectionError

logger = logging.getLogger(__name__)

class BinanceDataCollector(BaseDataCollector):
    def __init__(self):
        super().__init__("binance")
        self.exchange = None
        self.rate_limiter = time.time()
    
    def connect(self) -> bool:
        """Conectar a Binance"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': False,  # True for testnet
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Para futuros
                }
            })
            
            # Test connection
            markets = self.exchange.load_markets()
            self.logger.info(f"Connected to Binance. Found {len(markets)} markets")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    def _rate_limit(self):
        """Implementar rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.rate_limiter
        if time_diff < settings.REQUEST_DELAY:
            time.sleep(settings.REQUEST_DELAY - time_diff)
        self.rate_limiter = time.time()

    def _candidate_symbols(self, symbol: str) -> List[str]:
        """Devuelve variantes posibles para el símbolo solicitado."""
        candidates = [symbol]
        if ":" not in symbol and "/" in symbol:
            base, quote = symbol.split("/", 1)
            quote = quote.upper()
            colon_variant = f"{base}/{quote}:{quote}"
            if colon_variant not in candidates:
                candidates.append(colon_variant)
        return candidates

    def _resolve_market_symbol(self, symbol: str) -> str:
        """Encuentra el símbolo que CCXT reconoce para el mercado solicitado."""
        if not self.exchange:
            if not self.connect():
                raise DataCollectionError("Cannot connect to Binance")

        try:
            if not getattr(self.exchange, "markets", None):
                self.exchange.load_markets()
        except Exception as exc:
            raise DataCollectionError(f"Failed to load markets: {exc}") from exc

        candidates = self._candidate_symbols(symbol)
        for candidate in candidates:
            if candidate in self.exchange.symbols:
                if candidate != symbol:
                    self.logger.debug(
                        "Resolved market symbol variant",
                        extra={"requested": symbol, "resolved": candidate},
                    )
                return candidate
        raise DataCollectionError(
            f"Symbol {symbol} not available on Binance futures. Tried variants: {candidates}"
        )
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: datetime = None, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtener datos OHLCV de Binance"""
        if not self.exchange:
            if not self.connect():
                raise DataCollectionError("Cannot connect to Binance")
        
        try:
            market_symbol = self._resolve_market_symbol(symbol)
            self._rate_limit()
            
            # Convertir datetime a timestamp en ms
            since_ms = None
            if since:
                # Asegurar que since tenga timezone UTC
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
                since_ms = int(since.timestamp() * 1000)
            
            # Obtener datos
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=market_symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=min(limit, 1000)  # Binance limit
            )
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Formatear datos
            df = self.format_ohlcv_data(ohlcv)
            
            if not self.validate_data(df):
                raise DataCollectionError(f"Invalid data received for {symbol}")
            
            self.logger.info(
                f"Collected {len(df)} candles for {symbol} {timeframe}",
                extra={"market_symbol": market_symbol},
            )
            return df
            
        except ccxt.BaseError as e:
            self.logger.error(f"CCXT error collecting {symbol} {timeframe}: {e}")
            raise DataCollectionError(f"CCXT error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error collecting {symbol} {timeframe}: {e}")
            raise DataCollectionError(f"Unexpected error: {e}")
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Obtener datos históricos completos con manejo correcto de timezones"""
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        # Asegurar que las fechas tengan timezone UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        all_data = []
        current_date = start_date
        
        # Calcular tamaño de chunk según timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        
        if timeframe not in timeframe_minutes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # 1000 candles por request
        chunk_size = timedelta(minutes=timeframe_minutes[timeframe] * 1000)
        
        self.logger.info(f"Collecting historical data for {symbol} {timeframe} from {start_date} to {end_date}")
        
        while current_date < end_date:
            chunk_end = min(current_date + chunk_size, end_date)
            
            try:
                df_chunk = self.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=1000
                )
                
                if not df_chunk.empty:
                    # CORRECCIÓN: Convertir timestamps a timezone-aware para comparación
                    # Asegurar que df_chunk['timestamp'] sea timezone-aware
                    if df_chunk['timestamp'].dt.tz is None:
                        df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_localize('UTC')
                    
                    # Convertir current_date y chunk_end a pandas Timestamp con timezone
                    current_date_pd = pd.Timestamp(current_date).tz_convert('UTC')
                    chunk_end_pd = pd.Timestamp(chunk_end).tz_convert('UTC')
                    
                    # Filtrar datos dentro del rango
                    df_chunk = df_chunk[
                        (df_chunk['timestamp'] >= current_date_pd) & 
                        (df_chunk['timestamp'] < chunk_end_pd)
                    ]
                    
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                        
                        # Actualizar current_date al último timestamp + 1 periodo
                        last_timestamp = df_chunk['timestamp'].max()
                        # Convertir back to datetime for next iteration
                        current_date = last_timestamp.to_pydatetime() + timedelta(minutes=timeframe_minutes[timeframe])
                    else:
                        current_date = chunk_end
                else:
                    current_date = chunk_end
                
                # Progress logging
                if (end_date - start_date).total_seconds() > 0:
                    progress = ((current_date - start_date) / (end_date - start_date)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Current date: {current_date}")
                
            except Exception as e:
                self.logger.error(f"Error collecting chunk from {current_date}: {e}")
                current_date += chunk_size  # Skip problematic chunk
                continue
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.drop_duplicates(subset=['timestamp'], inplace=True)
            final_df.sort_values('timestamp', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Collected total {len(final_df)} candles for {symbol} {timeframe}")
            return final_df
        else:
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Obtener símbolos disponibles en futuros"""
        if not self.exchange:
            if not self.connect():
                return []
        
        try:
            markets = self.exchange.load_markets()
            future_symbols = [
                symbol for symbol, market in markets.items()
                if market.get('type') == 'future' and market.get('active')
            ]
            return future_symbols
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Obtener datos adicionales del mercado"""
        if not self.exchange:
            if not self.connect():
                return {}
        
        try:
            self._rate_limit()
            
            # Ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Funding rate (si está disponible)
            funding_rate = None
            try:
                funding_info = self.exchange.fetch_funding_rate(symbol)
                funding_rate = funding_info.get('fundingRate')
            except:
                pass
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'mark_price': ticker.get('last'),
                'index_price': ticker.get('index'),
                'funding_rate': funding_rate,
                'open_interest': ticker.get('info', {}).get('openInterest'),
                'volume_24h': ticker.get('baseVolume'),
                'price_change_24h': ticker.get('change')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
```

---

## 3. ML Training Pipeline

### File: `scripts/train_v2_production.py`

Trains 4 heterogeneous models per symbol: LSTM, TCN, XGBoost, Transformer.

```python
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
```

---

## 4. Ensemble Manager (Council of Sages)

### File: `ml/advanced_models/ensemble_manager.py`

Orchestrates multiple ML models with weighted voting for final prediction.

```python
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

# Importar arquitecturas
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer
from ml.advanced_models.tabular_model import XGBoostTradingModel

logger = logging.getLogger("EnsembleManager")

# Path to ensemble weights config
WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "v2_ensemble" / "ensemble_weights.json"

class EnsembleManager:
    """
    Orquestador del 'Comité de Sabios'.
    Gestiona múltiples modelos heterogéneos y combina sus predicciones.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.configs: Dict[str, Dict] = {}
        # Load default weights from config
        self.load_weights_from_config()
    
    def load_weights_from_config(self, version: str = "default"):
        """
        Carga pesos de votación desde JSON externo.
        Permite ajustar pesos sin recompilar.
        """
        try:
            if WEIGHTS_PATH.exists():
                with open(WEIGHTS_PATH, 'r') as f:
                    all_weights = json.load(f)
                weights_dict = all_weights.get(version, all_weights.get("default", {}))
                
                # Normalizar pesos para que sumen 1.0
                total = sum(weights_dict.values()) if weights_dict else 1.0
                for name, w in weights_dict.items():
                    if not name.startswith("_"):  # Ignorar comentarios
                        self.weights[name] = w / total
                        
                logger.info(f"✅ Loaded ensemble weights for version '{version}': {self.weights}")
            else:
                logger.warning(f"⚠️ Weights config not found at {WEIGHTS_PATH}, using defaults")
        except Exception as e:
            logger.error(f"❌ Failed to load weights config: {e}")
        
    def load_model(self, name: str, model_type: str, model_path: str, config_path: str, weight: float = 1.0):
        """
        Carga un modelo individual al ensemble.
        
        Args:
            name: Identificador único (ej. 'lstm_v1')
            model_type: 'lstm', 'tcn', 'transformer', 'xgboost'
            model_path: Ruta al archivo de pesos (.pt o .joblib)
            config_path: Ruta al json de configuración
            weight: Peso de voto en el ensemble
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Cargar config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.configs[name] = config
        self.weights[name] = weight
        
        # Instanciar arquitectura según tipo
        try:
            if model_type == 'lstm':
                # Extraer params relevantes del config
                # Asumimos que config tiene la estructura de production_training_results.json
                mc = config.get('model_config', config)
                model = DeepTemporalNet(
                    input_dim=mc.get('input_dim', 99), # Fallback si no está en config
                    hidden_dim=mc['hidden_dim'],
                    lstm_layers=mc['lstm_layers'],
                    dropout=mc.get('dropout', 0.2),
                    num_classes=3
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'tcn':
                mc = config.get('model_config', config)
                model = TCNTradingModel(
                    input_dim=mc.get('input_dim', 99),
                    num_channels=mc.get('num_channels', [64, 128, 256]),
                    kernel_size=mc.get('kernel_size', 3),
                    dropout=mc.get('dropout', 0.2)
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'transformer':
                mc = config.get('model_config', config)
                model = TradingTransformer(
                    input_dim=mc.get('input_dim', 99),
                    d_model=mc.get('d_model', 128),
                    nhead=mc.get('nhead', 4),
                    num_layers=mc.get('num_layers', 3)
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'xgboost':
                model = XGBoostTradingModel(use_gpu=(self.device.type == 'cuda'))
                model.load(str(path))
                self.models[name] = model
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            logger.info(f"✅ Loaded {name} ({model_type}) - Weight: {weight}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
            raise e

    def predict(self, x_tensor: torch.Tensor, x_numpy: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Genera predicción combinada.
        
        Args:
            x_tensor: Input para modelos PyTorch (Batch, Seq, Feat)
            x_numpy: Input para XGBoost (Batch, Seq, Feat) [Opcional, si es None se convierte x_tensor]
        """
        x_tensor = x_tensor.to(self.device)
        if x_numpy is None:
            x_numpy = x_tensor.cpu().numpy()
            
        individual_probs = {}
        weighted_probs_sum = torch.zeros(x_tensor.size(0), 3).to(self.device)
        total_weight = 0.0
        
        # Recolectar votos
        for name, model in self.models.items():
            weight = self.weights[name]
            total_weight += weight
            
            if isinstance(model, XGBoostTradingModel):
                # XGBoost output
                out = model.predict(x_numpy)
                probs = torch.from_numpy(out['probs']).to(self.device)
            else:
                # PyTorch output
                with torch.no_grad():
                    out = model(x_tensor)
                    # Convert logits to probs
                    probs = torch.softmax(out['logits'], dim=1)
            
            individual_probs[name] = probs
            weighted_probs_sum += probs * weight
            
        # Normalizar ensemble probs
        ensemble_probs = weighted_probs_sum / total_weight
        
        # Decisión final (Argmax)
        ensemble_class = torch.argmax(ensemble_probs, dim=1)
        
        # Confianza (Probabilidad de la clase elegida)
        confidence, _ = torch.max(ensemble_probs, dim=1)
        
        return {
            'ensemble_probs': ensemble_probs, # (Batch, 3)
            'ensemble_class': ensemble_class, # (Batch, )
            'confidence': confidence,         # (Batch, )
            'individual_votes': individual_probs
        }
        
    def get_consensus_level(self, prediction_result: Dict) -> float:
        """
        Calcula qué tan de acuerdo están los modelos.
        0.0 = Desacuerdo total
        1.0 = Unanimidad
        """
        votes = prediction_result['individual_votes']
        if not votes:
            return 0.0
            
        # Matriz de votos (Num_Models, 3)
        vote_matrix = torch.stack(list(votes.values()))
        
        # Desviación estándar entre las probabilidades de los modelos
        # Si todos dicen lo mismo, std es bajo.
        # Usamos 1 - std_promedio como métrica de consenso (simplificada)
        std_dev = torch.std(vote_matrix, dim=0).mean().item()
        
        # Normalizar un poco (std maximo es ~0.5)
        consensus = max(0.0, 1.0 - (std_dev * 2))
        return consensus
```

---

## 4.1 TCN Model Architecture

### File: `ml/advanced_models/tcn_model.py`

Temporal Convolutional Network with causal convolutions for time series.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Dict, Optional, Tuple

class Chomp1d(nn.Module):
    """
    Elimina el padding extra del futuro para asegurar causalidad.
    Si hacemos padding 'same' en conv1d, leemos del futuro. 
    Chomp corta esos valores extra del final.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Bloque residual básico de TCN:
    Dilated Conv -> WeightNorm -> ReLU -> Dropout -> Dilated Conv -> ...
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Segunda capa convolucional
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Conexión residual (downsample si cambian dimensiones)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Red TCN completa compuesta por múltiples bloques temporales.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i # 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding necesario para mantener la longitud de secuencia con dilatación
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNTradingModel(nn.Module):
    """
    Modelo de Trading basado en TCN.
    Compatible con la interfaz de DeepTemporalNet.
    """
    def __init__(
        self,
        input_dim: int,
        num_channels: list = [64, 128, 256, 512], # Canales por nivel (profundidad)
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        use_regression: bool = True
    ):
        super().__init__()
        
        # TCN Backbone
        # Input shape esperado por TCN: (Batch, Channels, Seq_Len)
        # Nuestros datos vienen como: (Batch, Seq_Len, Features)
        # Haremos transpose en el forward.
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        # Output Heads
        last_channel_dim = num_channels[-1]
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(last_channel_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Regression Head
        self.use_regression = use_regression
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(last_channel_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def forward(self, x, return_features=False) -> Dict[str, torch.Tensor]:
        # x shape: (Batch, Seq_Len, Features)
        # TCN espera: (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)
        
        y = self.tcn(x) # (Batch, Channels, Seq_Len)
        
        # Tomamos solo el último paso de tiempo (el más reciente)
        # TCN es causal, así que el último paso tiene info de toda la historia
        features = y[:, :, -1] # (Batch, Channels)
        
        outputs = {
            'logits': self.classifier(features)
        }
        
        if self.use_regression:
            outputs['regression'] = self.regressor(features)
            
        if return_features:
            outputs['features'] = features
            
        return outputs
```

---

## 5. ML Inference Service

### File: `services/ml_service_v2.py`

FastAPI service that serves ML predictions to the trading bot via HTTP.

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

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import V2 Models
from ml.advanced_models.ensemble_manager import EnsembleManager

# --- Logger Setup ---
class ServiceLogger:
    def __init__(self, name: str = "ml_service_v2") -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        self._logger = logging.getLogger(name)

    def info(self, msg: str, **kwargs): self._logger.info(f"{msg} {kwargs if kwargs else ''}")
    def error(self, msg: str, **kwargs): self._logger.error(f"{msg} {kwargs if kwargs else ''}")
    def warning(self, msg: str, **kwargs): self._logger.warning(f"{msg} {kwargs if kwargs else ''}")

LOGGER = ServiceLogger()

# --- Config ---
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble" # Carpeta futura para modelos V2

# --- Data Models ---
class ProbabilityRequestV2(BaseModel):
    symbol: str # Ej: "ADA/USDT:USDT" o "ADAUSDT"

class ProbabilityResponseV2(BaseModel):
    symbol: str
    long_prob: float
    short_prob: float
    neutral_prob: float
    consensus_level: float
    meta_verdict: str # "APPROVED" | "VETOED"

# --- Data Loader ---
def load_latest_data(symbol: str, limit: int = 60) -> pd.DataFrame:
    """Carga los últimos N registros de la DB V2."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    # Normalizar símbolo para DB (ADAUSDT -> ADA/USDT:USDT)
    # Asumimos que el bot envía formato CCXT o limpio.
    # La DB tiene formato CCXT: "ADA/USDT:USDT"
    db_symbol = symbol
    if "/" not in symbol:
        # Intento simple de conversión si viene como ADAUSDT
        # Esto es frágil, idealmente el bot envía el formato correcto
        pass 

    conn = sqlite3.connect(DB_PATH)
    
    # Query con JOIN y Taker Vol
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
    WHERE o.symbol = '{db_symbol}'
    ORDER BY o.timestamp DESC
    LIMIT {limit}
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        df = df.sort_values('timestamp') # Reordenar ascendente para secuencia
    finally:
        conn.close()
        
    return df

# --- Model Manager ---
class V2ModelManager:
    def __init__(self):
        self.ensembles: Dict[str, EnsembleManager] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # FIX: LRU Tracking para evicción de VRAM
        self.last_accessed: Dict[str, float] = {}
        self.MAX_MODELS_IN_VRAM = 12  # Optimized for GTX 1660 6GB (12 * 250MB ≈ 3GB, leaves 3GB headroom) 
        
        LOGGER.info(f"🚀 ML Service initialized on device: {self.device} | Max Models VRAM: {self.MAX_MODELS_IN_VRAM}")
        
        # ═══════════════════════════════════════════════════════
        # FASE 4.5: FILTRO NINJA (EMA ASIMÉTRICO)
        # ═══════════════════════════════════════════════════════
        self.smoothed_probs_cache = {} 
        # Alphas dinámicos se definen en predict()
        
    def _clean_symbol(self, symbol: str) -> str:
        # ADA/USDT:USDT -> ADAUSDT
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def get_ensemble(self, symbol: str) -> Optional[EnsembleManager]:
        clean_sym = self._clean_symbol(symbol)
        
        # FIX: Actualizar timestamp de acceso (LRU Hit)
        import time
        self.last_accessed[clean_sym] = time.time()
        
        if clean_sym in self.ensembles:
            return self.ensembles[clean_sym]
        
        # FIX: Chequeo de límite de VRAM antes de cargar nuevo modelo
        if len(self.ensembles) >= self.MAX_MODELS_IN_VRAM:
            self._evict_least_recently_used()
            
        # Try to load
        return self.load_model_for_symbol(clean_sym)

    def _evict_least_recently_used(self):
        """Elimina el modelo menos usado recientemente para liberar VRAM."""
        if not self.last_accessed:
            return
            
        oldest = min(self.last_accessed, key=self.last_accessed.get)
        LOGGER.info(f"🗑️ Evicting {oldest} from VRAM (LRU Strategy)")
        
        if oldest in self.ensembles:
            del self.ensembles[oldest]
        if oldest in self.scalers:
            del self.scalers[oldest]
        if oldest in self.feature_cols:
            del self.feature_cols[oldest]
        if oldest in self.last_accessed:
            del self.last_accessed[oldest]
        
        # CRITICAL: Forzar liberación de memoria de caché de PyTorch
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def load_model_for_symbol(self, clean_symbol: str) -> Optional[EnsembleManager]:
        symbol_dir = MODELS_DIR / clean_symbol
        if not symbol_dir.exists():
            LOGGER.warning(f"No models found for {clean_symbol} at {symbol_dir}")
            return None
            
        try:
            LOGGER.info(f"Loading models for {clean_symbol}...")
            ensemble = EnsembleManager(device=self.device)
            
            # Load Scaler
            self.scalers[clean_symbol] = joblib.load(symbol_dir / "scaler.pkl")
            
            # Load Feature Names
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_symbol] = json.load(f)
            
            # Detect version and load weights
            is_v2_1 = len(self.feature_cols[clean_symbol]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            # ensemble = EnsembleManager(device=self.device) # REMOVED: Double instantiation bug
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            # Load Transformer if available (backwards compatible)
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_symbol] = ensemble
            LOGGER.info(f"✅ Loaded {clean_symbol} ensemble.")
            return ensemble
            
        except Exception as e:
            LOGGER.error(f"Failed to load {clean_symbol}: {e}")
            return None

    def load_models(self):
        # Pre-load all available models in directory
        if not MODELS_DIR.exists():
            LOGGER.warning(f"Models dir {MODELS_DIR} not found.")
            return
            
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                self.load_model_for_symbol(item.name)

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        ensemble = self.get_ensemble(symbol)
        clean_sym = self._clean_symbol(symbol)
        
        if ensemble is None:
            # Dummy response if model missing (Fail Safe: Neutral)
            return {
                'ensemble_probs': torch.tensor([[0.0, 1.0, 0.0]]),
                'consensus': 0.0
            }
            
        # 1. Feature Engineering (Derived Features)
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # CONSEJO DE SABIOS v2.1: Agregar Meta-Features si el modelo las requiere
        # ═══════════════════════════════════════════════════════════════════════════
        cols = self.feature_cols.get(clean_sym, [])
        n_features = len(cols)
        
        # Detectar versión basada en número de features (13 = v2.0, 19 = v2.1)
        is_v2_1 = n_features >= 19 or 'mean_obi_12' in cols
        
        if is_v2_1:
            # Calcular meta-features en tiempo real
            window = 12
            df['mean_obi_12'] = df['obi'].rolling(window, min_periods=1).mean()
            df['max_obi_12'] = df['obi'].rolling(window, min_periods=1).max()
            # FIX: std con min_periods=2 para evitar NaN, luego fillna(0) por seguridad
            df['std_obi_12'] = df['obi'].rolling(window, min_periods=2).std().fillna(0)
            
            df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
            df['mean_volume_12'] = df['total_volume'].rolling(window, min_periods=1).mean()
            df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
            # FIX: fillna con método forward/backward para evitar NaN en primeras filas
            df['slope_price_12'] = (df['price'] - df['price'].shift(window).bfill()) / window
            
            # ═══════════════════════════════════════════════════════════════════
            # v2.2: CVD (Cumulative Volume Delta) - El "Medidor de Fuerza"
            # ═══════════════════════════════════════════════════════════════════
            df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window, min_periods=1).sum()
            df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)

            # ═══════════════════════════════════════════════════════════════════
            # v2.2: Volatilidad del Precio - El "Termómetro de Histeria"
            # ═══════════════════════════════════════════════════════════════════
            df['std_price_12'] = df['price'].rolling(window, min_periods=2).std().fillna(0)
            df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
            
            LOGGER.info(f"🧙 Consejo v2.1: Calculated meta-features for {clean_sym} ({n_features} features)")
        else:
            LOGGER.info(f"📊 Consejo v2.0: Using legacy features for {clean_sym} ({n_features} features)")
        
        # FIX #3: Usar orden de columnas EXACTO del features.json (guardado en training)
        try:
            X = df[cols].values
            
            # Sanity check: Detectar NaNs antes del scaler
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                LOGGER.warning(f"⚠️ Found {nan_count} NaNs in features for {clean_sym}, filling with 0")
                X = np.nan_to_num(X, nan=0.0)
                
        except KeyError as e:
            LOGGER.error(f"Missing columns/config for {clean_sym}: {e}")
            LOGGER.error(f"Available columns: {list(df.columns)}")
            LOGGER.error(f"Required columns: {cols}")
            raise e
            
        # 2. Scaling
        scaler = self.scalers[clean_sym]
        
        # FIX #1: Validar dimensiones del scaler vs features
        expected_features = scaler.n_features_in_
        actual_features = X.shape[1]
        
        if expected_features != actual_features:
            LOGGER.error(f"❌ Scaler/Feature mismatch for {clean_sym}!")
            LOGGER.error(f"   Scaler expects: {expected_features} features")
            LOGGER.error(f"   Data has: {actual_features} features")
            LOGGER.error(f"   This likely means model version mismatch (v2.0 scaler with v2.1 features)")
            raise ValueError(f"Feature dimension mismatch: scaler={expected_features}, data={actual_features}")
        
        X_scaled = scaler.transform(X)
        
        # 3. Sequence Creation
        SEQ_LEN = 12
        if len(X_scaled) < SEQ_LEN:
            LOGGER.warning(f"Not enough data for sequence. Need {SEQ_LEN}, got {len(X_scaled)}")
            pad_len = SEQ_LEN - len(X_scaled)
            X_scaled = np.pad(X_scaled, ((pad_len, 0), (0, 0)), mode='edge')
            
        X_seq = X_scaled[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        # 4. Predict
        with torch.no_grad():
            result = ensemble.predict(X_tensor)
            
        # ═══════════════════════════════════════════════════════
        # FASE 4.5: FILTRO NINJA (EMA ASIMÉTRICO)
        # Filosofía: "Subir lento (escéptico), Bajar rápido (paranoico)"
        # ═══════════════════════════════════════════════════════
        ensemble_probs = result['ensemble_probs'][0].tolist()
        
        raw_dict = {
            'short': float(ensemble_probs[0]),
            'neutral': float(ensemble_probs[1]),
            'long': float(ensemble_probs[2])
        }

        # 1. Obtener estado anterior (o usar cruda si es la primera vez)
        prev_smoothed = self.smoothed_probs_cache.get(clean_sym, raw_dict)

        # 2. Definir la personalidad del filtro
        ALPHA_SLOW = 0.15  # Escéptico: Si la señal sube, cuesta trabajo creerla
        ALPHA_FAST = 0.70  # Paranoico: Si la señal baja, reaccionamos YA

        smoothed_dict = {}

        # 3. Aplicar lógica asimétrica
        for key in ['short', 'neutral', 'long']:
            raw_val = raw_dict[key]
            prev_val = prev_smoothed[key]
            
            # Mágia aquí: ¿La señal está mejorando o empeorando?
            diff = raw_val - prev_val
            
            if diff > 0:
                # La probabilidad está subiendo -> PIDE CONFIRMACIÓN (Lento)
                alpha = ALPHA_SLOW
            else:
                # La probabilidad está bajando -> PÁNICO (Rápido)
                alpha = ALPHA_FAST
            
            # Fórmula EMA estándar
            new_val = (alpha * raw_val) + ((1 - alpha) * prev_val)
            smoothed_dict[key] = new_val

        # 4. Normalizar (asegurar que sumen 1.0)
        total = smoothed_dict['short'] + smoothed_dict['neutral'] + smoothed_dict['long']
        normalized_dict = {k: v / total for k, v in smoothed_dict.items()}

        # 5. Guardar en caché para el próximo tick
        self.smoothed_probs_cache[clean_sym] = normalized_dict

        # 6. Actualizar el resultado con el valor suavizado
        result['ensemble_probs'] = torch.tensor([[
            normalized_dict['short'],
            normalized_dict['neutral'],
            normalized_dict['long']
        ]])
            
        return result

MANAGER = V2ModelManager()

# --- API ---
router = APIRouter(prefix="/ml-v2", tags=["ml-v2"])

@router.on_event("startup")
async def startup_event():
    # REMOVED: Pre-loading all models bypassed LRU check
    # Models now load on-demand via get_ensemble() which respects MAX_MODELS_IN_VRAM
    LOGGER.info(f"🚀 ML Service V2 ready (Lazy Loading enabled, Max VRAM: {MANAGER.MAX_MODELS_IN_VRAM} models)")

@router.post("/predict", response_model=ProbabilityResponseV2)
async def predict_endpoint(request: ProbabilityRequestV2) -> ProbabilityResponseV2:
    symbol = request.symbol
    
    # 1. Load Data
    try:
        # Intentamos cargar con el símbolo tal cual, si falla probamos variantes
        df = load_latest_data(symbol)
        if df.empty:
            # Try converting ADAUSDT -> ADA/USDT:USDT
            alt_symbol = symbol.replace("USDT", "/USDT:USDT")
            df = load_latest_data(alt_symbol)
            
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No V2 data found for {symbol}")
            
    except Exception as e:
        LOGGER.error(f"Data load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Predict
    result = MANAGER.predict(symbol, df)
    probs = result['ensemble_probs'][0].tolist() # [Short, Neutral, Long]
    
    return ProbabilityResponseV2(
        symbol=symbol,
        short_prob=probs[0],
        neutral_prob=probs[1],
        long_prob=probs[2],
        consensus_level=float(result.get('consensus', 0.0)),
        meta_verdict="APPROVED" # Placeholder
    )

app = FastAPI(title="ML Service V2 (Ninja)", version="2.0.0")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Puerto 8001 para no chocar con el V1
    uvicorn.run("services.ml_service_v2:app", host="0.0.0.0", port=8001, reload=False)
```

---

## 6. Daily Retrain Script

### File: `scripts/daily_retrain.sh`

Cron-scheduled script that retrains all models using dual AMD GPUs.

```bash
#!/bin/bash

# Config
PROJECT_DIR="/home/jasan/Develop/trading_system"
LOG_FILE="$PROJECT_DIR/logs/daily_retrain.log"
DATE=$(date)

echo "[$DATE] 🚀 Starting Daily Retraining (Priority Mode)..." >> $LOG_FILE

# 1. Go to project dir
cd $PROJECT_DIR

# 2. Set AMD ROCm Environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY TRAINING: Production symbols first (9 active), then others (12)
# ═══════════════════════════════════════════════════════════════════════════

# PHASE 1: Production Symbols (HIGH PRIORITY)
# These are the 9 symbols actively trading - train them FIRST
# Source: binance-futures-bot-ts/.env SYMBOLS variable
# Format: Must match DB format (ADA/USDT:USDT, not ADAUSDT)
PRIORITY_SYMBOLS=(
    "DOGE/USDT:USDT"
    "LINK/USDT:USDT"
    "AVAX/USDT:USDT"
    "POL/USDT:USDT"
    "ETH/USDT:USDT"
    "XRP/USDT:USDT"
    "SOL/USDT:USDT"
    "ADA/USDT:USDT"
    "BTC/USDT:USDT"
)

echo "[$DATE] 🎯 PHASE 1: Training PRIORITY symbols (${#PRIORITY_SYMBOLS[@]} production symbols)..." >> $LOG_FILE

for symbol in "${PRIORITY_SYMBOLS[@]}"; do
    echo "[$DATE]   Training $symbol (Priority)..." >> $LOG_FILE
    
    # Alternate between GPU 0 and GPU 1 for load balancing
    export HIP_VISIBLE_DEVICES=$((RANDOM % 2))
    $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol" >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$DATE]   ✅ $symbol complete" >> $LOG_FILE
    else
        echo "[$DATE]   ❌ $symbol failed" >> $LOG_FILE
    fi
done

echo "[$DATE] ✅ PHASE 1 Complete: Priority symbols trained." >> $LOG_FILE

# PHASE 2: Secondary Symbols (LOWER PRIORITY)
# These are not actively trading but we keep models fresh
# Format: Must match DB format (ADA/USDT:USDT, not ADAUSDT)
SECONDARY_SYMBOLS=(
    "BNB/USDT:USDT"
    "DOT/USDT:USDT"
    "LTC/USDT:USDT"
    "UNI/USDT:USDT"
    "ATOM/USDT:USDT"
    "NEAR/USDT:USDT"
    "1000PEPE/USDT:USDT"
    "FET/USDT:USDT"
    "SEI/USDT:USDT"
    "WLD/USDT:USDT"
    "INJ/USDT:USDT"
    "APT/USDT:USDT"
)

echo "[$DATE] 📦 PHASE 2: Training SECONDARY symbols (${#SECONDARY_SYMBOLS[@]} backup symbols)..." >> $LOG_FILE

# ═══════════════════════════════════════════════════════════════════════════
# FIX: Train 2 symbols at a time (1 per GPU), wait for both before next pair
# This prevents GPU memory exhaustion on RX 6600
# ═══════════════════════════════════════════════════════════════════════════
TOTAL=${#SECONDARY_SYMBOLS[@]}
for ((i=0; i<TOTAL; i+=2)); do
    # Symbol for GPU 0 (even index)
    symbol_0="${SECONDARY_SYMBOLS[$i]}"
    
    # Symbol for GPU 1 (odd index, if exists)
    symbol_1=""
    if [ $((i+1)) -lt $TOTAL ]; then
        symbol_1="${SECONDARY_SYMBOLS[$((i+1))]}"
    fi
    
    echo "[$DATE]   Training pair: $symbol_0 (GPU 0) + $symbol_1 (GPU 1)..." >> $LOG_FILE
    
    # Launch GPU 0 job
    HIP_VISIBLE_DEVICES=0 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_0" >> $LOG_FILE 2>&1 &
    PID_0=$!
    
    # Launch GPU 1 job (if symbol exists)
    if [ -n "$symbol_1" ]; then
        HIP_VISIBLE_DEVICES=1 $PROJECT_DIR/.venv_rocm62/bin/python3 scripts/train_v2_production.py --symbol "$symbol_1" >> $LOG_FILE 2>&1 &
        PID_1=$!
        wait $PID_0 $PID_1
    else
        wait $PID_0
    fi
    
    echo "[$DATE]   ✅ Pair complete: $symbol_0 + $symbol_1" >> $LOG_FILE
done

echo "[$DATE] ✅ PHASE 2 Complete: Secondary symbols trained." >> $LOG_FILE

# 3. Reload ML Service to pick up new models
pm2 reload 03-ML-Service-V2 >> $LOG_FILE 2>&1
echo "[$DATE] 🔄 ML Service reloaded with new models." >> $LOG_FILE

echo "[$DATE] 🎉 Daily Retraining Complete!" >> $LOG_FILE
```

---

## 7. Backtest Engine

### File: `scripts/backtest_v6_real_ml.py`

Production-grade backtest that replicates live bot logic with real ML predictions.

```python
#!/usr/bin/env python3
"""
NINJA v6.0 REAL ML: Production Backtest Engine with Real Model Predictions
Replica EXACTAMENTE la lógica del bot v5.1.1 CON modelos ML entrenados.

Uso:
    source .venv_rocm62/bin/activate
    python scripts/backtest_v6_real_ml.py --symbol DOGEUSDT --days 7
    python scripts/backtest_v6_real_ml.py --symbol ALL --days 7
"""

import argparse
import sqlite3
import sys
import json
import os
import time as time_module
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════
# ROCm Environment Setup (MUST be before torch import)
# ═══════════════════════════════════════════════════════════════════════════
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Para RX 6600
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')  # Use first GPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import joblib

# Agregar rutas del proyecto
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import real ML models
from ml.advanced_models.ensemble_manager import EnsembleManager

# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIVERSAL PROFIT GUARDIAN (Port of TS)
# ═════════════════════════════════════════════════════════════════════════════════════
class UniversalProfitGuardian:
    def __init__(self, config: Dict):
        self.config = config

    def evaluate(self, ctx: Dict) -> bool:
        if ctx['peakRoe'] < self.config['peakThreshold']:
            return False
        drawdown = ctx['peakRoe'] - ctx['currentRoe']
        volMultiplier = 1.0
        if ctx['volatilityFactor'] >= 1.5:
            volMultiplier = 1.0 - (self.config['volatilitySensitivity'] * 0.5)
        elif ctx['volatilityFactor'] <= 0.8:
            volMultiplier = 1.0 + (self.config['volatilitySensitivity'] * 0.5)
        volMultiplier = max(0.5, min(1.5, volMultiplier))
        allowedDrawdown = self.config['baseDrawdown'] * volMultiplier
        if drawdown >= allowedDrawdown:
            if self.config['enableTrendProtection']:
                biasFavorsMe = (
                    (ctx['positionSide'] == 'LONG' and ctx['marketBias'] == 'BULL') or
                    (ctx['positionSide'] == 'SHORT' and ctx['marketBias'] == 'BEAR')
                )
                if biasFavorsMe:
                    return False
            return True
        return False

    @staticmethod
    def WHALE_CONFIG():
        return {'peakThreshold': 0.015, 'baseDrawdown': 0.40, 'volatilitySensitivity': 0.20, 'enableTrendProtection': True}
    @staticmethod
    def MONK_CONFIG():
        return {'peakThreshold': 0.01, 'baseDrawdown': 0.25, 'volatilitySensitivity': 0.30, 'enableTrendProtection': True}
    @staticmethod
    def BLOODBATH_CONFIG():
        return {'peakThreshold': 0.005, 'baseDrawdown': 0.15, 'volatilitySensitivity': 0.50, 'enableTrendProtection': False}

# ═══════════════════════════════════════════════════════════════════════════
# 2. REAL ML MODEL MANAGER (From ml_service_v2.py)
# ═══════════════════════════════════════════════════════════════════════════
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"

class RealMLPredictor:
    def __init__(self):
        self.ensembles: Dict[str, EnsembleManager] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        # Force CPU for backtest stability (GPU used by live service)
        self.device = "cpu"
        print(f"[ML] Device: {self.device}")

    def _clean_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def load_model(self, symbol: str) -> bool:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym in self.ensembles:
            return True
        
        symbol_dir = MODELS_DIR / clean_sym
        if not symbol_dir.exists():
            print(f"[ML] No models for {clean_sym}")
            return False
        
        try:
            self.scalers[clean_sym] = joblib.load(symbol_dir / "scaler.pkl")
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_sym] = json.load(f)
            
            is_v2_1 = len(self.feature_cols[clean_sym]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            ensemble = EnsembleManager(device=self.device)
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_sym] = ensemble
            print(f"[ML] ✅ Loaded {clean_sym} ({len(self.feature_cols[clean_sym])} features)")
            return True
        except Exception as e:
            print(f"[ML] ❌ Failed to load {clean_sym}: {e}")
            return False

    def predict(self, symbol: str, df: pd.DataFrame) -> Dict:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym not in self.ensembles:
            if not self.load_model(symbol):
                return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
        
        cols = self.feature_cols.get(clean_sym, [])
        is_v2_1 = len(cols) >= 19 or 'mean_obi_12' in cols
        
        if is_v2_1:
            window = 12
            df['mean_obi_12'] = df['obi'].rolling(window, min_periods=1).mean()
            df['max_obi_12'] = df['obi'].rolling(window, min_periods=1).max()
            df['std_obi_12'] = df['obi'].rolling(window, min_periods=2).std().fillna(0)
            df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
            df['mean_volume_12'] = df['total_volume'].rolling(window, min_periods=1).mean()
            df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
            df['slope_price_12'] = (df['price'] - df['price'].shift(window).bfill()) / window
            df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window, min_periods=1).sum()
            df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)
            df['std_price_12'] = df['price'].rolling(window, min_periods=2).std().fillna(0)
            df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
        
        try:
            X = df[cols].values
            X = np.nan_to_num(X, nan=0.0)
        except KeyError as e:
            print(f"[ML] Missing columns for {clean_sym}: {e}")
            return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        scaler = self.scalers[clean_sym]
        X_scaled = scaler.transform(X)
        
        SEQ_LEN = 12
        if len(X_scaled) < SEQ_LEN:
            pad_len = SEQ_LEN - len(X_scaled)
            X_scaled = np.pad(X_scaled, ((pad_len, 0), (0, 0)), mode='edge')
        
        X_seq = X_scaled[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.ensembles[clean_sym].predict(X_tensor)
        
        probs = result['ensemble_probs'][0].tolist()
        return {
            'shortProb': float(probs[0]),
            'neutralProb': float(probs[1]),
            'longProb': float(probs[2])
        }

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFIG & REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class NinjaConfigLoader:
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

    def get_regime_config(self, regime: str, symbol: str = None) -> Dict:
        regime_key = regime.upper()
        base = self.data['REGIMES'][regime_key]
        merged = base.copy()
        if symbol and self.data.get('SYMBOL_OVERRIDES'):
            sym_overrides = self.data['SYMBOL_OVERRIDES'].get(symbol, {})
            reg_overrides = sym_overrides.get(regime_key, {})
            if reg_overrides:
                merged.update(reg_overrides)
        return merged

class RegimeDetector:
    def __init__(self, config_loader):
        self.last_regime = 'BUNKER'
        self.regime_sticky_counter = 0
        self.config = config_loader.data['REGIME_DETECTOR']

    def analyze(self, snapshot: Dict) -> Dict:
        volatility = 'LOW'
        if snapshot['spreadPct'] > self.config['volatility_spread_high']:
            volatility = 'HIGH'
        elif snapshot['spreadPct'] > self.config['volatility_spread_low']:
            volatility = 'MED'
        
        bias = 'NEUTRAL'
        diff = snapshot['longProb'] - snapshot['shortProb']
        if diff > self.config['bias_strength_threshold']:
            bias = 'BULL'
        elif diff < -self.config['bias_strength_threshold']:
            bias = 'BEAR'
        
        raw_regime = 'BUNKER'
        if volatility == 'HIGH' and snapshot['neutralProb'] > 0.50:
            raw_regime = 'BLOODBATH'
        elif volatility == 'MED' and bias != 'NEUTRAL':
            raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias != 'NEUTRAL':
            raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias == 'NEUTRAL':
            raw_regime = 'MONK'
        
        thresholds = {'BLOODBATH': 3, 'WHALE': 12, 'MONK': 6, 'BUNKER': 2}
        sticky_threshold = thresholds.get(self.last_regime, 6)
        if raw_regime == self.last_regime:
            self.regime_sticky_counter = sticky_threshold
        else:
            self.regime_sticky_counter -= 1
        
        if self.regime_sticky_counter <= 0:
            self.last_regime = raw_regime
            self.regime_sticky_counter = thresholds.get(raw_regime, 6)
        else:
            raw_regime = self.last_regime
        
        return {'type': raw_regime, 'bias': bias, 'volatility': volatility}

# ═══════════════════════════════════════════════════════════════════════════
# 4. STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
class WhaleStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.WHALE_CONFIG())
    def get_config(self, symbol): return self.config_loader.get_regime_config('WHALE', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'WHALE_HARD_STOP'
        if ctx['opposingProb'] > 0.80: return 'WHALE_PANIC_EXTREME'
        if self.guardian.evaluate(ctx): return 'WHALE_DYNAMIC_LOCK'
        return None

class MonkStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.MONK_CONFIG())
    def get_config(self, symbol): return self.config_loader.get_regime_config('MONK', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'MONK_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']: return 'MONK_RANGE_TP'
        if self.guardian.evaluate(ctx): return 'MONK_DYNAMIC_LOCK'
        return None

class BloodbathStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
    def get_config(self, symbol): return self.config_loader.get_regime_config('BLOODBATH', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'BLOODBATH_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']: return 'BLOODBATH_MICRO_TP'
        if ctx['opposingProb'] > 0.55: return 'BLOODBATH_PANIC_FAST'
        return None

class BunkerStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
    def get_config(self, symbol): return self.config_loader.get_regime_config('BUNKER', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        if ctx['currentRoe'] < -0.05: return 'BUNKER_STOP_LOSS'
        if ctx['opposingProb'] > 0.60: return 'BUNKER_PANIC_EXIT'
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
class NinjaBacktester:
    def __init__(self, config_path=None):
        self.config_loader = NinjaConfigLoader(config_path)
        self.detector = RegimeDetector(self.config_loader)
        self.ml_predictor = RealMLPredictor()
        self.strategies = {
            'WHALE': WhaleStrategy(self.config_loader),
            'MONK': MonkStrategy(self.config_loader),
            'BLOODBATH': BloodbathStrategy(self.config_loader),
            'BUNKER': BunkerStrategy(self.config_loader)
        }
        # Post-Exit Gate state (matches live bot behavior)
        self.post_exit_data = {}

    def evaluate_post_exit_gate(self, symbol: str, current_price: float, timestamp: int) -> bool:
        """
        Post-Exit Gate Logic - Aggressive re-entry on pullback/breakout.
        Matches TypeScript bot's evaluatePostExitGate behavior.
        """
        if symbol not in self.post_exit_data:
            return True  # No gate, allow entry
        
        gate = self.post_exit_data[symbol]
        if gate.get('ready', False):
            return True  # Gate already cleared
        
        # Gate parameters (match live bot)
        pullback_pct = 0.006    # 0.6% drop required
        rebound_pct = 0.35      # 35% rebound of the drop
        breakout_pct = 0.0015   # 0.15% above exit price
        timeout_ms = 300_000    # 5 minutes timeout
        
        exit_price = gate.get('exit_price', current_price)
        exit_time = gate.get('exit_time', 0)
        
        # Timeout check
        if timestamp - exit_time > timeout_ms:
            gate['ready'] = True
            gate['reason'] = 'timeout'
            return True
        
        # Track min/max since exit
        gate['min_price'] = min(gate.get('min_price', exit_price), current_price)
        gate['max_price'] = max(gate.get('max_price', exit_price), current_price)
        
        # PULLBACK: Price dropped, then rebounded
        drop = exit_price - gate['min_price']
        if drop > 0:
            drop_threshold = exit_price * pullback_pct
            if drop >= drop_threshold:
                rebound_target = gate['min_price'] + (drop * rebound_pct)
                if current_price >= rebound_target:
                    gate['ready'] = True
                    gate['reason'] = 'pullback'
                    return True
        
        # BREAKOUT: Price broke above exit price
        breakout_target = exit_price * (1 + breakout_pct)
        if current_price >= breakout_target:
            gate['ready'] = True
            gate['reason'] = 'breakout'
            return True
        
        return False  # Still waiting

    def load_data(self, symbol: str, days: int) -> pd.DataFrame:
        DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        conn = sqlite3.connect(DB_PATH)
        # Full query with ALL columns needed for ML features
        query = f"""
        SELECT o.timestamp, o.mid_price as price, o.micro_price,
               o.spread_pct, o.spread_pct as bid_ask_spread,
               o.obi_5, o.obi_10, o.obi_20 as obi,
               o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth,
               d.funding_rate, d.open_interest,
               d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{symbol}' OR o.symbol = '{symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        # Fill NaN values
        df = df.fillna(0)
        return df

    def run(self, symbol: str, days: int, initial_capital: float = 1000.0):
        print(f"\n{'='*60}")
        print(f"BACKTEST V6.0 REAL ML: {symbol} ({days} Days)")
        print(f"{'='*60}")

        df = self.load_data(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"[Data] Loaded {len(df)} rows")
        
        if not self.ml_predictor.load_model(symbol):
            print(f"❌ No ML model for {symbol}")
            return None

        # Pre-compute predictions in batches for speed
        print(f"[ML] Computing predictions...")
        predictions = []
        SEQ_LEN = 12
        step = 10  # Predict every 10 ticks to speed up
        
        for i in range(SEQ_LEN, len(df), step):
            window_df = df.iloc[max(0, i-60):i+1].copy()
            pred = self.ml_predictor.predict(symbol, window_df)
            predictions.append((i, pred))
        
        print(f"[ML] Generated {len(predictions)} predictions")
        
        # Create prediction lookup
        pred_lookup = {}
        last_pred = {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        pred_idx = 0
        
        for i in range(len(df)):
            if pred_idx < len(predictions) and i >= predictions[pred_idx][0]:
                last_pred = predictions[pred_idx][1]
                pred_idx += 1
            pred_lookup[i] = last_pred

        # Simulation
        balance = initial_capital
        position = None
        entry_price = 0
        entry_time = None
        qty = 0
        peak_roe = 0
        trades = []
        equity_curve = []
        last_exit_time = 0
        commission_rate = 0.0004

        for i in range(SEQ_LEN, len(df)):
            row = df.iloc[i]
            timestamp = row['timestamp']
            price = row['price']
            preds = pred_lookup[i]
            
            snapshot = {
                'longProb': preds['longProb'],
                'shortProb': preds['shortProb'],
                'neutralProb': preds['neutralProb'],
                'spreadPct': row['spread_pct'] if pd.notna(row['spread_pct']) else 0.0004,
                'fundingRate': row['funding_rate'] if pd.notna(row['funding_rate']) else 0,
                'obi': row['obi'] if pd.notna(row['obi']) else 0
            }

            ctx = {
                'currentRoe': 0.0, 'peakRoe': 0.0, 'holdTimeMs': 0,
                'opposingProb': 0.0, 'neutralProb': snapshot['neutralProb'],
                'volatilityFactor': snapshot['spreadPct'] / 0.0004,
                'marketBias': 'NEUTRAL', 'positionSide': position
            }

            if position:
                roi = (price - entry_price) / entry_price * (1 if position == 'LONG' else -1)
                if roi > peak_roe:
                    peak_roe = roi
                ctx['currentRoe'] = roi
                ctx['peakRoe'] = peak_roe
                ctx['holdTimeMs'] = timestamp - entry_time
                ctx['marketBias'] = 'BULL' if snapshot['longProb'] > snapshot['shortProb'] else ('BEAR' if snapshot['shortProb'] > snapshot['longProb'] else 'NEUTRAL')
                ctx['opposingProb'] = snapshot['shortProb'] if position == 'LONG' else snapshot['longProb']

            exit_reason = None
            if position:
                regime_data = self.detector.analyze(snapshot)
                strategy = self.strategies[regime_data['type']]
                reason = strategy.evaluate_exit(ctx, symbol)
                if reason:
                    exit_reason = reason

            if position and exit_reason:
                pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                fee = abs(price * qty) * commission_rate * 2
                balance += pnl - fee
                trades.append({
                    'entry_time': entry_time, 'exit_time': timestamp,
                    'side': position, 'entry_price': entry_price, 'exit_price': price,
                    'roi_pct': ctx['currentRoe'] * 100, 'pnl': pnl - fee, 'reason': exit_reason
                })
                position = None
                peak_roe = 0
                last_exit_time = timestamp

            # v6.0 ORIGINAL: Fixed 30-minute cooldown (proven best results)
            if not position and (timestamp - last_exit_time) > 30 * 60 * 1000:
                regime_data = self.detector.analyze(snapshot)
                strategy = self.strategies[regime_data['type']]
                config = strategy.get_config(symbol)
                
                if config['leverage'] == 0:
                    continue

                thr = config['entry_threshold']
                if snapshot['longProb'] > thr:
                    position = 'LONG'
                elif snapshot['shortProb'] > thr:
                    position = 'SHORT'
                
                if position:
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * config['leverage']) / price
                    fee = (price * qty) * commission_rate
                    balance -= fee

            equity_curve.append(balance)

        # Report
        if trades:
            df_trades = pd.DataFrame(trades)
            wins = df_trades[df_trades['pnl'] > 0]
            total_pnl = df_trades['pnl'].sum()
            
            print(f"\n📊 RESULTS:")
            print(f"Capital Final: ${balance:.2f}")
            print(f"Retorno: {((balance - initial_capital)/initial_capital)*100:+.2f}%")
            print(f"Trades: {len(trades)} | WR: {len(wins)/len(trades)*100:.1f}%")
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"\nExit Reasons:\n{df_trades['reason'].value_counts()}")
            
            safe_sym = symbol.replace('/', '_').replace(':', '_')
            df_trades.to_csv(REPO_ROOT / f"backtest_v6_ml_{safe_sym}.csv", index=False)
            
            plt.figure(figsize=(12, 5))
            plt.plot(equity_curve)
            plt.title(f"Backtest V6.0 REAL ML - {symbol}")
            plt.xlabel("Ticks")
            plt.ylabel("Balance ($)")
            plt.grid(True, alpha=0.3)
            plt.savefig(REPO_ROOT / f"backtest_v6_ml_{safe_sym}.png")
            plt.close()
            print(f"💾 Saved: backtest_v6_ml_{safe_sym}.csv/.png")
            
            return {'symbol': symbol, 'return_pct': ((balance - initial_capital)/initial_capital)*100, 'trades': len(trades), 'win_rate': len(wins)/len(trades)*100}
        else:
            print("No trades executed.")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    config_path = REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
    backtester = NinjaBacktester(config_path=str(config_path))
    
    PROD_SYMBOLS = ["DOGEUSDT", "LINKUSDT", "AVAXUSDT", "POLUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "BTCUSDT"]
    
    if args.symbol == "ALL":
        print("🚀 Running REAL ML backtest for all production symbols...")
        results = []
        for sym in PROD_SYMBOLS:
            try:
                result = backtester.run(sym, args.days)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ {sym}: {e}")
        
        if results:
            print("\n" + "="*60)
            print("📊 FLEET SUMMARY")
            print("="*60)
            df_results = pd.DataFrame(results)
            print(df_results.to_string(index=False))
            print(f"\nAvg Return: {df_results['return_pct'].mean():+.2f}%")
            print(f"Avg Win Rate: {df_results['win_rate'].mean():.1f}%")
    else:
        backtester.run(args.symbol, args.days)
```

---

## 8. Grid Search Optimizer

### File: `scripts/optimize_grid_search.py`

Finds optimal trading parameters by testing 192 parameter combinations.

```python
#!/usr/bin/env python3
"""
NINJA v6.0: Grid Search Optimizer
Finds optimal trading parameters by testing parameter combinations.

Usage:
    python scripts/optimize_grid_search.py --symbol AVAXUSDT --days 12
"""

import os
# ROCm Setup (must be before torch import)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import numpy as np
import sqlite3
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import torch
from ml.advanced_models.ensemble_manager import EnsembleManager

# ═══════════════════════════════════════════════════════════════════════════
# GRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID = {
    'leverage': [3, 5, 10],
    'entry_threshold': [0.40, 0.45, 0.50, 0.55],
    'hard_stop_roe': [-0.02, -0.03, -0.04, -0.05],
    'tp_roe': [0.015, 0.02, 0.025, 0.03]
}

# ═══════════════════════════════════════════════════════════════════════════
# ML PREDICTOR (Reused from backtest)
# ═══════════════════════════════════════════════════════════════════════════
import json
import joblib

MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"

class RealMLPredictor:
    def __init__(self):
        self.ensembles = {}
        self.scalers = {}
        self.feature_cols = {}
        self.device = "cpu"  # CPU for stability
        print(f"[ML] Device: {self.device}")

    def _clean_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def load_model(self, symbol: str) -> bool:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym in self.ensembles:
            return True
        
        symbol_dir = MODELS_DIR / clean_sym
        if not symbol_dir.exists():
            print(f"[ML] No models for {clean_sym}")
            return False
        
        try:
            self.scalers[clean_sym] = joblib.load(symbol_dir / "scaler.pkl")
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_sym] = json.load(f)
            
            is_v2_1 = len(self.feature_cols[clean_sym]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            ensemble = EnsembleManager(device=self.device)
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_sym] = ensemble
            print(f"[ML] ✅ Loaded {clean_sym} ({len(self.feature_cols[clean_sym])} features)")
            return True
        except Exception as e:
            print(f"[ML] ❌ Failed to load {clean_sym}: {e}")
            return False

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym not in self.ensembles:
            if not self.load_model(symbol):
                return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['net_taker_flow'] = df['taker_buy_vol'] - df['taker_sell_vol']
        df['depth_imbalance'] = (df['bid_depth_20'] - df['ask_depth_20']) / (df['bid_depth_20'] + df['ask_depth_20'] + 1e-8)
        df['micro_price_delta'] = df['micro_price'] - df['price']
        
        feature_cols = self.feature_cols[clean_sym]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].values
        if len(X) < 12:
            return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        X_scaled = self.scalers[clean_sym].transform(X)
        X_seq = np.expand_dims(X_scaled[-12:], axis=0)
        
        # Convert to tensor for pytorch models
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        result = self.ensembles[clean_sym].predict(X_tensor)
        probs = result['ensemble_probs'][0].cpu().numpy()  # First batch item
        return {'longProb': float(probs[0]), 'shortProb': float(probs[1]), 'neutralProb': float(probs[2])}

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
class NinjaOptimizer:
    def __init__(self):
        self.ml_predictor = RealMLPredictor()
        self.df_cache = {}
        self.pred_cache = {}

    def load_data(self, symbol: str, days: int) -> pd.DataFrame:
        if symbol in self.df_cache:
            return self.df_cache[symbol]
            
        DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT o.timestamp, o.mid_price as price, o.spread_pct, o.obi_20 as obi,
               o.micro_price, o.obi_5, o.obi_10, o.spread_pct as bid_ask_spread,
               o.bid_depth_20, o.ask_depth_20, d.funding_rate, d.open_interest,
               d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{symbol}' OR o.symbol = '{symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = df.fillna(0)
        self.df_cache[symbol] = df
        return df

    def get_predictions(self, symbol: str, df: pd.DataFrame) -> dict:
        if symbol in self.pred_cache:
            return self.pred_cache[symbol]
            
        if not self.ml_predictor.load_model(symbol):
            return None
            
        SEQ_LEN = 12
        step = 20  # Faster for grid search
        predictions = []
        
        for i in range(SEQ_LEN, len(df), step):
            window_df = df.iloc[max(0, i-60):i+1].copy()
            pred = self.ml_predictor.predict(symbol, window_df)
            predictions.append((i, pred))
        
        # Build lookup
        pred_lookup = {}
        last_pred = {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        pred_idx = 0
        for i in range(len(df)):
            if pred_idx < len(predictions) and i >= predictions[pred_idx][0]:
                last_pred = predictions[pred_idx][1]
                pred_idx += 1
            pred_lookup[i] = last_pred
        
        self.pred_cache[symbol] = pred_lookup
        return pred_lookup

    def simulate(self, symbol: str, df: pd.DataFrame, pred_lookup: dict,
                 lev: int, entry_th: float, sl: float, tp: float,
                 initial_capital: float = 1000.0) -> dict:
        """Fast simulation with specific parameters."""
        balance = initial_capital
        position = None
        entry_price = 0
        entry_time = 0
        qty = 0
        trades = []
        last_exit_time = 0
        SEQ_LEN = 12

        for i in range(SEQ_LEN, len(df)):
            row = df.iloc[i]
            timestamp = row['timestamp']
            price = row['price']
            preds = pred_lookup[i]
            
            # EXIT LOGIC
            if position:
                roi = (price - entry_price) / entry_price * (1 if position == 'LONG' else -1)
                
                # Hard Stop
                if roi < sl:
                    pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                    balance += pnl - abs(price * qty) * 0.0008
                    trades.append({'roi': roi, 'pnl': pnl, 'reason': 'STOP'})
                    position = None
                    last_exit_time = timestamp
                    continue
                
                # Take Profit
                if roi >= tp:
                    pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                    balance += pnl - abs(price * qty) * 0.0008
                    trades.append({'roi': roi, 'pnl': pnl, 'reason': 'TP'})
                    position = None
                    last_exit_time = timestamp
                    continue
            
            # ENTRY LOGIC
            if not position and (timestamp - last_exit_time) > 30 * 60 * 1000:
                long_p = preds['longProb']
                short_p = preds['shortProb']
                
                if long_p > entry_th:
                    position = 'LONG'
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * lev) / price
                    balance -= qty * price * 0.0004
                elif short_p > entry_th:
                    position = 'SHORT'
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * lev) / price
                    balance -= qty * price * 0.0004

        if not trades:
            return None
        
        df_t = pd.DataFrame(trades)
        total_pnl = balance - initial_capital
        net_return = (total_pnl / initial_capital) * 100
        num_trades = len(df_t)
        
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'net_return': net_return,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'final_balance': balance
        }

    def optimize(self, symbol: str, days: int):
        print(f"\n{'='*60}")
        print(f"🔍 GRID SEARCH: {symbol} ({days} days)")
        print(f"{'='*60}")
        
        df = self.load_data(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        print(f"[Data] Loaded {len(df)} rows")
        
        pred_lookup = self.get_predictions(symbol, df)
        if pred_lookup is None:
            print(f"❌ No ML model for {symbol}")
            return None
        print(f"[ML] Predictions ready")
        
        # Generate all combinations
        combos = list(product(
            GRID['leverage'],
            GRID['entry_threshold'],
            GRID['hard_stop_roe'],
            GRID['tp_roe']
        ))
        print(f"[Grid] Testing {len(combos)} combinations...")
        
        results = []
        for i, (lev, entry_th, sl, tp) in enumerate(combos):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(combos)}")
            
            result = self.simulate(symbol, df, pred_lookup, lev, entry_th, sl, tp)
            if result and result['profit_factor'] > 0:
                result['params'] = {'leverage': lev, 'entry_threshold': entry_th, 
                                   'hard_stop_roe': sl, 'tp_roe': tp}
                results.append(result)
        
        if not results:
            print("❌ No profitable configurations found")
            return None
        
        # Sort by profit factor
        results.sort(key=lambda x: x['profit_factor'], reverse=True)
        
        # Show top 5
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 60)
        for i, r in enumerate(results[:5]):
            p = r['params']
            print(f"{i+1}. Lev={p['leverage']} | Entry={p['entry_threshold']} | Stop={p['hard_stop_roe']} | TP={p['tp_roe']}")
            print(f"   Return: {r['net_return']:+.1f}% | WR: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f} | Trades: {r['num_trades']}")
        
        best = results[0]
        bp = best['params']
        print(f"\n{'='*60}")
        print(f"✅ BEST CONFIG FOR {symbol}:")
        print(f"{'='*60}")
        print(f"  leverage: {bp['leverage']}")
        print(f"  entry_threshold: {bp['entry_threshold']}")
        print(f"  hard_stop_roe: {bp['hard_stop_roe']}")
        print(f"  tp_roe: {bp['tp_roe']}")
        print(f"\n  Expected: Return={best['net_return']:+.1f}% | WR={best['win_rate']:.1f}% | PF={best['profit_factor']:.2f}")
        
        print(f"\n💡 YAML Override:")
        print(f"  {symbol}:")
        print(f"    MONK: {{ leverage: {bp['leverage']}, entry_threshold: {bp['entry_threshold']}, hard_stop_roe: {bp['hard_stop_roe']}, tp_roe: {bp['tp_roe']} }}")
        
        return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--days", type=int, default=12)
    args = parser.parse_args()
    
    optimizer = NinjaOptimizer()
    optimizer.optimize(args.symbol, args.days)
```

---

## 9. System Summary

### Key Concepts

1. **Council of Sages (Comité de Sabios)**: Ensemble of heterogeneous models (TCN, XGBoost, Transformer) that vote on market direction.

2. **Regime Detection**: Classifies market into 4 states:
   - **WHALE**: Trending markets → Wide stops, ride the trend
   - **MONK**: Ranging markets → Tight TP, scalp the range
   - **BLOODBATH**: High volatility → Fast exits, micro profits
   - **BUNKER**: No trades → Protect capital

3. **Universal Profit Guardian**: Dynamic trailing stop that protects profits while allowing room to breathe.

4. **Ninja Filter (EMA Asimétrico)**: Asymmetric smoothing - slow to trust up moves, fast to panic on down moves.

### Trade Flow

```
1. Data Collector → SQLite (order book, funding, taker vol)
2. ML Service loads data, calculates meta-features
3. Ensemble predicts: P(long), P(short), P(neutral)
4. Bot receives probabilities via HTTP
5. Regime Detector classifies market state
6. Strategy evaluates entry (if P > threshold)
7. Guardian monitors position, locks profits
8. Exit on TP, Stop, or panic signal
```

### Performance (Backtest v6.0, 12 days)

| Symbol | Return | Win Rate |
|--------|--------|----------|
| AVAX   | +100%  | 90.9%    |
| BNB    | +73%   | 100%     |
| ETH    | +54%   | 100%     |
| BTC    | +17%   | 100%     |

---

*Generated by NINJA Trading System Documentation v6.0*
