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
import redis
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
# --- Data Loader ---
# Conexión Global a Redis
r_cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def load_latest_data(symbol: str, limit: int = 100) -> pd.DataFrame:
    """
    Carga datos ultrarrápidos desde Redis (Hot Path).
    """
    # Normalizar símbolo para Redis (ej: "market:BTC/USDT:USDT")
    # Si viene como "BTCUSDT", intentar convertir o usar tal cual si el collector usa ese formato.
    # El collector usa el símbolo que le pasen. Asumimos consistencia.
    redis_key = f"market:{symbol}"
    
    # LRANGE obtiene elementos de la lista. -limit es "los últimos N"
    raw_data = r_cache.lrange(redis_key, -limit, -1)
    
    if not raw_data or len(raw_data) < limit:
        # Fallback o Warm-up
        # LOGGER.warning(f"Redis miss/insufficient data for {symbol}. Found: {len(raw_data)}")
        return pd.DataFrame() 

    # Convertir lista de JSON strings a Lista de Dicts
    data = [json.loads(item) for item in raw_data]
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Conversión de tipos (Redis guarda todo como string en el JSON)
    # Asegurar que las columnas numéricas sean float
    numeric_cols = ['mid_price', 'obi_20', 'funding_rate', 'spread_pct', 'taker_buy_vol', 'taker_sell_vol', 'price', 'obi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # Asegurar timestamp correcto
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')

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
        
        # ═══════════════════════════════════════════════════════
        # 🛂 QUALITY GATE: Cache de metadatos (NINJA v7.8)
        # ═══════════════════════════════════════════════════════
        self.metadata_cache: Dict[str, dict] = {}
        self.QUALITY_THRESHOLD = 0.50  # 50% - Modelos por debajo son vetados
        
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
            
            # ═══════════════════════════════════════════════════════════════════
            # 🛂 QUALITY GATE: Cargar Metadata (Pasaporte de Calidad)
            # ═══════════════════════════════════════════════════════════════════
            meta_path = symbol_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        self.metadata_cache[clean_symbol] = json.load(f)
                        acc = self.metadata_cache[clean_symbol].get('accuracy', 0)
                        LOGGER.info(f"🛂 Passport loaded for {clean_symbol}: Accuracy={acc:.2%}")
                except Exception as e:
                    LOGGER.error(f"⚠️ Failed to load metadata for {clean_symbol}: {e}")
                    self.metadata_cache[clean_symbol] = {"accuracy": 0.0}
            else:
                LOGGER.warning(f"⚠️ No metadata.json for {clean_symbol}. Assuming untested (blocked).")
                self.metadata_cache[clean_symbol] = {"accuracy": 0.0}
            
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
        clean_sym = self._clean_symbol(symbol)
        
        # Load ensemble first (this populates metadata_cache)
        ensemble = self.get_ensemble(symbol)
        
        if ensemble is None:
            # Dummy response if model missing (Fail Safe: Neutral)
            return {
                'ensemble_probs': torch.tensor([[0.0, 1.0, 0.0]]),
                'consensus': 0.0
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # 🛂 QUALITY GATE (NINJA v7.8)
        # ═══════════════════════════════════════════════════════════════════
        # Check AFTER loading so metadata_cache is populated.
        # Si el modelo es "tonto" (<50% accuracy), forzamos NEUTRALIDAD.
        meta = self.metadata_cache.get(clean_sym, {})
        accuracy = meta.get('accuracy', 0.0)
        
        if accuracy < self.QUALITY_THRESHOLD:
            LOGGER.warning(f"⛔ VETOED {clean_sym}: Low Accuracy ({accuracy:.2%}). Forcing NEUTRAL.")
            return {
                'ensemble_probs': torch.tensor([[0.0, 1.0, 0.0]]),  # Short=0, Neutral=1, Long=0
                'consensus': 0.0,
                'verdict': 'VETOED_LOW_ACCURACY'
            }
            
        # 1. Feature Engineering (Derived Features)
        df = df.copy()
        
        # FIX: Map Redis 'spread_pct' to model's 'bid_ask_spread'
        if 'spread_pct' in df.columns and 'bid_ask_spread' not in df.columns:
            df['bid_ask_spread'] = df['spread_pct']
            
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
        SEQ_LEN = 60  # 60 ticks * 10s = 10 min context (v6.0 Audit Fix)
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

    def reload_model(self, symbol: str):
        """Fuerza la recarga del modelo eliminándolo de la caché."""
        clean_sym = self._clean_symbol(symbol)
        if clean_sym in self.ensembles:
            del self.ensembles[clean_sym]
        if clean_sym in self.scalers:
            del self.scalers[clean_sym]
        if clean_sym in self.feature_cols:
            del self.feature_cols[clean_sym]
        if clean_sym in self.last_accessed:
            del self.last_accessed[clean_sym]
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        LOGGER.info(f"♻️ Model for {clean_sym} evicted from cache. Will reload on next request.")

MANAGER = V2ModelManager()

# --- API ---
router = APIRouter(prefix="/ml-v2", tags=["ml-v2"])

@router.post("/reload")
async def reload_endpoint(request: ProbabilityRequestV2):
    """Endpoint para notificar que un modelo ha sido reentrenado."""
    try:
        MANAGER.reload_model(request.symbol)
        return {"status": "ok", "message": f"Reload scheduled for {request.symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        df = load_latest_data(symbol, limit=100)  # SEQ_LEN=60 + rolling buffer
        if df.empty:
            # Try converting ADAUSDT -> ADA/USDT:USDT
            alt_symbol = symbol.replace("USDT", "/USDT:USDT")
            df = load_latest_data(alt_symbol, limit=100)
            
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
