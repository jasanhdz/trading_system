#!/usr/bin/env python3
"""
Script de Entrenamiento "Berzerker AI"
Objetivo: Entrenar una Red Neuronal (o XGBoost) especializada en validar "Olas".
Filosofía: El patrón de 2-3 velas es el "Candidato". La IA es el "Juez".
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger("berzerker_trainer")

# Configurar nueva DB de velas
CANDLES_DB_URL = "sqlite:///data/binance_candles.db"
db_manager = DatabaseManager(CANDLES_DB_URL)

# Configuración
SYMBOL = 'XRP/USDT'
TIMEFRAME = '5m'
MODEL_DIR = Path("models/berzerker_v1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class BerzerkerNet(nn.Module):
    """
    Red Neuronal ligera para clasificación de patrones.
    Input: Features de la Ola (Volumen, Cuerpo, Mechas, RSI, etc.)
    Output: Probabilidad de éxito (0-1)
    """
    def __init__(self, input_size):
        super(BerzerkerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.layer3 = nn.Linear(32, 16)
        
        self.output = nn.Linear(16, 1)
        # self.sigmoid = nn.Sigmoid() # Not needed for forward if returning logits
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

def prepare_data():
    """
    Prepara el dataset:
    1. Encuentra candidatos (2 velas verdes con volumen).
    2. Etiqueta (1 si TP hit antes de SL, 0 si no).
    3. Extrae features.
    """
    logger.info(f"Cargando datos para {SYMBOL} {TIMEFRAME}...")
    df = db_manager.get_ohlcv_data(SYMBOL, TIMEFRAME)
    
    # Fallback logic removed as we enforce 5m data in binance_candles.db
    
    if df.empty:
        logger.error("No hay datos. Espera a que termine la descarga.")
        return None, None
        
    logger.info(f"Datos cargados: {len(df)} velas.")
    
    # --- Feature Engineering ---
    df['returns'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Volumen relativo
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_factor'] = df['volume'] / (df['vol_ma_20'] + 1e-8)
    
    # RSI (Simple)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- Identificación de Candidatos (La "Ola") ---
    # Definición: 2 velas verdes consecutivas con volumen creciente o alto
    df['is_green'] = df['close'] > df['open']
    df['prev_green'] = df['is_green'].shift(1)
    
    # Candidato: 2 verdes + volumen actual > 1.5x media
    df['is_candidate'] = (df['is_green']) & (df['prev_green']) & (df['vol_factor'] > 1.5)
    
    candidates = df[df['is_candidate']].copy()
    logger.info(f"Candidatos encontrados (Olas potenciales): {len(candidates)}")
    
    if len(candidates) < 100:
        logger.error("Insuficientes candidatos para entrenar.")
        return None, None
        
    # --- Etiquetado (Labelling) ---
    # Objetivo: +3% TP, -1.5% SL (Ratio 1:2) en los próximos 4 horas (48 velas)
    TP_PCT = 0.03
    SL_PCT = 0.015
    HORIZON = 48 
    
    labels = []
    features = []
    
    valid_indices = []
    
    for idx, row in candidates.iterrows():
        # Buscar índice numérico
        try:
            loc = df.index.get_loc(idx)
        except:
            continue
            
        if loc + HORIZON >= len(df):
            continue
            
        # Futuro
        future = df.iloc[loc+1 : loc+HORIZON+1]
        entry_price = row['close']
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        
        outcome = 0 # Fail
        
        for _, f_row in future.iterrows():
            if f_row['low'] <= sl_price:
                outcome = 0 # SL Hit first
                break
            if f_row['high'] >= tp_price:
                outcome = 1 # TP Hit first
                break
        
        labels.append(outcome)
        
        # Features para el modelo (lo que sabe la IA en el momento de entrada)
        feat_vector = [
            row['vol_factor'],
            row['body'] / row['open'], # Body %
            row['upper_wick'] / row['open'],
            row['lower_wick'] / row['open'],
            row['rsi'] / 100.0,
            df.iloc[loc-1]['vol_factor'], # Vol prev candle
            df.iloc[loc-1]['body'] / df.iloc[loc-1]['open'], # Body prev
            (row['close'] - df.iloc[loc-5]['close']) / df.iloc[loc-5]['close'] # Momentum 5m
        ]
        features.append(feat_vector)
        valid_indices.append(idx)
        
    X = np.array(features)
    y = np.array(labels)
    
    logger.info(f"Dataset final: {len(X)} muestras.")
    logger.info(f"Win Rate Base (sin IA): {y.mean():.2%}")
    
    return X, y

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight, fp_weight=1.5):
        super().__init__()
        self.fp_weight = fp_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets):
        loss = self.bce(logits, targets)
        weights = torch.where(targets == 0, self.fp_weight, 1.0)
        return (loss * weights).mean()

import argparse

def train_model():
    parser = argparse.ArgumentParser(description='Train Berzerker Model')
    parser.add_argument('--symbol', type=str, default='XRP/USDT', help='Symbol to train on')
    args = parser.parse_args()
    
    global SYMBOL, MODEL_DIR
    SYMBOL = args.symbol
    
    # Create symbol-specific model directory
    safe_symbol = SYMBOL.replace('/', '_')
    MODEL_DIR = Path(f"models/berzerker_{safe_symbol}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    X, y = prepare_data()
    
    if X is None:
        return
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar scaler
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Calculate Class Weights
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight_val = num_neg / (num_pos + 1e-8)
    pos_weight_tensor = torch.FloatTensor([pos_weight_val])
    
    logger.info(f"Class Balance: {num_pos} Positives, {num_neg} Negatives. Pos Weight: {pos_weight_val:.2f}")

    # Model
    model = BerzerkerNet(input_size=X.shape[1])
    
    # Custom Loss: Balance classes AND penalize FP
    criterion = WeightedBCELoss(pos_weight=pos_weight_tensor, fp_weight=1.5)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    EPOCHS = 100
    logger.info(f"Iniciando entrenamiento para {SYMBOL} con penalización de Falsos Positivos...")
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits) # Apply sigmoid for inference
        predicted = (test_probs > 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        logger.info(f"Accuracy en Test: {accuracy:.2%}")
        
        # Reporte detallado
        y_pred_np = predicted.numpy()
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred_np))
        
        # Guardar modelo
        torch.save(model.state_dict(), MODEL_DIR / "berzerker_net.pth")
        logger.info(f"Modelo guardado en {MODEL_DIR}")

if __name__ == "__main__":
    train_model()
