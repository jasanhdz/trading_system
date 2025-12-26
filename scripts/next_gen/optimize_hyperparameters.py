#!/usr/bin/env python3
import os
import sys
import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.preprocessing import StandardScaler

# Añadir path para importar módulos existentes
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.dataset import AdvancedDatasetConfig, load_sequence_dataset, SequenceDataset
from ml.advanced_models.temporal_model import SharpeLoss

# Configuración
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models/advanced/optuna_studies"
LOG_DIR = ROOT_DIR / "logs"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptunaOptimizer")

# Cache global para no recargar datos en cada trial
CACHED_DATA = {}

def get_data(symbol="ADA/USDT:USDT", timeframe="1h"):
    """Carga y cachea los datos para optimización."""
    key = f"{symbol}_{timeframe}"
    if key in CACHED_DATA:
        return CACHED_DATA[key]
    
    logger.info(f"📥 Cargando datos para {key}...")
    
    config = AdvancedDatasetConfig(
        symbol=symbol,
        timeframe=timeframe,
        sequence_length=96, # Max sequence length we might test
        prediction_horizon=12,
        target_return=0.005,
        min_records=1000,
        max_samples=10000 # Limitamos para velocidad de Optuna
    )
    
    try:
        # Ahora capturamos returns (regression_targets)
        features, labels, returns, _ = load_sequence_dataset(config)
        
        # Escalar features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Split simple Train/Val (80/20) para Optuna
        split_idx = int(len(features) * 0.8)
        
        train_X = features[:split_idx]
        train_y = labels[:split_idx]
        train_ret = returns[:split_idx]
        
        val_X = features[split_idx:]
        val_y = labels[split_idx:]
        val_ret = returns[split_idx:]
        
        CACHED_DATA[key] = (train_X, train_y, train_ret, val_X, val_y, val_ret)
        return CACHED_DATA[key]
        
    except Exception as e:
        logger.error(f"❌ Error cargando datos: {e}")
        raise e

def objective(trial):
    # 1. Cargar Datos (Desde caché)
    train_X, train_y, train_ret, val_X, val_y, val_ret = get_data()
    
    # 2. Sugerir Hiperparámetros
    params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256, step=64),
        'lstm_layers': trial.suggest_int('lstm_layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'sequence_length': trial.suggest_int('sequence_length', 24, 96, step=12)
    }
    
    # Ajustar secuencia (recortar si es necesario)
    seq_len = params['sequence_length']
    
    # Crear Datasets para este trial incluyendo RETORNOS
    # SequenceDataset espera: features, class_labels, regression_targets
    train_ds = SequenceDataset(train_X, train_y, regression_targets=train_ret, sequence_length=seq_len)
    val_ds = SequenceDataset(val_X, val_y, regression_targets=val_ret, sequence_length=seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
    
    # 3. Crear Modelo
    model = DeepTemporalNet(
        input_dim=train_X.shape[1],
        hidden_dim=params['hidden_dim'],
        lstm_layers=params['lstm_layers'],
        dropout=params['dropout'],
        num_classes=3
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    
    # USAMOS SHARPE LOSS AHORA
    # transaction_cost=0.001 (0.1% por trade aprox fees+slippage)
    criterion = SharpeLoss(transaction_cost=0.001).to(device)
    
    # 4. Entrenamiento "Flash" (Max 5 epochs)
    model.train()
    for epoch in range(5):
        for batch in train_loader:
            # Desempaquetar batch (features, labels, returns)
            if len(batch) == 3:
                batch_X, batch_y, batch_ret = batch
            else:
                # Fallback por si acaso
                batch_X, batch_y = batch
                batch_ret = torch.zeros_like(batch_y).float()

            batch_X = batch_X.to(device)
            batch_ret = batch_ret.to(device) # Necesitamos returns para SharpeLoss
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # El modelo devuelve un dict {'logits': ..., 'attention': ...}
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # SharpeLoss(logits, returns)
            loss = criterion(logits, batch_ret)
            
            loss.backward()
            optimizer.step()
            
        # Validación rápida
        model.eval()
        val_sharpe_accum = 0
        batches = 0
        
        # Para calcular Sharpe global correctamente deberíamos acumular todos los returns y logits
        # pero para velocidad en Optuna haremos un promedio de Sharpes por batch (aproximación)
        # O mejor: acumular PnL y calcular Sharpe al final de la epoch.
        
        all_val_returns = []
        all_val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_X, batch_y, batch_ret = batch
                else:
                    batch_X, batch_y = batch
                    batch_ret = torch.zeros_like(batch_y).float()
                    
                batch_X = batch_X.to(device)
                
                outputs = model(batch_X)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=1)
                
                all_val_probs.append(probs.cpu())
                all_val_returns.append(batch_ret.cpu())
        
        # Concatenar todo para calcular Sharpe Real del Validation Set
        full_probs = torch.cat(all_val_probs)
        full_returns = torch.cat(all_val_returns)
        
        # Calcular Posición: Long - Short
        position = full_probs[:, 1] - full_probs[:, 2]
        
        # Calcular PnL de Estrategia
        strategy_returns = position * full_returns
        
        # Calcular Sharpe Anualizado
        expected_ret = torch.mean(strategy_returns)
        std_ret = torch.std(strategy_returns)
        sharpe = (expected_ret / (std_ret + 1e-8)) * np.sqrt(365*24)
        
        sharpe_val = sharpe.item()
        
        # Reportar a Optuna para Pruning
        trial.report(sharpe_val, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        model.train()

    # Retornar Sharpe Final
    return sharpe_val

def run_optimization(study_name="study_ada_sharpe_v1", n_trials=15):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    storage_url = f"sqlite:///{MODELS_DIR}/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize", # Queremos maximizar Sharpe
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    logger.info(f"🚀 Iniciando optimización (Sharpe): {study_name}")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("✅ Optimización completada")
    logger.info(f"🏆 Mejores parámetros: {study.best_params}")
    logger.info(f"📈 Mejor Sharpe: {study.best_value}")
    
    df = study.trials_dataframe()
    df.to_csv(MODELS_DIR / f"{study_name}_results.csv")

if __name__ == "__main__":
    # Instalar optuna si no existe: pip install optuna
    run_optimization()
