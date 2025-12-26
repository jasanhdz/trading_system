#!/usr/bin/env python3
import sys
import shutil
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from ml.advanced_models.dataset import AdvancedDatasetConfig, load_sequence_dataset, SequenceDataset
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer
from ml.advanced_models.tabular_model import XGBoostTradingModel
from ml.advanced_models.ensemble_manager import EnsembleManager
from ml.advanced_models.meta_model import MetaLabelingModel
from ml.advanced_models.temporal_model import SharpeLoss
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnsembleTest")

TEMP_DIR = REPO_ROOT / "models" / "test_ensemble"

def train_pytorch_model(model, train_loader, device, name):
    logger.info(f"🏋️ Training {name}...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # Usamos CE simple para el test
    
    # Train 1 epoch
    for batch in train_loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        outputs = model(x)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        break # Solo 1 batch para verificar que corre
        
    logger.info(f"✅ {name} trained successfully.")
    return model

def main():
    logger.info("🚀 Starting Ensemble Pipeline Test")
    
    # 1. Prepare Data
    logger.info("📥 Loading dummy data...")
    config = AdvancedDatasetConfig(
        symbol="ADA/USDT:USDT",
        timeframe="1h",
        sequence_length=24,
        min_records=500,
        max_samples=1000 # Small sample
    )
    
    try:
        features, labels, returns, _ = load_sequence_dataset(config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Create directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    
    # Prepare PyTorch Dataset
    dataset = SequenceDataset(features, labels, regression_targets=returns, sequence_length=24)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) # Aumentamos batch para tener datos para el meta-modelo
    
    input_dim = features.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ Device: {device}")
    
    # 2. Train & Save Models
    
    # --- LSTM ---
    lstm = DeepTemporalNet(input_dim=input_dim, hidden_dim=64, lstm_layers=2, num_classes=3).to(device)
    lstm = train_pytorch_model(lstm, loader, device, "LSTM")
    torch.save(lstm.state_dict(), TEMP_DIR / "lstm.pt")
    with open(TEMP_DIR / "lstm_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'hidden_dim': 64, 'lstm_layers': 2, 'dropout': 0.2}}, f)
        
    # --- TCN ---
    tcn = TCNTradingModel(input_dim=input_dim, num_channels=[32, 64], kernel_size=3).to(device)
    tcn = train_pytorch_model(tcn, loader, device, "TCN")
    torch.save(tcn.state_dict(), TEMP_DIR / "tcn.pt")
    with open(TEMP_DIR / "tcn_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'num_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.2}}, f)

    # --- Transformer ---
    tf = TradingTransformer(input_dim=input_dim, d_model=32, nhead=2, num_layers=1).to(device)
    tf = train_pytorch_model(tf, loader, device, "Transformer")
    torch.save(tf.state_dict(), TEMP_DIR / "transformer.pt")
    with open(TEMP_DIR / "transformer_config.json", 'w') as f:
        json.dump({'model_config': {'input_dim': input_dim, 'd_model': 32, 'nhead': 2, 'num_layers': 1}}, f)
        
    # --- XGBoost ---
    logger.info("🏋️ Training XGBoost...")
    xgb_model = XGBoostTradingModel(use_gpu=(device == "cuda"))
    
    logger.info(f"   Features shape: {features.shape}")
    if len(features.shape) == 3:
        # (Batch, Seq, Feat) -> Take last step
        X_flat = features[:, -1, :]
    else:
        # Already 2D (Batch, Feat)
        X_flat = features
        
    xgb_model.train(X_flat, labels, X_flat, labels) # Train on itself for test
    xgb_model.save(str(TEMP_DIR / "xgboost.joblib"))
    with open(TEMP_DIR / "xgboost_config.json", 'w') as f:
        json.dump({}, f)
    logger.info("✅ XGBoost trained successfully.")

    # 3. Test Ensemble Manager
    logger.info("🧠 Testing EnsembleManager...")
    manager = EnsembleManager(device=device)
    
    manager.load_model("lstm_v1", "lstm", str(TEMP_DIR / "lstm.pt"), str(TEMP_DIR / "lstm_config.json"))
    manager.load_model("tcn_v1", "tcn", str(TEMP_DIR / "tcn.pt"), str(TEMP_DIR / "tcn_config.json"))
    manager.load_model("tf_v1", "transformer", str(TEMP_DIR / "transformer.pt"), str(TEMP_DIR / "transformer_config.json"))
    manager.load_model("xgb_v1", "xgboost", str(TEMP_DIR / "xgboost.joblib"), str(TEMP_DIR / "xgboost_config.json"))
    
    # 4. Predict (Ensemble)
    logger.info("🔮 Running Ensemble Prediction...")
    # Get a batch
    sample_batch = next(iter(loader))
    if len(sample_batch) == 3:
        x, y_true, _ = sample_batch
    else:
        x, y_true = sample_batch
        
    result = manager.predict(x)
    ensemble_probs = result['ensemble_probs'].cpu().numpy()
    
    logger.info("📊 Ensemble Results:")
    logger.info(f"   Ensemble Class: {result['ensemble_class'][:5].cpu().numpy()}")
    logger.info(f"   Consensus Level: {manager.get_consensus_level(result):.4f}")
    
    # 5. Test Meta-Model (The Judge)
    logger.info("⚖️ Testing Meta-Model (The Judge)...")
    
    # Simular datos de mercado (Contexto)
    batch_size = x.shape[0]
    market_features_df = pd.DataFrame({
        'atr': np.random.rand(batch_size),
        'volatility': np.random.rand(batch_size),
        'hour': np.random.randint(0, 24, batch_size),
        'spread': np.random.rand(batch_size) * 0.01
    })
    
    # Simular resultados de trades (Labels para el meta-modelo)
    # 1 = Trade fue bueno, 0 = Trade fue malo
    # Simulamos que si el ensemble acertó (y_true), es un trade bueno (simplificación)
    ensemble_preds = result['ensemble_class'].cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    trade_outcomes = (ensemble_preds == y_true_np).astype(int)
    
    meta_model = MetaLabelingModel(use_gpu=(device == "cuda"))
    
    # Entrenar Juez
    meta_model.train(
        ensemble_probs_train=ensemble_probs,
        market_features_train=market_features_df,
        trade_outcomes_train=trade_outcomes,
        ensemble_probs_val=ensemble_probs, # Usamos lo mismo para test simple
        market_features_val=market_features_df,
        trade_outcomes_val=trade_outcomes
    )
    
    # Pedir Veredicto
    veto_mask, approval_prob = meta_model.predict_veto(ensemble_probs, market_features_df, threshold=0.5)
    
    logger.info("👨‍⚖️ Judge's Verdict:")
    logger.info(f"   Approval Probs: {approval_prob[:5]}")
    logger.info(f"   Veto Mask: {veto_mask[:5]}")
    logger.info(f"   Trades Vetoed: {np.sum(veto_mask)} / {batch_size}")
    
    logger.info("✅ FULL SYSTEM TEST COMPLETED SUCCESSFULLY")
    
    # Cleanup
    # shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main()
