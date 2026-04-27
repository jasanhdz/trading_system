#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.phantom_v30.matrix_trainer import evaluate_model
from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.train_v30 import TransformerExtractor

def run():
    print("🏟️ Evaluando el viejo modelo (V30 original) en su ambiente nativo...")
    
    # Cargamos datos para CPU y creamos el env
    eval_data = load_tensor_data("cpu", split="val")
    features_np = eval_data['features'].numpy()
    close_np = eval_data['close'].numpy()
    
    eval_env = PhantomMatrixEnv(
        features=features_np,
        close_prices=close_np,
        num_envs=32,
    )
    
    CHAMPION_PATH = "/home/jasan/Develop/trading_system/models/phantom_v30_champion_v30_arch.zip.backup"
    
    # Tenemos que parchar dinámicamente TransformerExtractor para quitarle el pos_encoding 
    # y attention_pool, así PPO.load podrá cargar los pesos V30 perfectamente y darnos el PnL real.
    original_init = TransformerExtractor.__init__
    original_forward = TransformerExtractor.forward
    
    import torch
    import torch.nn as nn
    import gymnasium as gym
    
    def patch_init(self, observation_space: gym.spaces.Dict, d_model=64, nhead=4, num_layers=2, dropout=0.0):
        super(TransformerExtractor, self).__init__(observation_space, features_dim=d_model * 2)
        market_shape = observation_space['market'].shape
        self.window_size = market_shape[0]
        self.n_features = market_shape[1]
        self.market_embedding = nn.Linear(self.n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.account_embedding = nn.Linear(4, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def patch_forward(self, observations):
        market = observations['market']
        account = observations['account']
        x = self.market_embedding(market)
        x = self.transformer_encoder(x)
        market_out = x.mean(dim=1)
        market_out = self.dropout(market_out)
        account_out = self.account_embedding(account)
        return torch.cat([market_out, account_out], dim=1)
        
    TransformerExtractor.__init__ = patch_init
    TransformerExtractor.forward = patch_forward
    
    champ_score, champ_dd, _ = evaluate_model(CHAMPION_PATH, eval_env)
    
    print("\n--- RESULTADO REAL DEL CAMPEÓN V30 ---")
    print(f"🏆 PnL Verdadero: ${champ_score:.2f} (P95 DD: {champ_dd*100:.1f}%)")
    
if __name__ == '__main__':
    run()
