import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.env import PhantomEnv
from scripts.phantom_v30.data_loader import load_hybrid_data, augment_mirror

class TransformerExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor that uses a Transformer Encoder to process market data.
    V31: + Positional Encoding + Attention Pooling
    """
    def __init__(self, observation_space: gym.spaces.Dict, d_model=48, nhead=2, num_layers=2, dropout=0.0):
        # Market: (Window, Features) -> Transformer -> Attention Pool -> (d_model,)
        # Account: (4,) -> Linear -> (d_model,)
        # Concat -> (2 * d_model,)
        
        super().__init__(observation_space, features_dim=d_model * 2)
        
        market_shape = observation_space['market'].shape
        self.window_size = market_shape[0]
        self.n_features = market_shape[1]
        
        # Market Encoder
        self.market_embedding = nn.Linear(self.n_features, d_model)
        
        # Learned Positional Encoding: lets the Transformer know vela order
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.window_size, d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling: learn which candles matter most instead of blind averaging
        self.attention_pool = nn.Linear(d_model, 1)
        
        # Account Encoder
        self.account_embedding = nn.Linear(6, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, observations):
        market = observations['market'] # (Batch, Window, Features)
        account = observations['account'] # (Batch, 4)
        
        # Process Market
        x = self.market_embedding(market) # (Batch, Window, d_model)
        x = x + self.pos_encoding         # Inject temporal position info
        x = self.transformer_encoder(x)   # (Batch, Window, d_model)
        
        # Attention Pooling: weighted sum where the model learns what to focus on
        attn_weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)  # (Batch, Window)
        market_out = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (Batch, d_model)
        market_out = self.dropout(market_out)
        
        # Process Account
        account_out = self.account_embedding(account) # (Batch, d_model)
        
        # Concat
        return torch.cat([market_out, account_out], dim=1) # (Batch, 2*d_model)

def make_env():
    df = load_hybrid_data()
    # Apply Augmentation?
    # For training, we can concatenate mirrored data to original
    aug_df = augment_mirror(df)
    full_df = pd.concat([df, aug_df]).reset_index(drop=True)
    
    env = PhantomEnv(full_df, window_size=64)
    return env

def train():
    print("Setting up Environment...")
    # Vectorized Env not strictly necessary for 1 thread but good practice
    env = DummyVecEnv([make_env])
    
    print("Defining Policy...")
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(d_model=64, nhead=4, num_layers=2),
        net_arch=dict(pi=[64, 64], vf=[64, 64]) # Actor-Critic Heads
    )
    
    print("Initializing PPO Agent...")
    model = PPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Starting Training...")
    model.learn(total_timesteps=100_000, progress_bar=True)
    
    print("Saving Model...")
    model.save("models/phantom_v30_champion")
    print("Model Saved.")

if __name__ == "__main__":
    train()
