from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AegisTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, d_model: int = 32, nhead: int = 2, num_layers: int = 1, dropout: float = 0.1):
        super().__init__(observation_space, features_dim=d_model * 2)
        market_shape = observation_space["market"].shape
        account_shape = observation_space["account"].shape
        self.market_embedding = nn.Linear(market_shape[1], d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.account_net = nn.Sequential(nn.Linear(account_shape[0], d_model), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(d_model * 2, d_model * 2), nn.ReLU())

    def forward(self, obs):
        market = self.market_embedding(obs["market"])
        market_features = self.transformer(market).mean(dim=1)
        account_features = self.account_net(obs["account"])
        return self.out(torch.cat([market_features, account_features], dim=1))
