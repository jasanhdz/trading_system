"""Neural network used for XRP/USDT pattern classification."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class PatternNet(nn.Module):
    """Lightweight feed-forward network that outputs probability logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.2,
        num_classes: int = 2,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1] if dims else input_dim
        self.head = nn.Linear(last_dim, num_classes)
        self.output_activation = output_activation
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities using the configured activation."""
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_activation == "softmax":
                return torch.softmax(logits, dim=-1)
            return torch.sigmoid(logits)
