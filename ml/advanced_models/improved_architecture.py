"""
Arquitecturas mejoradas y más profundas para trading.

Incluye:
- DeepTemporalNet: Versión más profunda del modelo LSTM+Attention
- TransformerNet: Modelo basado en Transformer
- HybridCNNLSTM: Combinación de CNN + LSTM
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROCM_FORCE_LSTM_DROPOUT = os.environ.get("FORCE_LSTM_DROPOUT", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ROCM_DROPOUT_WARNING_EMITTED = False


class DeepTemporalNet(nn.Module):
    """
    Versión mejorada y más profunda del modelo LSTM+Attention.

    Mejoras sobre AdvancedTemporalNet:
    - Más capas LSTM (hasta 3)
    - Dense layers más profundas (3 capas: 384→256→128)
    - Residual connections en todas las capas densas
    - Layer Normalization para mejor convergencia
    - Más heads de atención (8 en lugar de 4)
    - Mejor regularización con dropout escalonado
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 48,  # Aumentado de 24
        hidden_dim: int = 192,      # Aumentado de 128
        lstm_layers: int = 3,       # Aumentado de 2
        dense_dims: Tuple[int, ...] = (384, 256, 128),  # Más profundo
        dropout: float = 0.35,      # Aumentado de 0.3
        use_attention: bool = True,
        bidirectional: bool = True,
        num_classes: int = 3,
        use_regression: bool = True,
        num_attention_heads: int = 8,  # Aumentado de 4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_regression = use_regression

        # Input projection con residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),  # Dropout más bajo en entrada
        )

        # Stacked LSTM con Layer Normalization
        is_rocm = bool(getattr(torch.version, "hip", None))
        lstm_dropout = dropout if lstm_layers > 1 else 0.0

        if is_rocm and not _ROCM_FORCE_LSTM_DROPOUT and lstm_dropout > 0.0:
            global _ROCM_DROPOUT_WARNING_EMITTED
            if not _ROCM_DROPOUT_WARNING_EMITTED:
                warnings.warn(
                    "Desactivando dropout en LSTM para ROCm. "
                    "Set FORCE_LSTM_DROPOUT=1 para forzar.",
                    RuntimeWarning,
                )
                _ROCM_DROPOUT_WARNING_EMITTED = True
            lstm_dropout = 0.0

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Layer Normalization después de LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        # Multi-head Attention mejorada
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        else:
            self.attention = None

        # Dense backbone con residual blocks
        dense_layers = []
        prev_dim = lstm_output_dim

        for i, dim in enumerate(dense_dims):
            # Linear + Norm + Activation + Dropout
            dense_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * (1 - i * 0.1)),  # Dropout escalonado
            ])

            # Residual block si las dimensiones coinciden
            if prev_dim == dim:
                dense_layers.append(ResidualBlock(dim, dropout))

            prev_dim = dim

        self.dense = nn.Sequential(*dense_layers)

        # Output heads
        self.classifier = nn.Linear(prev_dim, num_classes)

        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
            )

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        """Inicialización Xavier/He para mejor convergencia."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # orthogonal_ también requiere 2+ dimensiones
                    if param.ndim >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.normal_(param, mean=0, std=0.01)
                elif 'norm' in name:
                    # LayerNorm weights son vectores 1D, usar constant
                    nn.init.constant_(param, 1)
                else:
                    # kaiming_normal_ solo funciona con tensores de 2+ dimensiones
                    if param.ndim >= 2:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input projection
        x = self.input_proj(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        # Attention
        if self.attention is not None:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)  # Residual

        # Take last timestep
        features = lstm_out[:, -1, :]

        # Dense processing
        features = self.dense(features)

        # Outputs
        outputs = {
            'logits': self.classifier(features),
        }

        if self.use_regression:
            outputs['regression'] = self.regressor(features)

        return outputs


class ResidualBlock(nn.Module):
    """Residual block con LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = out + residual  # Residual connection
        out = F.relu(out)
        return out


class TransformerNet(nn.Module):
    """
    Modelo basado en Transformer para series temporales.

    Ventajas sobre LSTM:
    - Mejor captura de dependencias de largo alcance
    - Entrenamiento más rápido (paralelizable)
    - Positional encoding para capturar orden temporal
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 48,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        num_classes: int = 3,
        use_regression: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_regression = use_regression

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN para mejor estabilidad
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output layers
        self.norm = nn.LayerNorm(d_model)

        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Output heads
        self.classifier = nn.Linear(128, num_classes)

        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq, d_model)

        # Take last timestep
        x = x[:, -1, :]
        x = self.norm(x)

        # Dense processing
        features = self.dense(x)

        # Outputs
        outputs = {
            'logits': self.classifier(features),
        }

        if self.use_regression:
            outputs['regression'] = self.regressor(features)

        return outputs


class PositionalEncoding(nn.Module):
    """Positional encoding para Transformers."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridCNNLSTM(nn.Module):
    """
    Modelo híbrido CNN + LSTM.

    CNN captura patrones locales (corto plazo)
    LSTM captura dependencias temporales (largo plazo)
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 48,
        cnn_channels: Tuple[int, ...] = (64, 128, 192),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        use_regression: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_regression = use_regression

        # CNN para patrones locales
        # Input: (batch, seq_len, input_dim)
        # Conv1d espera: (batch, channels, length)

        cnn_layers = []
        prev_channels = input_dim

        for channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
            ])
            prev_channels = channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM para dependencias temporales
        is_rocm = bool(getattr(torch.version, "hip", None))
        lstm_dropout = dropout if lstm_layers > 1 else 0.0

        if is_rocm and not _ROCM_FORCE_LSTM_DROPOUT and lstm_dropout > 0.0:
            lstm_dropout = 0.0

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Output heads
        self.classifier = nn.Linear(128, num_classes)

        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # CNN espera (batch, channels, length)
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)

        # CNN processing
        x = self.cnn(x)  # (batch, channels, seq_len)

        # Volver a (batch, seq_len, channels) para LSTM
        x = x.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        features = lstm_out[:, -1, :]

        # Dense processing
        features = self.dense(features)

        # Outputs
        outputs = {
            'logits': self.classifier(features),
        }

        if self.use_regression:
            outputs['regression'] = self.regressor(features)

        return outputs


# Importar numpy para PositionalEncoding
import numpy as np
