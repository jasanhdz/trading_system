import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional

class PositionalEncoding(nn.Module):
    """
    Inyecta información sobre la posición relativa de los tokens en la secuencia.
    Esencial para Transformers ya que no tienen recurrencia.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class GatedResidualNetwork(nn.Module):
    """
    GRN: Componente clave del TFT. Permite al modelo decidir cuánto de la 
    entrada no lineal debe pasar vs la conexión residual.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        
        # Proyección residual si dimensiones no coinciden
        self.res_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        x_path = self.fc2(self.dropout(self.elu(self.fc1(x))))
        gate = torch.sigmoid(self.gate(x))
        return self.ln(gate * x_path + residual)

class TradingTransformer(nn.Module):
    """
    Transformer diseñado para Trading Institucional.
    
    Características:
    - Positional Encoding
    - Multi-Head Self Attention (captura dependencias a largo plazo)
    - Feed Forward Networks con Gating (GRN)
    - Layer Norm pre-activación para estabilidad
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
        use_regression: bool = True
    ):
        super().__init__()
        
        # 1. Input Embedding
        # Proyectamos features crudos a espacio latente d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True # Pre-LN es más estable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Heads
        # Usamos GRN antes de las cabezas finales para refinamiento
        self.grn_head = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        
        # Classification
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Regression
        self.use_regression = use_regression
        if use_regression:
            self.regressor = nn.Linear(d_model, 1)
            
    def forward(self, x, return_features=False) -> Dict[str, torch.Tensor]:
        # x: (Batch, Seq_Len, Features)
        
        # Embedding + Posición
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer Pass
        # mask=None (asumimos atención completa causal o bidireccional según se desee)
        # Para trading, a veces es útil ver toda la ventana pasada (bidireccional en el contexto de la ventana)
        encoded = self.transformer_encoder(x)
        
        # Pooling: Tomamos el último token (representa el estado más reciente)
        # Alternativa: Global Average Pooling
        last_token = encoded[:, -1, :] # (Batch, d_model)
        
        # Refinamiento final
        features = self.grn_head(last_token)
        
        outputs = {
            'logits': self.classifier(features)
        }
        
        if self.use_regression:
            outputs['regression'] = self.regressor(features)
            
        if return_features:
            outputs['features'] = features
            
        return outputs
