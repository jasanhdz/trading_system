import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Dict, Optional, Tuple

class Chomp1d(nn.Module):
    """
    Elimina el padding extra del futuro para asegurar causalidad.
    Si hacemos padding 'same' en conv1d, leemos del futuro. 
    Chomp corta esos valores extra del final.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Bloque residual básico de TCN:
    Dilated Conv -> WeightNorm -> ReLU -> Dropout -> Dilated Conv -> ...
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Segunda capa convolucional
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Conexión residual (downsample si cambian dimensiones)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Red TCN completa compuesta por múltiples bloques temporales.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i # 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding necesario para mantener la longitud de secuencia con dilatación
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNTradingModel(nn.Module):
    """
    Modelo de Trading basado en TCN.
    Compatible con la interfaz de DeepTemporalNet.
    """
    def __init__(
        self,
        input_dim: int,
        num_channels: list = [64, 128, 256, 512], # Canales por nivel (profundidad)
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        use_regression: bool = True
    ):
        super().__init__()
        
        # TCN Backbone
        # Input shape esperado por TCN: (Batch, Channels, Seq_Len)
        # Nuestros datos vienen como: (Batch, Seq_Len, Features)
        # Haremos transpose en el forward.
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        # Output Heads
        last_channel_dim = num_channels[-1]
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(last_channel_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Regression Head
        self.use_regression = use_regression
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(last_channel_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def forward(self, x, return_features=False) -> Dict[str, torch.Tensor]:
        # x shape: (Batch, Seq_Len, Features)
        # TCN espera: (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)
        
        y = self.tcn(x) # (Batch, Channels, Seq_Len)
        
        # Tomamos solo el último paso de tiempo (el más reciente)
        # TCN es causal, así que el último paso tiene info de toda la historia
        features = y[:, :, -1] # (Batch, Channels)
        
        outputs = {
            'logits': self.classifier(features)
        }
        
        if self.use_regression:
            outputs['regression'] = self.regressor(features)
            
        if return_features:
            outputs['features'] = features
            
        return outputs
