"""
Advanced temporal models for trading prediction using LSTM + Attention.

This module provides state-of-the-art architectures that capture:
- Temporal dependencies (LSTM/GRU)
- Multi-scale attention mechanisms
- Residual connections for better gradient flow
- Multiple prediction heads (regression + classification)
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


class AttentionLayer(nn.Module):
    """Self-attention mechanism for temporal sequences."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, hidden_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_out))
        return x


class TemporalEncoder(nn.Module):
    """LSTM-based encoder with attention for temporal patterns."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        is_rocm = bool(getattr(torch.version, "hip", None))
        lstm_dropout = dropout if num_layers > 1 else 0.0
        if (
            is_rocm
            and not _ROCM_FORCE_LSTM_DROPOUT
            and lstm_dropout > 0.0
        ):
            global _ROCM_DROPOUT_WARNING_EMITTED
            if not _ROCM_DROPOUT_WARNING_EMITTED:
                warnings.warn(
                    "Disabling in-LSTM dropout on ROCm to avoid MIOpen HIPRTC compilation "
                    "failures. Set FORCE_LSTM_DROPOUT=1 to override once the driver/toolchain "
                    "is patched.",
                    RuntimeWarning,
                )
                _ROCM_DROPOUT_WARNING_EMITTED = True
            lstm_dropout = 0.0
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim, num_heads=4, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * directions)
        
        if self.use_attention:
            lstm_out = self.attention(lstm_out)
        
        # Take last timestep or apply pooling
        output = lstm_out[:, -1, :]  # Last timestep
        output = self.dropout(output)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and dropout."""
    
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + residual  # Residual connection
        out = F.relu(out)
        return out


class AdvancedTemporalNet(nn.Module):
    """
    Advanced trading model with temporal encoding and multi-task learning.
    
    Features:
    - LSTM encoder with attention for temporal patterns
    - Residual connections for better gradient flow
    - Multi-task learning: classification + regression
    - Separate heads for long/short predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 24,  # lookback period
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dense_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        use_attention: bool = True,
        bidirectional: bool = True,
        num_classes: int = 3,  # neutral, long, short
        use_regression: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_regression = use_regression
        
        # Temporal encoder (LSTM + Attention)
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            use_attention=use_attention,
            bidirectional=bidirectional,
        )
        
        # Calculate encoder output dimension
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Dense layers with residual connections
        layers = []
        prev_dim = encoder_output_dim
        
        for dim in dense_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            # Add residual block if dimensions match
            if dim == prev_dim:
                layers.append(ResidualBlock(dim, dropout))
            
            prev_dim = dim
        
        self.dense_backbone = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dense_dims[-1] if dense_dims else encoder_output_dim
        
        # Classification head (direction prediction)
        self.classifier = nn.Linear(last_dim, num_classes)
        
        # Regression head (return prediction)
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(last_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_features: If True, return intermediate features
            
        Returns:
            Dictionary with keys:
                - 'logits': Classification logits (batch, num_classes)
                - 'regression': Predicted returns (batch, 1) if use_regression
                - 'features': Dense features (batch, last_dim) if return_features
        """
        # Temporal encoding
        encoded = self.encoder(x)
        
        # Dense processing
        features = self.dense_backbone(encoded)
        
        # Multi-task outputs
        outputs = {
            'logits': self.classifier(features),
        }
        
        if self.use_regression:
            outputs['regression'] = self.regressor(features)
        
        if return_features:
            outputs['features'] = features
        
        return outputs
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities using softmax."""
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs['logits'], dim=-1)
            return probs
    
    def predict_return(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted future return."""
        if not self.use_regression:
            raise ValueError("Model was not configured with regression head")
        
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['regression']


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple temporal models with weighted averaging.
    
    Combines predictions from multiple models trained with different:
    - Random seeds
    - Architectures
    - Training subsets (bagging)
    """
    
    def __init__(self, models: list[AdvancedTemporalNet], weights: Optional[list[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Average predictions across all models."""
        all_logits = []
        all_regression = []
        
        for model in self.models:
            outputs = model(x)
            all_logits.append(outputs['logits'])
            if 'regression' in outputs:
                all_regression.append(outputs['regression'])
        
        # Weighted average
        logits_stack = torch.stack(all_logits, dim=0)
        avg_logits = (logits_stack * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        result = {'logits': avg_logits}
        
        if all_regression:
            regression_stack = torch.stack(all_regression, dim=0)
            avg_regression = (regression_stack * self.weights.view(-1, 1, 1)).sum(dim=0)
            result['regression'] = avg_regression
        
        return result
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return ensemble class probabilities."""
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs['logits'], dim=-1)
            return probs


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for classification and regression tasks.
    
    Uses adaptive weighting to balance tasks during training.
    Now uses Focal Loss for classification to handle class imbalance better.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        classification_weight: float = 1.0,
        regression_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        # Use Focal Loss instead of standard Cross Entropy
        self.classification_criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.regression_criterion = nn.MSELoss()
        
        # Learnable task weights (log variance approach)
        self.log_var_class = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
    def forward(
        self,
        logits: torch.Tensor,
        class_targets: torch.Tensor,
        regression_output: Optional[torch.Tensor] = None,
        regression_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with automatic task balancing.
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        # Classification loss (Focal Loss)
        class_loss = self.classification_criterion(logits, class_targets)
        
        # Adaptive weighting using learned uncertainty
        weighted_class_loss = self.classification_weight * class_loss / (2 * torch.exp(self.log_var_class)) + self.log_var_class / 2
        
        total_loss = weighted_class_loss
        loss_dict = {'class_loss': class_loss.item()}
        
        # Regression loss (if provided)
        if regression_output is not None and regression_targets is not None:
            reg_loss = self.regression_criterion(regression_output.squeeze(), regression_targets)
            weighted_reg_loss = self.regression_weight * reg_loss / (2 * torch.exp(self.log_var_reg)) + self.log_var_reg / 2
            
            total_loss = total_loss + weighted_reg_loss
            loss_dict['reg_loss'] = reg_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
