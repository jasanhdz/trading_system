"""
Predictor for advanced temporal models.

Handles loading and inference with LSTM-based models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from ml.nn_pattern.features import build_feature_frame
from .temporal_model import AdvancedTemporalNet, EnsembleModel


class AdvancedPredictor:
    """
    Loads advanced temporal model and produces predictions.
    
    Handles:
    - Sequence preparation
    - Feature selection
    - Ensemble predictions
    - Multi-output predictions (classification + regression)
    """
    
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        meta_path: Path,
        feature_selector_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.meta_path = Path(meta_path)
        self.feature_selector_path = Path(feature_selector_path) if feature_selector_path else None
        self.device = torch.device(device)
        
        # Verify files exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
        
        # Load metadata
        self.meta = json.loads(self.meta_path.read_text())
        
        # Load scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Load feature selector (if exists)
        self.feature_selector = None
        if self.feature_selector_path and self.feature_selector_path.exists():
            self.feature_selector = joblib.load(self.feature_selector_path)
        
        # Extract configuration
        self.sequence_length = self.meta['sequence_length']
        self.selected_features = self.meta['selected_features']
        self.model_config = self.meta['model_config']
        self.ensemble_size = self.meta.get('ensemble_size', 1)
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Check if ensemble
        if self.ensemble_size > 1 and isinstance(checkpoint, dict) and 'models' in checkpoint:
            # Load ensemble
            models = []
            for state_dict in checkpoint['models']:
                model = AdvancedTemporalNet(
                    input_dim=len(self.selected_features),
                    sequence_length=self.sequence_length,
                    **self.model_config,
                )
                model.load_state_dict(state_dict)
                models.append(model)
            
            ensemble = EnsembleModel(models, weights=checkpoint.get('weights'))
            return ensemble
        else:
            # Load single model
            model = AdvancedTemporalNet(
                input_dim=len(self.selected_features),
                sequence_length=self.sequence_length,
                **self.model_config,
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
    
    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare input sequence from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns, sorted chronologically
            
        Returns:
            Prepared sequence of shape (sequence_length, n_features)
        """
        # Build features
        feature_frame, _ = build_feature_frame(df)
        
        if len(feature_frame) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows to make prediction, "
                f"got {len(feature_frame)}"
            )
        
        # Select features
        if self.feature_selector:
            features = feature_frame[self.selected_features].values
        else:
            # Ensure correct feature order
            features = feature_frame[self.selected_features].values
        
        # Take last sequence_length rows
        sequence = features[-self.sequence_length:]
        
        # Scale
        sequence_scaled = self.scaler.transform(sequence)
        
        return sequence_scaled.astype(np.float32)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate prediction from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns (needs at least sequence_length rows)
            
        Returns:
            Dictionary with predictions:
                - 'neutral', 'long', 'short': Class probabilities
                - 'predicted_return': Expected return (if regression enabled)
                - 'direction': Predicted direction ('neutral'/'long'/'short')
                - 'confidence': Confidence score (max probability)
        """
        # Prepare sequence
        sequence = self._prepare_sequence(df)
        
        # Convert to tensor (batch_size=1)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            
            # Classification probabilities
            probs = torch.softmax(outputs['logits'], dim=-1)
            probs_np = probs.cpu().numpy().flatten()
            
            # Build result
            result = {
                'neutral': float(probs_np[0]),
                'long': float(probs_np[1]),
                'short': float(probs_np[2]),
            }
            
            # Add regression prediction if available
            if 'regression' in outputs:
                predicted_return = outputs['regression'].cpu().numpy().flatten()[0]
                result['predicted_return'] = float(predicted_return)
            
            # Add direction and confidence
            direction_idx = int(probs_np.argmax())
            directions = ['neutral', 'long', 'short']
            result['direction'] = directions[direction_idx]
            result['confidence'] = float(probs_np[direction_idx])
        
        return result
    
    def predict_batch(self, sequences: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict for multiple sequences at once.
        
        Args:
            sequences: Array of shape (batch, sequence_length, n_features)
            
        Returns:
            Dictionary with batch predictions
        """
        # Convert to tensor
        sequences_tensor = torch.from_numpy(sequences).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            
            # Classification probabilities
            probs = torch.softmax(outputs['logits'], dim=-1)
            probs_np = probs.cpu().numpy()
            
            result = {
                'probabilities': probs_np,
                'predictions': probs_np.argmax(axis=-1),
            }
            
            # Add regression if available
            if 'regression' in outputs:
                result['predicted_returns'] = outputs['regression'].cpu().numpy()
        
        return result
