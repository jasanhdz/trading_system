"""
Predictor for advanced temporal models.

Handles loading and inference with LSTM-based models.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from ml.nn_pattern.features import build_feature_frame
from .improved_architecture import DeepTemporalNet
from .temporal_model import EnsembleModel


logger = logging.getLogger(__name__)


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
        
        logger.info(f"Predictor initialized: {len(self.selected_features)} features, "
                   f"sequence_length={self.sequence_length}")
        logger.debug(f"Features: {self.selected_features[:5]}...{self.selected_features[-2:]}")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load ensemble of models from directory."""
        # Ensure model_path is a directory containing folds
        if not self.model_path.is_dir():
             raise ValueError(f"Model path must be a directory containing fold models: {self.model_path}")

        fold_files = sorted(list(self.model_path.glob("best_model_fold*.pt")))
        if not fold_files:
            raise FileNotFoundError(f"No 'best_model_fold*.pt' files found in {self.model_path}")

        logger.info(f"Found {len(fold_files)} fold models for ensemble: {[f.name for f in fold_files]}")
        models = []
        for fold_file in fold_files:
            checkpoint = torch.load(fold_file, map_location=self.device)
            model = DeepTemporalNet(
                input_dim=len(self.selected_features),
                sequence_length=self.sequence_length,
                **self.model_config,
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            models.append(model)
        
        ensemble = EnsembleModel(models)
        ensemble.to(self.device)
        ensemble.eval()
        return ensemble
    
    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare input sequence from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns, sorted chronologically
            
        Returns:
            Prepared sequence of shape (sequence_length, n_features)
            
        Raises:
            ValueError: If insufficient data or missing features
        """
        # Build features
        feature_frame, _ = build_feature_frame(df)
        
        # Validate data length
        if len(feature_frame) < self.sequence_length:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length} rows, "
                f"got {len(feature_frame)}"
            )
        
        # Validate all required features exist
        missing_features = set(self.selected_features) - set(feature_frame.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Available features: {list(feature_frame.columns[:10])}..."
            )
        
        # Select features in correct order
        features = feature_frame[self.selected_features].values
        
        # Take last sequence_length rows
        sequence = features[-self.sequence_length:]
        
        # Validate shape before scaling
        expected_shape = (self.sequence_length, len(self.selected_features))
        if sequence.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {sequence.shape}"
            )
        
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
