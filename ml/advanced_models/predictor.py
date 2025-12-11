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
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.meta_path = Path(meta_path)
        self.device = torch.device(device)
        
        # Verify files exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
        
        # Load metadata first
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)

        # Extract configuration
        self.sequence_length = self.meta['sequence_length']
        self.selected_features = self.meta['selected_features']
        self.model_config = self.meta['model_config']
        self.ensemble_size = self.meta.get('ensemble_size', 1)
        
        logger.info(f"Predictor initialized: {len(self.selected_features)} features, "
                   f"sequence_length={self.sequence_length}")
        logger.debug(f"Features: {self.selected_features[:5]}...{self.selected_features[-2:]}")

        # Load ensemble pipelines (Model + Scaler + Selector)
        # Must be called AFTER extracting configuration
        self.ensemble_pipelines = self._load_ensemble()
        logger.info(f"Ensemble loaded successfully with {len(self.ensemble_pipelines)} models on {self.device}")

    def _load_ensemble(self) -> list:
        """Load ensemble of models, scalers, and selectors."""
        # Ensure model_path is a directory containing folds
        if not self.model_path.is_dir():
             raise ValueError(f"Model path must be a directory containing fold models: {self.model_path}")

        fold_files = sorted(list(self.model_path.glob("best_model_fold*.pt")))
        if not fold_files:
            raise FileNotFoundError(f"No 'best_model_fold*.pt' files found in {self.model_path}")

        logger.info(f"Found {len(fold_files)} fold models for ensemble: {[f.name for f in fold_files]}")
        
        pipelines = []
        
        for fold_file in fold_files:
            # Extract fold index from filename (e.g., best_model_fold0.pt -> 0)
            try:
                fold_idx = int(fold_file.stem.replace("best_model_fold", ""))
            except ValueError:
                logger.warning(f"Could not extract fold index from {fold_file.name}, skipping.")
                continue

            # Load Scaler for this fold
            scaler_file = self.model_path / f"scaler_fold{fold_idx}.pkl"
            if not scaler_file.exists():
                # Fallback to global scaler if per-fold not found (backward compatibility)
                if self.scaler_path.exists():
                    scaler = joblib.load(self.scaler_path)
                else:
                    raise FileNotFoundError(f"Scaler not found for fold {fold_idx}: {scaler_file}")
            else:
                scaler = joblib.load(scaler_file)

            # Load Selector for this fold
            selector_file = self.model_path / f"feature_selector_fold{fold_idx}.pkl"
            selector = None
            if selector_file.exists():
                selector = joblib.load(selector_file)
            elif self.feature_selector_path and self.feature_selector_path.exists():
                # Fallback to global selector
                selector = joblib.load(self.feature_selector_path)

            # Determine input dim
            # If selector exists, use its n_features, otherwise use meta or scaler mean
            if selector:
                input_dim = selector.n_features
            else:
                # If no selector, assume all features in scaler are used
                input_dim = scaler.mean_.shape[0]

            # Load Model
            checkpoint = torch.load(fold_file, map_location=self.device)
            model = DeepTemporalNet(
                input_dim=input_dim,
                sequence_length=self.sequence_length,
                **self.model_config,
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            pipelines.append({
                'model': model,
                'scaler': scaler,
                'selector': selector
            })
            
        return pipelines

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
        # Prepare sequence (FULL features)
        sequence_full = self._prepare_full_features(df)
        
        # Aggregate predictions
        total_probs = np.zeros(3)
        total_regression = 0.0
        regression_count = 0
        
        for pipeline in self.ensemble_pipelines:
            model = pipeline['model']
            scaler = pipeline['scaler']
            selector = pipeline['selector']
            
            # 1. Select features (if selector exists)
            seq_processed = sequence_full
            if selector:
                # Selector expects (n_samples, n_features), sequence is (seq_len, n_features)
                # But here we have (seq_len, n_features), so we transform directly
                seq_processed = selector.transform(seq_processed)
                
            # 2. Scale
            seq_processed = scaler.transform(seq_processed)
            
            # 3. Predict
            seq_tensor = torch.from_numpy(seq_processed).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                outputs = model(seq_tensor)
                probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy().flatten()
                total_probs += probs
                
                if 'regression' in outputs:
                    total_regression += outputs['regression'].cpu().numpy().flatten()[0]
                    regression_count += 1
        
        # Average
        avg_probs = total_probs / len(self.ensemble_pipelines)
        
        result = {
            'neutral': float(avg_probs[0]),
            'long': float(avg_probs[1]),
            'short': float(avg_probs[2]),
        }
        
        if regression_count > 0:
            result['predicted_return'] = float(total_regression / regression_count)
            
        # Add direction and confidence
        direction_idx = int(avg_probs.argmax())
        directions = ['neutral', 'long', 'short']
        result['direction'] = directions[direction_idx]
        result['confidence'] = float(avg_probs[direction_idx])
        
        return result
    def predict_batch(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict for multiple sequences at once (e.g. backtesting).
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Dictionary with batch predictions
        """
        # Prepare full feature matrix (n_samples, n_features)
        # Note: This method expects a DataFrame, not raw sequences, to ensure feature alignment
        features_full = self._prepare_full_features(df)
        
        # Create sliding windows
        # Shape: (n_windows, sequence_length, n_features)
        # This is expensive for large DF, but ensures correctness
        n_samples = len(features_full)
        if n_samples < self.sequence_length:
             raise ValueError("Insufficient data for batch prediction")
             
        # Efficient sliding window view
        stride = features_full.strides[0]
        n_windows = n_samples - self.sequence_length + 1
        windows = np.lib.stride_tricks.as_strided(
            features_full, 
            shape=(n_windows, self.sequence_length, features_full.shape[1]),
            strides=(stride, stride)
        )
        
        all_probs = []
        all_regs = []
        
        for pipeline in self.ensemble_pipelines:
            model = pipeline['model']
            scaler = pipeline['scaler']
            selector = pipeline['selector']
            
            # Process batch: windows shape (batch, seq_len, n_features)
            batch_size, seq_len, n_feat = windows.shape
            
            # Flatten for transform: (batch * seq_len, n_features)
            seq_flat = windows.reshape(-1, n_feat)
            
            if selector:
                seq_flat = selector.transform(seq_flat)
            
            seq_flat = scaler.transform(seq_flat)
            
            # Reshape back: (batch, seq_len, new_n_feat)
            new_n_feat = seq_flat.shape[1]
            seq_processed = seq_flat.reshape(batch_size, seq_len, new_n_feat)
            
            seq_tensor = torch.from_numpy(seq_processed).to(self.device).float()
            
            with torch.no_grad():
                # Process in mini-batches to avoid OOM
                mini_batch_size = 1024
                pipeline_probs = []
                pipeline_regs = []
                
                for i in range(0, batch_size, mini_batch_size):
                    batch_tensor = seq_tensor[i:i+mini_batch_size]
                    outputs = model(batch_tensor)
                    probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
                    pipeline_probs.append(probs)
                    
                    if 'regression' in outputs:
                        pipeline_regs.append(outputs['regression'].cpu().numpy())
                
                all_probs.append(np.concatenate(pipeline_probs, axis=0))
                if pipeline_regs:
                    all_regs.append(np.concatenate(pipeline_regs, axis=0))
        
        # Average across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        
        result = {
            'probabilities': avg_probs,
            'predictions': avg_probs.argmax(axis=-1),
        }
        
        if all_regs:
            avg_regs = np.mean(all_regs, axis=0)
            result['predicted_returns'] = avg_regs
        
        return result

    def _prepare_full_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare full feature matrix from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns, sorted chronologically
            
        Returns:
            Full feature matrix of shape (sequence_length, n_total_features)
            
        Raises:
            ValueError: If insufficient data or missing features
        """
        # Build features
        feature_frame, feature_names = build_feature_frame(df)
        
        # Validate data length
        if len(feature_frame) < self.sequence_length:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length} rows, "
                f"got {len(feature_frame)}"
            )
        
        # Validate all required features exist (using the superset from meta if available, or just check what we have)
        # En entrenamiento, 'selected_features' en meta.json son las features ANTES de la selección (feature_names)
        # O son las features DESPUES de la selección global (si se usó selección global antes de folds).
        # En nuestro script actual, 'feature_names' guardado en meta son las features ORIGINALES antes de selección per-fold.
        # Por tanto, debemos asegurar que el dataframe tenga esas columnas.
        
        required_features = self.selected_features # Estas son las features base que espera el selector
        missing_features = set(required_features) - set(feature_frame.columns)
        
        if missing_features:
            # A veces build_feature_frame genera nombres ligeramente distintos o el orden importa
            # Si faltan muchas, es un error.
            raise ValueError(
                f"Missing required features: {list(missing_features)[:5]}... "
                f"Available features: {list(feature_frame.columns[:10])}..."
            )
        
        # Select features in correct order (the order expected by the selector/scaler)
        features = feature_frame[required_features].values
        
        # Take last sequence_length rows
        sequence = features[-self.sequence_length:]
        
        return sequence.astype(np.float32)
