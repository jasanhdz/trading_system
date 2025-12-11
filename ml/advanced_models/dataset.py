"""
Advanced dataset with temporal sequences and feature engineering.

This module provides:
- Sliding window sequences for temporal models
- Advanced feature selection using multiple methods
- Walk-forward validation splits
- Data augmentation for time series
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from torch.utils.data import Dataset

from data.storage.database_manager import db_manager
from ml.nn_pattern.features import build_feature_frame


@dataclass
class AdvancedDatasetConfig:
    """Configuration for advanced temporal dataset."""
    
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    sequence_length: int = 24  # lookback window (e.g., 24 * 5min = 2 hours)
    prediction_horizon: int = 12  # bars ahead
    target_return: float = 0.002  # 0.2%
    min_records: int = 2000
    max_history_days: Optional[int] = None
    max_samples: Optional[int] = None
    
    # Feature selection
    use_feature_selection: bool = True
    n_features_to_select: int = 32  # from 64 original
    feature_selection_method: str = "mutual_info"  # or "correlation", "variance"
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_noise: float = 0.01  # 1% noise


class SequenceDataset(Dataset):
    """
    PyTorch Dataset that returns temporal sequences paired with targets.
    
    The dataset aligns each sequence ending at time `t` with the targets that
    were precomputed for that same timestamp (e.g. the future return between
    `t` and `t + prediction_horizon`).  This avoids shifting labels inside the
    Dataset and keeps both classification and regression targets consistent.
    """
    
    def __init__(
        self,
        features: np.ndarray,           # shape: (time_steps, n_features)
        class_labels: np.ndarray,       # shape: (time_steps,)
        regression_targets: Optional[np.ndarray] = None,  # shape: (time_steps,)
        sequence_length: int = 24,
        prediction_horizon: int = 12,
        augment: bool = False,
        augmentation_noise: float = 0.01,
        start_index: Optional[int] = None,
    ):
        if features.shape[0] != class_labels.shape[0]:
            raise ValueError(
                "Features and class labels must share the same number of rows "
                f"(got {features.shape[0]} vs {class_labels.shape[0]})"
            )
        if regression_targets is not None and features.shape[0] != regression_targets.shape[0]:
            raise ValueError(
                "Features and regression targets must share the same number of rows "
                f"(got {features.shape[0]} vs {regression_targets.shape[0]})"
            )
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augment = augment
        self.augmentation_noise = augmentation_noise
        
        # Convert to torch tensors
        self.features = torch.FloatTensor(features)
        self.class_labels = torch.LongTensor(class_labels)
        
        if regression_targets is not None:
            self.regression_targets = torch.FloatTensor(regression_targets)
        else:
            self.regression_targets = None
            
        # Create valid indices list
        # We can only start predicting when we have enough history
        # If start_index is provided (e.g. to skip buffer), use it
        start = start_index if start_index is not None else (sequence_length - 1)
        
        # Ensure start is at least sequence_length - 1
        start = max(start, sequence_length - 1)
        
        self.valid_indices = list(range(
            start,
            len(features)
        ))
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        
        # Get sequence (lookback window)
        start_idx = actual_idx - self.sequence_length + 1
        end_idx = actual_idx + 1
        sequence = self.features[start_idx:end_idx].clone() # Use clone() for tensors
        
        # Targets are already aligned with `actual_idx`
        class_label = self.class_labels[actual_idx]
        regression_value = None
        if self.regression_targets is not None:
            regression_value = self.regression_targets[actual_idx]
        
        # Data augmentation (add small noise)
        if self.augment and np.random.rand() < 0.5:
            noise = torch.randn_like(sequence) * self.augmentation_noise
            sequence = sequence + noise
        
        # Already tensors
        sequence_tensor = sequence.float()
        class_tensor = class_label.long()
        
        if regression_value is None:
            return sequence_tensor, class_tensor
        
        regression_tensor = regression_value.float()
        return sequence_tensor, class_tensor, regression_tensor


class FeatureSelector:
    """
    Advanced feature selection using multiple methods.
    
    Supports:
    - Mutual information
    - Correlation analysis
    - Variance thresholding
    - Recursive feature elimination
    """
    
    def __init__(
        self,
        method: str = "mutual_info",
        n_features: int = 32,
    ):
        self.method = method
        self.n_features = n_features
        self.selected_features: Optional[List[int]] = None
        self.feature_scores: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """
        Select best features based on the chosen method.
        
        Returns:
            List of selected feature names
        """
        if self.method == "mutual_info":
            return self._mutual_info_selection(X, y, feature_names)
        elif self.method == "correlation":
            return self._correlation_selection(X, y, feature_names)
        elif self.method == "variance":
            return self._variance_selection(X, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mutual_info_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> List[str]:
        """Select features based on mutual information with target."""
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        selector.fit(X, y)
        
        self.selected_features = selector.get_support(indices=True)
        self.feature_scores = selector.scores_
        
        selected_names = [feature_names[i] for i in self.selected_features]
        return selected_names
    
    def _correlation_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> List[str]:
        """Select features based on correlation with target."""
        # Compute absolute correlation with target
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        
        # Handle NaN correlations
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Select top k features
        self.selected_features = np.argsort(correlations)[-self.n_features:]
        self.feature_scores = correlations
        
        selected_names = [feature_names[i] for i in self.selected_features]
        return selected_names
    
    def _variance_selection(self, X: np.ndarray, feature_names: List[str]) -> List[str]:
        """Select features with highest variance (most informative)."""
        variances = np.var(X, axis=0)
        
        # Select top k features by variance
        self.selected_features = np.argsort(variances)[-self.n_features:]
        self.feature_scores = variances
        
        selected_names = [feature_names[i] for i in self.selected_features]
        return selected_names
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection to data."""
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[:, self.selected_features]
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Return DataFrame with feature importance scores."""
        if self.feature_scores is None:
            raise ValueError("Must call fit() first")
        
        df = pd.DataFrame({
            'feature': feature_names,
            'score': self.feature_scores,
            'selected': [i in self.selected_features for i in range(len(feature_names))]
        })
        
        return df.sort_values('score', ascending=False)


def _symbol_variants(symbol: str) -> List[str]:
    """Return possible DB symbol keys."""
    variants: List[str] = []
    seen: set[str] = set()

    def add(value: Optional[str]) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        variants.append(value)

    add(symbol)

    if ":" in symbol:
        add(symbol.split(":", 1)[0])
    elif "/" in symbol:
        base, quote = symbol.split("/", 1)
        add(f"{base}/{quote}:{quote}")

    return variants


def load_sequence_dataset(
    cfg: AdvancedDatasetConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load OHLCV data and prepare features for temporal model.
    
    Returns:
        features: Feature matrix (time_steps, n_features)
        class_labels: Class labels (time_steps,)
        regression_targets: Continuous returns (time_steps,)
        feature_names: List of feature names
    """
    # Calculate minimum required data
    effective_min = cfg.min_records + cfg.sequence_length
    if cfg.max_samples is not None:
        effective_min = min(effective_min, cfg.max_samples)
    
    required = effective_min + cfg.prediction_horizon
    
    # Try to load data from DB
    raw_df: Optional[pd.DataFrame] = None
    resolved_symbol: Optional[str] = None

    for candidate in _symbol_variants(cfg.symbol):
        candidate_df = db_manager.get_ohlcv_data(candidate, cfg.timeframe)
        if candidate_df.empty:
            continue

        candidate_df = candidate_df.sort_index()
        
        # Apply history window
        if cfg.max_history_days:
            latest_ts = candidate_df.index.max()
            if pd.isna(latest_ts):
                continue
            cutoff = latest_ts - pd.Timedelta(days=cfg.max_history_days)
            candidate_df = candidate_df[candidate_df.index >= cutoff]

        # Apply sample limit
        if cfg.max_samples is not None and len(candidate_df) > cfg.max_samples:
            candidate_df = candidate_df.tail(cfg.max_samples + cfg.prediction_horizon)

        if len(candidate_df) >= required:
            raw_df = candidate_df
            resolved_symbol = candidate
            break

        if raw_df is None or len(candidate_df) > len(raw_df):
            raw_df = candidate_df
            resolved_symbol = candidate

    if raw_df is None or resolved_symbol is None or len(raw_df) < required:
        available = len(raw_df) if raw_df is not None else 0
        tried = ", ".join(_symbol_variants(cfg.symbol))
        raise RuntimeError(
            f"Not enough {cfg.timeframe} data for {cfg.symbol} (tried: {tried}). "
            f"Need at least {required} rows, found {available}."
        )

    # Build features
    features_df, feature_names = build_feature_frame(raw_df)
    
    # Calculate targets
    aligned_close = raw_df.loc[features_df.index, "close"]
    future_close = aligned_close.shift(-cfg.prediction_horizon)
    future_return = (future_close / aligned_close) - 1.0

    # Classification labels (3 classes: neutral, long, short)
    class_labels = np.zeros(len(features_df), dtype=np.int64)
    long_mask = future_return >= cfg.target_return
    short_mask = future_return <= -cfg.target_return
    class_labels[long_mask] = 1  # Long
    class_labels[short_mask] = 2  # Short
    
    # Regression targets (actual returns)
    regression_targets = future_return.values.astype(np.float32)
    
    # Remove samples without valid targets
    mask = future_return.notna()
    features = features_df[mask].values.astype(np.float32)
    class_labels = class_labels[mask]
    regression_targets = regression_targets[mask]
    
    return features, class_labels, regression_targets, feature_names


def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward validation splits for time series.

    Instead of a single train/test split, creates multiple expanding windows:
    - Each split uses all previous data for training
    - Tests on the next time window
    - Simulates real trading where we retrain periodically

    Args:
        n_samples: Total number of samples
        n_splits: Number of validation folds
        train_ratio: Minimum proportion of data to use for first training
        gap: Number of samples to skip between train and test (avoid lookahead)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    min_train = int(n_samples * train_ratio)
    test_size = (n_samples - min_train) // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)

        if test_end <= test_start:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def walk_forward_split_with_validation(
    n_samples: int,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create walk-forward splits with VALIDATION set para early stopping.

    Divide los datos en: Train / Validation / Test
    - Train: Para entrenar el modelo
    - Validation: Para early stopping y ajustar hiperparámetros
    - Test: Nunca visto durante entrenamiento, solo para evaluación final

    Args:
        n_samples: Total number of samples
        n_splits: Number of validation folds
        train_ratio: Proportion for training (0.6 = 60%)
        val_ratio: Proportion for validation (0.2 = 20%)
        gap: Number of samples to skip between sets

    Returns:
        List of (train_indices, val_indices, test_indices) tuples
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) "
            f"debe ser < 1.0"
        )

    min_train = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = (n_samples - min_train - val_size) // n_splits

    splits = []
    for i in range(n_splits):
        # Train
        train_end = min_train + i * test_size

        # Validation
        val_start = train_end + gap
        val_end = val_start + val_size

        # Test
        test_start = val_end + gap
        test_end = min(test_start + test_size, n_samples)

        if test_end <= test_start or val_end <= val_start:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, val_idx, test_idx))

    return splits
