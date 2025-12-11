"""
Advanced trainer with walk-forward validation and ensemble learning.

Features:
- Walk-forward cross-validation for realistic performance estimation
- Hyperparameter optimization using Optuna (optional)
- Ensemble model training with bagging
- Learning rate scheduling and gradient clipping
- Advanced early stopping with multiple criteria
- Detailed metrics tracking and visualization
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler

from .dataset import (
    AdvancedDatasetConfig,
    FeatureSelector,
    SequenceDataset,
    load_sequence_dataset,
    walk_forward_split,
)
from .temporal_model import AdvancedTemporalNet, EnsembleModel, MultiTaskLoss
from .improved_architecture import TemporalConvNet, TransformerNet


class AdvancedTrainer:
    """
    Comprehensive trainer for temporal trading models.
    
    Includes:
    - Walk-forward validation
    - Automatic hyperparameter tuning
    - Ensemble training
    - Multi-task learning (classification + regression)
    """
    
    def __init__(
        self,
        config: AdvancedDatasetConfig,
        model_config: Dict,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        self._set_seed(seed)
        
        # Will be populated during training
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[FeatureSelector] = None
        self.selected_features: Optional[List[str]] = None
        
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_and_prepare_data(
        self,
        apply_selection: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load data and perform feature engineering + selection.
        
        Returns:
            features, class_labels, regression_targets, feature_names
        """
        print(f"Loading data for {self.config.symbol} {self.config.timeframe}...")
        features, class_labels, regression_targets, feature_names = load_sequence_dataset(
            self.config
        )
        
        print(f"Loaded {len(features)} samples with {len(feature_names)} features")
        
        # Feature selection (if enabled)
        if self.config.use_feature_selection and apply_selection:
            print(f"Selecting top {self.config.n_features_to_select} features...")
            self.feature_selector = FeatureSelector(
                method=self.config.feature_selection_method,
                n_features=self.config.n_features_to_select,
            )
            
            # Use a subset for feature selection (faster)
            sample_size = min(5000, len(features))
            sample_idx = np.random.choice(len(features), sample_size, replace=False)
            
            self.selected_features = self.feature_selector.fit(
                features[sample_idx],
                class_labels[sample_idx],
                feature_names,
            )
            
            print(f"Selected features: {self.selected_features}")
            
            # Transform features
            features = self.feature_selector.transform(features)
        else:
            self.selected_features = feature_names
        
        return features, class_labels, regression_targets, self.selected_features
    
    def train_single_model(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        class_weights: Optional[torch.Tensor] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> Tuple[AdvancedTemporalNet, Dict]:
        """
        Train a single model with early stopping and LR scheduling.
        
        Returns:
            Trained model and training history
        """
        # Create model based on type
        model_type = self.model_config.get("type", "lstm")
        model_params = {k: v for k, v in self.model_config.items() if k != "type"}
        
        if model_type == "tcn":
            model = TemporalConvNet(
                input_dim=len(self.selected_features),
                **model_params,
            ).to(self.device)
        elif model_type == "transformer":
            model = TransformerNet(
                input_dim=len(self.selected_features),
                sequence_length=self.config.sequence_length,
                **model_params,
            ).to(self.device)
        else:
            # Default to LSTM (AdvancedTemporalNet)
            model = AdvancedTemporalNet(
                input_dim=len(self.selected_features),
                sequence_length=self.config.sequence_length,
                **model_params,
            ).to(self.device)
        
        # Loss and optimizer
        criterion = MultiTaskLoss(
            class_weights=class_weights.to(self.device) if class_weights is not None else None,
            classification_weight=1.0,
            regression_weight=0.5,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            # verbose=True,
        )
        
        # Training tracking
        best_metric = -float('inf')
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        history = {
            'train_loss': [],
            'train_class_loss': [],
            'train_reg_loss': [],
            'valid_loss': [],
            'valid_f1': [],
            'valid_accuracy': [],
            'valid_regression_mse': [],
            'valid_regression_mae': [],
        }
        
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            component_sums = defaultdict(float)
            saw_regression = False

            for batch in train_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    batch_seq, class_targets, regression_targets = batch
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    batch_seq, class_targets = batch
                    regression_targets = None
                else:
                    raise ValueError("Expected SequenceDataset to return 2 or 3 tensors per sample.")

                batch_seq = batch_seq.to(self.device)
                class_targets = class_targets.to(self.device)
                regression_targets = (
                    regression_targets.to(self.device) if regression_targets is not None else None
                )

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_seq)
                
                # Compute loss
                loss, loss_dict = criterion(
                    outputs['logits'],
                    class_targets,
                    outputs.get('regression'),
                    regression_targets,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_seq.size(0)
                train_samples += batch_seq.size(0)
                component_sums['class_loss'] += loss_dict['class_loss'] * batch_seq.size(0)
                if 'reg_loss' in loss_dict:
                    component_sums['reg_loss'] += loss_dict['reg_loss'] * batch_seq.size(0)
                    saw_regression = True
            
            avg_train_loss = train_loss / max(train_samples, 1)
            avg_class_loss = component_sums['class_loss'] / max(train_samples, 1)
            avg_reg_loss = (
                component_sums['reg_loss'] / max(train_samples, 1) if saw_regression else None
            )
            history['train_loss'].append(avg_train_loss)
            history['train_class_loss'].append(avg_class_loss)
            history['train_reg_loss'].append(avg_reg_loss)
            
            # Validation phase
            if valid_loader:
                valid_metrics = self.evaluate_model(model, valid_loader)
                history['valid_loss'].append(valid_metrics['loss'])
                history['valid_f1'].append(valid_metrics['macro_f1'])
                history['valid_accuracy'].append(valid_metrics['accuracy'])
                reg_metrics = valid_metrics.get('regression')
                history['valid_regression_mse'].append(reg_metrics['mse'] if reg_metrics else None)
                history['valid_regression_mae'].append(reg_metrics['mae'] if reg_metrics else None)
                
                # Update learning rate
                scheduler.step(valid_metrics['macro_f1'])
                
                # Early stopping check
                metric = valid_metrics['macro_f1']
                if metric - best_metric > min_delta:
                    best_metric = metric
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Valid Loss: {valid_metrics['loss']:.4f} | "
                          f"Valid F1: {valid_metrics['macro_f1']:.4f} | "
                          f"Valid Acc: {valid_metrics['accuracy']:.4f}")
                
                if patience and epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Restored best model from epoch {best_epoch}")
        
        return model, history
    
    def evaluate_model(
        self,
        model: AdvancedTemporalNet,
        loader: DataLoader,
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        regression_predictions = []
        regression_truth = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    batch_seq, class_targets = batch[:2]
                    regression_targets = batch[2] if len(batch) >= 3 else None
                else:
                    raise ValueError("Expected SequenceDataset to return at least sequence and class label.")

                batch_seq = batch_seq.to(self.device)
                class_targets = class_targets.to(self.device)

                outputs = model(batch_seq)
                logits = outputs['logits']
                
                # Simple cross entropy for evaluation
                loss = nn.functional.cross_entropy(logits, class_targets)
                
                all_logits.append(logits.cpu())
                all_targets.append(class_targets.cpu())
                
                total_loss += loss.item() * batch_seq.size(0)
                total_samples += batch_seq.size(0)

                if regression_targets is not None and 'regression' in outputs:
                    regression_predictions.append(outputs['regression'].cpu().numpy().reshape(-1))
                    regression_truth.append(regression_targets.cpu().numpy().reshape(-1))
        
        # Concatenate all batches
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute probabilities and predictions
        probs = torch.softmax(logits, dim=1).numpy()
        predictions = probs.argmax(axis=1)
        y_true = targets.numpy()
        
        # Metrics
        accuracy = float((predictions == y_true).mean())
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[0, 1, 2],
            zero_division=0,
        )
        
        macro_f1 = float(f1.mean())
        
        # Per-class metrics
        per_class = {}
        class_names = ['neutral', 'long', 'short']
        for idx, name in enumerate(class_names):
            per_class[name] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx]),
            }
        
        # Average precision for long/short
        ap_long = ap_short = 0.0
        if (y_true == 1).any():
            ap_long = float(average_precision_score((y_true == 1).astype(int), probs[:, 1]))
        if (y_true == 2).any():
            ap_short = float(average_precision_score((y_true == 2).astype(int), probs[:, 2]))
        
        result = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class': per_class,
            'ap_long': ap_long,
            'ap_short': ap_short,
        }

        if regression_predictions and regression_truth:
            preds = np.concatenate(regression_predictions)
            trues = np.concatenate(regression_truth)
            result['regression'] = {
                'mse': float(np.mean((preds - trues) ** 2)),
                'mae': float(np.mean(np.abs(preds - trues))),
            }
        else:
            result['regression'] = None

        return result
    
    def walk_forward_validation(
        self,
        features: np.ndarray,
        class_labels: np.ndarray,
        regression_targets: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5,
        **train_kwargs,
    ) -> List[Dict]:
        """
        Perform walk-forward validation with proper Train/Validation/Test splits.
        
        Returns:
            List of metrics for each fold
        """
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION ({n_splits} folds)")
        print(f"{'='*60}\n")
        
        # Use the new split method with validation set
        splits = walk_forward_split_with_validation(
            n_samples=len(features),
            n_splits=n_splits,
            train_ratio=0.6,  # 60% initial train
            val_ratio=0.2,    # 20% validation
            gap=self.config.prediction_horizon,
        )
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")
            print(f"Train: {len(train_idx)} | Valid: {len(val_idx)} | Test: {len(test_idx)}")
            
            # Split data
            train_features = features[train_idx]
            train_class = class_labels[train_idx]
            
            # Add lookback buffer to Val and Test to prevent data loss at boundaries
            lookback = self.config.sequence_length
            
            # Val needs lookback from train
            val_start_idx = max(0, val_idx[0] - lookback)
            val_features_buffer = features[val_start_idx : val_idx[-1] + 1]
            val_class_buffer = class_labels[val_start_idx : val_idx[-1] + 1]
            val_reg_buffer = regression_targets[val_start_idx : val_idx[-1] + 1]
            
            # Test needs lookback from val (or train if val is skipped/empty)
            test_start_idx = max(0, test_idx[0] - lookback)
            test_features_buffer = features[test_start_idx : test_idx[-1] + 1]
            test_class_buffer = class_labels[test_start_idx : test_idx[-1] + 1]
            test_reg_buffer = regression_targets[test_start_idx : test_idx[-1] + 1]
            
            # Feature Selection (Fit on Train ONLY)
            current_feature_names = feature_names
            if self.config.use_feature_selection:
                # Create a new selector for this fold
                fold_selector = FeatureSelector(
                    method=self.config.feature_selection_method,
                    n_features=self.config.n_features_to_select,
                )
                
                # Fit on TRAIN data only
                # Use a subset if train is too large to speed up
                sample_size = min(5000, len(train_features))
                if len(train_features) > sample_size:
                    sample_idx = np.random.choice(len(train_features), sample_size, replace=False)
                    fit_features = train_features[sample_idx]
                    fit_labels = train_class[sample_idx]
                else:
                    fit_features = train_features
                    fit_labels = train_class
                
                selected = fold_selector.fit(
                    fit_features,
                    fit_labels,
                    feature_names,
                )
                
                # Transform all sets (including buffers)
                train_features = fold_selector.transform(train_features)
                val_features_buffer = fold_selector.transform(val_features_buffer)
                test_features_buffer = fold_selector.transform(test_features_buffer)
                current_feature_names = selected
            
            # Scale features (fit on train only)
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features_buffer = scaler.transform(val_features_buffer)
            test_features_buffer = scaler.transform(test_features_buffer)
            
            # Create datasets
            train_dataset = SequenceDataset(
                train_features,
                train_class,
                regression_targets[train_idx],
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                augment=self.config.use_augmentation,
                augmentation_noise=self.config.augmentation_noise,
            )
            
            val_dataset = SequenceDataset(
                val_features_buffer,
                val_class_buffer,
                val_reg_buffer,
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                augment=False,
                start_index=lookback if val_start_idx < val_idx[0] else None, # Only skip if we added buffer
            )
            
            test_dataset = SequenceDataset(
                test_features_buffer,
                test_class_buffer,
                test_reg_buffer,
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                augment=False,
                start_index=lookback if test_start_idx < test_idx[0] else None, # Only skip if we added buffer
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=512,
                shuffle=True,
                drop_last=False,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=512,
                shuffle=False,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=512,
                shuffle=False,
            )
            
            # Compute class weights based on TRAIN set
            class_counts = np.bincount(train_class, minlength=3)
            if (class_counts > 0).all():
                inv_weights = class_counts.sum() / class_counts
                class_weights = torch.from_numpy((inv_weights / inv_weights.mean()).astype(np.float32))
            else:
                class_weights = None
            
            # Update selected features for this fold so the model knows the input dimension
            self.selected_features = current_feature_names
            
            # Train model using VALIDATION set for early stopping
            model, history = self.train_single_model(
                train_loader,
                val_loader,  # Pass validation loader here!
                class_weights=class_weights,
                **train_kwargs,
            )
            
            # Evaluate on TEST set (unseen)
            test_metrics = self.evaluate_model(model, test_loader)
            
            print(f"\nFold {fold_idx} Test Results (Unseen Data):")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"  AP Long: {test_metrics['ap_long']:.4f}")
            print(f"  AP Short: {test_metrics['ap_short']:.4f}")
            
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'metrics': test_metrics,
                'history': history,
            })
        
        # Aggregate results
        avg_f1 = np.mean([r['metrics']['macro_f1'] for r in fold_results])
        
        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Average Macro F1: {avg_f1:.4f} ± {np.std([r['metrics']['macro_f1'] for r in fold_results]):.4f}")
        print(f"{'='*60}\n")
        
        return fold_results


def walk_forward_split_with_validation(
    n_samples: int,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create walk-forward splits with VALIDATION set for early stopping.
    
    Returns:
        List of (train_indices, val_indices, test_indices) tuples
    """
    # Calculate size of the test block
    # We reserve train_ratio + val_ratio for the initial window
    initial_window_size = int(n_samples * (train_ratio + val_ratio))
    remaining_samples = n_samples - initial_window_size
    test_size = remaining_samples // n_splits
    
    if test_size <= 0:
         # Fallback for small datasets: simple split
         test_size = n_samples // (n_splits + 2)
         initial_window_size = n_samples - (test_size * n_splits)

    splits = []
    current_end = initial_window_size
    
    for i in range(n_splits):
        # Define windows relative to current_end
        # Test window is the next block
        test_start = current_end + gap
        test_end = min(test_start + test_size, n_samples)
        
        if test_end <= test_start:
            break
            
        # Validation window is the block immediately preceding test
        # Size is roughly val_ratio of total, or proportional
        val_size_indices = int(n_samples * val_ratio)
        val_end = current_end
        val_start = max(0, val_end - val_size_indices)
        
        # Train window is everything before validation
        train_end = val_start - gap
        train_start = 0
        
        if train_end <= train_start:
            # Not enough data for this split configuration
            continue
            
        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_end)
        
        splits.append((train_idx, val_idx, test_idx))
        
        # Move window forward
        current_end = test_end

    return splits

