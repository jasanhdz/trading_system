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
        self
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
        if self.config.use_feature_selection:
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
        # Create model
        model = AdvancedTemporalNet(
            input_dim=len(self.selected_features),
            sequence_length=self.config.sequence_length,
            **self.model_config,
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
        n_splits: int = 5,
        **train_kwargs,
    ) -> List[Dict]:
        """
        Perform walk-forward validation to get realistic performance estimates.
        
        Returns:
            List of metrics for each fold
        """
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION ({n_splits} folds)")
        print(f"{'='*60}\n")
        
        splits = walk_forward_split(
            n_samples=len(features),
            n_splits=n_splits,
            train_ratio=0.7,
            gap=self.config.prediction_horizon,
        )
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")
            print(f"Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
            
            # Split data
            train_features = features[train_idx]
            train_class = class_labels[train_idx]
            test_features = features[test_idx]
            test_class = class_labels[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)
            
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
            
            test_dataset = SequenceDataset(
                test_features,
                test_class,
                regression_targets[test_idx],
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                augment=False,
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=512,
                shuffle=True,
                drop_last=False,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=512,
                shuffle=False,
            )
            
            # Compute class weights
            class_counts = np.bincount(train_class, minlength=3)
            if (class_counts > 0).all():
                inv_weights = class_counts.sum() / class_counts
                class_weights = torch.from_numpy((inv_weights / inv_weights.mean()).astype(np.float32))
            else:
                class_weights = None
            
            # Train model
            model, history = self.train_single_model(
                train_loader,
                None,  # No validation set in walk-forward (use test as final eval)
                class_weights=class_weights,
                **train_kwargs,
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, test_loader)
            
            print(f"\nFold {fold_idx} Test Results:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"  AP Long: {test_metrics['ap_long']:.4f}")
            print(f"  AP Short: {test_metrics['ap_short']:.4f}")
            reg_metrics = test_metrics.get('regression')
            if reg_metrics:
                print(f"  Regression MSE: {reg_metrics['mse']:.6f}")
                print(f"  Regression MAE: {reg_metrics['mae']:.6f}")
            
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': test_metrics,
                'history': history,
            })
        
        # Aggregate results
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['metrics']['macro_f1'] for r in fold_results])
        avg_ap_long = np.mean([r['metrics']['ap_long'] for r in fold_results])
        avg_ap_short = np.mean([r['metrics']['ap_short'] for r in fold_results])
        regression_metrics = [r['metrics'].get('regression') for r in fold_results]
        regression_metrics = [m for m in regression_metrics if m]
        
        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std([r['metrics']['accuracy'] for r in fold_results]):.4f}")
        print(f"Average Macro F1: {avg_f1:.4f} ± {np.std([r['metrics']['macro_f1'] for r in fold_results]):.4f}")
        print(f"Average AP Long: {avg_ap_long:.4f}")
        print(f"Average AP Short: {avg_ap_short:.4f}")
        if regression_metrics and len(regression_metrics) == len(fold_results):
            avg_mse = np.mean([m['mse'] for m in regression_metrics])
            std_mse = np.std([m['mse'] for m in regression_metrics])
            avg_mae = np.mean([m['mae'] for m in regression_metrics])
            std_mae = np.std([m['mae'] for m in regression_metrics])
            print(f"Average Regression MSE: {avg_mse:.6f} ± {std_mse:.6f}")
            print(f"Average Regression MAE: {avg_mae:.6f} ± {std_mae:.6f}")
        print(f"{'='*60}\n")
        
        return fold_results
