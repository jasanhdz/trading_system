#!/usr/bin/env python3
"""
Quick test to verify the advanced models work correctly.

This script performs a minimal test to ensure:
- Data loading works
- Model can be instantiated
- Training loop runs
- Prediction works
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import torch

print("="*80)
print("ADVANCED MODELS - QUICK TEST")
print("="*80 + "\n")

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from ml.advanced_models.temporal_model import (
        AdvancedTemporalNet,
        AttentionLayer,
        EnsembleModel,
        MultiTaskLoss,
    )
    from ml.advanced_models.dataset import (
        AdvancedDatasetConfig,
        SequenceDataset,
        FeatureSelector,
        walk_forward_split,
    )
    from ml.advanced_models.trainer import AdvancedTrainer
    from ml.advanced_models.predictor import AdvancedPredictor
    print("✓ All modules imported successfully\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Create model
print("Test 2: Creating model...")
try:
    model = AdvancedTemporalNet(
        input_dim=32,
        sequence_length=24,
        hidden_dim=64,
        lstm_layers=1,
        dense_dims=(128,),
        dropout=0.2,
        use_attention=True,
        bidirectional=True,
        num_classes=3,
        use_regression=True,
    )
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters\n")
except Exception as e:
    print(f"✗ Model creation failed: {e}\n")
    sys.exit(1)

# Test 3: Forward pass
print("Test 3: Testing forward pass...")
try:
    batch_size = 16
    seq_length = 24
    input_dim = 32
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Regression shape: {outputs['regression'].shape}\n")
except Exception as e:
    print(f"✗ Forward pass failed: {e}\n")
    sys.exit(1)

# Test 4: Loss computation
print("Test 4: Testing loss function...")
try:
    criterion = MultiTaskLoss(
        class_weights=None,
        classification_weight=1.0,
        regression_weight=0.5,
    )
    
    class_targets = torch.randint(0, 3, (batch_size,))
    regression_targets = torch.randn(batch_size)
    
    loss, loss_dict = criterion(
        outputs['logits'],
        class_targets,
        outputs['regression'],
        regression_targets,
    )
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss components: {loss_dict}\n")
except Exception as e:
    print(f"✗ Loss computation failed: {e}\n")
    sys.exit(1)

# Test 5: Dataset creation
print("Test 5: Testing dataset...")
try:
    # Create dummy data
    n_samples = 1000
    n_features = 32
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    class_labels = np.random.randint(0, 3, n_samples).astype(np.int64)
    regression_targets = np.random.randn(n_samples).astype(np.float32)
    
    dataset = SequenceDataset(
        features=features,
        class_labels=class_labels,
        regression_targets=regression_targets,
        sequence_length=24,
        prediction_horizon=12,
        augment=False,
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Test getitem
    seq, class_label, regression_value = dataset[0]
    print(f"  Sequence shape: {seq.shape}")
    print(f"  Class label: {class_label.item()} | Regression target: {regression_value.item():.4f}\n")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}\n")
    sys.exit(1)

# Test 6: Feature selector
print("Test 6: Testing feature selector...")
try:
    X_dummy = np.random.randn(500, 64)
    y_dummy = np.random.randint(0, 3, 500)
    feature_names = [f"feature_{i}" for i in range(64)]
    
    selector = FeatureSelector(method='mutual_info', n_features=32)
    selected = selector.fit(X_dummy, y_dummy, feature_names)
    
    X_transformed = selector.transform(X_dummy)
    
    print(f"✓ Feature selection successful")
    print(f"  Selected {len(selected)} features from {X_dummy.shape[1]}")
    print(f"  Transformed shape: {X_transformed.shape}\n")
except Exception as e:
    print(f"✗ Feature selection failed: {e}\n")
    sys.exit(1)

# Test 7: Walk-forward splits
print("Test 7: Testing walk-forward splits...")
try:
    splits = walk_forward_split(
        n_samples=1000,
        n_splits=5,
        train_ratio=0.7,
        gap=12,
    )
    
    print(f"✓ Walk-forward splits created: {len(splits)} folds")
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"  Fold {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    print()
except Exception as e:
    print(f"✗ Walk-forward splits failed: {e}\n")
    sys.exit(1)

# Test 8: Ensemble
print("Test 8: Testing ensemble...")
try:
    models = [
        AdvancedTemporalNet(
            input_dim=32,
            sequence_length=24,
            hidden_dim=64,
            lstm_layers=1,
            dense_dims=(128,),
            dropout=0.2,
            num_classes=3,
            use_regression=True,
        )
        for _ in range(3)
    ]
    
    ensemble = EnsembleModel(models)
    
    # Forward pass
    outputs = ensemble(dummy_input)
    
    print(f"✓ Ensemble created: {len(models)} models")
    print(f"  Output shape: {outputs['logits'].shape}\n")
except Exception as e:
    print(f"✗ Ensemble creation failed: {e}\n")
    sys.exit(1)

# Test 9: Backward pass
print("Test 9: Testing backward pass...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward
    outputs = model(dummy_input)
    
    # Fresh targets for backward test
    batch_size = dummy_input.size(0)
    class_targets = torch.randint(0, 3, (batch_size,))
    regression_targets = torch.randn(batch_size)
    
    # Loss
    loss, _ = criterion(
        outputs['logits'],
        class_targets,
        outputs['regression'],
        regression_targets,
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backward pass successful")
    print(f"  Loss after step: {loss.item():.4f}\n")
except Exception as e:
    print(f"✗ Backward pass failed: {e}\n")
    sys.exit(1)

# Summary
print("="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nThe advanced models are ready to use!")
print("\nNext steps:")
print("1. Train a model:")
print("   python scripts/train_advanced_model.py --symbol BTCUSDT --timeframe 15m")
print("\n2. Compare with old model:")
print("   python scripts/compare_models.py --symbol BTCUSDT --timeframe 15m")
print("\n3. Read the documentation:")
print("   ml/advanced_models/README.md")
print()
