import numpy as np
import pytest
from ml.advanced_models.trainer import walk_forward_split_with_validation

def test_walk_forward_split_logic():
    # Setup
    n_samples = 1000
    n_splits = 5
    train_ratio = 0.6
    val_ratio = 0.2
    gap = 0
    
    splits = walk_forward_split_with_validation(
        n_samples=n_samples,
        n_splits=n_splits,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        gap=gap
    )
    
    # Verify number of splits
    assert len(splits) == n_splits
    
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"Fold {i}: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
        
        # Verify no overlap
        # Train vs Val
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        # Val vs Test
        assert len(np.intersect1d(val_idx, test_idx)) == 0
        # Train vs Test
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        
        # Verify temporal order
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()
        
        # Verify sizes (approximate)
        # Val size should be around 20% of total
        expected_val = int(n_samples * val_ratio)
        assert abs(len(val_idx) - expected_val) < 5
        
    # Verify expanding window (train set grows or shifts)
    # In this implementation, train window grows because start is always 0?
    # Let's check the implementation.
    # Implementation says: train_start = 0. So it is expanding window.
    
    first_train_size = len(splits[0][0])
    last_train_size = len(splits[-1][0])
    assert last_train_size > first_train_size

def test_walk_forward_gap():
    n_samples = 100
    gap = 5
    splits = walk_forward_split_with_validation(
        n_samples=n_samples,
        n_splits=2,
        train_ratio=0.5,
        val_ratio=0.2,
        gap=gap
    )
    
    train_idx, val_idx, test_idx = splits[0]
    
    # Check gap between train and val
    assert val_idx.min() - train_idx.max() > gap
    
    # Check gap between val and test
    assert test_idx.min() - val_idx.max() > gap

if __name__ == "__main__":
    test_walk_forward_split_logic()
    test_walk_forward_gap()
    print("All tests passed!")
