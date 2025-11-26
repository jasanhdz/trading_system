
import sys
import os
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from ml.advanced_models.dataset import AdvancedDatasetConfig, load_sequence_dataset
from ml.advanced_models.trainer import AdvancedTrainer
from ml.advanced_models.improved_architecture import TemporalConvNet

def test_pipeline():
    print("🚀 Testing New Features & Architecture Pipeline")
    
    # 1. Test Feature Generation
    print("\n1. Testing Feature Generation...")
    config = AdvancedDatasetConfig(
        symbol="BTC/USDT:USDT", # Correct DB format
        timeframe="15m",
        sequence_length=48,
        prediction_horizon=3,
        target_return=0.002,
        max_samples=1000, # Small sample for speed
        use_feature_selection=False
    )
    
    try:
        features, class_labels, reg_targets, feature_names = load_sequence_dataset(config)
        print(f"✅ Data Loaded: {features.shape}")
        
        # Check for new features
        new_feats = ["atr_pct", "volume_flow", "price_location"]
        found = [f for f in new_feats if f in feature_names]
        if len(found) == len(new_feats):
            print(f"✅ New Features Found: {found}")
        else:
            print(f"❌ Missing Features: {set(new_feats) - set(found)}")
            
    except Exception as e:
        print(f"❌ Feature Generation Failed: {e}")
        return

    # 2. Test TCN Model Initialization
    print("\n2. Testing TCN Model Initialization...")
    try:
        model = TemporalConvNet(
            input_dim=len(feature_names),
            num_channels=[32, 64, 128],
            kernel_size=3,
            num_classes=3
        )
        print("✅ TCN Model Initialized")
        
        # Test forward pass
        dummy_input = torch.randn(2, 48, len(feature_names))
        output = model(dummy_input)
        print(f"✅ Forward Pass Output Shape: {output['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Model Initialization Failed: {e}")
        return

    # 3. Test Trainer Integration (Short Run)
    print("\n3. Testing Trainer Integration...")
    try:
        trainer = AdvancedTrainer(
            config=config,
            model_config={
                "type": "tcn",
                "num_channels": [32, 64],
                "kernel_size": 3,
                "dropout": 0.2
            },
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Manually set selected features since we skipped load_and_prepare_data
        trainer.selected_features = feature_names
        
        # Use SequenceDataset to create windows
        from ml.advanced_models.dataset import SequenceDataset
        from torch.utils.data import DataLoader
        
        dataset = SequenceDataset(
            features,
            class_labels,
            reg_targets,
            sequence_length=48,
            prediction_horizon=3,
            augment=False
        )
        loader = DataLoader(dataset, batch_size=16)
        
        print("   Starting training loop (1 epoch)...")
        model, history = trainer.train_single_model(
            train_loader=loader,
            valid_loader=loader, # Reuse for test
            epochs=1
        )
        print("✅ Training Loop Completed")
        
    except Exception as e:
        print(f"❌ Trainer Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
