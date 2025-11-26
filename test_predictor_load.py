#!/usr/bin/env python3
"""Test predictor loading to debug the 500 errors."""
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import os
os.environ["ML_MODELS_ROOT"] = str(REPO_ROOT / "models" / "advanced")

from ml.advanced_models.predictor import AdvancedPredictor

# Test loading ETH model
symbol = "ETHUSDT"
timeframe = "15m"

model_dir = REPO_ROOT / "models" / "advanced" / symbol / timeframe
model_path = model_dir / "model.pt"
scaler_path = model_dir / "scaler.pkl"
meta_path = model_dir / "meta.json"
feature_selector_path = model_dir / "feature_selector.pkl"

print(f"Testing predictor loading for {symbol}/{timeframe}")
print(f"Model path: {model_path} (exists: {model_path.exists()})")
print(f"Scaler path: {scaler_path} (exists: {scaler_path.exists()})")
print(f"Meta path: {meta_path} (exists: {meta_path.exists()})")
print(f"Feature selector path: {feature_selector_path} (exists: {feature_selector_path.exists()})")
print()

try:
    predictor = AdvancedPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        meta_path=meta_path,
        feature_selector_path=feature_selector_path if feature_selector_path.exists() else None,
        device="cpu",
    )
    print("✅ Predictor loaded successfully!")
    print(f"Sequence length: {predictor.sequence_length}")
    print(f"Feature count: {len(predictor.selected_features)}")
except Exception as e:
    print(f"❌ Failed to load predictor: {e}")
    import traceback
    traceback.print_exc()
