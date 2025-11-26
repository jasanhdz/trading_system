#!/usr/bin/env python3
"""
Phase 1: Extract real feature names from feature_selector.pkl

This script loads the feature selector (which requires ml imports)
and extracts the real feature names for each model.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add repo root to path for ml imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import torch
from ml.nn_pattern.features import ALL_FEATURES


def extract_features_for_model(symbol: str, timeframe: str):
    """Extract real feature names from feature_selector.pkl."""
    base_dir = Path(f"models/advanced/{symbol}/{timeframe}")
    
    print(f"\n📦 Processing {symbol} {timeframe}...")
    
    # Load feature selector
    selector_path = base_dir / "feature_selector.pkl"
    if not selector_path.exists():
        print(f"   ❌ feature_selector.pkl not found")
        return None
    
    selector = joblib.load(selector_path)
    
    # Extract selected feature INDICES
    if not hasattr(selector, 'selected_features'):
        print(f"   ❌ Selector doesn't have 'selected_features' attribute")
        return None
    
    selected_indices = [int(idx) for idx in selector.selected_features]  # Convert to int
    print(f"   ✅ Extracted {len(selected_indices)} feature indices")
    
    # Map indices to actual feature names
    selected_feature_names = [ALL_FEATURES[idx] for idx in selected_indices]
    print(f"   ✅ Mapped to feature names")
    print(f"   First 5: {selected_feature_names[:5]}")
    print(f"   Last 5: {selected_feature_names[-5:]}")
    
    # Verify against model checkpoint
    model_path = base_dir / "model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    expected_features = checkpoint['input_proj.0.weight'].shape[1]
    
    if len(selected_feature_names) != expected_features:
        print(f"   ⚠️  WARNING: Feature count mismatch!")
        print(f"      Selector: {len(selected_feature_names)}, Model: {expected_features}")
        return None
    
    print(f"   ✅ Feature count matches model: {expected_features}")
    
    return selected_feature_names


def main():
    print("=" * 80)
    print("PHASE 1: Extracting Real Feature Names")
    print("=" * 80)
    
    models = [
        ("ETHUSDT", "15m"),
        ("XRPUSDT", "15m"),
        ("LTCUSDT", "15m"),
    ]
    
    results = {}
    
    for symbol, timeframe in models:
        features = extract_features_for_model(symbol, timeframe)
        if features:
            results[f"{symbol}_{timeframe}"] = features
    
    # Save extracted features to JSON for reference
    output_file = Path("models/advanced/extracted_features.json")
    with open(output_file, 'w') as f:
        json.dump({
            "extracted_at": datetime.now().isoformat(),
            "models": results
        }, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Extracted features for {len(results)}/{len(models)} models")
    print(f"📄 Saved to: {output_file}")
    print(f"{'=' * 80}")
    
    if len(results) == len(models):
        print("\n🚀 Ready for Phase 2: Updating meta.json files")
        return True
    else:
        print("\n⚠️  Some models failed - check errors above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
