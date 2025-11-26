#!/usr/bin/env python3
"""
Phase 2: Update meta.json files with real feature names.

Uses the extracted features from Phase 1 to update all meta.json files.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def update_meta_json(symbol: str, timeframe: str, selected_features: list):
    """Update meta.json with real feature names."""
    base_dir = Path(f"models/advanced/{symbol}/{timeframe}")
    
    print(f"\n📦 Updating {symbol} {timeframe}...")
    
    # Load production_training_results.json for model_config
    results_path = base_dir / "production_training_results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    # Create new meta.json with real feature names
    meta = {
        "sequence_length": results["config"]["sequence_length"],
        "selected_features": selected_features,
        "model_config": {
            'hidden_dim': results["model_config"]["hidden_dim"],
            'lstm_layers': results["model_config"]["lstm_layers"],
            'dense_dims': results["model_config"]["dense_dims"],
            'dropout': results["model_config"]["dropout"],
            'use_attention': results["model_config"]["use_attention"],
            'bidirectional': results["model_config"]["bidirectional"],
            'num_classes': results["model_config"]["num_classes"],
            'use_regression': results["model_config"]["use_regression"]
        },
        "ensemble_size": 1,
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_count": len(selected_features),
        "created_at": datetime.now().isoformat(),
        "version": "2.0_real_features"
    }
    
    # Backup old meta.json
    meta_path = base_dir / "meta.json"
    if meta_path.exists():
        backup_path = base_dir / f"meta.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        meta_path.rename(backup_path)
        print(f"   📦 Backed up old meta.json")
    
    # Save new meta.json
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"   ✅ Updated meta.json with {len(selected_features)} real features")
    return True


def main():
    print("=" * 80)
    print("PHASE 2: Updating meta.json with Real Feature Names")
    print("=" * 80)
    
    # Load extracted features from Phase 1
    extracted_path = Path("models/advanced/extracted_features.json")
    if not extracted_path.exists():
        print("❌ extracted_features.json not found. Run Phase 1 first!")
        return False
    
    with open(extracted_path) as f:
        data = json.load(f)
    
    models_data = data.get("models", {})
    
    # Update each model
    success_count = 0
    for key, features in models_data.items():
        symbol, timeframe = key.rsplit('_', 1)
        if update_meta_json(symbol, timeframe, features):
            success_count += 1
    
    print(f"\n{'=' * 80}")
    print(f"✅ Updated {success_count}/{len(models_data)} meta.json files")
    print(f"{'=' * 80}")
    
    if success_count == len(models_data):
        print("\n🚀 Ready for Phase 3: Improving Predictor Logic")
        return True
    else:
        print("\n⚠️  Some updates failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
