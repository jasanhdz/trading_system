#!/usr/bin/env python3
"""Deep debug of the exact failure point."""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.nn_pattern.features import build_feature_frame

# Load model metadata
meta_path = REPO_ROOT / "models" / "advanced" / "ETHUSDT" / "15m" / "meta.json"
meta = json.loads(meta_path.read_text())
expected_features = meta['selected_features']

print("Expected features from model:")
for i, f in enumerate(expected_features, 1):
    print(f"  {i:2d}. {f}")
print()

# Generate sample features
dates = pd.date_range('2024-01-01', periods=200, freq='15min')
df = pd.DataFrame({
    'open': np.random.rand(200) * 100 + 2000,
    'high': np.random.rand(200) * 100 + 2050,
    'low': np.random.rand(200) * 100 + 1950,
    'close': np.random.rand(200) * 100 + 2000,
    'volume': np.random.rand(200) * 1000000,
}, index=dates)

feature_frame, _ = build_feature_frame(df)

print(f"Generated {len(feature_frame.columns)} features total")
print(f"Model expects {len(expected_features)} features")
print()

# Try to select expected features
try:
    selected = feature_frame[expected_features]
    print(f"✅ Successfully selected {len(selected.columns)} features")
    print(f"   Shape: {selected.shape}")
except KeyError as e:
    print(f"❌ Failed to select expected features!")
    print(f"   Error: {e}")
    
    # Find which features are missing
    missing = set(expected_features) - set(feature_frame.columns)
    if missing:
        print(f"\n   Missing features ({len(missing)}):")
        for f in sorted(missing):
            print(f"      - {f}")
