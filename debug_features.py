#!/usr/bin/env python3
"""Debug feature mismatch between model and generated features."""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.nn_pattern.features import build_feature_frame, ALL_FEATURES

# Load model metadata
meta_path = REPO_ROOT / "models" / "advanced" / "ETHUSDT" / "15m" / "meta.json"
meta = json.loads(meta_path.read_text())

print("=" * 80)
print("FEATURE MISMATCH ANALYSIS")
print("=" * 80)
print()

expected_features = meta['selected_features']
print(f"Model expects: {len(expected_features)} features")
print(f"Code generates: {len(ALL_FEATURES)} features")
print()

# Generate sample features to see what's produced
# Create fake OHLCV data
dates = pd.date_range('2024-01-01', periods=200, freq='15min')
df = pd.DataFrame({
    'open': np.random.rand(200) * 100 + 2000,
    'high': np.random.rand(200) * 100 + 2050,
    'low': np.random.rand(200) * 100 + 1950,
    'close': np.random.rand(200) * 100 + 2000,
    'volume': np.random.rand(200) * 1000000,
}, index=dates)

feature_frame, feature_names = build_feature_frame(df)
print(f"Actually generated: {len(feature_frame.columns)} features")
print()

# Find differences
expected_set = set(expected_features)
generated_set = set(feature_frame.columns)

missing_in_generated = expected_set - generated_set
extra_in_generated = generated_set - expected_set

if missing_in_generated:
    print(f"❌ Missing in generated features ({len(missing_in_generated)}):")
    for f in sorted(missing_in_generated):
        print(f"   - {f}")
    print()

if extra_in_generated:
    print(f"⚠️  Extra features not in model ({len(extra_in_generated)}):")
    for f in sorted(extra_in_generated):
        print(f"   - {f}")
    print()

if not missing_in_generated and not extra_in_generated:
    print("✅ Perfect match! All features align.")
else:
    print()
    print("Suggested fix:")
    print("  The predictor should select only the expected features from the generated set.")
    print("  This is already done in predictor.py line 152, but there might be an issue.")
