#!/usr/bin/env python3
"""Find duplicate features."""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.nn_pattern.features import build_feature_frame

# Load model metadata
meta_path = REPO_ROOT / "models" / "advanced" / "ETHUSDT" / "15m" / "meta.json"
meta = json.loads(meta_path.read_text())
expected_features = meta['selected_features']

print("Checking for duplicates in expected_features list:")
print()

counter = Counter(expected_features)
duplicates = {feat: count for feat, count in counter.items() if count > 1}

if duplicates:
    print(f"❌ Found {len(duplicates)} duplicate features:")
    for feat, count in duplicates.items():
        print(f"   - {feat}: appears {count} times")
    print()
    print(f"Total unique features: {len(set(expected_features))}")
    print(f"Total in list: {len(expected_features)}")
else:
    print("✅ No duplicates found in expected_features")
    
print()
print("Now checking generated features...")

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

# Check for duplicates in generated features
gen_counter = Counter(feature_frame.columns)
gen_duplicates = {feat: count for feat, count in gen_counter.items() if count > 1}

if gen_duplicates:
    print(f"❌ Found {len(gen_duplicates)} duplicate columns in generated features:")
    for feat, count in gen_duplicates.items():
        print(f"   - {feat}: appears {count} times")
else:
    print("✅ No duplicate columns in generated features")
