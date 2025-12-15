import joblib
import numpy as np
from pathlib import Path

model_path = Path("models/advanced/ETHUSDT/1h")
scaler_path = model_path / "scaler.pkl"
selector_path = model_path / "feature_selector.pkl"

print(f"Inspecting artifacts in {model_path}")

if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'mean_'):
        print(f"Scaler expected features: {scaler.mean_.shape[0]}")
    else:
        print("Scaler has no mean_ attribute")

if selector_path.exists():
    selector = joblib.load(selector_path)
    print(f"Selector type: {type(selector)}")
    if hasattr(selector, 'n_features_in_'):
        print(f"Selector input features: {selector.n_features_in_}")
    if hasattr(selector, 'n_features_to_select'):
        print(f"Selector output features: {selector.n_features_to_select}")
    if hasattr(selector, 'get_support'):
        print(f"Selector support sum: {sum(selector.get_support())}")
