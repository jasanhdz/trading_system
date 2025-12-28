import xgboost as xgb
import joblib
import json
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models/advanced")
SYMBOL = "AVAXUSDT"
XGB_PATH = MODELS_DIR / f"{SYMBOL}_xgb.joblib"
FEATURES_PATH = MODELS_DIR / f"{SYMBOL}_features.json"

def check_importance():
    if not XGB_PATH.exists():
        print(f"Model not found at {XGB_PATH}")
        return

    # Load Model
    model = joblib.load(XGB_PATH)
    
    # Load Feature Names
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, 'r') as f:
            feature_names = json.load(f)
    else:
        print("Features file not found, using generic names")
        feature_names = [f"f{i}" for i in range(100)] # Fallback

    # Get Importance
    # XGBoost sklearn API
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        # Map to names
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        
        print(f"--- Feature Importance for {SYMBOL} ---")
        print(feat_imp)
    else:
        # Booster object
        scores = model.get_score(importance_type='gain')
        # Map f0, f1... to names if keys are generic
        # If keys are 'bid_depth', we are good.
        print("--- Feature Importance (Gain) ---")
        # Sort dict
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_scores:
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    check_importance()
