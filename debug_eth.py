import torch
import numpy as np
import json
from pathlib import Path
from ml.advanced_models.predictor import AdvancedPredictor
from ml.advanced_models.dataset import load_sequence_dataset, AdvancedDatasetConfig
import sys

# Force ROCm settings
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def debug_predictions():
    symbol = "ETHUSDT"
    timeframe = "15m"
    model_path = Path(f"models/advanced/{symbol}/{timeframe}")
    
    print(f"Debugging {symbol} {timeframe}...")
    
    # Load predictor
    predictor = AdvancedPredictor(
        model_path=model_path,
        scaler_path=model_path / "scaler.pkl",
        meta_path=model_path / "meta.json"
    )
    
    # Load data
    meta = json.loads((model_path / "meta.json").read_text())
    config = AdvancedDatasetConfig(
        symbol=meta['symbol'],
        timeframe=meta['timeframe'],
        sequence_length=meta['sequence_length'],
        prediction_horizon=meta['prediction_horizon'],
        target_return=meta['target_return'],
        max_history_days=1000,
    )
    
    features, _, _, _ = load_sequence_dataset(config)
    
    # Use last 15%
    n_samples = len(features)
    test_start = int(n_samples * 0.85)
    X_test = features[test_start:]
    
    print(f"Loaded {len(X_test)} test samples")
    
    # Predict a batch
    batch_size = 100
    window_size = config.sequence_length
    
    max_probs = []
    all_probs = []
    
    print("Running inference on first 100 samples...")
    
    for i in range(window_size, window_size + batch_size):
        window = X_test[i-window_size:i]
        
        # Manual pipeline simulation (like in optimize_threshold)
        probs_sum = np.zeros(3)
        
        for pipeline in predictor.ensemble_pipelines:
            model = pipeline['model']
            selector = pipeline['selector']
            scaler = pipeline['scaler']
            
            window_input = window
            if selector:
                try:
                    window_input = selector.transform(window)
                except:
                    if hasattr(selector, 'support_'):
                        window_input = window[:, selector.support_]
            
            if scaler:
                window_input = scaler.transform(window_input)
                
            window_tensor = torch.FloatTensor(window_input).unsqueeze(0).to(predictor.device)
            
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(window_tensor).cpu().numpy()[0]
                else:
                    outputs = model(window_tensor)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            probs_sum += probs
            
        avg_probs = probs_sum / len(predictor.ensemble_pipelines)
        all_probs.append(avg_probs)
        max_probs.append(np.max(avg_probs))
        
    all_probs = np.array(all_probs)
    max_probs = np.array(max_probs)
    
    print("\n--- Statistics ---")
    print(f"Max Probability observed: {np.max(max_probs):.4f}")
    print(f"Mean Max Probability: {np.mean(max_probs):.4f}")
    print(f"Min Max Probability: {np.min(max_probs):.4f}")
    
    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"Sample {i}: {all_probs[i]}")

if __name__ == "__main__":
    debug_predictions()
