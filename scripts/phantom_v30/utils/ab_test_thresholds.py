import os
import sys
from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.train_v30 import TransformerExtractor # Required for model loading

MODEL_PATH = "models/phantom_v30_champion_OVERFITTED.zip"

def evaluate_with_threshold(model, eval_env, threshold: float):
    obs = eval_env.reset()
    done = False
    
    total_trades = 0
    
    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Get probabilities
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        if hasattr(model.policy, "get_distribution"):
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            
            act_val = action[0].item() if isinstance(action[0], torch.Tensor) else int(action[0])
            confidence = probs[act_val]
            
            # Solo vetar entradas reales (LONG=1, SHORT=2)
            if act_val in [1, 2] and confidence < threshold:
                # Override action to IDLE (0)
                action[0] = 0
                
        # Step
        obs, rewards, dones, infos = eval_env.step(action)
        done = dones[0]
        
    final_info = infos[0]
    final_balance = final_info.get('balance', 0)
    
    # Calculate PnL
    pnl = final_balance - 20.0
    return final_balance, pnl

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
        
    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device="cpu")
    
    print("Loading Validation Data...")
    eval_data = load_tensor_data("cpu", split="val")
    features_np = eval_data['features'].numpy()
    close_np = eval_data['close'].numpy()
    
    print("\n--- A/B TEST: Entry Thresholds (Validation Set) ---")
    thresholds_to_test = [0.0, 0.35, 0.55, 0.65, 0.75]
    
    results = []
    
    for t in thresholds_to_test:
        print(f"\nEvaluating Threshold: {t:.2f} ...")
        
        eval_env = PhantomMatrixEnv(
            features=features_np,
            close_prices=close_np,
            num_envs=1
        )
        
        final_balance, pnl = evaluate_with_threshold(model, eval_env, t)
        
        print(f"Result for Threshold {t:.2f}:")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  Net PnL:       ${pnl:.2f}")
        print(f"  ROI:           {(pnl / 20.0) * 100:.2f}%")
        
        results.append((t, final_balance, pnl))
        
    print("\n--- SUMMARY ---")
    for r in results:
        print(f"Threshold {r[0]:.2f} -> Balance: ${r[1]:.2f} | PnL: ${r[2]:.2f}")

if __name__ == "__main__":
    main()
