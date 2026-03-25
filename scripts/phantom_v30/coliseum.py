import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.data_loader import load_hybrid_data
from scripts.phantom_v30.train_v30 import TransformerExtractor # Keep import for loading

MODEL_DIR = "models"
CHAMPION_PATH = f"{MODEL_DIR}/phantom_v30_champion.zip"
CHALLENGER_PATH = f"{MODEL_DIR}/phantom_v30_challenger.zip" # Hypothetical

def evaluate_model(model_path, eval_env, n_episodes=1):
    """
    Evaluates a model on the Matrix Environment.
    Returns: Mean Reward, Final Balance, Maximum Drawdown
    """
    if not os.path.exists(model_path):
        return -np.inf, 0, {0:0, 1:0, 2:0, 3:0}
        
    try:
        model = PPO.load(model_path, env=eval_env, device="cpu")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return -np.inf, 0, {0:0, 1:0, 2:0, 3:0}
    
    action_counts = {0:0, 1:0, 2:0, 3:0} # Idle, Long, Short, Close
    
    obs = eval_env.reset()
    done = np.zeros(eval_env.num_envs, dtype=bool)
    max_steps = 1500
    steps = 0
    total_reward = 0.0
    
    while not done.all() and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        for a in action:
            action_counts[int(a)] += 1
            
        obs, reward, d, infos = eval_env.step(action)
        done = done | d
        total_reward += np.sum(reward)
        steps += 1
        
    final_balance = np.mean([info['balance'] for info in infos])
    
    return total_reward / steps, final_balance, action_counts

def main():
    print("🏟️ Welcome to The Coliseum 🏟️")
    print("Loading Verification Data (Last 20% of data or specific set)...")
    # Ideally load OOS data. For now, we load same data but maybe different slice?
    # Let's assume we use the last chunk of hybrid data as "Validation Set"
    full_df = load_hybrid_data()
    # ⚔️ KAMIKAZE MODE: Evaluate only on the last 60 days (60 * 288 = 17280 candles)
    if len(full_df) > 17280:
        full_df = full_df.iloc[-17280:].reset_index(drop=True)
        
    val_split = int(len(full_df) * 0.8)
    val_df = full_df.iloc[val_split:].reset_index(drop=True)
    # OPTIMIZATION: Slice to last 5000 for speed
    if len(val_df) > 5000:
        val_df = val_df.iloc[-5000:].reset_index(drop=True)
    
    print(f"Validation Set: {len(val_df)} candles.")
    
    # Needs to match standard numpy extraction for Matrix Env
    df = val_df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['high_norm'] = np.log(df['high'] / df['close']).fillna(0)
    df['low_norm'] = np.log(df['low'] / df['close']).fillna(0)
    vol_ma = df['volume'].rolling(window=24).mean()
    df['vol_norm'] = (df['volume'] / (vol_ma + 1e-8)).fillna(0).clip(0, 10)
    
    feats = df[['log_ret', 'high_norm', 'low_norm', 'vol_norm']].values.astype(np.float32)
    prices = df['close'].values.astype(np.float32)
    
    env = PhantomMatrixEnv(features=feats, close_prices=prices, num_envs=8)
    
    # 1. Evaluate Champion
    print(f"Evaluating Champion: {CHAMPION_PATH}")
    champ_reward, champ_balance, champ_actions = evaluate_model(CHAMPION_PATH, env)
    print(f"🏆 Champion Score: Reward={champ_reward:.4f}, Balance=${champ_balance:.2f}")
    print(f"   Actions: Idle={champ_actions[0]}, Long={champ_actions[1]}, Short={champ_actions[2]}, Close={champ_actions[3]}")
    
    # 2. Evaluate Challenger (Dummy logic for now)
    # In a real loop, we would train a challenger here.
    # For now, let's just pretend we have one or re-eval champion as challenger to test logic
    print(f"Evaluating Challenger...")
    try:
        chall_reward, chall_balance, chall_actions = evaluate_model(CHALLENGER_PATH, env) 
    except ValueError:
        # Fallback if Challenger model is old structure or fails
        print("⚠️ Challenger Model incompatible or failed.")
        return

    print(f"⚔️ Challenger Score: Reward={chall_reward:.4f}, Balance=${chall_balance:.2f}")
    
    # 3. Decision
    
    # 3. Decision
    if chall_balance > champ_balance: # Simple Profit Logic
        print("🚀 PROMOTION! Challenger defeats Champion.")
        # os.rename(CHALLENGER_PATH, CHAMPION_PATH) # Dangerous in demo
    else:
        print("🛡️ DEFENSE! Champion retains title.")

if __name__ == "__main__":
    main()
