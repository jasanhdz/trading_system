
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append('/home/jasan/Develop/trading_system')
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv

def test_gpu_env():
    print("Testing Env on GPU...")
    num_envs = 64
    n_candles = 1000
    features = np.random.randn(n_candles, 18).astype(np.float32)
    close_prices = np.random.uniform(2000, 3000, n_candles).astype(np.float32)
    
    env = PhantomMatrixEnv(features, close_prices, num_envs)
    obs = env.reset()
    
    for i in range(10):
        actions = np.random.randint(0, 4, num_envs)
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step {i} successful. Avg Reward: {np.mean(rewards):.4f}")
    
    print("Test passed!")

if __name__ == "__main__":
    test_gpu_env()
