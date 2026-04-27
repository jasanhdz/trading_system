import os
import sys
from pathlib import Path
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.phantom_v30.exit_tensor_loader import load_exit_tensors
from scripts.phantom_v30.matrix_exit_env import MatrixExitEnv

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        # Print info every 10240 steps to show it's alive
        if self.num_timesteps % 10240 == 0:
            print(f"🔄 Checkpoint: {self.num_timesteps} steps processed...")
        return True

def train():
    print("🧠 Initiating Phantom Exit Agent Training (CPU Mode)")
    print("   Synchronized with Champion data split (4yr train / 14d val / 200 gap)")
    
    # Backup old model if exists
    old_model = "models/phantom_exit_champion.zip"
    if os.path.exists(old_model):
        import shutil
        backup = "models/phantom_exit_champion_OVERFITTED.zip"
        if not os.path.exists(backup):
            shutil.move(old_model, backup)
            print(f"   📦 Backed up old model → {backup}")
    
    # Load data with SAME split as Champion trainer
    tensors = load_exit_tensors(device="cpu", split="train")
    
    # Define environment factory
    def make_env():
        return MatrixExitEnv(tensors, leverage=20, max_steps=24, fee=0.0004)
        
    # Create vectorized environment
    num_envs = 4
    print(f"⚔️ Spawning {num_envs} Parallel CPU Trade Environments...")
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    
    # Explicitly force PyTorch to use CPU to avoid conflict with GPU V30 trainers
    device = torch.device('cpu')
    
    # Define CPU PPO Model — Kamikaze Exit Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,    # Lower LR for finer exit decisions
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,         # Less exploration, more exploitation of CVD signals
        vf_coef=0.5,
        device=device,
        verbose=1
    )
    
    print("🚀 Starting CPU Training Loop...")
    
    try:
        model.learn(total_timesteps=1_000_000, callback=ProgressCallback())
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User")
        
    # Save the model
    save_path = "models/phantom_exit_champion.zip"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"🏆 Saved Exit Champion Model: {save_path}")

if __name__ == "__main__":
    train()
