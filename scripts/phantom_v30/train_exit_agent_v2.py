#!/usr/bin/env python3
"""
Phantom Exit Agent V2 Trainer — "Ride the Wave"
================================================
Trains a PPO agent to master momentum-based exits.

Features:
  - 12D observation space (momentum, acceleration, drawdown, CVD, volume)
  - Synchronized with Phase 1 live brackets (150% TP, -50% SL, 8h max)
  - CPU training (runs alongside GPU V30 champion training)
  - Coliseum evaluation with multi-seed validation
  - Auto-crowning with survival filter

Usage:
  python scripts/phantom_v30/train_exit_agent_v2.py
"""
import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.phantom_v30.exit_tensor_loader_v2 import load_exit_tensors_v2
from scripts.phantom_v30.matrix_exit_env_v2 import MatrixExitEnvV2

# === Config ===
CHAMPION_PATH = "models/phantom_exit_champion_v2.zip"
CHALLENGER_PATH = "models/phantom_exit_challenger_v2.zip"
TOTAL_TIMESTEPS = 500_000       # 500k steps per iteration
NUM_ENVS = 8                    # 8 parallel CPU environments
N_ITERATIONS = 10               # Total training iterations
EVAL_EPISODES = 200             # Evaluation episodes per seed
EVAL_SEEDS = [42, 137, 256, 1337, 7777]


class ProgressCallback(BaseCallback):
    """Print progress every N steps."""
    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        
    def _on_step(self):
        if self.num_timesteps % 25_000 == 0:
            elapsed = time.time() - self.start_time
            pct = (self.num_timesteps / self.total_steps) * 100
            rate = self.num_timesteps / max(elapsed, 1)
            eta = (self.total_steps - self.num_timesteps) / max(rate, 1)
            print(f"   📊 {self.num_timesteps:,}/{self.total_steps:,} ({pct:.0f}%) | "
                  f"{rate:.0f} steps/s | ETA: {eta:.0f}s")
            
            # Heartbeat for watchdog
            try:
                Path("logs/exit_training_v2.log").touch()
            except Exception:
                pass
        return True


def evaluate_model(model_path: str, tensors, n_episodes: int = EVAL_EPISODES) -> dict:
    """
    Multi-seed evaluation of exit model.
    Returns dict with avg_pnl, win_rate, avg_duration, action_distribution.
    """
    if not os.path.exists(model_path):
        return {'avg_pnl': -np.inf, 'win_rate': 0, 'details': 'not_found'}
    
    env = MatrixExitEnvV2(tensors)
    
    try:
        model = PPO.load(model_path, env=DummyVecEnv([lambda: env]), device="cpu")
    except Exception as e:
        print(f"  ⚠️ Failed to load {model_path}: {e}")
        return {'avg_pnl': -np.inf, 'win_rate': 0, 'details': str(e)}
    
    all_pnls = []
    all_durations = []
    all_reasons = []
    action_counts = {0: 0, 1: 0}
    
    for seed in EVAL_SEEDS:
        np.random.seed(seed)
        
        for ep in range(n_episodes // len(EVAL_SEEDS)):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                action_counts[action] = action_counts.get(action, 0) + 1
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                steps += 1
            
            all_pnls.append(total_reward)
            all_durations.append(steps)
            all_reasons.append(info.get('reason', 'unknown'))
    
    avg_pnl = np.mean(all_pnls)
    win_rate = np.mean([1 if p > 0 else 0 for p in all_pnls])
    avg_duration = np.mean(all_durations)
    
    # Reason distribution
    reason_counts = {}
    for r in all_reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    
    total_actions = sum(action_counts.values())
    close_rate = action_counts.get(1, 0) / max(total_actions, 1) * 100
    
    return {
        'avg_pnl': float(avg_pnl),
        'median_pnl': float(np.median(all_pnls)),
        'win_rate': float(win_rate),
        'avg_duration': float(avg_duration),
        'close_rate': close_rate,
        'p25_pnl': float(np.percentile(all_pnls, 25)),
        'p75_pnl': float(np.percentile(all_pnls, 75)),
        'reasons': reason_counts,
        'n_episodes': len(all_pnls),
    }


def print_eval_report(name: str, results: dict):
    """Pretty-print evaluation results."""
    print(f"\n  {'='*50}")
    print(f"  📊 {name}")
    print(f"  {'='*50}")
    print(f"  Avg PnL:      {results['avg_pnl']:.4f} ROE")
    print(f"  Median PnL:   {results.get('median_pnl', 0):.4f} ROE")
    print(f"  P25/P75:      {results.get('p25_pnl', 0):.4f} / {results.get('p75_pnl', 0):.4f}")
    print(f"  Win Rate:     {results['win_rate']*100:.1f}%")
    print(f"  Avg Duration: {results.get('avg_duration', 0):.1f} candles ({results.get('avg_duration', 0)*5:.0f} min)")
    print(f"  AI Close %:   {results.get('close_rate', 0):.1f}%")
    print(f"  Exit Reasons: {results.get('reasons', {})}")
    print(f"  Episodes:     {results.get('n_episodes', 0)}")


def train_iteration(tensors_train, tensors_val, iteration: int):
    """Run one training + evaluation iteration."""
    print(f"\n{'='*60}")
    print(f"🔄 Exit Agent V2 — Training Iteration {iteration}")
    print(f"{'='*60}")
    
    # === TRAINING ===
    def make_env():
        return MatrixExitEnvV2(tensors_train)
    
    print(f"\n⚔️ Spawning {NUM_ENVS} parallel CPU environments...")
    env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    
    # Load existing model or create new
    if os.path.exists(CHAMPION_PATH):
        print(f"  📂 Loading champion for continued training...")
        model = PPO.load(CHAMPION_PATH, env=env, device="cpu")
        model.learning_rate = 3e-5  # Lower LR for fine-tuning
    else:
        print(f"  🆕 Creating fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,      # More exploration for V2 (12D space is larger)
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            verbose=0,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 64], vf=[128, 64])  # Larger network for 12D
            )
        )
    
    print(f"🚀 Training {TOTAL_TIMESTEPS:,} steps on CPU...")
    start = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=ProgressCallback(TOTAL_TIMESTEPS),
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    
    elapsed = time.time() - start
    print(f"⏱️ Training completed in {elapsed:.0f}s ({TOTAL_TIMESTEPS/elapsed:.0f} steps/s)")
    
    # Save challenger
    os.makedirs("models", exist_ok=True)
    model.save(CHALLENGER_PATH)
    print(f"💾 Saved challenger: {CHALLENGER_PATH}")
    
    env.close()
    
    # === EVALUATION (Validation Set) ===
    print(f"\n🏟️ COLISEUM: Evaluating on validation data...")
    
    champ_results = evaluate_model(CHAMPION_PATH, tensors_val)
    print_eval_report("Champion (Current)", champ_results)
    
    challenger_results = evaluate_model(CHALLENGER_PATH, tensors_val)
    print_eval_report("Challenger (New)", challenger_results)
    
    # === PROMOTION DECISION ===
    champ_score = champ_results['avg_pnl']
    challenger_score = challenger_results['avg_pnl']
    challenger_wr = challenger_results['win_rate']
    
    print(f"\n🏆 Champion:    {champ_score:.4f} avg PnL")
    print(f"⚔️  Challenger:  {challenger_score:.4f} avg PnL | {challenger_wr*100:.1f}% WR")
    
    if champ_score <= -np.inf or champ_score <= -100:
        # No valid champion
        if challenger_score > -0.5 and challenger_wr > 0.3:
            print(f"🆕 No valid champion. Crowning challenger!")
            shutil.copy2(CHALLENGER_PATH, CHAMPION_PATH)
            print(f"✅ First Exit V2 Champion crowned!")
        else:
            print(f"⚠️ Challenger too weak (PnL: {challenger_score:.4f}, WR: {challenger_wr*100:.1f}%). Keeping as-is.")
    elif challenger_score > champ_score and challenger_wr >= 0.35:
        improvement = ((challenger_score - champ_score) / max(abs(champ_score), 0.01)) * 100
        print(f"🚀 PROMOTION! Challenger wins by {improvement:.1f}% improvement!")
        
        # Backup old champion
        backup = f"{CHAMPION_PATH}.backup_{int(time.time())}"
        if os.path.exists(CHAMPION_PATH):
            os.rename(CHAMPION_PATH, backup)
        shutil.copy2(CHALLENGER_PATH, CHAMPION_PATH)
        print(f"✅ New Exit V2 Champion crowned! (Old backed up)")
    else:
        print(f"🛡️ Champion holds title.")
    
    # Cleanup
    if os.path.exists(CHALLENGER_PATH):
        os.remove(CHALLENGER_PATH)
    
    return challenger_results


def main():
    print("=" * 60)
    print("🧠 Phantom Exit Agent V2 — 'Ride the Wave' Trainer")
    print("=" * 60)
    print(f"   Features:    12D (momentum, acceleration, CVD, volume)")
    print(f"   Brackets:    SL -50% ROE | TP 150% ROE | Trail 30%/15%")
    print(f"   Max Duration: 8 hours (96 candles)")
    print(f"   Steps/iter:  {TOTAL_TIMESTEPS:,}")
    print(f"   Envs:        {NUM_ENVS} (CPU)")
    print(f"   Iterations:  {N_ITERATIONS}")
    print()
    
    # Fetch fresh data
    print("📡 Fetching freshest market data...")
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/update_ml_candles.py"], check=True, timeout=120)
        print("✅ Data sync complete.")
    except Exception as e:
        print(f"⚠️ Data sync failed ({e}). Using cached data.")
    
    # Load data
    tensors_train = load_exit_tensors_v2(device="cpu", split="train")
    tensors_val = load_exit_tensors_v2(device="cpu", split="val")
    
    print(f"\n📊 Train: {tensors_train['n_candles']:,} candles")
    print(f"📊 Val:   {tensors_val['n_candles']:,} candles")
    
    # Training loop
    for iteration in range(1, N_ITERATIONS + 1):
        try:
            results = train_iteration(tensors_train, tensors_val, iteration)
            
            # Heartbeat
            try:
                Path("logs/exit_training_v2.log").touch()
            except Exception:
                pass
            
            if iteration < N_ITERATIONS:
                print(f"\n⏸️ Sleeping 5s before iteration {iteration + 1}...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Training stopped by user at iteration {iteration}")
            break
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)
            continue
    
    print(f"\n{'='*60}")
    print(f"🏁 Exit Agent V2 Training Complete!")
    print(f"   Champion: {CHAMPION_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
