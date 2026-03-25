#!/usr/bin/env python3
"""
PASO 1: Behavioral Cloning — Generate Teacher Dataset
Runs the Champion V8 model over all training data and records every
(observation, action) pair. This creates the "teacher" dataset for
supervised pre-training.

Output: data/bc_teacher_dataset.npz
"""
import sys
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv

CHAMPION_PATH = "models/phantom_v30_champion.zip"
OUTPUT_PATH = "data/bc_teacher_dataset.npz"
NUM_ENVS = 64  # Small batch for deterministic rollout, but enough for speed
MAX_STEPS_PER_EPISODE = 5000
NUM_PASSES = 2  # Multiple passes to cover different starting points


def generate_teacher_dataset():
    print("=" * 60)
    print("🧑‍🏫 PASO 1: Generating Teacher Dataset from Champion V8")
    print("=" * 60)

    # Load training data
    data = load_tensor_data("cpu", days=None, split="train")
    features = data['features'].numpy()
    close_prices = data['close'].numpy()

    # Load Champion V8
    env = PhantomMatrixEnv(features=features, close_prices=close_prices, num_envs=NUM_ENVS)
    model = PPO.load(CHAMPION_PATH, env=env, device="cpu")
    print(f"✅ Champion V8 loaded from {CHAMPION_PATH}")

    all_market_obs = []
    all_account_obs = []
    all_actions = []

    total_samples = 0

    for pass_idx in range(NUM_PASSES):
        print(f"\n📡 Pass {pass_idx + 1}/{NUM_PASSES}...")
        obs = env.reset()
        done_all = np.zeros(NUM_ENVS, dtype=bool)
        steps = 0

        while not done_all.all() and steps < MAX_STEPS_PER_EPISODE:
            # Get Champion's action (deterministic = what it WOULD do)
            action, _ = model.predict(obs, deterministic=True)

            # Record observation + action pair
            all_market_obs.append(obs['market'].copy())      # (num_envs, 64, 11)
            all_account_obs.append(obs['account'].copy())     # (num_envs, 4)
            all_actions.append(action.copy())                 # (num_envs,)

            total_samples += NUM_ENVS

            # Step environment
            obs, reward, done, infos = env.step(action)
            done_all = done_all | done
            steps += 1

        print(f"   ✅ Pass {pass_idx + 1} done: {steps} steps, {total_samples:,} total samples")

    # Stack into single arrays
    market_dataset = np.concatenate(all_market_obs, axis=0)    # (N, 64, 11)
    account_dataset = np.concatenate(all_account_obs, axis=0)  # (N, 4)
    action_dataset = np.concatenate(all_actions, axis=0)       # (N,)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(action_dataset):,}")
    print(f"   Market shape: {market_dataset.shape}")
    print(f"   Account shape: {account_dataset.shape}")

    # Action distribution
    unique, counts = np.unique(action_dataset, return_counts=True)
    print(f"   Action distribution:")
    action_names = {0: 'Idle', 1: 'Long', 2: 'Short', 3: 'Close'}
    for a, c in zip(unique, counts):
        pct = c / len(action_dataset) * 100
        print(f"     {action_names.get(a, a)}: {c:,} ({pct:.1f}%)")

    # Save dataset
    np.savez(
        OUTPUT_PATH,
        market=market_dataset,
        account=account_dataset,
        actions=action_dataset,
    )
    print(f"\n💾 Saved teacher dataset → {OUTPUT_PATH}")
    print(f"   File size: {Path(OUTPUT_PATH).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    generate_teacher_dataset()
