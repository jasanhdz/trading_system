#!/usr/bin/env python3
"""
PASO 2: Behavioral Cloning — Supervised Pre-training
Trains a fresh PPO policy network to IMITATE the Champion V8's actions using
CrossEntropyLoss (supervised classification). This gives the new model a
"muscle memory" of how good trading looks BEFORE we let RL improve it.

Input:  data/bc_teacher_dataset.npz
Output: models/phantom_v30_bc_pretrained.zip (SB3-compatible PPO model)
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import BaseBuffer

# MONKEY-PATCH for ROCm gfx1032
def _safe_to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
    if copy:
        return torch.tensor(array).to(self.device)
    return torch.as_tensor(array).to(self.device)
BaseBuffer.to_torch = _safe_to_torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.train_v30 import TransformerExtractor

DATASET_PATH = "data/bc_teacher_dataset.npz"
OUTPUT_PATH = "models/phantom_v30_bc_pretrained.zip"

# Training hyperparameters
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 1
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
NUM_EPOCHS = 12
DEVICE = "cuda:0"


def pretrain_behavioral_cloning():
    print("=" * 60)
    print("🧠 PASO 2: Behavioral Cloning — Supervised Pre-training")
    print("=" * 60)

    # === Load teacher dataset ===
    print("\n📦 Loading teacher dataset...")
    dataset = np.load(DATASET_PATH)
    market_data = torch.tensor(dataset['market'], dtype=torch.float32)
    account_data = torch.tensor(dataset['account'], dtype=torch.float32)
    action_labels = torch.tensor(dataset['actions'], dtype=torch.long)

    n_samples = len(action_labels)
    print(f"   Samples: {n_samples:,}")
    print(f"   Market shape: {market_data.shape}")
    print(f"   Account shape: {account_data.shape}")

    # Action distribution
    unique, counts = np.unique(dataset['actions'], return_counts=True)
    action_names = {0: 'Idle', 1: 'Long', 2: 'Short', 3: 'Close'}
    for a, c in zip(unique, counts):
        print(f"   {action_names.get(a, a)}: {c:,} ({c/n_samples*100:.1f}%)")

    # === Create class weights for imbalanced dataset ===
    # Champion V8 does Idle ~93% of the time, so we weight trading actions more
    total = len(action_labels)
    class_counts = torch.bincount(action_labels, minlength=4).float()
    class_weights = total / (4.0 * class_counts + 1e-10)
    class_weights = class_weights.to(DEVICE)
    print(f"\n   Class weights: {class_weights.cpu().numpy()}")

    # === Train/val split (90/10) ===
    perm = torch.randperm(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = perm[:split], perm[split:]

    train_dataset = TensorDataset(
        market_data[train_idx],
        account_data[train_idx],
        action_labels[train_idx]
    )
    val_dataset = TensorDataset(
        market_data[val_idx],
        account_data[val_idx],
        action_labels[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"\n   Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # === Create fresh PPO model (to get the network architecture) ===
    print("\n🏗️ Creating fresh PPO model with 32D architecture...")
    data = load_tensor_data("cpu", days=None, split="train")
    dummy_env = PhantomMatrixEnv(
        features=data['features'].numpy(),
        close_prices=data['close'].numpy(),
        num_envs=4,
    )

    POLICY_KWARGS = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS, dropout=0.1),
        net_arch=dict(pi=[D_MODEL, D_MODEL], vf=[D_MODEL, D_MODEL])
    )

    ppo_model = PPO(
        "MultiInputPolicy",
        dummy_env,
        policy_kwargs=POLICY_KWARGS,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        seed=42,
        device=DEVICE,
    )

    # === Extract the policy network for supervised training ===
    policy = ppo_model.policy
    policy.train()

    # We'll train the features_extractor + action_net (pi head) together
    # The value head will be fine-tuned during RL phase

    # Build a simple forward function that outputs action logits
    def get_action_logits(market_batch, account_batch):
        """Forward pass through features extractor + action head."""
        obs = {
            'market': market_batch,
            'account': account_batch,
        }
        features = policy.extract_features(obs, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        action_logits = policy.action_net(latent_pi)
        return action_logits

    # === Training loop ===
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    print(f"\n🏋️ Training for {NUM_EPOCHS} epochs...")
    print(f"   Device: {DEVICE}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for market_b, account_b, labels_b in train_loader:
            market_b = market_b.to(DEVICE)
            account_b = account_b.to(DEVICE)
            labels_b = labels_b.to(DEVICE)

            logits = get_action_logits(market_b, account_b)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels_b.size(0)
            train_correct += (logits.argmax(dim=1) == labels_b).sum().item()
            train_total += labels_b.size(0)

        scheduler.step()
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total * 100

        # --- Validate ---
        policy.eval()
        val_correct = 0
        val_total = 0
        val_action_counts = torch.zeros(4)

        with torch.no_grad():
            for market_b, account_b, labels_b in val_loader:
                market_b = market_b.to(DEVICE)
                account_b = account_b.to(DEVICE)
                labels_b = labels_b.to(DEVICE)

                logits = get_action_logits(market_b, account_b)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels_b).sum().item()
                val_total += labels_b.size(0)

                for a in range(4):
                    val_action_counts[a] += (preds == a).sum().item()

        val_acc = val_correct / val_total * 100

        # Per-class accuracy on validation
        action_str = " | ".join([
            f"{action_names[i]}:{int(val_action_counts[i])}"
            for i in range(4)
        ])

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        star = "⭐" if is_best else "  "
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
              f"Loss={avg_train_loss:.4f} | "
              f"Train={train_acc:.1f}% | "
              f"Val={val_acc:.1f}% {star} | "
              f"[{action_str}]")

        # Save best model
        if is_best:
            ppo_model.save(OUTPUT_PATH)

    # Final save
    ppo_model.save(OUTPUT_PATH)
    print("-" * 60)
    print(f"\n✅ Behavioral Cloning complete!")
    print(f"   Best validation accuracy: {best_val_acc:.1f}%")
    print(f"   Model saved → {OUTPUT_PATH}")
    print(f"\n   🔑 This model now 'knows' how V8 trades.")
    print(f"   Next: Run PASO 3 (RL fine-tuning) to improve upon V8.")


if __name__ == "__main__":
    pretrain_behavioral_cloning()
