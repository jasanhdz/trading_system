from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from aegis_alpha.rl.policy import AegisTransformerExtractor


class DummyAegisGym(gym.Env):
    def __init__(self, window_size: int, n_features: int, account_dim: int):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(low=-10, high=10, shape=(window_size, n_features), dtype=np.float32),
                "account": spaces.Box(low=-np.inf, high=np.inf, shape=(account_dim,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {
            "market": np.zeros(self.observation_space["market"].shape, dtype=np.float32),
            "account": np.zeros(self.observation_space["account"].shape, dtype=np.float32),
        }, {}

    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}


def _action_logits(policy, market_batch, account_batch):
    obs = {"market": market_batch, "account": account_batch}
    features = policy.extract_features(obs, policy.pi_features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


def train_bc(
    dataset_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_samples: int | None,
    device: str,
    sampler_power: float,
) -> None:
    data = np.load(dataset_path)
    market_np = data["market"]
    account_np = data["account"]
    actions_np = data["actions"]
    if max_samples and len(actions_np) > max_samples:
        rng = np.random.default_rng(4667)
        idx = rng.choice(len(actions_np), size=max_samples, replace=False)
        market_np = market_np[idx]
        account_np = account_np[idx]
        actions_np = actions_np[idx]

    market = torch.tensor(market_np, dtype=torch.float32)
    account = torch.tensor(account_np, dtype=torch.float32)
    actions = torch.tensor(actions_np, dtype=torch.long)
    n_samples = len(actions)
    class_counts = torch.bincount(actions, minlength=4).float()
    class_weights = (n_samples / (4.0 * class_counts + 1e-10)).pow(0.75).clamp(0.35, 8.0).to(device)

    perm = torch.randperm(n_samples)
    split = int(n_samples * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]
    pin_memory = device.startswith("cuda")
    train_dataset = TensorDataset(market[train_idx], account[train_idx], actions[train_idx])
    train_sample_weights = (n_samples / (4.0 * class_counts[actions[train_idx]] + 1e-10)).pow(sampler_power)
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights.double(),
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(market[val_idx], account[val_idx], actions[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    env = DummyAegisGym(market.shape[1], market.shape[2], account.shape[1])
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": AegisTransformerExtractor,
            "features_extractor_kwargs": {"d_model": 32, "nhead": 2, "num_layers": 1, "dropout": 0.1},
            "net_arch": {"pi": [32, 32], "vf": [32, 32]},
        },
        learning_rate=3e-4,
        n_steps=32,
        batch_size=128,
        n_epochs=1,
        gamma=0.95,
        ent_coef=0.02,
        seed=4667,
        device=device,
        verbose=0,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_names = ["IDLE", "LONG", "SHORT", "CLOSE"]
    print(f"Samples: {n_samples:,}")
    print(f"Class counts: {dict(zip(action_names, class_counts.int().tolist()))}")
    print(f"Class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")
    print(f"Sampler power: {sampler_power:.2f}")

    best_balanced = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for market_b, account_b, labels_b in train_loader:
            market_b = market_b.to(device)
            account_b = account_b.to(device)
            labels_b = labels_b.to(device)
            logits = _action_logits(policy, market_b, account_b)
            loss = criterion(logits, labels_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels_b.size(0)
            total += labels_b.size(0)
            correct += (logits.argmax(dim=1) == labels_b).sum().item()

        policy.eval()
        val_total = 0
        val_correct = 0
        pred_counts = torch.zeros(4)
        class_total = torch.zeros(4)
        class_correct = torch.zeros(4)
        with torch.no_grad():
            for market_b, account_b, labels_b in val_loader:
                market_b = market_b.to(device)
                account_b = account_b.to(device)
                labels_b = labels_b.to(device)
                preds = _action_logits(policy, market_b, account_b).argmax(dim=1)
                val_total += labels_b.size(0)
                val_correct += (preds == labels_b).sum().item()
                for action in range(4):
                    pred_counts[action] += (preds == action).sum().item()
                    label_mask = labels_b == action
                    class_total[action] += label_mask.sum().item()
                    class_correct[action] += ((preds == action) & label_mask).sum().item()
        recalls = class_correct / torch.clamp(class_total, min=1.0)
        balanced = recalls.mean().item() * 100
        val_acc = val_correct / max(val_total, 1) * 100
        train_acc = correct / max(total, 1) * 100
        if balanced > best_balanced:
            best_balanced = balanced
            best_epoch = epoch
            model.save(str(output_path))
        pred_text = " | ".join(f"{action_names[i]}:{int(pred_counts[i])}" for i in range(4))
        recall_text = " | ".join(f"{action_names[i]}R:{recalls[i].item()*100:.0f}%" for i in range(4))
        print(
            f"Epoch {epoch:02d}/{epochs}: loss={total_loss/max(total,1):.4f} "
            f"train={train_acc:.1f}% val={val_acc:.1f}% bal={balanced:.1f}% "
            f"[{pred_text}] [{recall_text}]"
        )

    print(f"Saved best model -> {output_path} (balanced={best_balanced:.1f}% epoch={best_epoch})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/bc_prudent_dataset.npz")
    parser.add_argument("--output", default="aegis_alpha/models/bc/aegis_bc_prudent.zip")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampler-power", type=float, default=0.65)
    args = parser.parse_args()
    train_bc(
        Path(args.dataset),
        Path(args.output),
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.max_samples,
        args.device,
        args.sampler_power,
    )


if __name__ == "__main__":
    main()
