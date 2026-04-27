#!/usr/bin/env python3
"""
Train a single challenger model in an isolated process.
Called by matrix_trainer.py via subprocess.Popen.

GPU isolation via HIP_VISIBLE_DEVICES → each process sees cuda:0.
Uses 1024 envs + short n_steps (128) to keep backward pass within gfx1032 limits.
"""
import argparse
import torch
import numpy as np
import os
import sys
import time
import shutil
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.buffers import BaseBuffer

# MONKEY-PATCH for ROCm gfx1032: CPU-first tensor creation
def _safe_to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
    if copy:
        return torch.tensor(array).to(self.device)
    return torch.as_tensor(array).to(self.device)
BaseBuffer.to_torch = _safe_to_torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.train_v30 import TransformerExtractor


class ColiseoEarlyStop(BaseCallback):
    """
    Guillotina de tiempo inteligente.
    Si la política deja de moverse (KL baja + Explicación alta), 
    no quemamos GPU en vano.
    """
    def __init__(self, kl_threshold: float = 0.0007, 
                 ev_threshold: float = 0.88, 
                 patience: int = 10):
        super().__init__()
        self.kl_threshold = kl_threshold
        self.ev_threshold = ev_threshold
        self.patience = patience
        self.stagnant_count = 0
        self.last_check_step = 0
        
    def _on_step(self) -> bool:
        # Solo chequear cada ~1 rollout (evita spamear logs)
        check_interval = self.model.n_steps * self.model.n_envs
        if self.num_timesteps - self.last_check_step < check_interval:
            return True
            
        self.last_check_step = self.num_timesteps
        
        # SB3 deposita estos valores durante model.train()
        if not hasattr(self.model, "logger") or self.model.logger is None:
            return True
        kl = self.model.logger.name_to_value.get('train/approx_kl', 999.0)
        ev = self.model.logger.name_to_value.get('train/explained_variance', -1.0)
        loss = self.model.logger.name_to_value.get('train/loss', 999.0)
        
        # Condición de estancamiento: política congelada Y value function explicada
        frozen_policy = (kl < self.kl_threshold)
        solved_value = (ev > self.ev_threshold)
        # Bonus: si la loss total no ha bajado en las últimas N comprobaciones, también es señal
        if frozen_policy and solved_value:
            self.stagnant_count += 1
            print(f"  🐢 Stagnation {self.stagnant_count}/{self.patience} | "
                  f"KL:{kl:.6f} EV:{ev:.3f} Loss:{loss:.4f}")
            if self.stagnant_count >= self.patience:
                print(f"🛑 EARLY STOP en step {self.num_timesteps:,}. "
                      f"Política convergida. Enviando al Coliseo AHORA.")
                return False  # SB3 detiene model.learn() inmediatamente
        else:
            # Resetear contador si hay movimiento (evita acumulación por ruido)
            if self.stagnant_count > 0 and not frozen_policy:
                print(f"  🌱 Recovery | KL:{kl:.6f} EV:{ev:.3f}")
            self.stagnant_count = 0
            
        return True

def cosine_lr_schedule(progress_remaining: float) -> float:
    """Cosine annealing: 3e-4 → 3e-5 with 5% linear warmup.
    progress_remaining goes from 1.0 (start) to 0.0 (end)."""
    warmup_pct = 0.05
    progress = 1.0 - progress_remaining  # 0.0 → 1.0
    
    if progress < warmup_pct:
        # Linear warmup from 5e-5 to 5e-4
        return 5e-5 + (progress / warmup_pct) * (5e-4 - 5e-5)
    
    # Cosine decay from 5e-4 to 5e-5
    decay_progress = (progress - warmup_pct) / (1.0 - warmup_pct)
    return 5e-5 + 0.5 * (5e-4 - 5e-5) * (1 + np.cos(np.pi * decay_progress))

# === MUTATION 2: Smart Entropy Schedule ===
class RiskSeekingCallback(BaseCallback):
    """Smart exploration based on Champion quality."""
    def __init__(self, champ_pnl: float):
        super().__init__()
        self.champ_pnl = champ_pnl
        self.starting_ent = 0.08 if champ_pnl >= 25.0 else 0.15

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        # Entropy decays from starting_ent to 0.01 to keep model from stalling
        self.model.ent_coef = max(self.starting_ent * (1 - progress ** 1.5), 0.01)
        return True

def adaptive_clip_schedule(progress_remaining: float) -> float:
    """Clip: 0.20 → 0.12 (allow bigger policy updates early). progress_remaining goes 1.0 -> 0.0"""
    progress = 1.0 - progress_remaining
    return 0.20 - 0.08 * progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--champ-pnl", type=float, default=0.0)
    parser.add_argument("--mirror", type=int, default=0)
    args = parser.parse_args()
    
    # V31: Evidence-based architecture (32D is too small for 15 features, up to 48D)
    n_heads = 2 if args.d_model <= 48 else 4 # Keep 2 heads for stability
    n_layers = 1 if args.d_model <= 32 else 2
    
    POLICY_KWARGS = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(d_model=args.d_model, nhead=n_heads, num_layers=n_layers, dropout=0.1),
        net_arch=dict(pi=[args.d_model, args.d_model], vf=[args.d_model, args.d_model])
    )

    gpu_id = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    device = "cuda:0"

    # === IRON SHIELD v2: Anti-Hang + Anti-Fragmentation ===
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["RCCL_P2P_DISABLE"] = "1"
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    # ======================================================

    print(f"🏋️ Challenger Training (GPU isolated)")
    print(f"   HIP_VISIBLE_DEVICES={gpu_id} → {device}")
    print(f"   🛡️ Iron Shield v2: P2P Off, Expandable Segments")
    print(f"   Seed={args.seed}, Envs={args.num_envs}, Steps={args.timesteps:,}, Model={args.d_model}D")
    sys.stdout.flush()

    # Load data
    data = load_tensor_data("cpu", days=None, split="train", mirror=args.mirror)
    np.random.seed(args.seed)
    env = PhantomMatrixEnv(
        features=data['features'].numpy(),
        close_prices=data['close'].numpy(),
        num_envs=args.num_envs,
    )

    # === IRON SHIELD v4: Clean Start (Fix Bug #1 & #2) ===
    # Always train from scratch. Stale checkpoints caused iterations 2+
    # to skip training entirely (the model thought it was already done).
    checkpoint_dir = f"./models/checkpoints/gpu_{gpu_id}"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"   🛡️ Iron Shield v4: Clean checkpoint dir for fresh training")

    # === PASO 3: Load Champion Model (Hill Climbing) ===
    # To compound our improvements, Challengers should now mutate from the reigning Champion
    # (e.g. the $65 model) instead of always starting from the bare V8 BC baseline.
    champion_path = "models/phantom_v30_champion.zip"
    latest_path = "models/phantom_v31_latest_challenger.zip"
    bc_model_path = "models/phantom_v30_bc_pretrained.zip"
    
    # Priority 0: Latest Challenger (Accumulate learning even if not promoted)
    if os.path.exists(latest_path):
        base_path = latest_path
        print(f"   🧠 Loading Ongoing Challenger from {base_path} to accumulate learning...")
    # Priority 1: Reigning Champion
    elif os.path.exists(champion_path):
        base_path = champion_path
        print(f"   🧠 Loading Reigning Champion from {base_path} for Hill Climbing...")
    # Priority 2: BC Pretrained
    elif os.path.exists(bc_model_path):
        base_path = bc_model_path
        print(f"   🧠 Loading Pre-Trained BC Model from {base_path}...")
    else:
        base_path = None
        
    model = None
    if base_path:
        try:
            starting_ent = 0.08 if args.champ_pnl >= 25.0 else 0.15
            model = PPO.load(
                base_path,
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": cosine_lr_schedule,
                    "n_steps": 256,       # V40: 512 envs × 256 = 131,072 buffer (expert recommendation)
                    "batch_size": 2048,   # V40: 64 mini-batches to match larger buffer
                    "n_epochs": 6,        
                    "gamma": 0.95,
                    "gae_lambda": 0.95,   
                    "ent_coef": starting_ent, 
                    "verbose": 1,
                    "seed": args.seed,
                }
            )
        except (ValueError, RuntimeError) as e:
            print(f"   ⚠️ Architecture mismatch loading {base_path}: {e}")
            print(f"   🆕 Starting fresh with new V31 architecture (PosEnc + AttnPool)...")
            model = None

    if model is None:
        print(f"   🆕 Creating new model from scratch with V31 architecture...")
        starting_ent = 0.08 if args.champ_pnl >= 25.0 else 0.15
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            learning_rate=cosine_lr_schedule,
            n_steps=256,         # V40: 512 envs × 256 = 131,072 buffer
            batch_size=2048,     # V40: 64 mini-batches
            n_epochs=6,          
            gamma=0.95,
            clip_range=adaptive_clip_schedule,     
            gae_lambda=0.95,     
            ent_coef=starting_ent,       
            seed=args.seed,
            device=device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // (args.num_envs * 256), 1),
        save_path=checkpoint_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    start = time.time()
    print(f"   ⚡ Training started...")
    sys.stdout.flush()

    # V40: Ensure LR scheduler fully resets for cyclic SGDR behavior
    model.num_timesteps = 0
    model._current_progress_remaining = 1.0
    model.lr_schedule = cosine_lr_schedule

    risk_cb = RiskSeekingCallback(champ_pnl=args.champ_pnl)
    early_cb = ColiseoEarlyStop(kl_threshold=0.0007, ev_threshold=0.88, patience=10)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, risk_cb, early_cb], reset_num_timesteps=True)

    elapsed = time.time() - start
    fps = args.timesteps / elapsed
    print(f"   ✅ Done in {elapsed:.0f}s ({fps:,.0f} effective FPS)")

    model.save(args.save_path)
    print(f"   💾 Saved → {args.save_path}")
    sys.stdout.flush()

    # HARD EXIT: Bypasses Python/ROCm cleanup handlers
    os._exit(0)


if __name__ == "__main__":
    main()
