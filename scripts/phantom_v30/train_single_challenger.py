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

def conservative_lr_schedule(progress_remaining: float) -> float:
    """V46.4 conservative fine-tuning: 3e-4 -> 3e-5 with short warmup."""
    warmup_pct = 0.05
    progress = 1.0 - progress_remaining
    if progress < warmup_pct:
        return 3e-5 + (progress / warmup_pct) * (3e-4 - 3e-5)
    decay_progress = (progress - warmup_pct) / (1.0 - warmup_pct)
    return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + np.cos(np.pi * decay_progress))

# === MUTATION 2: Smart Entropy Schedule v2 ===
class RiskSeekingCallback(BaseCallback):
    """Smart exploration based on Champion quality (V43: tiered + slower decay)."""
    def __init__(self, champ_pnl: float, starting_ent_override: float = None, ent_floor: float = None):
        super().__init__()
        self.champ_pnl = champ_pnl
        self.ent_floor = float(os.environ.get("PHANTOM_ENTROPY_FLOOR", "0.01")) if ent_floor is None else ent_floor
        if starting_ent_override is not None:
            self.starting_ent = starting_ent_override
        else:
            if champ_pnl < 8.0:
                self.starting_ent = 0.18
            elif champ_pnl < 15.0:
                self.starting_ent = 0.14
            elif champ_pnl < 25.0:
                self.starting_ent = 0.10
            else:
                self.starting_ent = 0.08

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        # V46.4: lower entropy floor so challengers stop churning once a policy emerges.
        self.model.ent_coef = max(self.starting_ent * (1 - progress ** 2.5), self.ent_floor)
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
    parser.add_argument(
        "--base-source",
        choices=["auto", "latest", "champion", "bc", "fresh"],
        default=os.environ.get("PHANTOM_BASE_SOURCE", "auto"),
        help="Model lineage source. Use bc/fresh to break contaminated champion lineage.",
    )
    args = parser.parse_args()
    champion_mutation_entropy = float(os.environ.get("PHANTOM_CHAMPION_MUTATION_ENTROPY", "0.10"))
    champion_mutation_lr = float(os.environ.get("PHANTOM_CHAMPION_MUTATION_LR", "0.00025"))
    latest_mutation_entropy = float(os.environ.get("PHANTOM_LATEST_MUTATION_ENTROPY", "0.10"))
    bc_mutation_entropy = float(os.environ.get("PHANTOM_BC_MUTATION_ENTROPY", "0.12"))
    fresh_mutation_entropy = float(os.environ.get("PHANTOM_FRESH_MUTATION_ENTROPY", "0.18"))
    entropy_floor = float(os.environ.get("PHANTOM_ENTROPY_FLOOR", "0.01"))
    
    # V45-FAST: Light viable architecture for 21 features + 6 account dims
    args.d_model = 32
    n_heads = 2
    n_layers = 1
    
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
    leverage_label = f"{float(os.environ.get('PHANTOM_LEVERAGE', '5.0')):g}x".replace(".", "p")
    champion_path = "models/phantom_v30_champion.zip"
    latest_path = f"models/phantom_v31_latest_challenger_{leverage_label}.zip"
    bc_model_path = "models/phantom_v30_bc_pretrained.zip"
    
    if args.base_source == "fresh":
        base_path = None
        print("   🆕 Base source forced to FRESH. Creating challenger from scratch.")
    elif args.base_source == "bc":
        if os.path.exists(bc_model_path):
            base_path = bc_model_path
            print(f"   🧬 Base source forced to BC from {base_path}...")
        else:
            base_path = None
            print("   ⚠️ BC base requested but not found. Falling back to fresh model.")
    elif args.base_source == "champion":
        if os.path.exists(champion_path):
            base_path = champion_path
            print(f"   🧠 Base source forced to Champion from {base_path}...")
        else:
            base_path = None
            print("   ⚠️ Champion base requested but not found. Falling back to fresh model.")
    elif args.base_source == "latest":
        if os.path.exists(latest_path):
            base_path = latest_path
            print(f"   🧠 Base source forced to Latest from {base_path}...")
        else:
            base_path = None
            print("   ⚠️ Latest base requested but not found. Falling back to fresh model.")
    # Priority 0: Latest Challenger (Accumulate learning even if not promoted)
    elif os.path.exists(latest_path):
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
    custom_lr = conservative_lr_schedule
    starting_ent_override = None

    if base_path:
        try:
            if base_path == champion_path:
                starting_ent = champion_mutation_entropy
                starting_ent_override = champion_mutation_entropy
                custom_lr = lambda p: champion_mutation_lr
                print(f"   🧊 V46.4 CONSERVATIVE MUTATION: entropy={starting_ent:.2f}, LR={champion_mutation_lr:g}")
            elif base_path == latest_path:
                starting_ent = latest_mutation_entropy
                starting_ent_override = latest_mutation_entropy
                custom_lr = conservative_lr_schedule
                print(f"   🧊 V46.4 Latest fine-tune: entropy={starting_ent:.2f}, conservative LR schedule")
            elif base_path == bc_model_path:
                starting_ent = bc_mutation_entropy
                starting_ent_override = bc_mutation_entropy
                custom_lr = conservative_lr_schedule
                print(f"   🧬 V46.4 BC fine-tune: entropy={starting_ent:.2f}, conservative LR schedule")
            else:
                starting_ent = latest_mutation_entropy
                starting_ent_override = latest_mutation_entropy

            model = PPO.load(
                base_path,
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": custom_lr,
                    "n_steps": 256,       # V40: 512 envs × 256 = 131,072 buffer (expert recommendation)
                    "batch_size": 2048,   # Revertido por OOM VRAM en 8GB
                    "n_epochs": 4,        
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
        starting_ent = fresh_mutation_entropy
        starting_ent_override = fresh_mutation_entropy
        print(f"   🧊 V46.4 Fresh policy: entropy={starting_ent:.2f}, conservative LR schedule")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            learning_rate=conservative_lr_schedule,
            n_steps=256,         # V40: 512 envs × 256 = 131,072 buffer
            batch_size=2048,     # Revertido por OOM
            n_epochs=4,          
            gamma=0.95,
            clip_range=adaptive_clip_schedule,     
            gae_lambda=0.95,     
            ent_coef=starting_ent,       
            seed=args.seed,
            device=device,
        )

    checkpoint_every_timesteps = int(os.environ.get("PHANTOM_CHECKPOINT_EVERY_TIMESTEPS", "1000000"))
    checkpoint_cb = CheckpointCallback(
        # CheckpointCallback counts VecEnv steps, not PPO rollout iterations.
        # With 1024 envs, dividing by n_steps saved on every callback call and
        # produced thousands of checkpoints on the HDD.
        save_freq=max(checkpoint_every_timesteps // args.num_envs, 1),
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
    model.lr_schedule = custom_lr if 'custom_lr' in locals() else conservative_lr_schedule

    risk_cb = RiskSeekingCallback(
        champ_pnl=args.champ_pnl,
        starting_ent_override=starting_ent_override if 'starting_ent_override' in locals() else None,
        ent_floor=entropy_floor,
    )
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
