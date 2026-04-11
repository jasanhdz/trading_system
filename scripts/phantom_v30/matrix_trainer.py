#!/usr/bin/env python3
"""
Phantom V30 Matrix Trainer — Dual-GPU Coliseum
Uses subprocess.Popen for GPU isolation (fixes ROCm gfx1032 multiprocessing deadlock).

Architecture:
  - Orchestrator (this process): Launches subprocesses, runs Coliseum on CPU
  - Challenger A: Separate Python process, HIP_VISIBLE_DEVICES=0
  - Challenger B: Separate Python process, HIP_VISIBLE_DEVICES=1
"""
import torch
import numpy as np
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from stable_baselines3 import PPO
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

# Paths
CHAMPION_PATH = "models/phantom_v30_champion.zip"
CHALLENGER_A_PATH = "models/phantom_v30_challenger_a.zip"
CHALLENGER_B_PATH = "models/phantom_v30_challenger_b.zip"
SAFE_CHECKPOINT_PATH = "models/phantom_v31_safe_checkpoint.zip"  # Best survivor vault

# Training Config
NUM_ENVS = 512  # 512 envs × 128 steps = 65,536 buffer (4x more context)
TOTAL_TIMESTEPS = 16_000_000  # V33: 16M steps (15 features need more experience to converge)
MIN_VIABLE_SCORE = 25.0  # V10: $25 minimum ($5 ROI from $20)
MAX_DD_THRESHOLD = 0.96  # V13.1: Kamikaze filter — 96% DD allowed (let them surf the wicks)


def evaluate_model_single(model_path: str, eval_env, seed: int = 42):
    """Single-seed evaluation. Returns (balance, p95_drawdown, action_counts)."""
    if not os.path.exists(model_path):
        return -np.inf, 0.0, {}

    try:
        model = PPO.load(model_path, env=eval_env, device="cpu")
    except Exception as e:
        print(f"  ⚠️ Failed to load {model_path}: {e}")
        return -np.inf, 0.0, {}

    np.random.seed(seed)

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    MAX_STEPS = 8000  # V33.1: Cover full 14-day validation set
    
    obs = eval_env.reset()
    peak_equities = np.full(eval_env.num_envs, float(eval_env.balances[0]))
    max_drawdowns = np.zeros(eval_env.num_envs, dtype=np.float32)
    
    done_all = np.zeros(eval_env.num_envs, dtype=bool)
    steps = 0

    while not done_all.all() and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        for a in action:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1
        obs, reward, done, infos = eval_env.step(action)
        done_all = done_all | done
        steps += 1

        equities = np.array([info.get('equity', info['balance']) for info in infos])
        peak_equities = np.maximum(peak_equities, equities)
        safe_peaks = np.maximum(peak_equities, 1e-10)
        dds = (safe_peaks - equities) / safe_peaks
        max_drawdowns = np.maximum(max_drawdowns, dds)

    final_balance = np.mean([info['balance'] for info in infos])
    # P95 DD: worst-case risk (catches the outlier that would liquidate you in production)
    p95_drawdown = float(np.percentile(max_drawdowns, 95))

    return final_balance, p95_drawdown, action_counts


def evaluate_model(model_path: str, eval_env, n_episodes: int = 1, seed: int = 42):
    """Multi-seed robust evaluation. Runs 5 seeds, reports P75 balance and P95 DD.
    Prevents overfitting to any single starting configuration."""
    EVAL_SEEDS = [42, 137, 256, 1337, 7777]
    
    balances = []
    drawdowns = []
    total_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for s in EVAL_SEEDS:
        bal, dd, ac = evaluate_model_single(model_path, eval_env, seed=s)
        if bal == -np.inf:
            return -np.inf, 0.0, {}
        balances.append(bal)
        drawdowns.append(dd)
        for k, v in ac.items():
            total_action_counts[k] = total_action_counts.get(k, 0) + v
    
    # P25 balance: conservative estimate (bottom quartile — premia consistencia)
    p25_balance = float(np.percentile(balances, 25))
    # P95 DD: worst-case risk across all seeds
    p95_dd = float(np.max(drawdowns))  # Worst DD across all seeds
    
    total_actions = sum(total_action_counts.values())
    tr = (total_action_counts.get(1, 0) + total_action_counts.get(2, 0)) / max(total_actions, 1) * 100
    
    print(f"  Balances across seeds: [{', '.join(f'${b:.2f}' for b in balances)}]")
    print(f"  P25 Balance: ${p25_balance:.2f} | P95 DD: {p95_dd*100:.1f}%")
    print(f"  Actions: Idle={total_action_counts[0]}, Long={total_action_counts[1]}, Short={total_action_counts[2]}, Close={total_action_counts[3]}")
    print(f"  Trading Rate: {tr:.1f}%")

    return p25_balance, p95_dd, total_action_counts


def launch_challenger(gpu_id: int, save_path: str, seed: int, d_model: int = 32) -> subprocess.Popen:
    """Launch a challenger training in a completely isolated subprocess."""
    script = Path(__file__).parent / "train_single_challenger.py"
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    cmd = [
        sys.executable,
        str(script),
        "--save-path", save_path,
        "--seed", str(seed),
        "--num-envs", str(NUM_ENVS),
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--d-model", str(d_model),
    ]

    print(f"  🚀 Launching Challenger on GPU {gpu_id} (PID isolation)...")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    return proc


def continuous_train():
    """Main dual-GPU training loop using subprocess isolation."""
    n_gpus = torch.cuda.device_count()
    print(f"🚀 Phantom V30 Matrix Trainer (Subprocess Mode)")
    print(f"   GPUs: {n_gpus}")
    print(f"   Envs per GPU: {NUM_ENVS}")
    print(f"   Total parallel agents: {NUM_ENVS * min(n_gpus, 2)}")
    print(f"   Timesteps per iteration: {TOTAL_TIMESTEPS:,}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"🔄 Training Iteration {iteration}")
        print(f"{'='*60}")
        
        # === DYNAMIC DATA COLLECTION ===
        print("\n📡 Fetching freshest market data (update_ml_candles.py)...")
        try:
            subprocess.run([sys.executable, "scripts/update_ml_candles.py"], check=True)
            print("✅ Data synchronization complete.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Data collector failed ({e}). Proceeding with cached database data.")

        seed_a = int(time.time()) % 100000
        seed_b = seed_a + 31337

        # === DUAL-GPU TRAINING (subprocess isolation) ===
        if n_gpus >= 2:
            print(f"\n⚔️ Dual-GPU V10 Mode: 32D × 2 (different seeds)...")

            start = time.time()
            proc_a = launch_challenger(0, CHALLENGER_A_PATH, seed_a, d_model=48)
            proc_b = launch_challenger(1, CHALLENGER_B_PATH, seed_b, d_model=48)

            # Wait for both with HEARTBEAT (touch log every 10s so Watchdog knows we're alive)
            while proc_a.poll() is None or proc_b.poll() is None:
                try:
                    Path("/home/jasan/Develop/trading_system/logs/training.log").touch()
                except Exception:
                    pass
                time.sleep(10)

            rc_a = proc_a.returncode
            rc_b = proc_b.returncode
            total_time = time.time() - start

            print(f"\n⏱️ Both GPUs finished in {total_time:.0f}s (rc: {rc_a}, {rc_b})")
            sys.stdout.flush()

            challengers = []
            if rc_a == 0:
                challengers.append(("Challenger A (GPU:0)", CHALLENGER_A_PATH))
            else:
                print(f"  ⚠️ Challenger A failed (exit code {rc_a})")
            if rc_b == 0:
                challengers.append(("Challenger B (GPU:1)", CHALLENGER_B_PATH))
            else:
                print(f"  ⚠️ Challenger B failed (exit code {rc_b})")

            if not challengers:
                print("  ❌ Both challengers failed! Retrying in 30s...")
                time.sleep(30)
                continue
        else:
            print(f"\n🏋️ Single-GPU Mode: Training 1 challenger...")
            proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a)
            rc = proc.wait()
            if rc != 0:
                print(f"  ❌ Challenger failed (exit code {rc}). Retrying in 30s...")
                time.sleep(30)
                continue
            challengers = [("Challenger A", CHALLENGER_A_PATH)]

        # === COLISEUM EVALUATION (CPU) ===
        try:
            print(f"\n🏟️ COLISEUM: Evaluating all fighters on CPU...")
            sys.stdout.flush()

            eval_data = load_tensor_data("cpu", split="val")
            features_np = eval_data['features'].numpy()
            close_np = eval_data['close'].numpy()
            
            eval_env = PhantomMatrixEnv(
                features=features_np,
                close_prices=close_np,
                num_envs=32,  # V10: 32 envs for 4x more stable evaluation
            )

            # Evaluate Champion (multi-seed robust)
            print(f"\n  📊 Evaluating Champion (5 seeds)...")
            champ_score, champ_dd, _ = evaluate_model(CHAMPION_PATH, eval_env)

            # Evaluate all challengers
            best_chall_score = -np.inf
            best_chall_path = None
            best_chall_name = None
            best_chall_dd = 0.0

            for name, path in challengers:
                print(f"\n  📊 Evaluating {name} (5 seeds)...")
                score, dd, _ = evaluate_model(path, eval_env)
                if score > best_chall_score:
                    best_chall_score = score
                    best_chall_path = path
                    best_chall_name = name
                    best_chall_dd = dd

            eval_env.close()

            # === PROMOTION DECISION ===
            print(f"\n🏆 Champion:       ${champ_score:.2f} (P95 DD: {champ_dd*100:.1f}%)")
            print(f"⚔️  Best Challenger: ${best_chall_score:.2f} (P95 DD: {best_chall_dd*100:.1f}%) [{best_chall_name}]")
            print(f"📏 Survival Filter: Max DD allowed = {MAX_DD_THRESHOLD*100:.0f}%")

            # === SURVIVAL FILTER: Block kamikazes ===
            challenger_survives = best_chall_dd <= MAX_DD_THRESHOLD
            if not challenger_survives and best_chall_score > champ_score:
                print(f"💀 BLOCKED! {best_chall_name} earned ${best_chall_score:.2f} but P95 DD {best_chall_dd*100:.1f}% exceeds {MAX_DD_THRESHOLD*100:.0f}% limit.")
                print(f"🛡️ Champion survives by survival filter.")
            elif champ_score == -np.inf and best_chall_score > -np.inf and best_chall_path:
                if best_chall_score >= 15.0:  # Any surviving V31 seed becomes the new baseline
                    print(f"🆕 Retiring V30 Champion. Auto-promoting {best_chall_name} to start V31 lineage.")
                    shutil.copy2(best_chall_path, CHAMPION_PATH)
                    print(f"✅ Nuevo Campeón V31 semilla coronado para acumular aprendizaje.")
                else:
                    reason = f"DD {best_chall_dd*100:.1f}% > {MAX_DD_THRESHOLD*100:.0f}%" if not challenger_survives else f"PnL ${best_chall_score:.2f} < ${MIN_VIABLE_SCORE:.0f}"
                    print(f"⚠️ No Champion & Challenger rejected ({reason}). Skipping.")
            elif best_chall_score > champ_score and best_chall_score >= MIN_VIABLE_SCORE and challenger_survives and best_chall_path:
                print(f"🚀 PROMOTION! {best_chall_name} defeats Champion!")
                print(f"   PnL: ${best_chall_score:.2f} > ${champ_score:.2f} ✅")
                print(f"   DD:  {best_chall_dd*100:.1f}% ≤ {MAX_DD_THRESHOLD*100:.0f}% ✅")
                backup_path = f"{CHAMPION_PATH}.backup_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New Champion crowned! (Old backed up)")
            elif best_chall_score > champ_score and best_chall_score < MIN_VIABLE_SCORE:
                print(f"🛡️ Challenger better (${best_chall_score:.2f}) but below ${MIN_VIABLE_SCORE:.0f} threshold. No promotion.")
            else:
                print(f"🛡️ DEFENSE! Champion retains title.")

            # --- ONGOING LEARNING: 3-Tier Survivor Vault ---
            if best_chall_path and os.path.exists(best_chall_path):
                if best_chall_dd < 0.90:
                    # ✅ SAFE: Save as ongoing baseline AND update the safe checkpoint vault
                    shutil.copy2(best_chall_path, "models/phantom_v31_latest_challenger.zip")
                    shutil.copy2(best_chall_path, SAFE_CHECKPOINT_PATH)
                    print(f"📥 Saved {best_chall_name} (${best_chall_score:.2f} | DD: {best_chall_dd*100:.1f}%) as ongoing baseline + safe checkpoint.")
                else:
                    # 💀 SUICIDAL: Restore from safe checkpoint (preserves progress) or champion (last resort)
                    if os.path.exists(SAFE_CHECKPOINT_PATH):
                        shutil.copy2(SAFE_CHECKPOINT_PATH, "models/phantom_v31_latest_challenger.zip")
                        print(f"💀 MUTATION REJECTED: {best_chall_name} ({best_chall_dd*100:.1f}% DD). Restoring from SAFE CHECKPOINT (preserving progress).")
                    else:
                        shutil.copy2(CHAMPION_PATH, "models/phantom_v31_latest_challenger.zip")
                        print(f"� MUTATION REJECTED: {best_chall_name} ({best_chall_dd*100:.1f}% DD). No safe checkpoint found, restoring Champion baseline.")

            # Cleanup challenger files
            for _, path in challengers:
                if os.path.exists(path):
                    os.remove(path)

        except Exception as e:
            print(f"\n❌ COLISEUM ERROR: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Skipping evaluation, continuing to next iteration...")

        sys.stdout.flush()

        # HEARTBEAT: Touch the log file so Watchdog knows we are alive
        try:
            Path("logs/training.log").touch()
        except Exception as e:
            print(f"⚠️ Failed to touch heartbeat file: {e}")

        print(f"\n⏸️  Sleeping 10s before next iteration...")
        time.sleep(10)


if __name__ == "__main__":
    continuous_train()
