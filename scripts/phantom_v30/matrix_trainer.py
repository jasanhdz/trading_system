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
import requests
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load TELEGRAM credentials from .env
load_dotenv(Path(__file__).parent.parent.parent / "binance-futures-bot-ts" / ".env")

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
ENVS_PER_GPU = 890   # Ajustado agresivamente para 8GB (48D)
TOTAL_TIMESTEPS = 4_000_000  # V39: Bajado a 4M para evaluar y GUARDAR PROGRESO 4 veces más rápido contra reinicios inesperados

# === IRON SHIELD MEMORY MANAGEMENT ===
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ROC_ENABLE_PRE_VEGA"] = "1" 

MIN_VIABLE_SCORE = 25.0  # V10: $25 minimum ($5 ROI from $20)
MAX_DD_THRESHOLD = 0.90  # V13.1: Kamikaze filter — 90% DD allowed (let them surf the wicks)


def send_telegram_message(message: str):
    import datetime
    log_path = Path(__file__).parent.parent.parent / "logs" / "telegram_reports.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\n{message}\n{'-'*60}\n")
    except Exception as e:
        print(f"Error saving telegram log locally: {e}")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

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
    MAX_STEPS = 8000
    
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
    """Multi-seed robust evaluation. Runs 7 seeds, reports P75 balance and P95 DD.
    Prevents overfitting to any single starting configuration."""
    EVAL_SEEDS = [42, 137, 256, 1337, 7777, 9999, 12345]
    
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
    close_rate = total_action_counts.get(3, 0) / max(total_actions, 1)
    
    print(f"  Balances across seeds: [{', '.join(f'${b:.2f}' for b in balances)}]")
    print(f"  P25 Balance: ${p25_balance:.2f} | P95 DD: {p95_dd*100:.1f}%")
    print(f"  Actions: Idle={total_action_counts[0]}, Long={total_action_counts[1]}, Short={total_action_counts[2]}, Close={total_action_counts[3]}")
    print(f"  Trading Rate: {tr:.1f}% | Close Rate: {close_rate:.1%}")

    return p25_balance, p95_dd, total_action_counts


def launch_challenger(gpu_id: int, save_path: str, seed: int, d_model: int = 32, champ_pnl: float = 0.0, mirror: int = 0) -> subprocess.Popen:
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
        "--num-envs", str(ENVS_PER_GPU),
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--d-model", str(d_model),
        "--champ-pnl", str(champ_pnl),
        "--mirror", str(mirror),
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
    print(f"   Envs per GPU: {ENVS_PER_GPU}")
    print(f"   Total parallel agents: {ENVS_PER_GPU * min(n_gpus, 2)}")
    print(f"   Timesteps per iteration: {TOTAL_TIMESTEPS:,}")

    iteration = 0
    current_champ_score = 0.0  # Tracks champion score for smart entropy
    while True:
        iteration += 1
        mirror_mode = int(iteration % 2 == 0)  # Par=1(Espejo), Impar=0(Real)
        print(f"\n{'='*60}")
        print(f"🔄 Training Iteration {iteration} | Mirror: {'ON' if mirror_mode else 'OFF'}")
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
            print(f"\n⚔️ Dual-GPU V11 Mode: 48D × 2 (different seeds)...")

            start = time.time()
            proc_a = launch_challenger(0, CHALLENGER_A_PATH, seed_a, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode)
            proc_b = launch_challenger(1, CHALLENGER_B_PATH, seed_b, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode)

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
                msg_err = "  ❌ Both challengers failed! Retrying in 30s..."
                print(msg_err)
                send_telegram_message(msg_err)
                time.sleep(30)
                continue
        else:
            print(f"\n🏋️ Single-GPU Mode: Training 1 challenger...")
            proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a, champ_pnl=current_champ_score, mirror=mirror_mode)
            rc = proc.wait()
            if rc != 0:
                msg_err = f"  ❌ Challenger failed (exit code {rc}). Retrying in 30s..."
                print(msg_err)
                send_telegram_message(msg_err)
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
                num_envs=64,  # V43: 64 envs for even more stable evaluation
            )

            # Evaluate Champion (multi-seed robust)
            print(f"\n  📊 Evaluating Champion (7 seeds)...")
            champ_score, champ_dd, champ_actions = evaluate_model(CHAMPION_PATH, eval_env)
            
            # Update the tracked champion score for the next iteration's Smart Entropy
            if champ_score != -np.inf:
                current_champ_score = champ_score

            # Evaluate all challengers
            best_chall_score = -np.inf
            best_chall_path = None
            best_chall_name = None
            best_chall_dd = 0.0
            best_chall_actions = {}
            
            all_challenger_reports = ""

            for name, path in challengers:
                print(f"\n  📊 Evaluating {name} (5 seeds)...")
                score, dd, chall_actions = evaluate_model(path, eval_env)
                
                all_challenger_reports += f"⚔️ *{name}*\n"
                all_challenger_reports += f"PnL: ${score:.2f} | P95 DD: {dd*100:.1f}%\n"
                all_challenger_reports += f"Idle: {chall_actions.get(0,0)} | Long: {chall_actions.get(1,0)} | Short: {chall_actions.get(2,0)} | Close: {chall_actions.get(3,0)}\n\n"

                if score > best_chall_score:
                    best_chall_score = score
                    best_chall_path = path
                    best_chall_name = name
                    best_chall_dd = dd
                    best_chall_actions = chall_actions

            eval_env.close()
            
            msg = f"🔄 *Iteration {iteration} Complete*\n\n"
            msg += f"🏆 *Champion*\n"
            msg += f"PnL: ${champ_score:.2f} | P95 DD: {champ_dd*100:.1f}%\n"
            msg += f"Idle: {champ_actions.get(0,0)} | Long: {champ_actions.get(1,0)} | Short: {champ_actions.get(2,0)} | Close: {champ_actions.get(3,0)}\n\n"
            msg += all_challenger_reports
            msg += "📝 *Result:*\n"

            # === PROMOTION DECISION ===
            print(f"\n🏆 Champion:       ${champ_score:.2f} (P95 DD: {champ_dd*100:.1f}%)")
            print(f"⚔️  Best Challenger: ${best_chall_score:.2f} (P95 DD: {best_chall_dd*100:.1f}%) [{best_chall_name}]")
            print(f"📏 Survival Filter: Max DD allowed = {MAX_DD_THRESHOLD*100:.0f}%")

            champ_survives = champ_dd <= MAX_DD_THRESHOLD
            challenger_survives = best_chall_dd <= MAX_DD_THRESHOLD
            
            # === RISK-ADJUSTED SCORING (Sortino-Proxy) ===
            # V44: Utility = PnL * (1 - DD)^2.0. Exponente subido de 1.5→2.0
            # para castigar DD más agresivamente y romper el equilibrio evolutivo.
            # Champion actual ($12.52, DD 82.9%): Utility baja de ~0.93 a ~0.51
            # Challenger eficiente ($12.50, DD 75%): Utility sube a ~0.78 → PROMOVIBLE
            def get_utility(pnl, dd):
                return pnl * np.power(np.maximum(1.0 - dd, 0.001), 2.0)
            
            champ_utility = get_utility(champ_score, champ_dd)
            chall_utility = get_utility(best_chall_score, best_chall_dd)
            
            print(f"📊 Risk-Adjusted Utility: Champ={champ_utility:.2f} | Best Challenger={chall_utility:.2f}")

            # === REGLA 0: DETHRONE INCONDICIONAL ===
            if not champ_survives and challenger_survives and best_chall_path:
                print(f"💀 CHAMPION DETHRONED! Legacy kamikaze DD {champ_dd*100:.1f}% > {MAX_DD_THRESHOLD*100:.0f}% limit.")
                print(f"👑 CROWNING {best_chall_name} as new CLEAN baseline!")
                backup_path = f"{CHAMPION_PATH}.kamikaze_banned_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New clean Champion crowned! Kamikaze lineage terminated.")
                msg += f"💀 Champion Dethroned (DD > {MAX_DD_THRESHOLD*100:.0f}% limit)\n👑 Crowning {best_chall_name}!"
            
            # === REGLA 1: PROMOTION POR UTILIDAD (EFICIENCIA) ===
            elif challenger_survives and chall_utility > champ_utility and best_chall_path:
                print(f"🚀 PROMOTION! {best_chall_name} defeats Champion on RISK-ADJUSTED basis!")
                print(f"   Utility: {chall_utility:.2f} > {champ_utility:.2f} ✅")
                print(f"   PnL:     ${best_chall_score:.2f} vs ${champ_score:.2f}")
                print(f"   DD:      {best_chall_dd*100:.1f}% vs {champ_dd*100:.1f}%")
                
                backup_path = f"{CHAMPION_PATH}.backup_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New Efficiency Champion crowned! (Old backed up)")
                msg += f"🚀 PROMOTION! {best_chall_name} defeats Champion!\nUtil: {chall_utility:.2f} > {champ_utility:.2f}"
                
            # === REGLA 2: Bloqueo de Kamikazes retadores ===
            elif not challenger_survives and best_chall_score > champ_score:
                print(f"💀 BLOCKED! {best_chall_name} earned ${best_chall_score:.2f} but P95 DD {best_chall_dd*100:.1f}% exceeds {MAX_DD_THRESHOLD*100:.0f}% limit.")
                print(f"🛡️ Champion survives by survival filter.")
                msg += f"💀 BLOCKED! {best_chall_name} DD {best_chall_dd*100:.1f}% > limit. Champion survives."
            
            # === REGLA 3: Baseline inicial si no hay campeón ===
            elif champ_score == -np.inf and best_chall_score > -np.inf and best_chall_path:
                if challenger_survives and best_chall_score >= 10.0:
                    print(f"🆕 No champion. Auto-promoting {best_chall_name} as founding baseline.")
                    shutil.copy2(best_chall_path, CHAMPION_PATH)
                    msg += f"🆕 No champion. Auto-promoting {best_chall_name} as baseline."
                else:
                    print(f"⚠️ No valid baseline found.")
                    msg += "⚠️ No valid baseline found."
            else:
                print(f"🛡️ DEFENSE! Champion retains title by efficiency.")
                msg += "🛡️ DEFENSE! Champion retains title by efficiency."

            send_telegram_message(msg)

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
            send_telegram_message(f"❌ *Trainer Error on Iteration {iteration}*\n```\n{e}\n```")

        sys.stdout.flush()

        # HEARTBEAT: Touch the log file so Watchdog knows we are alive
        try:
            Path("logs/training.log").touch()
        except Exception as e:
            print(f"⚠️ Failed to touch heartbeat file: {e}")

        print(f"\n⏸️  Sleeping 10s before next iteration...")
        time.sleep(10)


if __name__ == "__main__":
    send_telegram_message("🟢 *Matrix Trainer [03-V30-Trainer] Iniciado / Reiniciado* 🚀\nIniciando entrenamiento continuo en las GPUs...")
    continuous_train()
