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
HEARTBEAT_PATH = Path("/home/jasan/Develop/trading_system/logs/training.log")

# Training Config
ENVS_PER_GPU = 1024  # Fast path when VRAM is clean.
FALLBACK_ENVS_PER_GPU = 512  # Retry guardrail after HIP OOM/GPU hang.
TOTAL_TIMESTEPS = 4_000_000  # V39: Bajado a 4M para evaluar y GUARDAR PROGRESO 4 veces más rápido contra reinicios inesperados

# === IRON SHIELD MEMORY MANAGEMENT ===
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ROC_ENABLE_PRE_VEGA"] = "1" 

MIN_VIABLE_SCORE = 25.0  # V10: $25 minimum ($5 ROI from $20)
MAX_DD_THRESHOLD = 0.65  # V46.0: Survivor filter — discard kamikaze drawdowns.
MIN_FINAL_BALANCE = 21.0  # V46.0: At least +$1 from the $20 baseline.
PROMOTION_MARGIN = 1.10  # V46.0: Require a material utility improvement.


def heartbeat():
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_PATH.touch()
    except Exception as e:
        print(f"⚠️ Failed to touch heartbeat file: {e}")


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
    MAX_STEPS = 4032  # V45-FAST: 14 días exactos de 5m (evaluación más rápida)
    
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


def evaluate_model(model_path: str, windows: list, n_episodes: int = 1, seed: int = 42):
    """Walk-Forward Multiverse Evaluation. Evaluates over multiple random historical windows."""
    balances = []
    drawdowns = []
    total_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i, window in enumerate(windows):
        heartbeat()
        eval_env = PhantomMatrixEnv(
            features=window['features'],
            close_prices=window['close'],
            num_envs=64,
        )
        bal, dd, ac = evaluate_model_single(model_path, eval_env, seed=seed + i)
        eval_env.close()
        
        if bal == -np.inf:
            return -np.inf, 0.0, {}
            
        balances.append(bal)
        drawdowns.append(dd)
        for k, v in ac.items():
            total_action_counts[k] = total_action_counts.get(k, 0) + v
            
    # Score final: media de metrícas para premiar consistencia en el "multiverso"
    avg_balance = float(np.mean(balances))
    avg_dd = float(np.mean(drawdowns))
    
    total_actions = sum(total_action_counts.values())
    tr = (total_action_counts.get(1, 0) + total_action_counts.get(2, 0)) / max(total_actions, 1) * 100
    close_rate = total_action_counts.get(3, 0) / max(total_actions, 1)
    
    print(f"  Balances in Multiverse: [{', '.join(f'${b:.2f}' for b in balances)}]")
    print(f"  Avg Balance: ${avg_balance:.2f} | Avg DD: {avg_dd*100:.1f}%")
    print(f"  Actions: Idle={total_action_counts[0]}, Long={total_action_counts[1]}, Short={total_action_counts[2]}, Close={total_action_counts[3]}")
    print(f"  Trading Rate: {tr:.1f}% | Close Rate: {close_rate:.1%}")

    return avg_balance, avg_dd, total_action_counts


def launch_challenger(gpu_id: int, save_path: str, seed: int, d_model: int = 32, champ_pnl: float = 0.0, mirror: int = 0, num_envs: int = ENVS_PER_GPU) -> subprocess.Popen:
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
        "--num-envs", str(num_envs),
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--d-model", str(d_model),
        "--champ-pnl", str(champ_pnl),
        "--mirror", str(mirror),
    ]

    print(f"  🚀 Launching Challenger on GPU {gpu_id} (PID isolation, envs={num_envs})...")
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
    print(f"   Fallback envs per GPU: {FALLBACK_ENVS_PER_GPU}")
    print(f"   Total parallel agents: {ENVS_PER_GPU * min(n_gpus, 2)}")
    print(f"   Timesteps per iteration: {TOTAL_TIMESTEPS:,}")

    iteration = 0
    current_champ_score = 0.0  # Tracks champion score for smart entropy
    forced_retirement_counter = 0

    while True:
        heartbeat()
        iteration += 1
        mirror_mode = int(iteration % 2 == 0)  # Par=1(Espejo), Impar=0(Real)
        print(f"\n{'='*60}")
        print(f"🔄 Training Iteration {iteration} | Mirror: {'ON' if mirror_mode else 'OFF'}")
        print(f"{'='*60}")
        
        # === DYNAMIC DATA COLLECTION ===
        print("\n📡 Fetching freshest market data (update_ml_candles.py)...")
        try:
            heartbeat()
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
                heartbeat()
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

            if not challengers and FALLBACK_ENVS_PER_GPU < ENVS_PER_GPU:
                print(f"  ⚠️ Both challengers failed at {ENVS_PER_GPU} envs. Retrying once at {FALLBACK_ENVS_PER_GPU} envs...")
                send_telegram_message(f"⚠️ Both challengers failed at {ENVS_PER_GPU} envs. Retrying at {FALLBACK_ENVS_PER_GPU} envs to avoid a full trainer restart.")

                proc_a = launch_challenger(0, CHALLENGER_A_PATH, seed_a + 101, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU)
                proc_b = launch_challenger(1, CHALLENGER_B_PATH, seed_b + 101, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU)
                while proc_a.poll() is None or proc_b.poll() is None:
                    heartbeat()
                    time.sleep(10)

                rc_a = proc_a.returncode
                rc_b = proc_b.returncode
                print(f"\n⏱️ Fallback GPUs finished (rc: {rc_a}, {rc_b})")
                sys.stdout.flush()

                if rc_a == 0:
                    challengers.append(("Challenger A (GPU:0 fallback)", CHALLENGER_A_PATH))
                else:
                    print(f"  ⚠️ Fallback Challenger A failed (exit code {rc_a})")
                if rc_b == 0:
                    challengers.append(("Challenger B (GPU:1 fallback)", CHALLENGER_B_PATH))
                else:
                    print(f"  ⚠️ Fallback Challenger B failed (exit code {rc_b})")

            if not challengers:
                msg_err = "  ❌ Both challengers failed after fallback! Retrying in 30s..."
                print(msg_err)
                send_telegram_message(msg_err)
                time.sleep(30)
                continue
        else:
            print(f"\n🏋️ Single-GPU Mode: Training 1 challenger...")
            proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a, champ_pnl=current_champ_score, mirror=mirror_mode)
            rc = proc.wait()
            if rc != 0 and FALLBACK_ENVS_PER_GPU < ENVS_PER_GPU:
                print(f"  ⚠️ Challenger failed at {ENVS_PER_GPU} envs. Retrying once at {FALLBACK_ENVS_PER_GPU} envs...")
                proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a + 101, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU)
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
            heartbeat()

            # === MULTIVERSE EVALUATION (V45) ===
            all_data = load_tensor_data("cpu", split="all")
            heartbeat()
            n_candles = all_data['features'].shape[0]
            window_size = 14 * 288
            max_start = n_candles - window_size - 500
            
            windows = []
            for _ in range(7):
                start_idx = np.random.randint(264, max_start)
                end_idx = start_idx + window_size
                windows.append({
                    'features': all_data['features'][start_idx:end_idx].numpy(),
                    'close': all_data['close'][start_idx:end_idx].numpy()
                })

            # Evaluate Champion (Walk-Forward robust)
            print(f"\n  📊 Evaluating Champion (7 Multiverse Windows)...")
            heartbeat()
            champ_score, champ_dd, champ_actions = evaluate_model(CHAMPION_PATH, windows)
            
            # Update the tracked champion score for the next iteration's Smart Entropy
            if champ_score != -np.inf:
                current_champ_score = champ_score - 20.0

            # Evaluate all challengers
            best_chall_score = -np.inf
            best_chall_path = None
            best_chall_name = None
            best_chall_dd = 0.0
            best_chall_actions = {}
            # === SURVIVOR UTILITY (V46.0) ===
            def get_utility(final_balance, max_dd):
                initial_balance = 20.0
                net_profit = final_balance - initial_balance

                if final_balance < MIN_FINAL_BALANCE:
                    return -10.0 * np.log1p(max(0.1, initial_balance - final_balance))

                utility = net_profit * np.power(max(1.0 - max_dd, 0.001), 3.0)

                if max_dd > 0.45:
                    utility *= np.exp(-8.0 * (max_dd - 0.45))

                if max_dd > MAX_DD_THRESHOLD:
                    utility = -abs(utility)

                return utility

            # Evaluate all challengers
            best_chall_score = -np.inf
            best_chall_utility = -np.inf
            best_chall_path = None
            best_chall_name = None
            best_chall_dd = 0.0
            best_chall_actions = {}
            
            all_challenger_reports = ""

            for name, path in challengers:
                print(f"\n  📊 Evaluating {name} (7 Multiverse Windows)...")
                heartbeat()
                score, dd, chall_actions = evaluate_model(path, windows)
                
                util = get_utility(score, dd)
                net_profit = score - 20.0
                
                all_challenger_reports += f"⚔️ *{name}*\n"
                all_challenger_reports += f"Balance: ${score:.2f} | Net: ${net_profit:+.2f} | P95 DD: {dd*100:.1f}%\n"
                all_challenger_reports += f"Idle: {chall_actions.get(0,0)} | Long: {chall_actions.get(1,0)} | Short: {chall_actions.get(2,0)} | Close: {chall_actions.get(3,0)}\n\n"

                if util > best_chall_utility:
                    best_chall_utility = util
                    best_chall_score = score
                    best_chall_path = path
                    best_chall_name = name
                    best_chall_dd = dd
                    best_chall_actions = chall_actions
            
            msg = f"🔄 *Iteration {iteration} Complete*\n\n"
            msg += f"🏆 *Champion*\n"
            champ_net = champ_score - 20.0
            msg += f"Balance: ${champ_score:.2f} | Net: ${champ_net:+.2f} | P95 DD: {champ_dd*100:.1f}%\n"
            msg += f"Idle: {champ_actions.get(0,0)} | Long: {champ_actions.get(1,0)} | Short: {champ_actions.get(2,0)} | Close: {champ_actions.get(3,0)}\n\n"
            msg += all_challenger_reports
            msg += "📝 *Result:*\n"

            # === PROMOTION DECISION ===
            print(f"\n🏆 Champion:       ${champ_score:.2f} (P95 DD: {champ_dd*100:.1f}%)")
            print(f"⚔️  Best Challenger: ${best_chall_score:.2f} (P95 DD: {best_chall_dd*100:.1f}%) [{best_chall_name}]")
            print(f"📏 Survival Filter: Max DD allowed = {MAX_DD_THRESHOLD*100:.0f}%")

            champ_survives = (
                champ_dd <= MAX_DD_THRESHOLD and
                champ_score >= MIN_FINAL_BALANCE
            )
            challenger_survives = (
                best_chall_dd <= MAX_DD_THRESHOLD and
                best_chall_score >= MIN_FINAL_BALANCE
            )
            
            champ_utility = get_utility(champ_score, champ_dd)
            chall_utility = best_chall_utility
            promote = (
                challenger_survives and
                chall_utility > max(
                    champ_utility * PROMOTION_MARGIN,
                    champ_utility + 0.05,
                )
            )
            
            print(f"📊 Risk-Adjusted Utility: Champ={champ_utility:.2f} | Best Challenger={chall_utility:.2f}")

            # Tracking champion DD for Forced Retirement
            if champ_dd > 0.80:
                forced_retirement_counter += 1
            else:
                forced_retirement_counter = 0

            # === REGLA ESPECIAL: CHAMPION FORCED RETIREMENT ===
            if forced_retirement_counter >= 5 and challenger_survives and best_chall_path:
                print(f"👴 CHAMPION FORCED RETIREMENT! DD > 80% for 5+ iterations.")
                print(f"👑 CROWNING {best_chall_name} as new CLEAN baseline!")
                backup_path = f"{CHAMPION_PATH}.forced_retirement_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ Frozen legacy terminated. Evolution restored.")
                msg += f"👴 Champion Forced Retirement (DD > 80% 5x)\n👑 Crowning {best_chall_name}!"
                forced_retirement_counter = 0  # reset for next champion
            
            # === REGLA 0: DETHRONE INCONDICIONAL ===
            elif not champ_survives and challenger_survives and best_chall_path:
                print(f"💀 CHAMPION DETHRONED! Legacy baseline failed Survivor filter: ${champ_score:.2f}, DD {champ_dd*100:.1f}%.")
                print(f"👑 CROWNING {best_chall_name} as new CLEAN baseline!")
                backup_path = f"{CHAMPION_PATH}.kamikaze_banned_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New clean Champion crowned! Kamikaze lineage terminated.")
                msg += f"💀 Champion Dethroned (failed Survivor filter)\n👑 Crowning {best_chall_name}!"
            
            # === REGLA 1: PROMOTION POR UTILIDAD (EFICIENCIA) ===
            elif promote and best_chall_path:
                print(f"🚀 PROMOTION! {best_chall_name} defeats Champion on SURVIVOR utility!")
                print(f"   Utility: {chall_utility:.2f} > {champ_utility:.2f} ✅")
                print(f"   Balance: ${best_chall_score:.2f} vs ${champ_score:.2f}")
                print(f"   DD:      {best_chall_dd*100:.1f}% vs {champ_dd*100:.1f}%")
                
                backup_path = f"{CHAMPION_PATH}.backup_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New Efficiency Champion crowned! (Old backed up)")
                msg += f"🚀 PROMOTION! {best_chall_name} defeats Champion!\nUtil: {chall_utility:.2f} > {champ_utility:.2f}"
                
            # === REGLA 2: Bloqueo de Kamikazes retadores ===
            elif not challenger_survives and best_chall_score > champ_score:
                print(f"💀 BLOCKED! {best_chall_name} earned ${best_chall_score:.2f} with P95 DD {best_chall_dd*100:.1f}%, failing Survivor filter.")
                print(f"🛡️ Champion survives by survival filter.")
                msg += f"💀 BLOCKED! {best_chall_name} failed Survivor filter. Champion survives."
            
            # === REGLA 3: Baseline inicial si no hay campeón ===
            elif champ_score == -np.inf and best_chall_score > -np.inf and best_chall_path:
                if challenger_survives:
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
            heartbeat()

            # --- ONGOING LEARNING: 3-Tier Survivor Vault ---
            if best_chall_path and os.path.exists(best_chall_path):
                if best_chall_dd <= MAX_DD_THRESHOLD and best_chall_score >= MIN_FINAL_BALANCE:
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
        heartbeat()

        print(f"\n⏸️  Sleeping 10s before next iteration...")
        time.sleep(10)


if __name__ == "__main__":
    send_telegram_message("🟢 *Matrix Trainer [03-V30-Trainer] Iniciado / Reiniciado* 🚀\nIniciando entrenamiento continuo en las GPUs...")
    continuous_train()
