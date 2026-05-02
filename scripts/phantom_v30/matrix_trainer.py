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
import json
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
CURRENT_LEVERAGE = float(os.environ.get("PHANTOM_LEVERAGE", "5.0"))
LEVERAGE_LABEL = f"{CURRENT_LEVERAGE:g}x".replace(".", "p")
POSITION_FRACTION = float(os.environ.get("PHANTOM_POSITION_FRACTION", "0.25"))
HARD_STOP_ROE = float(os.environ.get("PHANTOM_HARD_STOP_ROE", "0.15"))
MIN_HOLD_STEPS = int(os.environ.get("PHANTOM_MIN_HOLD_STEPS", "6"))
MIN_FLAT_STEPS = int(os.environ.get("PHANTOM_MIN_FLAT_STEPS", "12"))
INVALID_ACTION_PENALTY = float(os.environ.get("PHANTOM_INVALID_ACTION_PENALTY", "-0.02"))
ENTRY_PENALTY = float(os.environ.get("PHANTOM_ENTRY_PENALTY", "0.04"))
IDLE_FLAT_BONUS = float(os.environ.get("PHANTOM_IDLE_FLAT_BONUS", "0.004"))
DIRECTION_DOMINANCE_LIMIT = float(os.environ.get("PHANTOM_DIRECTION_DOMINANCE_LIMIT", "0.95"))
DIRECTION_GATE_MIN_BALANCE = float(os.environ.get("PHANTOM_DIRECTION_GATE_MIN_BALANCE", "25.0"))
DIRECTION_GATE_MIN_SIGNALQ = int(os.environ.get("PHANTOM_DIRECTION_GATE_MIN_SIGNALQ", "1"))
CHALLENGER_A_SOURCE = os.environ.get("PHANTOM_CHALLENGER_A_SOURCE", "auto")
CHALLENGER_B_SOURCE = os.environ.get("PHANTOM_CHALLENGER_B_SOURCE", "bc")
CHAMPION_PATH = "models/phantom_v30_champion.zip"
CHALLENGER_A_PATH = "models/phantom_v30_challenger_a.zip"
CHALLENGER_B_PATH = "models/phantom_v30_challenger_b.zip"
LATEST_CHALLENGER_PATH = f"models/phantom_v31_latest_challenger_{LEVERAGE_LABEL}.zip"
SAFE_CHECKPOINT_PATH = f"models/phantom_v31_safe_checkpoint_{LEVERAGE_LABEL}.zip"  # Best survivor vault for this leverage stage
HEARTBEAT_PATH = Path("/home/jasan/Develop/trading_system/logs/training.log")
STATE_PATH = Path(f"/home/jasan/Develop/trading_system/logs/phantom_v30_trainer_state_{LEVERAGE_LABEL}.json")

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
EVAL_NUM_ENVS = max(1, int(os.environ.get("PHANTOM_EVAL_NUM_ENVS", "64")))
EVAL_MAX_STEPS = max(1, int(os.environ.get("PHANTOM_EVAL_MAX_STEPS", str(14 * 288))))
SIGNALQ_SAMPLE_EVERY = max(1, int(os.environ.get("PHANTOM_SIGNALQ_SAMPLE_EVERY", "24")))


def load_trainer_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load trainer state: {e}")
        return {}


def save_trainer_state(iteration: int, forced_retirement_counter: int, last_mirror_mode: int, current_champ_score: float):
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATE_PATH.with_suffix(".json.tmp")
        payload = {
            "iteration": int(iteration),
            "forced_retirement_counter": int(forced_retirement_counter),
            "last_mirror_mode": int(last_mirror_mode),
            "current_champ_score": float(current_champ_score),
            "updated_at": int(time.time()),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, STATE_PATH)
    except Exception as e:
        print(f"⚠️ Failed to save trainer state: {e}")


def heartbeat():
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_PATH.touch()
    except Exception as e:
        print(f"⚠️ Failed to touch heartbeat file: {e}")


def format_run_config(mirror_mode: int | None = None) -> str:
    mirror_part = ""
    if mirror_mode is not None:
        mirror_part = f" | Mirror: {'ON' if mirror_mode else 'OFF'}"
    return (
        f"• Leverage: *{LEVERAGE_LABEL}*{mirror_part}\n"
        f"• Position fraction: *{POSITION_FRACTION:.2f}* | Hard stop ROE: *{HARD_STOP_ROE*100:.0f}%*\n"
        f"• Cooldown: *hold ≥ {MIN_HOLD_STEPS}* | *flat ≥ {MIN_FLAT_STEPS}* | Invalid *{INVALID_ACTION_PENALTY:+.2f}*\n"
        f"• Scarcity: Entry *-{ENTRY_PENALTY:.2f}* | Idle flat *+{IDLE_FLAT_BONUS:.3f}*\n"
        f"• Direction gate: max side *≤ {DIRECTION_DOMINANCE_LIMIT*100:.0f}%* unless SignalQ + balance are strong\n"
        f"• Eval: *{EVAL_NUM_ENVS} envs x {EVAL_MAX_STEPS} steps*\n"
        f"• SignalQ: *cada {SIGNALQ_SAMPLE_EVERY} steps*\n"
        f"• Survivor: *DD ≤ {MAX_DD_THRESHOLD*100:.0f}%* + *Balance ≥ ${MIN_FINAL_BALANCE:.2f}*"
    )


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
        response = requests.post(url, json=payload, timeout=10)
        if not response.ok:
            print(f"Failed to send Telegram message: HTTP {response.status_code}: {response.text[:500]}")
    except Exception as e:
        safe_error = str(e).replace(bot_token, "<redacted>")
        print(f"Failed to send Telegram message: {safe_error}")


def empty_signal_quality() -> dict:
    return {
        "predictions": 0,
        "top_prob_sum": 0.0,
        "long_short_gap_sum": 0.0,
        "high_conf_signals": 0,
        "high_conf_wins": 0,
        "mfe_sum": {3: 0.0, 6: 0.0, 12: 0.0},
        "mae_sum": {3: 0.0, 6: 0.0, 12: 0.0},
    }


def empty_trade_metrics() -> dict:
    return {
        "opens": 0,
        "long_opens": 0,
        "short_opens": 0,
        "manual_closes": 0,
        "hard_stops": 0,
        "trailing_stops": 0,
        "bracket_closes": 0,
        "closed_trades": 0,
        "flips": 0,
        "liquidations": 0,
        "invalid_actions": 0,
        "fees": 0.0,
        "closed_hold_steps_sum": 0,
        "closed_hold_steps_count": 0,
        "opened_flat_steps_sum": 0,
        "opened_flat_steps_count": 0,
    }


def merge_trade_metrics(total: dict, part: dict):
    for key in ("opens", "long_opens", "short_opens", "manual_closes", "hard_stops", "trailing_stops", "bracket_closes",
                "closed_trades",
                "flips", "liquidations", "invalid_actions", "closed_hold_steps_sum", "closed_hold_steps_count",
                "opened_flat_steps_sum", "opened_flat_steps_count"):
        total[key] += part.get(key, 0)
    total["fees"] += part.get("fees", 0.0)


def collect_step_trade_metrics(infos: list) -> dict:
    metrics = empty_trade_metrics()
    for info in infos:
        if info.get("opened", False):
            metrics["opens"] += 1
            metrics["opened_flat_steps_sum"] += int(info.get("opened_flat_steps", 0))
            metrics["opened_flat_steps_count"] += 1
        if info.get("opened_long", False):
            metrics["long_opens"] += 1
        if info.get("opened_short", False):
            metrics["short_opens"] += 1
        if info.get("manual_closed", False):
            metrics["manual_closes"] += 1
        if info.get("hard_stop", False):
            metrics["hard_stops"] += 1
        if info.get("trailing_stop", False):
            metrics["trailing_stops"] += 1
        if info.get("bracket_closed", False):
            metrics["bracket_closes"] += 1
        if info.get("flipped", False):
            metrics["flips"] += 1
        if info.get("liquidated", False):
            metrics["liquidations"] += 1
        if info.get("invalid_action", False):
            metrics["invalid_actions"] += 1
        closed_event = (
            info.get("manual_closed", False) or
            info.get("bracket_closed", False) or
            info.get("flipped", False) or
            info.get("liquidated", False)
        )
        fees = float(info.get("fees", 0.0))
        metrics["fees"] += fees
        if closed_event:
            closed_hold_steps = int(info.get("closed_hold_steps", 0))
            metrics["closed_trades"] += 1
            metrics["closed_hold_steps_sum"] += closed_hold_steps
            metrics["closed_hold_steps_count"] += 1
    return metrics


def finalize_trade_metrics(raw: dict) -> dict:
    hold_count = max(raw.get("closed_hold_steps_count", 0), 1)
    flat_count = max(raw.get("opened_flat_steps_count", 0), 1)
    long_opens = raw.get("long_opens", 0)
    short_opens = raw.get("short_opens", 0)
    directional_opens = long_opens + short_opens
    direction_dominance = max(long_opens, short_opens) / directional_opens if directional_opens else 0.0
    return {
        "opens": raw.get("opens", 0),
        "long_opens": long_opens,
        "short_opens": short_opens,
        "direction_dominance": direction_dominance,
        "manual_closes": raw.get("manual_closes", 0),
        "hard_stops": raw.get("hard_stops", 0),
        "trailing_stops": raw.get("trailing_stops", 0),
        "bracket_closes": raw.get("bracket_closes", 0),
        "closed_trades": raw.get("closed_trades", 0),
        "flips": raw.get("flips", 0),
        "liquidations": raw.get("liquidations", 0),
        "invalid_actions": raw.get("invalid_actions", 0),
        "fees": raw.get("fees", 0.0),
        "avg_hold_steps": raw.get("closed_hold_steps_sum", 0) / hold_count if raw.get("closed_hold_steps_count", 0) else 0.0,
        "avg_flat_steps": raw.get("opened_flat_steps_sum", 0) / flat_count if raw.get("opened_flat_steps_count", 0) else 0.0,
    }


def merge_signal_quality(total: dict, part: dict):
    total["predictions"] += part.get("predictions", 0)
    total["top_prob_sum"] += part.get("top_prob_sum", 0.0)
    total["long_short_gap_sum"] += part.get("long_short_gap_sum", 0.0)
    total["high_conf_signals"] += part.get("high_conf_signals", 0)
    total["high_conf_wins"] += part.get("high_conf_wins", 0)
    for horizon in (3, 6, 12):
        total["mfe_sum"][horizon] += part.get("mfe_sum", {}).get(horizon, 0.0)
        total["mae_sum"][horizon] += part.get("mae_sum", {}).get(horizon, 0.0)


def finalize_signal_quality(raw: dict) -> dict:
    predictions = max(raw.get("predictions", 0), 1)
    signals = max(raw.get("high_conf_signals", 0), 1)
    return {
        "predictions_count": raw.get("predictions", 0),
        "top_prob_avg": raw.get("top_prob_sum", 0.0) / predictions,
        "signals_gt_65_pct": raw.get("high_conf_signals", 0) / predictions,
        "signals_gt_65_count": raw.get("high_conf_signals", 0),
        "signals_gt_65_wins": raw.get("high_conf_wins", 0),
        "win_rate_gt_65": raw.get("high_conf_wins", 0) / signals if raw.get("high_conf_signals", 0) else 0.0,
        "long_short_gap_avg": raw.get("long_short_gap_sum", 0.0) / predictions,
        "mfe_avg": {h: raw["mfe_sum"][h] / signals if raw.get("high_conf_signals", 0) else 0.0 for h in (3, 6, 12)},
        "mae_avg": {h: raw["mae_sum"][h] / signals if raw.get("high_conf_signals", 0) else 0.0 for h in (3, 6, 12)},
    }


def format_signal_quality(quality: dict) -> str:
    if not quality:
        return "SignalQ: n/a"
    mfe = quality["mfe_avg"]
    mae = quality["mae_avg"]
    return (
        f"SignalQ: top_prob {quality['top_prob_avg']*100:.1f}% | "
        f">65 {quality['signals_gt_65_pct']*100:.2f}% ({quality['signals_gt_65_count']}) | "
        f"win>65 {quality['win_rate_gt_65']*100:.1f}% | "
        f"gap {quality['long_short_gap_avg']*100:.1f}%\n"
        f"MFE/MAE 3: {mfe[3]*100:+.2f}%/{mae[3]*100:+.2f}% | "
        f"6: {mfe[6]*100:+.2f}%/{mae[6]*100:+.2f}% | "
        f"12: {mfe[12]*100:+.2f}%/{mae[12]*100:+.2f}%"
    )


def format_signal_quality_compact(quality: dict) -> str:
    if not quality:
        return "📡 SignalQ: n/a"

    text = (
        f"📡 SignalQ: Top *{quality['top_prob_avg']*100:.1f}%* | "
        f">65 *{quality['signals_gt_65_pct']*100:.2f}% ({quality['signals_gt_65_count']})* | "
        f"Gap *{quality['long_short_gap_avg']*100:.1f}%*"
    )

    if quality["signals_gt_65_count"] > 0:
        mfe = quality["mfe_avg"]
        mae = quality["mae_avg"]
        text += f"\n📈 MFE/MAE 12: *{mfe[12]*100:+.2f}%* / *{mae[12]*100:+.2f}%*"

    return text


def format_trade_metrics_compact(metrics: dict) -> str:
    if not metrics:
        return "🔁 Trade: n/a"
    return (
        f"🔁 Trade: O *{metrics.get('opens', 0):,}* | "
        f"L/S *{metrics.get('long_opens', 0):,}/{metrics.get('short_opens', 0):,}* | "
        f"Dom *{metrics.get('direction_dominance', 0.0)*100:.0f}%* | "
        f"MC *{metrics.get('manual_closes', 0):,}* | "
        f"HS *{metrics.get('hard_stops', 0):,}* | "
        f"FL *{metrics.get('flips', 0):,}* | "
        f"Inv *{metrics.get('invalid_actions', 0):,}* | "
        f"Hold *{metrics.get('avg_hold_steps', 0.0):.1f}* | "
        f"Flat *{metrics.get('avg_flat_steps', 0.0):.1f}* | "
        f"Fees *${metrics.get('fees', 0.0):.2f}*"
    )


def format_fighter_report(icon: str, name: str, balance: float, dd: float, actions: dict, signal_quality: dict, trade_metrics: dict | None = None, gpu_label: str | None = None) -> str:
    net = balance - 20.0
    title = f"{icon} *{name}*"
    if gpu_label:
        title += f" `{gpu_label}`"

    return (
        f"{title}\n"
        f"💰 Balance: *${balance:.2f}* | Net: *${net:+.2f}*\n"
        f"🛡️ P95 DD: *{dd*100:.1f}%*\n"
        f"🎮 Actions: I *{actions.get(0,0):,}* | "
        f"L *{actions.get(1,0):,}* | "
        f"S *{actions.get(2,0):,}* | "
        f"C *{actions.get(3,0):,}*\n"
        f"{format_trade_metrics_compact(trade_metrics or {})}\n"
        f"{format_signal_quality_compact(signal_quality)}"
    )


def parse_challenger_display(name: str) -> tuple[str, str | None]:
    if "GPU:0" in name:
        return "Challenger A", "GPU:0"
    if "GPU:1" in name:
        return "Challenger B", "GPU:1"
    return name, None


def get_policy_probs(model, obs):
    try:
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()
    except Exception as e:
        print(f"  ⚠️ Failed to collect signal probabilities: {e}")
        return None

    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return probs


def collect_signal_quality(probs, eval_env) -> dict:
    quality = empty_signal_quality()
    if probs is None:
        return quality
    top_actions = np.argmax(probs, axis=1)
    top_probs = np.max(probs, axis=1)
    long_short_gaps = np.abs(probs[:, 1] - probs[:, 2])
    quality["predictions"] += int(len(top_probs))
    quality["top_prob_sum"] += float(np.sum(top_probs))
    quality["long_short_gap_sum"] += float(np.sum(long_short_gaps))

    high_conf_trade = (top_probs >= 0.65) & ((top_actions == 1) | (top_actions == 2))
    if not np.any(high_conf_trade):
        return quality

    idx = np.clip(eval_env.current_steps, 0, eval_env.n_candles - 1)
    entry_prices = eval_env.close_prices[idx]
    safe_entry = np.maximum(entry_prices, 1e-10)
    directions = np.where(top_actions == 1, 1.0, -1.0)
    signal_idx = np.where(high_conf_trade)[0]
    quality["high_conf_signals"] += int(len(signal_idx))

    for horizon in (3, 6, 12):
        future_offsets = np.arange(1, horizon + 1)
        future_idx = np.clip(idx[:, None] + future_offsets[None, :], 0, eval_env.n_candles - 1)
        future_prices = eval_env.close_prices[future_idx]
        directional_returns = ((future_prices - safe_entry[:, None]) / safe_entry[:, None]) * directions[:, None]
        quality["mfe_sum"][horizon] += float(np.sum(np.max(directional_returns[signal_idx], axis=1)))
        quality["mae_sum"][horizon] += float(np.sum(np.min(directional_returns[signal_idx], axis=1)))
        if horizon == 12:
            quality["high_conf_wins"] += int(np.sum(directional_returns[signal_idx, -1] > 0))

    return quality


def evaluate_model_single(model_path: str, eval_env, seed: int = 42):
    """Single-seed evaluation. Returns (balance, p95_drawdown, action_counts, signal_quality)."""
    if not os.path.exists(model_path):
        return -np.inf, 0.0, {}, finalize_signal_quality(empty_signal_quality()), finalize_trade_metrics(empty_trade_metrics())

    try:
        model = PPO.load(model_path, env=eval_env, device="cpu")
    except Exception as e:
        print(f"  ⚠️ Failed to load {model_path}: {e}")
        return -np.inf, 0.0, {}, finalize_signal_quality(empty_signal_quality()), finalize_trade_metrics(empty_trade_metrics())

    np.random.seed(seed)

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    signal_quality = empty_signal_quality()
    trade_metrics = empty_trade_metrics()
    obs = eval_env.reset()
    peak_equities = np.full(eval_env.num_envs, float(eval_env.balances[0]))
    max_drawdowns = np.zeros(eval_env.num_envs, dtype=np.float32)
    
    done_all = np.zeros(eval_env.num_envs, dtype=bool)
    steps = 0

    while not done_all.all() and steps < EVAL_MAX_STEPS:
        if steps % SIGNALQ_SAMPLE_EVERY == 0:
            probs = get_policy_probs(model, obs)
            if probs is None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.argmax(probs, axis=1)
                merge_signal_quality(signal_quality, collect_signal_quality(probs, eval_env))
        else:
            action, _ = model.predict(obs, deterministic=True)
        action = np.atleast_1d(action)
        for a in action:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1
        obs, reward, done, infos = eval_env.step(action)
        merge_trade_metrics(trade_metrics, collect_step_trade_metrics(infos))
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

    return final_balance, p95_drawdown, action_counts, finalize_signal_quality(signal_quality), finalize_trade_metrics(trade_metrics)


def evaluate_model(model_path: str, windows: list, n_episodes: int = 1, seed: int = 42):
    """Walk-Forward Multiverse Evaluation. Evaluates over multiple random historical windows."""
    balances = []
    drawdowns = []
    total_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_signal_quality = empty_signal_quality()
    total_trade_metrics = empty_trade_metrics()
    
    for i, window in enumerate(windows):
        heartbeat()
        start_time = time.time()
        print(f"    Window {i + 1}/{len(windows)} starting...", flush=True)
        eval_env = PhantomMatrixEnv(
            features=window['features'],
            close_prices=window['close'],
            num_envs=EVAL_NUM_ENVS,
        )
        bal, dd, ac, sq, tm = evaluate_model_single(model_path, eval_env, seed=seed + i)
        eval_env.close()
        
        if bal == -np.inf:
            return -np.inf, 0.0, {}, finalize_signal_quality(empty_signal_quality()), finalize_trade_metrics(empty_trade_metrics())
            
        balances.append(bal)
        drawdowns.append(dd)
        for k, v in ac.items():
            total_action_counts[k] = total_action_counts.get(k, 0) + v
        merge_trade_metrics(total_trade_metrics, {
            "opens": tm["opens"],
            "long_opens": tm["long_opens"],
            "short_opens": tm["short_opens"],
            "manual_closes": tm["manual_closes"],
            "hard_stops": tm["hard_stops"],
            "trailing_stops": tm["trailing_stops"],
            "bracket_closes": tm["bracket_closes"],
            "closed_trades": tm["closed_trades"],
            "flips": tm["flips"],
            "liquidations": tm["liquidations"],
            "invalid_actions": tm["invalid_actions"],
            "fees": tm["fees"],
            "closed_hold_steps_sum": tm["avg_hold_steps"] * tm["closed_trades"],
            "closed_hold_steps_count": tm["closed_trades"],
            "opened_flat_steps_sum": tm["avg_flat_steps"] * tm["opens"],
            "opened_flat_steps_count": tm["opens"],
        })
        # Rehydrate finalized per-window quality into weighted raw totals.
        window_predictions = sq["predictions_count"]
        window_signals = sq["signals_gt_65_count"]
        total_signal_quality["predictions"] += window_predictions
        total_signal_quality["top_prob_sum"] += sq["top_prob_avg"] * window_predictions
        total_signal_quality["long_short_gap_sum"] += sq["long_short_gap_avg"] * window_predictions
        total_signal_quality["high_conf_signals"] += sq["signals_gt_65_count"]
        total_signal_quality["high_conf_wins"] += sq["signals_gt_65_wins"]
        for horizon in (3, 6, 12):
            total_signal_quality["mfe_sum"][horizon] += sq["mfe_avg"][horizon] * window_signals
            total_signal_quality["mae_sum"][horizon] += sq["mae_avg"][horizon] * window_signals
        signal_line = format_signal_quality(sq).splitlines()[0]
        elapsed = time.time() - start_time
        print(
            f"    Window {i + 1}/{len(windows)} done in {elapsed:.0f}s | "
            f"Balance ${bal:.2f} | DD {dd*100:.1f}% | "
            f"Opens {tm['opens']:,} | Manual closes {tm['manual_closes']:,} | "
            f"Invalid {tm['invalid_actions']:,} | Fees ${tm['fees']:.2f} | "
            f"{signal_line}",
            flush=True,
        )
            
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
    signal_quality = finalize_signal_quality(total_signal_quality)
    trade_metrics = finalize_trade_metrics(total_trade_metrics)
    print(
        f"  Trade: Opens={trade_metrics['opens']:,}, ManualCloses={trade_metrics['manual_closes']:,}, "
        f"Long/Short={trade_metrics['long_opens']:,}/{trade_metrics['short_opens']:,}, "
        f"Dominance={trade_metrics['direction_dominance']*100:.1f}%, "
        f"HardStops={trade_metrics['hard_stops']:,}, Flips={trade_metrics['flips']:,}, "
        f"Invalid={trade_metrics['invalid_actions']:,}, "
        f"AvgHold={trade_metrics['avg_hold_steps']:.1f}, AvgFlat={trade_metrics['avg_flat_steps']:.1f}, "
        f"Fees=${trade_metrics['fees']:.2f}"
    )
    print(f"  {format_signal_quality(signal_quality)}")

    return avg_balance, avg_dd, total_action_counts, signal_quality, trade_metrics


def launch_challenger(
    gpu_id: int,
    save_path: str,
    seed: int,
    d_model: int = 32,
    champ_pnl: float = 0.0,
    mirror: int = 0,
    num_envs: int = ENVS_PER_GPU,
    base_source: str = "auto",
) -> subprocess.Popen:
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
        "--base-source", base_source,
    ]

    print(f"  🚀 Launching Challenger on GPU {gpu_id} (PID isolation, envs={num_envs}, base={base_source})...")
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
    print(f"   Curriculum leverage: {CURRENT_LEVERAGE:g}x")
    print(f"   Position fraction: {POSITION_FRACTION:.2f} | Hard stop ROE: {HARD_STOP_ROE*100:.0f}%")
    print(f"   Challenger sources: A={CHALLENGER_A_SOURCE} | B={CHALLENGER_B_SOURCE}")
    print(f"   Latest lineage: {LATEST_CHALLENGER_PATH}")
    print(f"   Safe vault: {SAFE_CHECKPOINT_PATH}")
    print(f"   Eval envs: {EVAL_NUM_ENVS} | Eval steps: {EVAL_MAX_STEPS} | SignalQ sample every: {SIGNALQ_SAMPLE_EVERY} steps")

    state = load_trainer_state()
    iteration = int(state.get("iteration", 0))
    current_champ_score = float(state.get("current_champ_score", 0.0))  # Tracks champion score for smart entropy
    forced_retirement_counter = int(state.get("forced_retirement_counter", 0))
    last_mirror_mode = int(state.get("last_mirror_mode", 0))
    if state:
        print(
            f"♻️ Resuming trainer state: iteration={iteration}, "
            f"forced_retirement_counter={forced_retirement_counter}, "
            f"last_mirror_mode={last_mirror_mode}, current_champ_score={current_champ_score:.2f}"
        )

    while True:
        heartbeat()
        iteration += 1
        mirror_mode = int(iteration % 2 == 0)  # Par=1(Espejo), Impar=0(Real)
        save_trainer_state(iteration, forced_retirement_counter, mirror_mode, current_champ_score)
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
            proc_a = launch_challenger(0, CHALLENGER_A_PATH, seed_a, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, base_source=CHALLENGER_A_SOURCE)
            proc_b = launch_challenger(1, CHALLENGER_B_PATH, seed_b, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, base_source=CHALLENGER_B_SOURCE)

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
                send_telegram_message(
                    f"⚠️ *Trainer Warning* — ⚙️ *Curriculum {LEVERAGE_LABEL}*\n\n"
                    f"{format_run_config(mirror_mode)}\n\n"
                    f"Both challengers failed at {ENVS_PER_GPU} envs. Retrying at {FALLBACK_ENVS_PER_GPU} envs to avoid a full trainer restart."
                )

                proc_a = launch_challenger(0, CHALLENGER_A_PATH, seed_a + 101, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU, base_source=CHALLENGER_A_SOURCE)
                proc_b = launch_challenger(1, CHALLENGER_B_PATH, seed_b + 101, d_model=48, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU, base_source=CHALLENGER_B_SOURCE)
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
                send_telegram_message(
                    f"❌ *Trainer Warning* — ⚙️ *Curriculum {LEVERAGE_LABEL}*\n\n"
                    f"{format_run_config(mirror_mode)}\n\n"
                    f"{msg_err.strip()}"
                )
                time.sleep(30)
                continue
        else:
            print(f"\n🏋️ Single-GPU Mode: Training 1 challenger...")
            proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a, champ_pnl=current_champ_score, mirror=mirror_mode, base_source=CHALLENGER_A_SOURCE)
            rc = proc.wait()
            if rc != 0 and FALLBACK_ENVS_PER_GPU < ENVS_PER_GPU:
                print(f"  ⚠️ Challenger failed at {ENVS_PER_GPU} envs. Retrying once at {FALLBACK_ENVS_PER_GPU} envs...")
                proc = launch_challenger(0, CHALLENGER_A_PATH, seed_a + 101, champ_pnl=current_champ_score, mirror=mirror_mode, num_envs=FALLBACK_ENVS_PER_GPU, base_source=CHALLENGER_A_SOURCE)
                rc = proc.wait()
            if rc != 0:
                msg_err = f"  ❌ Challenger failed (exit code {rc}). Retrying in 30s..."
                print(msg_err)
                send_telegram_message(
                    f"❌ *Trainer Warning* — ⚙️ *Curriculum {LEVERAGE_LABEL}*\n\n"
                    f"{format_run_config(mirror_mode)}\n\n"
                    f"{msg_err.strip()}"
                )
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
            champ_score, champ_dd, champ_actions, champ_signal_quality, champ_trade_metrics = evaluate_model(CHAMPION_PATH, windows)
            
            # Update the tracked champion score for the next iteration's Smart Entropy
            if champ_score != -np.inf:
                current_champ_score = champ_score - 20.0
                save_trainer_state(iteration, forced_retirement_counter, mirror_mode, current_champ_score)

            # Evaluate all challengers
            best_chall_score = -np.inf
            best_chall_path = None
            best_chall_name = None
            best_chall_dd = 0.0
            best_chall_actions = {}
            best_chall_signal_quality = {}
            best_chall_trade_metrics = {}
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
            best_chall_trade_metrics = {}
            
            all_challenger_reports = ""

            for name, path in challengers:
                print(f"\n  📊 Evaluating {name} (7 Multiverse Windows)...")
                heartbeat()
                score, dd, chall_actions, chall_signal_quality, chall_trade_metrics = evaluate_model(path, windows)
                
                util = get_utility(score, dd)
                net_profit = score - 20.0
                
                display_name, gpu_label = parse_challenger_display(name)
                all_challenger_reports += format_fighter_report(
                    "⚔️",
                    display_name,
                    score,
                    dd,
                    chall_actions,
                    chall_signal_quality,
                    chall_trade_metrics,
                    gpu_label,
                )
                all_challenger_reports += "\n\n"

                if util > best_chall_utility:
                    best_chall_utility = util
                    best_chall_score = score
                    best_chall_path = path
                    best_chall_name = name
                    best_chall_dd = dd
                    best_chall_actions = chall_actions
                    best_chall_signal_quality = chall_signal_quality
                    best_chall_trade_metrics = chall_trade_metrics
            
            msg = f"🔄 *Iteration {iteration} Complete* — ⚙️ *Curriculum {LEVERAGE_LABEL}*\n\n"
            msg += "🧪 *Setup*\n"
            msg += f"{format_run_config(mirror_mode)}\n\n"
            msg += "━━━━━━━━━━━━━━━━━━\n\n"
            msg += format_fighter_report("🏆", "Champion", champ_score, champ_dd, champ_actions, champ_signal_quality, champ_trade_metrics)
            msg += "\n\n"
            msg += all_challenger_reports
            msg += "━━━━━━━━━━━━━━━━━━\n\n"
            msg += "📝 *Result*\n"

            # === PROMOTION DECISION ===
            print(f"\n🏆 Champion:       ${champ_score:.2f} (P95 DD: {champ_dd*100:.1f}%)")
            print(f"⚔️  Best Challenger: ${best_chall_score:.2f} (P95 DD: {best_chall_dd*100:.1f}%) [{best_chall_name}]")
            print(f"🏆 Champion {format_signal_quality(champ_signal_quality)}")
            print(f"⚔️  Best Challenger {format_signal_quality(best_chall_signal_quality)}")
            print(f"🏆 Champion {format_trade_metrics_compact(champ_trade_metrics)}")
            print(f"⚔️  Best Challenger {format_trade_metrics_compact(best_chall_trade_metrics)}")
            print(f"📏 Survival Filter: Max DD allowed = {MAX_DD_THRESHOLD*100:.0f}%")

            champ_survives = (
                champ_dd <= MAX_DD_THRESHOLD and
                champ_score >= MIN_FINAL_BALANCE
            )
            challenger_survives = (
                best_chall_dd <= MAX_DD_THRESHOLD and
                best_chall_score >= MIN_FINAL_BALANCE
            )
            challenger_directional = best_chall_trade_metrics.get("opens", 0) > 0
            challenger_direction_dominance = best_chall_trade_metrics.get("direction_dominance", 0.0)
            challenger_direction_gate = (
                not challenger_directional or
                challenger_direction_dominance <= DIRECTION_DOMINANCE_LIMIT or
                (
                    best_chall_score >= DIRECTION_GATE_MIN_BALANCE and
                    best_chall_signal_quality.get("signals_gt_65_count", 0) >= DIRECTION_GATE_MIN_SIGNALQ
                )
            )
            
            champ_utility = get_utility(champ_score, champ_dd)
            chall_utility = best_chall_utility
            promote = (
                challenger_survives and
                challenger_direction_gate and
                chall_utility > max(
                    champ_utility * PROMOTION_MARGIN,
                    champ_utility + 0.05,
                )
            )
            
            print(f"📊 Risk-Adjusted Utility: Champ={champ_utility:.2f} | Best Challenger={chall_utility:.2f}")
            if not challenger_direction_gate:
                print(
                    f"🧭 Direction gate blocked promotion: dominance={challenger_direction_dominance*100:.1f}% "
                    f"> {DIRECTION_DOMINANCE_LIMIT*100:.0f}% without strong SignalQ/balance."
                )

            # Tracking champion DD for Forced Retirement
            if champ_dd > 0.80:
                forced_retirement_counter += 1
            else:
                forced_retirement_counter = 0
            save_trainer_state(iteration, forced_retirement_counter, mirror_mode, current_champ_score)

            # === REGLA ESPECIAL: CHAMPION FORCED RETIREMENT ===
            if forced_retirement_counter >= 5 and challenger_survives and challenger_direction_gate and best_chall_path:
                print(f"👴 CHAMPION FORCED RETIREMENT! DD > 80% for 5+ iterations.")
                print(f"👑 CROWNING {best_chall_name} as new CLEAN baseline!")
                backup_path = f"{CHAMPION_PATH}.forced_retirement_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ Frozen legacy terminated. Evolution restored.")
                msg += f"👴 *FORCED RETIREMENT* — Champion DD stayed above 80% for 5+ iterations.\n"
                msg += f"👑 Crowning {best_chall_name} as new baseline."
                forced_retirement_counter = 0  # reset for next champion
                save_trainer_state(iteration, forced_retirement_counter, mirror_mode, current_champ_score)
            
            # === REGLA 0: DETHRONE INCONDICIONAL ===
            elif not champ_survives and challenger_survives and challenger_direction_gate and best_chall_path:
                print(f"💀 CHAMPION DETHRONED! Legacy baseline failed Survivor filter: ${champ_score:.2f}, DD {champ_dd*100:.1f}%.")
                print(f"👑 CROWNING {best_chall_name} as new CLEAN baseline!")
                backup_path = f"{CHAMPION_PATH}.kamikaze_banned_{int(time.time())}"
                if os.path.exists(CHAMPION_PATH):
                    os.rename(CHAMPION_PATH, backup_path)
                shutil.copy2(best_chall_path, CHAMPION_PATH)
                print(f"✅ New clean Champion crowned! Kamikaze lineage terminated.")
                msg += f"💀 *DETHRONED* — Champion failed Survivor filter.\n"
                msg += f"👑 Crowning {best_chall_name} as new baseline."
            
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
                msg += f"🚀 *PROMOTION* — {best_chall_name} becomes new Champion.\n"
                msg += f"Utility: *{chall_utility:.2f}* > *{champ_utility:.2f}*."
                
            # === REGLA 2: Bloqueo de Kamikazes retadores ===
            elif not challenger_survives and best_chall_score > champ_score:
                print(f"💀 BLOCKED! {best_chall_name} earned ${best_chall_score:.2f} with P95 DD {best_chall_dd*100:.1f}%, failing Survivor filter.")
                print(f"🛡️ Champion survives by survival filter.")
                msg += f"💀 *BLOCKED* — {best_chall_name} failed Survivor filter.\n"
                if best_chall_dd > MAX_DD_THRESHOLD:
                    msg += f"Reason: DD *{best_chall_dd*100:.1f}%* > *{MAX_DD_THRESHOLD*100:.0f}%* allowed.\n"
                if best_chall_score < MIN_FINAL_BALANCE:
                    msg += f"Reason: Balance *${best_chall_score:.2f}* < *${MIN_FINAL_BALANCE:.2f}* required.\n"
                msg += "🛡️ Champion remains as temporary baseline."

            elif challenger_survives and not challenger_direction_gate and best_chall_score > champ_score:
                print(f"🧭 BLOCKED! {best_chall_name} passed Survivor but is directionally degenerate.")
                msg += f"🧭 *BLOCKED* — {best_chall_name} passed Survivor but failed direction gate.\n"
                msg += (
                    f"Reason: one side is *{challenger_direction_dominance*100:.1f}%* of executed entries "
                    f"(limit *{DIRECTION_DOMINANCE_LIMIT*100:.0f}%*) without strong SignalQ/balance.\n"
                )
                msg += "🛡️ Champion remains as temporary baseline."
            
            # === REGLA 3: Baseline inicial si no hay campeón ===
            elif champ_score == -np.inf and best_chall_score > -np.inf and best_chall_path:
                if challenger_survives and challenger_direction_gate:
                    print(f"🆕 No champion. Auto-promoting {best_chall_name} as founding baseline.")
                    shutil.copy2(best_chall_path, CHAMPION_PATH)
                    msg += f"🆕 *FOUNDING BASELINE* — Auto-promoting {best_chall_name}."
                else:
                    print(f"⚠️ No valid baseline found.")
                    msg += "⚠️ *NO VALID BASELINE* — No fighter passed Survivor."
            else:
                print(f"🛡️ DEFENSE! Champion retains title by efficiency.")
                msg += "🛡️ *DEFENSE* — Champion retains title by Survivor efficiency."

            send_telegram_message(msg)
            heartbeat()

            # --- ONGOING LEARNING: 3-Tier Survivor Vault ---
            if best_chall_path and os.path.exists(best_chall_path):
                if best_chall_dd <= MAX_DD_THRESHOLD and best_chall_score >= MIN_FINAL_BALANCE:
                    # ✅ SAFE: Save as ongoing baseline AND update the safe checkpoint vault
                    shutil.copy2(best_chall_path, LATEST_CHALLENGER_PATH)
                    shutil.copy2(best_chall_path, SAFE_CHECKPOINT_PATH)
                    print(f"📥 Saved {best_chall_name} (${best_chall_score:.2f} | DD: {best_chall_dd*100:.1f}%) as ongoing baseline + safe checkpoint.")
                else:
                    # 💀 SUICIDAL: Restore from safe checkpoint (preserves progress) or champion (last resort)
                    if os.path.exists(SAFE_CHECKPOINT_PATH):
                        shutil.copy2(SAFE_CHECKPOINT_PATH, LATEST_CHALLENGER_PATH)
                        print(f"💀 MUTATION REJECTED: {best_chall_name} ({best_chall_dd*100:.1f}% DD). Restoring from SAFE CHECKPOINT (preserving progress).")
                    else:
                        shutil.copy2(CHAMPION_PATH, LATEST_CHALLENGER_PATH)
                        print(f"� MUTATION REJECTED: {best_chall_name} ({best_chall_dd*100:.1f}% DD). No safe checkpoint found, restoring Champion baseline.")

            # Cleanup challenger files
            for _, path in challengers:
                if os.path.exists(path):
                    os.remove(path)

            save_trainer_state(iteration, forced_retirement_counter, mirror_mode, current_champ_score)

        except Exception as e:
            print(f"\n❌ COLISEUM ERROR: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Skipping evaluation, continuing to next iteration...")
            send_telegram_message(
                f"❌ *Trainer Error on Iteration {iteration}* — ⚙️ *Curriculum {LEVERAGE_LABEL}*\n\n"
                f"{format_run_config(mirror_mode)}\n\n"
                f"`{e}`"
            )

        sys.stdout.flush()

        # HEARTBEAT: Touch the log file so Watchdog knows we are alive
        heartbeat()

        print(f"\n⏸️  Sleeping 10s before next iteration...")
        time.sleep(10)


if __name__ == "__main__":
    send_telegram_message(
        f"🟢 *Matrix Trainer [03-V30-Trainer] Iniciado / Reiniciado* 🚀\n\n"
        f"🧪 *Setup*\n"
        f"{format_run_config()}\n\n"
        "Iniciando entrenamiento continuo en las GPUs..."
    )
    continuous_train()
