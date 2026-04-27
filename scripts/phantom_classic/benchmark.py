#!/usr/bin/env python3
"""
Phantom Classic Challenge — Fair Benchmark
Evaluates RL Champion vs Classical Agent on identical market data using the same env.

Usage:
    python -m scripts.phantom_classic.benchmark

Output: side-by-side PnL, drawdown, action distribution, and trade stats.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.phantom_v30.tensor_loader import load_tensor_data
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_classic.classical_agent import ClassicalAgent

try:
    from stable_baselines3 import PPO
except ImportError:
    print("⚠️  SB3 not available — will only evaluate Classical Agent")
    PPO = None


# ─── Config ───
CHAMPION_PATH = "models/phantom_v30_champion.zip"
NUM_EVAL_ENVS = 64        # Parallel eval environments
EVAL_SEEDS = [42, 137, 256, 512, 1024]   # Multiple seeds for robust comparison


def evaluate_agent(agent, env, agent_name: str, seed: int = 42):
    """
    Run a full episode and collect:
      - final_balance (per env)
      - peak_drawdown (per env)
      - action_counts
    """
    np.random.seed(seed)
    obs = env.reset()

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    max_steps = env.n_candles - 2

    for step in range(max_steps):
        if hasattr(agent, 'predict'):
            actions, _ = agent.predict(obs, deterministic=True)
        else:
            actions = np.zeros(env.num_envs, dtype=np.int64)

        obs, rewards, dones, infos = env.step(actions)

        for a in actions:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1

    # ── Collect results ──
    balances = env.balances.copy()
    peak_equity = env.peak_equity.copy()
    drawdowns = (peak_equity - balances) / np.maximum(peak_equity, 1e-10)

    return {
        "name": agent_name,
        "seed": seed,
        "mean_balance": np.mean(balances),
        "median_balance": np.median(balances),
        "p25_balance": np.percentile(balances, 25),
        "p75_balance": np.percentile(balances, 75),
        "best_balance": np.max(balances),
        "worst_balance": np.min(balances),
        "mean_dd": np.mean(drawdowns) * 100,
        "p95_dd": np.percentile(drawdowns, 95) * 100,
        "action_counts": action_counts,
        "total_actions": sum(action_counts.values()),
    }


def print_comparison(rl_results, classic_results):
    """Pretty-print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("⚔️  PHANTOM CLASSIC CHALLENGE — RESULTS")
    print("=" * 70)

    # Aggregate across seeds
    for label, results_list in [("🤖 RL Champion", rl_results), ("📐 Classical Agent", classic_results)]:
        if not results_list:
            print(f"\n{label}: NO DATA")
            continue

        balances = [r["mean_balance"] for r in results_list]
        dds = [r["p95_dd"] for r in results_list]

        print(f"\n{label}")
        print(f"  {'Metric':<25} {'Mean':>10} {'Best':>10} {'Worst':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Balance (mean of envs)':<25} ${np.mean(balances):>9.2f} ${np.max(balances):>9.2f} ${np.min(balances):>9.2f}")
        print(f"  {'P95 Drawdown':<25} {np.mean(dds):>9.1f}% {np.min(dds):>9.1f}% {np.max(dds):>9.1f}%")

        # Average action distribution
        total_actions = sum(r["total_actions"] for r in results_list)
        avg_idle  = sum(r["action_counts"][0] for r in results_list) / total_actions * 100
        avg_long  = sum(r["action_counts"][1] for r in results_list) / total_actions * 100
        avg_short = sum(r["action_counts"][2] for r in results_list) / total_actions * 100
        avg_close = sum(r["action_counts"][3] for r in results_list) / total_actions * 100

        print(f"\n  Action Distribution (avg across {len(results_list)} seeds):")
        print(f"    Idle:  {avg_idle:5.1f}%  |  Long:  {avg_long:5.1f}%  |  Short: {avg_short:5.1f}%  |  Close: {avg_close:5.1f}%")

    # ── Winner ──
    if rl_results and classic_results:
        rl_avg = np.mean([r["mean_balance"] for r in rl_results])
        cl_avg = np.mean([r["mean_balance"] for r in classic_results])
        diff = rl_avg - cl_avg
        winner = "🤖 RL CHAMPION" if diff > 0 else "📐 CLASSICAL AGENT"
        print(f"\n{'=' * 70}")
        print(f"🏆 WINNER: {winner}  (${abs(diff):.2f} edge)")
        print(f"   RL: ${rl_avg:.2f}  vs  Classical: ${cl_avg:.2f}")
        print(f"{'=' * 70}")


def main():
    print("🔧 Loading market data...")
    data = load_tensor_data(split="val")
    features_np = data["features"].cpu().numpy()
    close_np = data["close"].cpu().numpy()

    rl_results = []
    classic_results = []

    for seed in EVAL_SEEDS:
        print(f"\n🎲 Seed {seed}:")

        # ── Evaluate Classical Agent ──
        env = PhantomMatrixEnv(
            features=features_np,
            close_prices=close_np,
            num_envs=NUM_EVAL_ENVS,
        )
        classic_agent = ClassicalAgent()
        result = evaluate_agent(classic_agent, env, "ClassicCVD_MTF", seed=seed)
        classic_results.append(result)
        print(f"  📐 Classical: ${result['mean_balance']:.2f} (P95 DD: {result['p95_dd']:.1f}%)")

        # ── Evaluate RL Champion ──
        if PPO is not None and os.path.exists(CHAMPION_PATH):
            env2 = PhantomMatrixEnv(
                features=features_np,
                close_prices=close_np,
                num_envs=NUM_EVAL_ENVS,
            )
            try:
                rl_model = PPO.load(CHAMPION_PATH, env=env2, device="cpu")
                result = evaluate_agent(rl_model, env2, "RL_Champion", seed=seed)
                rl_results.append(result)
                print(f"  🤖 RL Champ:  ${result['mean_balance']:.2f} (P95 DD: {result['p95_dd']:.1f}%)")
            except Exception as e:
                print(f"  ⚠️ RL Champion failed: {e}")
        else:
            print(f"  ⚠️ RL Champion not found at {CHAMPION_PATH}")

    print_comparison(rl_results, classic_results)


if __name__ == "__main__":
    main()
