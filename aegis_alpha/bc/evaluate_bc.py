from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from aegis_alpha.config import load_config
from aegis_alpha.env.aegis_env import AegisEnv
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from data.storage.database_manager import DatabaseManager


def _load_market(config_path: str):
    cfg = load_config(config_path)
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    df = db.get_ohlcv_data(symbol, cfg.timeframe)
    if df.empty:
        df = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    frame = build_feature_frame(df)
    return cfg, frame[FEATURE_COLUMNS].values.astype(np.float32), frame["close"].values.astype(np.float32)


def _probs(model: PPO, obs: dict[str, np.ndarray]) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        return dist.distribution.probs.detach().cpu().numpy()[0]


def evaluate(model_path: Path, config_path: str, max_steps: int | None, deterministic: bool) -> dict[str, float | str]:
    cfg, features, close = _load_market(config_path)
    model = PPO.load(str(model_path), device="cpu")
    env = AegisEnv(features, close, risk=cfg.risk, window_size=cfg.model.window_size)
    obs = env.reset()
    done = False
    steps = 0
    equity_curve = []
    top_probs = []
    gaps = []
    gt65 = 0
    while not done and (max_steps is None or steps < max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        prob = _probs(model, obs)
        top_prob = float(np.max(prob))
        top_probs.append(top_prob)
        gaps.append(float(abs(prob[1] - prob[2])))
        gt65 += int(top_prob >= 0.65)
        obs, _, done, info = env.step(int(action))
        equity_curve.append(float(info["equity"]))
        steps += 1

    equity = np.array(equity_curve, dtype=np.float32)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        final_balance = float(equity[-1])
    else:
        dd = np.array([0.0], dtype=np.float32)
        final_balance = cfg.risk.initial_balance
    dominance = max(env.long_opens, env.short_opens) / max(env.opens, 1)
    return {
        "balance": final_balance,
        "net": final_balance - cfg.risk.initial_balance,
        "p95_dd": float(np.quantile(dd, 0.95)),
        "max_dd": float(np.max(dd)),
        "opens": float(env.opens),
        "long_opens": float(env.long_opens),
        "short_opens": float(env.short_opens),
        "manual_closes": float(env.closes),
        "invalid_actions": float(env.invalid_actions),
        "fees": float(env.total_fees),
        "avg_hold_steps": float(env.hold_sum / max(env.step_count, 1)),
        "avg_flat_steps": float(env.flat_sum / max(env.step_count, 1)),
        "direction_dominance": float(dominance),
        "top_prob_avg": float(np.mean(top_probs)) if top_probs else 0.0,
        "signals_gt_65_pct": float(gt65 / max(len(top_probs), 1)),
        "long_short_gap_avg": float(np.mean(gaps)) if gaps else 0.0,
        "last_regime": detect_regime(features[-64:]).type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aegis_alpha/models/bc/aegis_bc_prudent.zip")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--max-steps", type=int, default=4032)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(Path(args.model), args.config, args.max_steps, deterministic=not args.stochastic)
    print("Aegis BC Evaluation")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
