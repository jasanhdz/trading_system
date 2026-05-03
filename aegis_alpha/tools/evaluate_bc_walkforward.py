#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from aegis_alpha.config import AegisConfig, load_config
from aegis_alpha.env.aegis_env import AegisEnv
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from data.storage.database_manager import DatabaseManager


@dataclass(frozen=True)
class MarketData:
    cfg: AegisConfig
    features: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray
    regimes: np.ndarray


ACTION_NAMES = ("IDLE", "LONG", "SHORT", "CLOSE")
REGIME_ORDER = ("trend_up", "trend_down", "chop", "compression", "high_vol", "mixed")


def _load_market(config_path: str) -> MarketData:
    cfg = load_config(config_path)
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    candles = db.get_ohlcv_data(symbol, cfg.timeframe)
    if candles.empty and symbol != cfg.symbol:
        candles = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    if candles.empty:
        raise RuntimeError(f"No candles found for {cfg.symbol} {cfg.timeframe}")

    frame = build_feature_frame(candles)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = frame["close"].values.astype(np.float32)
    timestamps = frame.index.astype(str).values
    regimes = _detect_regimes(features, cfg.model.window_size)
    return MarketData(cfg=cfg, features=features, close=close, timestamps=timestamps, regimes=regimes)


def _detect_regimes(features: np.ndarray, window_size: int) -> np.ndarray:
    regimes = np.empty((len(features),), dtype="U16")
    for idx in range(len(features)):
        start = max(0, idx - window_size + 1)
        regimes[idx] = detect_regime(features[start : idx + 1]).type
    return regimes


def _model_probs(model: PPO, obs: dict[str, np.ndarray]) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        return dist.distribution.probs.detach().cpu().numpy()[0]


def _dominant_regime(regimes: np.ndarray) -> str:
    if len(regimes) == 0:
        return "unknown"
    return Counter(regimes.tolist()).most_common(1)[0][0]


def _safe_float(value: float | np.floating) -> float:
    return float(np.asarray(value).item())


def _entry_quality(close: np.ndarray, entry_step: int, side: int, horizon: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    end = min(entry_step + horizon, len(close) - 1)
    future = close[entry_step + 1 : end + 1]
    if len(future) == 0 or entry_price <= 0:
        return 0.0, 0.0
    returns = future / entry_price - 1.0
    if side > 0:
        return float(np.max(returns)), float(max(0.0, -np.min(returns)))
    return float(max(0.0, -np.min(returns))), float(max(0.0, np.max(returns)))


def _trade_result(position, close_price: float, total_fee: float) -> dict[str, float]:
    entry_notional = abs(position.size) * position.entry_price
    close_notional = abs(position.size) * close_price
    gross = abs(position.size) * (
        close_price - position.entry_price if position.side > 0 else position.entry_price - close_price
    )
    fees = (entry_notional + close_notional) * total_fee
    net = gross - fees
    return {
        "net": float(net),
        "return": float(net / max(entry_notional, 1e-10)),
        "fees": float(fees),
    }


def _evaluate_window(
    model: PPO,
    market: MarketData,
    start_step: int,
    window_steps: int,
    source: str,
    deterministic: bool,
    entry_quality_horizon: int,
) -> dict[str, Any]:
    env = AegisEnv(market.features, market.close, risk=market.cfg.risk, window_size=market.cfg.model.window_size)
    obs = env.reset(start_step=start_step)
    done = False
    steps = 0
    equity_curve: list[float] = []
    top_probs: list[float] = []
    gaps: list[float] = []
    gt65 = 0
    action_counts = Counter()
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    trade_fees: list[float] = []
    entry_mfe: list[float] = []
    entry_mae: list[float] = []

    while not done and steps < window_steps:
        prev_position = env.position
        prev_closes = env.closes
        prev_step = env.step_idx
        prev_price = float(market.close[prev_step])
        action, _ = model.predict(obs, deterministic=deterministic)
        action_id = int(np.asarray(action).item())
        probs = _model_probs(model, obs)
        top_prob = float(np.max(probs))
        top_probs.append(top_prob)
        gaps.append(float(abs(probs[1] - probs[2])))
        gt65 += int(top_prob >= 0.65)
        action_counts[ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else "UNKNOWN"] += 1
        obs, _, done, info = env.step(action_id)
        if prev_position.side == 0 and env.position.side != 0:
            mfe, mae = _entry_quality(market.close, prev_step, env.position.side, entry_quality_horizon)
            entry_mfe.append(mfe)
            entry_mae.append(mae)
        if prev_position.side != 0 and env.closes > prev_closes:
            result = _trade_result(prev_position, prev_price, market.cfg.risk.total_fee)
            trade_returns.append(result["return"])
            trade_pnls.append(result["net"])
            trade_fees.append(result["fees"])
        equity_curve.append(float(info["equity"]))
        steps += 1

    end_step = min(env.step_idx, len(market.close) - 1)
    equity = np.asarray(equity_curve, dtype=np.float32)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        final_balance = float(equity[-1])
    else:
        dd = np.asarray([0.0], dtype=np.float32)
        final_balance = market.cfg.risk.initial_balance

    window_regimes = market.regimes[start_step:end_step]
    dominance = max(env.long_opens, env.short_opens) / max(env.opens, 1)
    wins = [pnl for pnl in trade_pnls if pnl > 0.0]
    losses = [pnl for pnl in trade_pnls if pnl < 0.0]
    gross_win = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(-np.sum(losses)) if losses else 0.0
    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "start_timestamp": str(market.timestamps[start_step]),
        "end_timestamp": str(market.timestamps[end_step]),
        "regime_dominant": _dominant_regime(window_regimes),
        "regime_distribution": dict(Counter(window_regimes.tolist())),
        "steps": int(steps),
        "balance": _safe_float(final_balance),
        "net": _safe_float(final_balance - market.cfg.risk.initial_balance),
        "p95_dd": _safe_float(np.quantile(dd, 0.95)),
        "max_dd": _safe_float(np.max(dd)),
        "opens": int(env.opens),
        "entry_count": int(env.opens),
        "long_opens": int(env.long_opens),
        "short_opens": int(env.short_opens),
        "manual_closes": int(env.closes),
        "invalid_actions": int(env.invalid_actions),
        "fees": _safe_float(env.total_fees),
        "fees_per_trade": _safe_float(env.total_fees / max(env.opens, 1)),
        "avg_return_per_trade": _safe_float(np.mean(trade_returns)) if trade_returns else 0.0,
        "avg_trade_pnl": _safe_float(np.mean(trade_pnls)) if trade_pnls else 0.0,
        "win_rate": _safe_float(len(wins) / max(len(trade_pnls), 1)),
        "profit_factor": _safe_float(gross_win / gross_loss) if gross_loss > 0.0 else (0.0 if gross_win == 0.0 else 999.0),
        "avg_win": _safe_float(np.mean(wins)) if wins else 0.0,
        "avg_loss": _safe_float(np.mean(losses)) if losses else 0.0,
        "entry_mfe_avg": _safe_float(np.mean(entry_mfe)) if entry_mfe else 0.0,
        "entry_mae_avg": _safe_float(np.mean(entry_mae)) if entry_mae else 0.0,
        "entry_mfe_median": _safe_float(np.median(entry_mfe)) if entry_mfe else 0.0,
        "entry_mae_median": _safe_float(np.median(entry_mae)) if entry_mae else 0.0,
        "entry_quality_horizon": int(entry_quality_horizon),
        "avg_hold_steps": _safe_float(env.hold_sum / max(env.step_count, 1)),
        "avg_flat_steps": _safe_float(env.flat_sum / max(env.step_count, 1)),
        "direction_dominance": _safe_float(dominance),
        "top_prob_avg": _safe_float(np.mean(top_probs)) if top_probs else 0.0,
        "signals_gt_65_pct": _safe_float(gt65 / max(len(top_probs), 1)),
        "long_short_gap_avg": _safe_float(np.mean(gaps)) if gaps else 0.0,
        "action_counts": dict(action_counts),
    }


def _add_window(windows: dict[int, set[str]], start: int, source: str, min_start: int, max_start: int) -> None:
    if min_start <= start <= max_start:
        windows[int(start)].add(source)


def select_windows(
    market: MarketData,
    window_steps: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    seed: int,
) -> list[tuple[int, str]]:
    min_start = market.cfg.model.window_size
    max_start = len(market.close) - window_steps - 1
    if max_start < min_start:
        raise RuntimeError(
            f"Not enough history for window_steps={window_steps}; available={len(market.close)}, min_start={min_start}"
        )

    windows: dict[int, set[str]] = defaultdict(set)

    for idx in range(recent_windows):
        _add_window(windows, max_start - idx * window_steps, "recent", min_start, max_start)

    rng = np.random.default_rng(seed)
    if random_windows > 0:
        candidates = np.arange(min_start, max_start + 1, dtype=np.int64)
        picks = rng.choice(candidates, size=min(random_windows, len(candidates)), replace=False)
        for pick in picks:
            _add_window(windows, int(pick), "random", min_start, max_start)

    for regime in REGIME_ORDER:
        idxs = np.flatnonzero(market.regimes[min_start:max_start + 1] == regime) + min_start
        if len(idxs) == 0:
            continue
        picks = rng.choice(idxs, size=min(regime_windows_per_regime, len(idxs)), replace=False)
        for pick in picks:
            _add_window(windows, int(pick), f"regime:{regime}", min_start, max_start)

    return [(start, "+".join(sorted(sources))) for start, sources in sorted(windows.items())]


def _summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if not windows:
        return {}
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    p95_dd = np.asarray([w["p95_dd"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    fees = np.asarray([w["fees"] for w in windows], dtype=np.float32)
    fees_per_trade = np.asarray([w["fees_per_trade"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    win_rates = np.asarray([w["win_rate"] for w in windows], dtype=np.float32)
    profit_factors = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    entry_counts = np.asarray([w["entry_count"] for w in windows], dtype=np.float32)
    entry_mfe = np.asarray([w["entry_mfe_avg"] for w in windows], dtype=np.float32)
    entry_mae = np.asarray([w["entry_mae_avg"] for w in windows], dtype=np.float32)
    dominance = np.asarray([w["direction_dominance"] for w in windows], dtype=np.float32)
    regimes = Counter(w["regime_dominant"] for w in windows)
    return {
        "window_count": len(windows),
        "median_balance": _safe_float(np.median(balances)),
        "p25_balance": _safe_float(np.quantile(balances, 0.25)),
        "worst_balance": _safe_float(np.min(balances)),
        "median_p95_dd": _safe_float(np.median(p95_dd)),
        "worst_max_dd": _safe_float(np.max(max_dd)),
        "median_fees": _safe_float(np.median(fees)),
        "median_fees_per_trade": _safe_float(np.median(fees_per_trade)),
        "median_entry_count": _safe_float(np.median(entry_counts)),
        "median_avg_return_per_trade": _safe_float(np.median(avg_returns)),
        "median_win_rate": _safe_float(np.median(win_rates)),
        "median_profit_factor": _safe_float(np.median(profit_factors)),
        "median_entry_mfe_avg": _safe_float(np.median(entry_mfe)),
        "median_entry_mae_avg": _safe_float(np.median(entry_mae)),
        "dominance_distribution": {
            "median": _safe_float(np.median(dominance)),
            "p25": _safe_float(np.quantile(dominance, 0.25)),
            "p75": _safe_float(np.quantile(dominance, 0.75)),
            "max": _safe_float(np.max(dominance)),
            "gt_80_pct": _safe_float(np.mean(dominance > 0.80)),
            "gt_95_pct": _safe_float(np.mean(dominance > 0.95)),
        },
        "regime_window_counts": dict(regimes),
    }


def run_walkforward(
    model_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    seed: int,
    deterministic: bool,
    entry_quality_horizon: int,
    report_name: str | None = None,
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"BC model not found: {model_path}")

    market = _load_market(config_path)
    model = PPO.load(str(model_path), device="cpu")
    selected = select_windows(
        market,
        window_steps=window_steps,
        recent_windows=recent_windows,
        random_windows=random_windows,
        regime_windows_per_regime=regime_windows_per_regime,
        seed=seed,
    )
    print(f"Selected windows: {len(selected)}")

    results = []
    for idx, (start_step, source) in enumerate(selected, start=1):
        metrics = _evaluate_window(
            model,
            market,
            start_step,
            window_steps,
            source,
            deterministic,
            entry_quality_horizon,
        )
        results.append(metrics)
        print(
            f"[{idx:02d}/{len(selected):02d}] {metrics['source']} {metrics['start_timestamp']} -> "
            f"{metrics['end_timestamp']} regime={metrics['regime_dominant']} "
            f"balance={metrics['balance']:.2f} p95_dd={metrics['p95_dd']:.2%} "
            f"opens={metrics['opens']} fees={metrics['fees']:.2f} "
            f"wr={metrics['win_rate']:.1%} pf={metrics['profit_factor']:.2f}"
        )

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_bc_walkforward_v1",
        "created_at": created_at,
        "model_path": str(model_path),
        "config_path": config_path,
        "symbol": market.cfg.symbol,
        "timeframe": market.cfg.timeframe,
        "window_steps": window_steps,
        "selection": {
            "recent_windows": recent_windows,
            "random_windows": random_windows,
            "regime_windows_per_regime": regime_windows_per_regime,
            "seed": seed,
        },
        "entry_quality_horizon": entry_quality_horizon,
        "summary": _summary(results),
        "windows": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    name = report_name or model_path.stem
    output_path = output_dir / f"bc_walkforward_{name}_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    return output_path


def _parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(spec)
    return path.stem, path


def _comparative_summary(report_paths: list[Path], output_dir: Path) -> Path:
    rows = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report["summary"]
        rows.append(
            {
                "model": Path(report["model_path"]).stem,
                "report_path": str(path),
                "median_balance": summary.get("median_balance", 0.0),
                "p25_balance": summary.get("p25_balance", 0.0),
                "worst_balance": summary.get("worst_balance", 0.0),
                "median_p95_dd": summary.get("median_p95_dd", 0.0),
                "worst_max_dd": summary.get("worst_max_dd", 0.0),
                "median_fees": summary.get("median_fees", 0.0),
                "median_fees_per_trade": summary.get("median_fees_per_trade", 0.0),
                "dominance_median": summary.get("dominance_distribution", {}).get("median", 0.0),
                "dominance_gt_80_pct": summary.get("dominance_distribution", {}).get("gt_80_pct", 0.0),
                "median_entry_count": summary.get("median_entry_count", 0.0),
                "median_avg_return_per_trade": summary.get("median_avg_return_per_trade", 0.0),
                "median_win_rate": summary.get("median_win_rate", 0.0),
                "median_profit_factor": summary.get("median_profit_factor", 0.0),
                "median_entry_mfe_avg": summary.get("median_entry_mfe_avg", 0.0),
                "median_entry_mae_avg": summary.get("median_entry_mae_avg", 0.0),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["median_balance"]),
            float(row["p25_balance"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
            -float(row["median_fees"]),
            float(row["median_avg_return_per_trade"]),
        )

    ranked = sorted(rows, key=score, reverse=True)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"bc_walkforward_comparison_{created_at}.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "aegis_bc_walkforward_comparison_v1",
                "created_at": created_at,
                "ranking_basis": [
                    "median_balance",
                    "p25_balance",
                    "worst_balance",
                    "worst_max_dd",
                    "median_fees",
                    "median_avg_return_per_trade",
                ],
                "best_by_composite": ranked[0]["model"] if ranked else None,
                "models": ranked,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Comparison saved -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aegis_alpha/models/bc/aegis_bc_prudent.zip")
    parser.add_argument("--models", nargs="*", default=None, help="Optional list of name=path specs for comparative runs")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default="aegis_alpha/logs/coliseum")
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--recent-windows", type=int, default=4)
    parser.add_argument("--random-windows", type=int, default=4)
    parser.add_argument("--regime-windows-per-regime", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4667)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--entry-quality-horizon", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.models:
        report_paths = []
        for name, model_path in (_parse_model_spec(spec) for spec in args.models):
            report_paths.append(
                run_walkforward(
                    model_path=model_path,
                    config_path=args.config,
                    output_dir=output_dir,
                    window_steps=args.window_steps,
                    recent_windows=args.recent_windows,
                    random_windows=args.random_windows,
                    regime_windows_per_regime=args.regime_windows_per_regime,
                    seed=args.seed,
                    deterministic=not args.stochastic,
                    entry_quality_horizon=args.entry_quality_horizon,
                    report_name=name,
                )
            )
        _comparative_summary(report_paths, output_dir)
    else:
        run_walkforward(
            model_path=Path(args.model),
            config_path=args.config,
            output_dir=output_dir,
            window_steps=args.window_steps,
            recent_windows=args.recent_windows,
            random_windows=args.random_windows,
            regime_windows_per_regime=args.regime_windows_per_regime,
            seed=args.seed,
            deterministic=not args.stochastic,
            entry_quality_horizon=args.entry_quality_horizon,
        )


if __name__ == "__main__":
    main()
