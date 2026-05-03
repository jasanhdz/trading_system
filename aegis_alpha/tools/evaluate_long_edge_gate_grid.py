#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, close_position, current_roe, open_position
from aegis_alpha.tools.evaluate_long_edge_gate import MarketData, _load_market, select_windows


REGIME_ORDER = ("trend_up", "trend_down", "chop", "high_vol", "compression", "mixed")


@dataclass(frozen=True)
class ExitConfig:
    max_hold_steps: int
    take_profit: float
    stop_loss: float
    edge_name: str
    edge_threshold: float

    @property
    def key(self) -> str:
        tp = f"{self.take_profit * 100:.2f}".replace(".", "p")
        sl = f"{abs(self.stop_loss) * 100:.2f}".replace(".", "p")
        return f"mh{self.max_hold_steps}_tp{tp}_sl{sl}_{self.edge_name}"


@dataclass
class OpenTrade:
    balance_before_open: float
    entry_notional: float
    entry_step: int
    entry_price: float
    entry_score: float
    entry_fee: float
    entry_regime: str


def _trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def _close_trade(
    market: MarketData,
    position: Position,
    balance: float,
    step: int,
    trade: OpenTrade,
    reason: str,
) -> tuple[float, dict[str, Any]]:
    price = float(market.close[step])
    new_balance, _, close_fee = close_position(balance, position, price, market.cfg.risk)
    net = new_balance - trade.balance_before_open
    trade_return = net / max(trade.entry_notional, 1e-10)
    mfe, mae = _trade_mfe_mae(market.close, trade.entry_step, step)
    return new_balance, {
        "entry_step": int(trade.entry_step),
        "exit_step": int(step),
        "entry_regime": trade.entry_regime,
        "entry_score": safe_float(trade.entry_score),
        "exit_score": safe_float(market.expected_long_return[step]),
        "hold_steps": int(step - trade.entry_step),
        "net": safe_float(net),
        "return": safe_float(trade_return),
        "fees": safe_float(trade.entry_fee + close_fee),
        "mfe": safe_float(mfe),
        "mae": safe_float(mae),
        "reason": reason,
    }


def _dominant_regime(regimes: np.ndarray) -> str:
    if len(regimes) == 0:
        return "unknown"
    return Counter(regimes.tolist()).most_common(1)[0][0]


def _evaluate_window(
    market: MarketData,
    gate_threshold: float,
    exit_cfg: ExitConfig,
    start_step: int,
    window_steps: int,
    source: str,
) -> dict[str, Any]:
    risk = market.cfg.risk
    balance = risk.initial_balance
    position = Position()
    open_trade: OpenTrade | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    equity_curve: list[float] = []
    exposure_steps = 0
    total_fees = 0.0
    trades: list[dict[str, Any]] = []
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])

        if position.side == 0:
            if flat_steps >= risk.min_flat_steps and score >= gate_threshold:
                before = balance
                balance, position, fee = open_position(balance, 1, price, step, risk)
                if position.side > 0:
                    total_fees += fee
                    open_trade = OpenTrade(
                        balance_before_open=before,
                        entry_notional=abs(position.size) * position.entry_price,
                        entry_step=step,
                        entry_price=price,
                        entry_score=score,
                        entry_fee=fee,
                        entry_regime=str(market.regimes[step]),
                    )
                    hold_steps = 0
                    flat_steps = 0
                else:
                    flat_steps += 1
            else:
                flat_steps += 1
        else:
            exposure_steps += 1
            raw_return = (price / max(position.entry_price, 1e-10)) - 1.0
            roe = current_roe(position, price, risk)
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif raw_return <= exit_cfg.stop_loss:
                close_reason = "stop_loss"
            elif hold_steps >= risk.min_hold_steps and raw_return >= exit_cfg.take_profit:
                close_reason = "take_profit"
            elif hold_steps >= exit_cfg.max_hold_steps:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and score < exit_cfg.edge_threshold:
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                balance, trade = _close_trade(market, position, balance, step, open_trade, close_reason)
                total_fees += float(trade["fees"]) - open_trade.entry_fee
                trades.append(trade)
                position = Position()
                open_trade = None
                hold_steps = 0
                flat_steps = 0
            else:
                hold_steps += 1
                flat_steps = 0

        if position.side == 0:
            equity = balance
        else:
            pnl = abs(position.size) * (price - position.entry_price)
            equity = balance + pnl
        equity_curve.append(float(equity))

    if position.side != 0 and open_trade is not None:
        balance, trade = _close_trade(market, position, balance, end_limit, open_trade, "window_end")
        total_fees += float(trade["fees"]) - open_trade.entry_fee
        trades.append(trade)
        equity_curve.append(float(balance))

    equity = np.asarray(equity_curve, dtype=np.float32)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        final_balance = float(equity[-1])
    else:
        dd = np.asarray([0.0], dtype=np.float32)
        final_balance = risk.initial_balance

    returns = np.asarray([trade["return"] for trade in trades], dtype=np.float32)
    wins = returns[returns > 0.0]
    window_regimes = market.regimes[start_step:end_limit]
    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "start_timestamp": str(market.timestamps[start_step]),
        "end_timestamp": str(market.timestamps[end_limit]),
        "regime_dominant": _dominant_regime(window_regimes),
        "balance": safe_float(final_balance),
        "net": safe_float(final_balance - risk.initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(len(wins) / max(len(returns), 1)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "fees": safe_float(total_fees),
        "exposure_time": safe_float(exposure_steps / max(window_steps, 1)),
        "mfe_avg": safe_float(np.mean([trade["mfe"] for trade in trades])) if trades else 0.0,
        "mae_avg": safe_float(np.mean([trade["mae"] for trade in trades])) if trades else 0.0,
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
        "trades_detail": trades,
    }


def _summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    return {
        "median_balance": safe_float(np.median(balances)),
        "p25_balance": safe_float(np.quantile(balances, 0.25)),
        "worst_balance": safe_float(np.min(balances)),
        "median_pf": safe_float(np.median(pfs)),
        "p25_pf": safe_float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": safe_float(np.mean(balances > initial_balance)),
        "median_trades": safe_float(np.median(trades)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "median_exposure_time": safe_float(np.median(exposure)),
    }


def _regime_report(config_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in config_reports:
        for window in report["windows"]:
            for trade in window["trades_detail"]:
                by_regime[str(trade["entry_regime"])].append(trade)

    all_losses = sum(max(0.0, -float(trade["return"])) for trades in by_regime.values() for trade in trades)
    regimes: dict[str, dict[str, Any]] = {}
    for regime in REGIME_ORDER:
        trades = by_regime.get(regime, [])
        returns = np.asarray([trade["return"] for trade in trades], dtype=np.float32)
        losses = float(np.sum(np.maximum(0.0, -returns))) if len(returns) else 0.0
        regimes[regime] = {
            "trade_count": int(len(trades)),
            "avg_return": safe_float(np.mean(returns)) if len(returns) else 0.0,
            "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
            "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
            "drawdown_contribution": safe_float(losses / max(all_losses, 1e-10)),
            "loss_return_sum": safe_float(losses),
        }
    return regimes


def _rank_configs(config_reports: list[dict[str, Any]], baseline: dict[str, float], min_trade_ratio: float) -> list[dict[str, Any]]:
    min_trades = baseline["median_trades"] * min_trade_ratio
    rows = []
    for report in config_reports:
        summary = report["summary"]
        improves = (
            summary["p25_pf"] > baseline["p25_pf"]
            and summary["worst_balance"] > baseline["worst_balance"]
            and summary["profitable_window_pct"] > baseline["profitable_window_pct"]
            and summary["median_trades"] >= min_trades
        )
        row = {
            "config_id": report["config_id"],
            **report["config"],
            **summary,
            "improves_baseline": bool(improves),
        }
        rows.append(row)

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            float(row["profitable_window_pct"]),
            float(row["median_balance"]),
            -float(row["worst_max_dd"]),
            float(row["median_trades"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_grid(
    model_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    seed: int,
    min_trade_ratio: float,
) -> tuple[Path, Path]:
    market = _load_market(config_path, model_path)
    valid_scores = market.expected_long_return[np.isfinite(market.expected_long_return)]
    gate_threshold = float(np.quantile(valid_scores, 0.98))
    edge_thresholds = {
        "edge_lt_0": 0.0,
        "edge_lt_top5": float(np.quantile(valid_scores, 0.95)),
        "edge_lt_top10": float(np.quantile(valid_scores, 0.90)),
    }
    selected = select_windows(
        market,
        window_steps=window_steps,
        recent_windows=recent_windows,
        random_windows=random_windows,
        regime_windows_per_regime=regime_windows_per_regime,
        seed=seed,
    )
    print(f"Selected windows: {len(selected)}")
    print(f"Entry gate top 2% threshold: {gate_threshold:.8f}")
    print(f"Edge deterioration thresholds: {edge_thresholds}")

    configs = [
        ExitConfig(max_hold, take_profit, stop_loss, edge_name, edge_threshold)
        for max_hold, take_profit, stop_loss, (edge_name, edge_threshold) in product(
            (6, 12, 18, 24),
            (0.0015, 0.0025, 0.0035, 0.0050),
            (-0.0015, -0.0025, -0.0035),
            edge_thresholds.items(),
        )
    ]

    config_reports: list[dict[str, Any]] = []
    for idx, exit_cfg in enumerate(configs, start=1):
        windows = [
            _evaluate_window(market, gate_threshold, exit_cfg, start_step, window_steps, source)
            for start_step, source in selected
        ]
        summary = _summary(windows, market.cfg.risk.initial_balance)
        config_reports.append(
            {
                "config_id": exit_cfg.key,
                "config": {
                    "max_hold": exit_cfg.max_hold_steps,
                    "take_profit": exit_cfg.take_profit,
                    "stop_loss": exit_cfg.stop_loss,
                    "edge_deterioration": exit_cfg.edge_name,
                    "edge_threshold": exit_cfg.edge_threshold,
                },
                "summary": summary,
                "windows": windows,
            }
        )
        print(
            f"[{idx:03d}/{len(configs):03d}] {exit_cfg.key} "
            f"med={summary['median_balance']:.2f} p25pf={summary['p25_pf']:.2f} "
            f"worst={summary['worst_balance']:.2f} prof={summary['profitable_window_pct']:.1%} "
            f"trades={summary['median_trades']:.1f}"
        )

    baseline = {
        "p25_pf": 0.6945650577545166,
        "worst_balance": 17.904136657714844,
        "profitable_window_pct": 0.5714285714285714,
        "median_trades": 11.0,
    }
    ranking = _rank_configs(config_reports, baseline, min_trade_ratio)
    best = ranking[0] if ranking else None
    best_report = next((report for report in config_reports if report["config_id"] == best["config_id"]), None) if best else None
    regime_payload = {
        "schema_version": "aegis_long_edge_regime_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "best_config_id": best["config_id"] if best else None,
        "best_config": best_report["config"] if best_report else None,
        "regimes": _regime_report([best_report]) if best_report else {},
    }

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    grid_payload = {
        "schema_version": "aegis_long_edge_gate_grid_v1",
        "created_at": created_at,
        "model_path": str(model_path),
        "config_path": config_path,
        "entry_gate": "top_2pct_expected_return_long",
        "entry_gate_threshold": gate_threshold,
        "edge_deterioration_thresholds": edge_thresholds,
        "window_steps": window_steps,
        "selection": {
            "recent_windows": recent_windows,
            "random_windows": random_windows,
            "regime_windows_per_regime": regime_windows_per_regime,
            "seed": seed,
        },
        "baseline_v031_top2": baseline,
        "success_rule": {
            "improve_p25_pf": True,
            "improve_worst_balance": True,
            "improve_profitable_window_pct": True,
            "median_trades_min_ratio": min_trade_ratio,
        },
        "best_config_id": best["config_id"] if best else None,
        "best_config": best,
        "ranking": ranking,
        "configs": config_reports,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"long_edge_gate_grid_{created_at}.json"
    regime_path = output_dir / f"long_edge_regime_report_{created_at}.json"
    grid_path.write_text(json.dumps(grid_payload, indent=2, sort_keys=True), encoding="utf-8")
    regime_path.write_text(json.dumps(regime_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Grid report saved -> {grid_path}")
    print(f"Regime report saved -> {regime_path}")
    if best:
        print(
            f"Best: {best['config_id']} median={best['median_balance']:.2f} "
            f"p25pf={best['p25_pf']:.2f} worst={best['worst_balance']:.2f} "
            f"prof={best['profitable_window_pct']:.1%} trades={best['median_trades']:.1f}"
        )
    return grid_path, regime_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default="aegis_alpha/logs/edge")
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--recent-windows", type=int, default=4)
    parser.add_argument("--random-windows", type=int, default=4)
    parser.add_argument("--regime-windows-per-regime", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4667)
    parser.add_argument("--min-trade-ratio", type=float, default=0.60)
    args = parser.parse_args()
    run_grid(
        model_path=Path(args.model),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        recent_windows=args.recent_windows,
        random_windows=args.random_windows,
        regime_windows_per_regime=args.regime_windows_per_regime,
        seed=args.seed,
        min_trade_ratio=args.min_trade_ratio,
    )


if __name__ == "__main__":
    main()
