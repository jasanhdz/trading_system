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

from aegis_alpha.edge.common import profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, current_roe, position_notional
from aegis_alpha.tools.evaluate_long_edge_gate import MarketData, _load_market


ALLOWED_REGIMES = {"mixed", "chop", "high_vol"}
THRESHOLD_PCTS = (0.015, 0.020, 0.025, 0.030)
COST_MULTIPLIERS = (1.0, 1.5, 2.0)


@dataclass
class OpenTrade:
    balance_before_open: float
    entry_notional: float
    entry_step: int
    entry_price: float
    entry_score: float
    entry_fee: float
    entry_regime: str


def _open_position_with_fee_multiplier(
    balance: float,
    price: float,
    step: int,
    market: MarketData,
    fee_multiplier: float,
) -> tuple[float, Position, float]:
    notional = position_notional(balance, market.cfg.risk)
    fee = notional * market.cfg.risk.total_fee * fee_multiplier
    if balance <= fee * 1.5:
        return balance, Position(), 0.0
    size = notional / max(price, 1e-10)
    return balance - fee, Position(side=1, size=size, entry_price=price, entry_step=step), fee


def _close_position_with_fee_multiplier(
    balance: float,
    position: Position,
    price: float,
    market: MarketData,
    fee_multiplier: float,
) -> tuple[float, float, float]:
    if position.side == 0:
        return balance, 0.0, 0.0
    pnl = abs(position.size) * (price - position.entry_price)
    fee = abs(position.size) * price * market.cfg.risk.total_fee * fee_multiplier
    return max(0.0, balance + pnl - fee), pnl - fee, fee


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
    fee_multiplier: float,
) -> tuple[float, dict[str, Any]]:
    price = float(market.close[step])
    new_balance, _, close_fee = _close_position_with_fee_multiplier(balance, position, price, market, fee_multiplier)
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


def _add_window(windows: dict[int, set[str]], start: int, source: str, min_start: int, max_start: int) -> None:
    if min_start <= start <= max_start:
        windows[int(start)].add(source)


def select_robust_windows(
    market: MarketData,
    window_steps: int,
    seed: int,
    target_max: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    non_overlap_windows: int,
) -> list[tuple[int, str]]:
    min_start = market.cfg.model.window_size
    max_start = len(market.close) - window_steps - 1
    if max_start < min_start:
        raise RuntimeError(f"Not enough history for window_steps={window_steps}")

    windows: dict[int, set[str]] = defaultdict(set)

    non_overlap = np.arange(min_start, max_start + 1, window_steps, dtype=np.int64)
    if len(non_overlap) > non_overlap_windows:
        picks = np.linspace(0, len(non_overlap) - 1, non_overlap_windows).round().astype(int)
        non_overlap = non_overlap[picks]
    for start in non_overlap:
        _add_window(windows, int(start), "non_overlap", min_start, max_start)

    for idx in range(recent_windows):
        _add_window(windows, max_start - idx * window_steps, "recent", min_start, max_start)

    rng = np.random.default_rng(seed)
    candidates = np.arange(min_start, max_start + 1, dtype=np.int64)
    random_picks = rng.choice(candidates, size=min(random_windows, len(candidates)), replace=False)
    for start in random_picks:
        _add_window(windows, int(start), "random", min_start, max_start)

    for regime in ("trend_up", "trend_down", "chop", "compression", "high_vol", "mixed"):
        idxs = np.flatnonzero(market.regimes[min_start : max_start + 1] == regime) + min_start
        if len(idxs) == 0:
            continue
        picks = rng.choice(idxs, size=min(regime_windows_per_regime, len(idxs)), replace=False)
        for start in picks:
            _add_window(windows, int(start), f"regime:{regime}", min_start, max_start)

    items = [(start, "+".join(sorted(sources))) for start, sources in windows.items()]
    if len(items) > target_max:
        starts = np.asarray([item[0] for item in items], dtype=np.int64)
        keep = np.linspace(0, len(starts) - 1, target_max).round().astype(int)
        items = [sorted(items)[idx] for idx in keep]
    items = sorted(items)
    if len(items) < 50:
        raise RuntimeError(f"Robustness selection produced only {len(items)} windows; need >=50")
    return items


def _evaluate_window(
    market: MarketData,
    gate_threshold: float,
    fee_multiplier: float,
    start_step: int,
    window_steps: int,
    source: str,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> dict[str, Any]:
    risk = market.cfg.risk
    balance = risk.initial_balance
    position = Position()
    open_trade: OpenTrade | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    exposure_steps = 0
    total_fees = 0.0
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    blocked_by_regime = Counter()
    allowed_count = 0
    blocked_count = 0
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])
        regime = str(market.regimes[step])

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
            if entry_signal and regime not in ALLOWED_REGIMES:
                blocked_count += 1
                blocked_by_regime[regime] += 1
                flat_steps += 1
            elif entry_signal:
                before = balance
                balance, position, fee = _open_position_with_fee_multiplier(balance, price, step, market, fee_multiplier)
                if position.side > 0:
                    total_fees += fee
                    allowed_count += 1
                    open_trade = OpenTrade(
                        balance_before_open=before,
                        entry_notional=abs(position.size) * position.entry_price,
                        entry_step=step,
                        entry_price=price,
                        entry_score=score,
                        entry_fee=fee,
                        entry_regime=regime,
                    )
                    hold_steps = 0
                    flat_steps = 0
                else:
                    flat_steps += 1
            else:
                flat_steps += 1
        else:
            exposure_steps += 1
            roe = current_roe(position, price, risk)
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and roe >= take_profit_roe:
                close_reason = "take_profit"
            elif hold_steps >= max_hold_steps:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and score <= close_edge_threshold:
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                balance, trade = _close_trade(market, position, balance, step, open_trade, close_reason, fee_multiplier)
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
            equity = balance + abs(position.size) * (price - position.entry_price)
        equity_curve.append(float(equity))

    if position.side != 0 and open_trade is not None:
        balance, trade = _close_trade(market, position, balance, end_limit, open_trade, "window_end", fee_multiplier)
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
    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "start_timestamp": str(market.timestamps[start_step]),
        "end_timestamp": str(market.timestamps[end_limit]),
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
        "trades_per_month": safe_float(len(trades) / max(window_steps * 5.0 / 60.0 / 24.0 / 30.4375, 1e-10)),
        "allowed_count": int(allowed_count),
        "blocked_count": int(blocked_count),
        "skipped_trend_down_count": int(blocked_by_regime["trend_down"]),
        "blocked_by_regime": dict(blocked_by_regime),
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
    }


def _summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    trades_month = np.asarray([w["trades_per_month"] for w in windows], dtype=np.float32)
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
        "median_trades_per_month": safe_float(np.median(trades_month)),
        "blocked_count": int(sum(w["blocked_count"] for w in windows)),
        "allowed_count": int(sum(w["allowed_count"] for w in windows)),
        "skipped_trend_down_count": int(sum(w["skipped_trend_down_count"] for w in windows)),
    }


def _rank(config_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in config_reports:
        summary = report["summary"]
        rows.append(
            {
                "config_id": report["config_id"],
                **report["config"],
                **summary,
                "passes_success_criteria": bool(
                    summary["profitable_window_pct"] >= 0.65
                    and summary["p25_pf"] >= 1.0
                    and summary["worst_balance"] >= 19.0
                    and summary["worst_max_dd"] <= 0.12
                ),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["p25_pf"]),
            float(row["profitable_window_pct"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
            float(row["median_balance"]),
            float(row["median_trades"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_robustness(
    model_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    seed: int,
    target_max_windows: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    non_overlap_windows: int,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> Path:
    market = _load_market(config_path, model_path)
    valid_scores = market.expected_long_return[np.isfinite(market.expected_long_return)]
    thresholds = {f"top_{pct * 100:g}pct": float(np.quantile(valid_scores, 1.0 - pct)) for pct in THRESHOLD_PCTS}
    windows = select_robust_windows(
        market,
        window_steps=window_steps,
        seed=seed,
        target_max=target_max_windows,
        recent_windows=recent_windows,
        random_windows=random_windows,
        regime_windows_per_regime=regime_windows_per_regime,
        non_overlap_windows=non_overlap_windows,
    )
    print(f"Selected robustness windows: {len(windows)}")
    print(f"Thresholds: {thresholds}")

    config_reports: list[dict[str, Any]] = []
    for threshold_name, threshold in thresholds.items():
        for fee_multiplier in COST_MULTIPLIERS:
            evals = [
                _evaluate_window(
                    market,
                    threshold,
                    fee_multiplier,
                    start_step,
                    window_steps,
                    source,
                    max_hold_steps,
                    close_edge_threshold,
                    take_profit_roe,
                )
                for start_step, source in windows
            ]
            summary = _summary(evals, market.cfg.risk.initial_balance)
            config_id = f"{threshold_name}_fee{fee_multiplier:g}x"
            config_reports.append(
                {
                    "config_id": config_id,
                    "config": {
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        "fee_multiplier": fee_multiplier,
                        "allowed_regimes": sorted(ALLOWED_REGIMES),
                    },
                    "summary": summary,
                    "windows": evals,
                }
            )
            print(
                f"{config_id}: med={summary['median_balance']:.2f} p25pf={summary['p25_pf']:.2f} "
                f"worst={summary['worst_balance']:.2f} prof={summary['profitable_window_pct']:.1%} "
                f"dd={summary['worst_max_dd']:.1%} trades/mo={summary['median_trades_per_month']:.1f}"
            )

    ranking = _rank(config_reports)
    fee_15 = [row for row in ranking if abs(float(row["fee_multiplier"]) - 1.5) < 1e-9]
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_robustness_v1",
        "created_at": created_at,
        "model_path": str(model_path),
        "config_path": config_path,
        "policy": {
            "side": "LONG_ONLY",
            "allowed_regimes": sorted(ALLOWED_REGIMES),
            "max_hold_steps": max_hold_steps,
            "close_edge_threshold": close_edge_threshold,
            "take_profit_roe": take_profit_roe,
            "short_entries": False,
        },
        "thresholds": thresholds,
        "cost_multipliers": list(COST_MULTIPLIERS),
        "window_steps": window_steps,
        "window_count": len(windows),
        "window_selection": {
            "target_max_windows": target_max_windows,
            "recent_windows": recent_windows,
            "random_windows": random_windows,
            "regime_windows_per_regime": regime_windows_per_regime,
            "non_overlap_windows": non_overlap_windows,
            "seed": seed,
        },
        "success_criteria": {
            "window_count": ">=50",
            "profitable_window_pct": ">=65%",
            "p25_pf": ">=1.0",
            "worst_balance": ">=19.0",
            "worst_max_dd": "<=12%",
            "fee_1_5x_must_not_collapse": True,
        },
        "passes_any": bool(any(row["passes_success_criteria"] for row in ranking)),
        "passes_fee_1_5x": bool(any(row["passes_success_criteria"] for row in fee_15)),
        "best_config": ranking[0] if ranking else None,
        "best_fee_1_5x_config": fee_15[0] if fee_15 else None,
        "ranking": ranking,
        "configs": config_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"long_edge_robustness_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    if ranking:
        best = ranking[0]
        print(
            f"Best: {best['config_id']} p25pf={best['p25_pf']:.2f} "
            f"worst={best['worst_balance']:.2f} prof={best['profitable_window_pct']:.1%} "
            f"dd={best['worst_max_dd']:.1%} passes={best['passes_success_criteria']}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default="aegis_alpha/logs/edge")
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--seed", type=int, default=4667)
    parser.add_argument("--target-max-windows", type=int, default=96)
    parser.add_argument("--recent-windows", type=int, default=16)
    parser.add_argument("--random-windows", type=int, default=24)
    parser.add_argument("--regime-windows-per-regime", type=int, default=6)
    parser.add_argument("--non-overlap-windows", type=int, default=48)
    parser.add_argument("--max-hold-steps", type=int, default=24)
    parser.add_argument("--close-edge-threshold", type=float, default=0.0)
    parser.add_argument("--take-profit-roe", type=float, default=0.06)
    args = parser.parse_args()
    run_robustness(
        model_path=Path(args.model),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        seed=args.seed,
        target_max_windows=args.target_max_windows,
        recent_windows=args.recent_windows,
        random_windows=args.random_windows,
        regime_windows_per_regime=args.regime_windows_per_regime,
        non_overlap_windows=args.non_overlap_windows,
        max_hold_steps=args.max_hold_steps,
        close_edge_threshold=args.close_edge_threshold,
        take_profit_roe=args.take_profit_roe,
    )


if __name__ == "__main__":
    main()
