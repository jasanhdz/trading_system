#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, close_position, current_roe, open_position
from aegis_alpha.tools.evaluate_long_edge_gate import MarketData, _load_market, select_windows


REGIME_VARIANTS: dict[str, set[str]] = {
    "allow_all_except_trend_down": {"trend_up", "chop", "high_vol", "compression", "mixed"},
    "allow_trend_up_mixed_chop_high_vol": {"trend_up", "mixed", "chop", "high_vol"},
    "allow_mixed_chop_high_vol": {"mixed", "chop", "high_vol"},
    "allow_mixed_chop": {"mixed", "chop"},
}


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
    variant_name: str,
    allowed_regimes: set[str],
    gate_threshold: float,
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
    equity_curve: list[float] = []
    exposure_steps = 0
    total_fees = 0.0
    trades: list[dict[str, Any]] = []
    blocked_by_regime = Counter()
    allowed_signal_count = 0
    blocked_count = 0
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])
        regime = str(market.regimes[step])

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
            if entry_signal and regime not in allowed_regimes:
                blocked_count += 1
                blocked_by_regime[regime] += 1
                flat_steps += 1
            elif entry_signal:
                before = balance
                balance, position, fee = open_position(balance, 1, price, step, risk)
                if position.side > 0:
                    total_fees += fee
                    allowed_signal_count += 1
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
        "variant": variant_name,
        "allowed_regimes": sorted(allowed_regimes),
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
        "allowed_count": int(allowed_signal_count),
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
        "blocked_count": int(sum(w["blocked_count"] for w in windows)),
        "allowed_count": int(sum(w["allowed_count"] for w in windows)),
        "skipped_trend_down_count": int(sum(w["skipped_trend_down_count"] for w in windows)),
    }


def _rank(variant_reports: dict[str, dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for name, report in variant_reports.items():
        summary = report["summary"]
        rows.append(
            {
                "variant": name,
                "allowed_regimes": report["allowed_regimes"],
                **summary,
                "improves_baseline": bool(
                    summary["p25_pf"] > baseline["p25_pf"]
                    and summary["worst_balance"] > baseline["worst_balance"]
                    and summary["profitable_window_pct"] > baseline["profitable_window_pct"]
                ),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            float(row["profitable_window_pct"]),
            float(row["median_balance"]),
            -float(row["worst_max_dd"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_validation(
    model_path: Path,
    config_path: str,
    output_dir: Path,
    gate_threshold: float,
    window_steps: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    seed: int,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> Path:
    market = _load_market(config_path, model_path)
    selected = select_windows(
        market,
        window_steps=window_steps,
        recent_windows=recent_windows,
        random_windows=random_windows,
        regime_windows_per_regime=regime_windows_per_regime,
        seed=seed,
    )
    print(f"Selected windows: {len(selected)}")
    print(f"Gate threshold: {gate_threshold:.8f}")

    variant_reports: dict[str, dict[str, Any]] = {}
    for variant_name, allowed_regimes in REGIME_VARIANTS.items():
        windows = [
            _evaluate_window(
                market,
                variant_name,
                allowed_regimes,
                gate_threshold,
                start_step,
                window_steps,
                source,
                max_hold_steps,
                close_edge_threshold,
                take_profit_roe,
            )
            for start_step, source in selected
        ]
        summary = _summary(windows, market.cfg.risk.initial_balance)
        variant_reports[variant_name] = {
            "allowed_regimes": sorted(allowed_regimes),
            "summary": summary,
            "windows": windows,
        }
        print(
            f"{variant_name}: med={summary['median_balance']:.2f} p25pf={summary['p25_pf']:.2f} "
            f"worst={summary['worst_balance']:.2f} prof={summary['profitable_window_pct']:.1%} "
            f"trades={summary['median_trades']:.1f} blocked={summary['blocked_count']}"
        )

    baseline = {
        "p25_pf": 0.6945650577545166,
        "worst_balance": 17.904136657714844,
        "profitable_window_pct": 0.5714285714285714,
        "median_trades": 11.0,
    }
    ranking = _rank(variant_reports, baseline)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_regime_gate_v1",
        "created_at": created_at,
        "model_path": str(model_path),
        "config_path": config_path,
        "entry_gate": "top_2pct_expected_return_long",
        "gate_threshold": gate_threshold,
        "policy": {
            "side": "LONG_ONLY",
            "max_hold_steps": max_hold_steps,
            "close_edge_threshold": close_edge_threshold,
            "take_profit_roe": take_profit_roe,
            "short_entries": False,
        },
        "risk_budget": {
            "leverage": market.cfg.risk.leverage,
            "position_fraction": market.cfg.risk.position_fraction,
            "hard_stop_roe": market.cfg.risk.hard_stop_roe,
            "min_hold_steps": market.cfg.risk.min_hold_steps,
            "min_flat_steps": market.cfg.risk.min_flat_steps,
        },
        "window_steps": window_steps,
        "selection": {
            "recent_windows": recent_windows,
            "random_windows": random_windows,
            "regime_windows_per_regime": regime_windows_per_regime,
            "seed": seed,
        },
        "baseline_v031_top2": baseline,
        "variants": variant_reports,
        "ranking": ranking,
        "improver_count": int(sum(1 for row in ranking if row["improves_baseline"])),
        "best_variant": ranking[0] if ranking else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"long_edge_regime_gate_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    if ranking:
        best = ranking[0]
        print(
            f"Best: {best['variant']} p25pf={best['p25_pf']:.2f} worst={best['worst_balance']:.2f} "
            f"prof={best['profitable_window_pct']:.1%} trades={best['median_trades']:.1f} "
            f"improves={best['improves_baseline']}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default="aegis_alpha/logs/edge")
    parser.add_argument("--gate-threshold", type=float, default=0.00072011)
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--recent-windows", type=int, default=4)
    parser.add_argument("--random-windows", type=int, default=4)
    parser.add_argument("--regime-windows-per-regime", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4667)
    parser.add_argument("--max-hold-steps", type=int, default=24)
    parser.add_argument("--close-edge-threshold", type=float, default=0.0)
    parser.add_argument("--take-profit-roe", type=float, default=0.06)
    args = parser.parse_args()
    run_validation(
        model_path=Path(args.model),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        gate_threshold=args.gate_threshold,
        window_steps=args.window_steps,
        recent_windows=args.recent_windows,
        random_windows=args.random_windows,
        regime_windows_per_regime=args.regime_windows_per_regime,
        seed=args.seed,
        max_hold_steps=args.max_hold_steps,
        close_edge_threshold=args.close_edge_threshold,
        take_profit_roe=args.take_profit_roe,
    )


if __name__ == "__main__":
    main()
