#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, current_roe
from aegis_alpha.tools.build_long_edge_candidate_dataset import (
    BASE_GUARD,
    MetaMarketData,
    candidate_features,
    compact_feature_names,
    load_meta_market,
)
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES, select_robust_windows


BENCHMARK_V036 = {
    "config_id": "loss7_pause48_pause2_48_maxday3_fee1x",
    "p25_pf": 0.8786,
    "worst_balance": 19.19,
    "worst_max_dd": 0.0929,
    "profitable_window_pct": 0.6667,
    "median_trades": 8.0,
}
BENCHMARK_V040_META_065 = {
    "threshold": 0.65,
    "p25_pf": 2.220350742340088,
    "worst_balance": 18.01103973388672,
    "worst_max_dd": 0.10444813221693039,
    "profitable_window_pct": 0.7916666666666666,
    "median_trades": 4.0,
}


@dataclass(frozen=True)
class AdaptiveVariant:
    name: str
    mode: str
    meta_soft_threshold: float = 0.0
    meta_defensive_threshold: float = 0.65
    defensive_after_loss_steps: int = 0
    defensive_drawdown: float = 0.0
    defensive_recent_losses: bool = False
    dynamic_sizing: bool = False


@dataclass
class OpenTrade:
    balance_before_open: float
    entry_notional: float
    entry_step: int
    entry_price: float
    entry_score: float
    entry_fee: float
    entry_regime: str
    meta_filter_prob: float
    position_fraction: float
    reduced_size: bool


VARIANTS: tuple[AdaptiveVariant, ...] = (
    AdaptiveVariant(name="A_loss_defensive_48", mode="after_loss", defensive_after_loss_steps=48),
    AdaptiveVariant(name="A_loss_defensive_72", mode="after_loss", defensive_after_loss_steps=72),
    AdaptiveVariant(name="B_drawdown_defensive_2pct", mode="drawdown", defensive_drawdown=0.02),
    AdaptiveVariant(name="B_drawdown_defensive_3pct", mode="drawdown", defensive_drawdown=0.03),
    AdaptiveVariant(name="C_two_losses_last5", mode="recent_losses", defensive_recent_losses=True),
    AdaptiveVariant(
        name="D_soft55_defensive65",
        mode="soft_defensive",
        meta_soft_threshold=0.55,
        meta_defensive_threshold=0.65,
        defensive_after_loss_steps=48,
        defensive_drawdown=0.02,
        defensive_recent_losses=True,
    ),
    AdaptiveVariant(name="E_dynamic_sizing", mode="dynamic_sizing", dynamic_sizing=True),
    AdaptiveVariant(
        name="D_plus_E_soft_dynamic",
        mode="soft_defensive_dynamic",
        meta_soft_threshold=0.55,
        meta_defensive_threshold=0.65,
        defensive_after_loss_steps=48,
        defensive_drawdown=0.02,
        defensive_recent_losses=True,
        dynamic_sizing=True,
    ),
)


def _day_key(step: int) -> int:
    return step // 288


def _predict_meta_prob(classifier: Any, market: MetaMarketData, step: int, gate_threshold: float) -> float:
    x = candidate_features(market, step, gate_threshold).reshape(1, -1)
    return float(classifier.predict_proba(x)[0, 1])


def _open_position(
    balance: float,
    price: float,
    step: int,
    market: MetaMarketData,
    position_fraction: float,
) -> tuple[float, Position, float]:
    notional = balance * market.cfg.risk.leverage * position_fraction
    fee = notional * market.cfg.risk.total_fee * BASE_GUARD["fee_multiplier"]
    if balance <= fee * 1.5:
        return balance, Position(), 0.0
    return balance - fee, Position(side=1, size=notional / max(price, 1e-10), entry_price=price, entry_step=step), fee


def _close_position(balance: float, position: Position, price: float, market: MetaMarketData) -> tuple[float, float, float]:
    pnl = abs(position.size) * (price - position.entry_price)
    fee = abs(position.size) * price * market.cfg.risk.total_fee * BASE_GUARD["fee_multiplier"]
    return max(0.0, balance + pnl - fee), pnl - fee, fee


def _trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def _close_trade(
    market: MetaMarketData,
    position: Position,
    balance: float,
    step: int,
    trade: OpenTrade,
    reason: str,
) -> tuple[float, dict[str, Any]]:
    price = float(market.close[step])
    new_balance, _, close_fee = _close_position(balance, position, price, market)
    net = new_balance - trade.balance_before_open
    trade_return = net / max(trade.entry_notional, 1e-10)
    mfe, mae = _trade_mfe_mae(market.close, trade.entry_step, step)
    return new_balance, {
        "entry_step": int(trade.entry_step),
        "exit_step": int(step),
        "entry_timestamp": str(market.timestamps[trade.entry_step]),
        "exit_timestamp": str(market.timestamps[step]),
        "entry_regime": trade.entry_regime,
        "entry_score": safe_float(trade.entry_score),
        "exit_score": safe_float(market.expected_long_return[step]),
        "meta_filter_prob": safe_float(trade.meta_filter_prob),
        "position_fraction": safe_float(trade.position_fraction),
        "reduced_size": bool(trade.reduced_size),
        "hold_steps": int(step - trade.entry_step),
        "net": safe_float(net),
        "return": safe_float(trade_return),
        "fees": safe_float(trade.entry_fee + close_fee),
        "mfe": safe_float(mfe),
        "mae": safe_float(mae),
        "reason": reason,
    }


def _recent_loss_defensive(recent_wins: deque[int]) -> bool:
    if len(recent_wins) < 5:
        return False
    return sum(1 for win in recent_wins if win == 0) >= 2


def _should_use_defensive_mode(
    variant: AdaptiveVariant,
    step: int,
    defensive_until: int,
    current_drawdown: float,
    recent_wins: deque[int],
) -> bool:
    if variant.mode == "after_loss":
        return step < defensive_until
    if variant.mode == "drawdown":
        return current_drawdown > variant.defensive_drawdown
    if variant.mode == "recent_losses":
        return _recent_loss_defensive(recent_wins)
    if variant.mode == "soft_defensive":
        return (
            step < defensive_until
            or current_drawdown > variant.defensive_drawdown
            or _recent_loss_defensive(recent_wins)
        )
    return False


def _entry_decision(
    variant: AdaptiveVariant,
    meta_prob: float,
    defensive_mode: bool,
    base_position_fraction: float,
) -> tuple[bool, float, bool, str]:
    if variant.mode == "soft_defensive_dynamic":
        threshold = variant.meta_defensive_threshold if defensive_mode else variant.meta_soft_threshold
        if meta_prob < threshold:
            return False, base_position_fraction, False, "meta_soft_dynamic"
        if meta_prob < variant.meta_defensive_threshold:
            return True, base_position_fraction * 0.5, True, "reduced_size"
        return True, base_position_fraction, False, "full_size"

    if variant.dynamic_sizing:
        if meta_prob >= 0.65:
            return True, base_position_fraction, False, "full_size"
        return True, base_position_fraction * 0.5, True, "reduced_size"

    if variant.mode == "soft_defensive":
        threshold = variant.meta_defensive_threshold if defensive_mode else variant.meta_soft_threshold
        if meta_prob >= threshold:
            return True, base_position_fraction, False, "meta_pass"
        return False, base_position_fraction, False, "meta_soft_defensive"

    if defensive_mode and meta_prob < variant.meta_defensive_threshold:
        return False, base_position_fraction, False, "meta_defensive"
    return True, base_position_fraction, False, "base_pass"


def _evaluate_window(
    market: MetaMarketData,
    classifier: Any,
    variant: AdaptiveVariant,
    gate_threshold: float,
    start_step: int,
    window_steps: int,
    source: str,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> dict[str, Any]:
    risk = market.cfg.risk
    initial_balance = risk.initial_balance
    loss_floor = initial_balance * (1.0 - BASE_GUARD["max_window_loss_pct"])
    balance = initial_balance
    position = Position()
    open_trade: OpenTrade | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    pause_until = -1
    consecutive_losses = 0
    defensive_until = -1
    recent_wins: deque[int] = deque(maxlen=5)
    trades_by_day: Counter[int] = Counter()
    guard_counts: Counter[str] = Counter()
    skipped_by_meta = 0
    reduced_size_trades = 0
    defensive_candidate_count = 0
    meta_candidate_count = 0
    exposure_steps = 0
    total_fees = 0.0
    equity_peak = initial_balance
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])
        regime = str(market.regimes[step])
        day = _day_key(step)
        current_equity = balance if position.side == 0 else balance + abs(position.size) * (price - position.entry_price)
        equity_peak = max(equity_peak, current_equity)
        current_drawdown = (equity_peak - current_equity) / max(equity_peak, 1e-10)

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
            if entry_signal and balance <= loss_floor:
                guard_counts["max_window_loss"] += 1
                flat_steps += 1
            elif entry_signal and step < pause_until:
                guard_counts["pause"] += 1
                flat_steps += 1
            elif entry_signal and trades_by_day[day] >= BASE_GUARD["max_trades_per_day"]:
                guard_counts["max_trades_per_day"] += 1
                flat_steps += 1
            elif entry_signal and regime not in ALLOWED_REGIMES:
                guard_counts["regime"] += 1
                flat_steps += 1
            elif entry_signal:
                meta_candidate_count += 1
                meta_prob = _predict_meta_prob(classifier, market, step, gate_threshold)
                defensive_mode = _should_use_defensive_mode(variant, step, defensive_until, current_drawdown, recent_wins)
                defensive_candidate_count += int(defensive_mode)
                should_open, position_fraction, reduced_size, decision_reason = _entry_decision(
                    variant, meta_prob, defensive_mode, risk.position_fraction
                )
                if not should_open:
                    skipped_by_meta += 1
                    guard_counts[decision_reason] += 1
                    flat_steps += 1
                    continue

                before = balance
                balance, position, fee = _open_position(balance, price, step, market, position_fraction)
                if position.side > 0:
                    total_fees += fee
                    trades_by_day[day] += 1
                    reduced_size_trades += int(reduced_size)
                    open_trade = OpenTrade(
                        balance_before_open=before,
                        entry_notional=abs(position.size) * position.entry_price,
                        entry_step=step,
                        entry_price=price,
                        entry_score=score,
                        entry_fee=fee,
                        entry_regime=regime,
                        meta_filter_prob=meta_prob,
                        position_fraction=position_fraction,
                        reduced_size=reduced_size,
                    )
                    hold_steps = 0
                    flat_steps = 0
                else:
                    guard_counts["open_failed"] += 1
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
                is_win = int(float(trade["net"]) > 0.0)
                recent_wins.append(is_win)
                if is_win == 0:
                    consecutive_losses += 1
                    pause_until = max(pause_until, step + BASE_GUARD["pause_after_loss_steps"])
                    if variant.defensive_after_loss_steps > 0:
                        defensive_until = max(defensive_until, step + variant.defensive_after_loss_steps)
                    if consecutive_losses >= 2:
                        pause_until = max(pause_until, step + BASE_GUARD["pause_after_2_losses_steps"])
                else:
                    consecutive_losses = 0
                position = Position()
                open_trade = None
                hold_steps = 0
                flat_steps = 0
            else:
                hold_steps += 1
                flat_steps = 0

        equity = balance if position.side == 0 else balance + abs(position.size) * (price - position.entry_price)
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
        final_balance = initial_balance
    returns = np.asarray([trade["return"] for trade in trades], dtype=np.float32)
    wins = returns[returns > 0.0]
    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "balance": safe_float(final_balance),
        "net": safe_float(final_balance - initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(len(wins) / max(len(returns), 1)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "fees": safe_float(total_fees),
        "exposure_time": safe_float(exposure_steps / max(window_steps, 1)),
        "trades_per_month": safe_float(len(trades) / max(window_steps * 5.0 / 60.0 / 24.0 / 30.4375, 1e-10)),
        "skipped_by_meta": int(skipped_by_meta),
        "reduced_size_trades": int(reduced_size_trades),
        "meta_candidate_count": int(meta_candidate_count),
        "defensive_candidate_count": int(defensive_candidate_count),
        "skipped_by_guard": int(sum(guard_counts.values())),
        "guard_counts": dict(guard_counts),
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
        "avg_mfe": safe_float(np.mean([trade["mfe"] for trade in trades])) if trades else 0.0,
        "avg_mae": safe_float(np.mean([trade["mae"] for trade in trades])) if trades else 0.0,
        "trades_detail": trades,
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
        "median_trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "median_exposure_time": safe_float(np.median(exposure)),
        "skipped_by_meta": int(sum(w["skipped_by_meta"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_guard": int(sum(w["skipped_by_guard"] for w in windows)),
        "meta_candidate_count": int(sum(w["meta_candidate_count"] for w in windows)),
        "defensive_candidate_count": int(sum(w["defensive_candidate_count"] for w in windows)),
    }


def _rank(variant_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in variant_reports:
        summary = report["summary"]
        rows.append(
            {
                "variant": report["variant"],
                "mode": report["config"]["mode"],
                **report["config"],
                **summary,
                "beats_v036_p25_pf": bool(summary["p25_pf"] > BENCHMARK_V036["p25_pf"]),
                "passes_success_criteria": bool(
                    summary["worst_balance"] >= 19.00
                    and summary["worst_max_dd"] <= 0.10
                    and summary["median_trades"] >= 6.0
                    and summary["profitable_window_pct"] >= 0.70
                    and summary["p25_pf"] > BENCHMARK_V036["p25_pf"]
                ),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["passes_success_criteria"]),
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
            float(row["profitable_window_pct"]),
            float(row["median_trades"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_adaptive_meta_eval(
    edge_model_path: Path,
    meta_filter_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    seed: int,
    target_max_windows: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    non_overlap_windows: int,
    gate_threshold: float | None,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> Path:
    market = load_meta_market(config_path, edge_model_path)
    valid_scores = market.expected_long_return[np.isfinite(market.expected_long_return)]
    threshold = float(np.quantile(valid_scores, 0.97)) if gate_threshold is None else gate_threshold
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
    bundle = load_model_bundle(meta_filter_path)
    expected_features = bundle.get("feature_names", [])
    if expected_features and list(expected_features) != compact_feature_names():
        raise RuntimeError("Meta-filter feature schema mismatch")
    classifier = bundle["classifier"]
    print(f"Selected windows: {len(windows)}")
    print(f"Gate threshold top 3%: {threshold:.8f}")

    variant_reports: list[dict[str, Any]] = []
    for variant in VARIANTS:
        evals = [
            _evaluate_window(
                market=market,
                classifier=classifier,
                variant=variant,
                gate_threshold=threshold,
                start_step=start_step,
                window_steps=window_steps,
                source=source,
                max_hold_steps=max_hold_steps,
                close_edge_threshold=close_edge_threshold,
                take_profit_roe=take_profit_roe,
            )
            for start_step, source in windows
        ]
        summary = _summary(evals, market.cfg.risk.initial_balance)
        config = {
            "mode": variant.mode,
            "meta_soft_threshold": variant.meta_soft_threshold,
            "meta_defensive_threshold": variant.meta_defensive_threshold,
            "defensive_after_loss_steps": variant.defensive_after_loss_steps,
            "defensive_drawdown": variant.defensive_drawdown,
            "defensive_recent_losses": variant.defensive_recent_losses,
            "dynamic_sizing": variant.dynamic_sizing,
        }
        variant_reports.append({"variant": variant.name, "config": config, "summary": summary, "windows": evals})
        print(
            f"{variant.name} p25pf={summary['p25_pf']:.2f} worst={summary['worst_balance']:.2f} "
            f"dd={summary['worst_max_dd']:.1%} prof={summary['profitable_window_pct']:.1%} "
            f"trades={summary['median_trades']:.1f} skip_meta={summary['skipped_by_meta']} "
            f"reduced={summary['reduced_size_trades']}"
        )

    ranking = _rank(variant_reports)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_adaptive_meta_v1",
        "created_at": created_at,
        "edge_model_path": str(edge_model_path),
        "meta_filter_path": str(meta_filter_path),
        "config_path": config_path,
        "policy": {
            "side": "LONG_ONLY",
            "entry_gate": "top_3pct_expected_return_long",
            "gate_threshold": threshold,
            "allowed_regimes": sorted(ALLOWED_REGIMES),
            "guard": BASE_GUARD,
            "max_hold_steps": max_hold_steps,
            "close_edge_threshold": close_edge_threshold,
            "take_profit_roe": take_profit_roe,
            "short_entries": False,
        },
        "window_count": len(windows),
        "window_steps": window_steps,
        "benchmarks": {
            "v036": BENCHMARK_V036,
            "v040_meta_065": BENCHMARK_V040_META_065,
        },
        "success_criteria": {
            "worst_balance": ">=19.00",
            "worst_max_dd": "<=10%",
            "median_trades": ">=6",
            "profitable_window_pct": ">=70%",
            "p25_pf": ">0.8786",
        },
        "passes_any": bool(any(row["passes_success_criteria"] for row in ranking)),
        "beats_v036_p25_pf_count": int(sum(row["beats_v036_p25_pf"] for row in ranking)),
        "best_variant": ranking[0] if ranking else None,
        "ranking": ranking,
        "variants": variant_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"long_edge_adaptive_meta_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    if ranking:
        best = ranking[0]
        print(
            f"Best {best['variant']} p25pf={best['p25_pf']:.2f} worst={best['worst_balance']:.2f} "
            f"dd={best['worst_max_dd']:.1%} prof={best['profitable_window_pct']:.1%} "
            f"trades={best['median_trades']:.1f} passes={best['passes_success_criteria']}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--meta-filter", default="aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default="aegis_alpha/logs/edge")
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--seed", type=int, default=4667)
    parser.add_argument("--target-max-windows", type=int, default=96)
    parser.add_argument("--recent-windows", type=int, default=16)
    parser.add_argument("--random-windows", type=int, default=24)
    parser.add_argument("--regime-windows-per-regime", type=int, default=6)
    parser.add_argument("--non-overlap-windows", type=int, default=48)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--max-hold-steps", type=int, default=24)
    parser.add_argument("--close-edge-threshold", type=float, default=0.0)
    parser.add_argument("--take-profit-roe", type=float, default=0.06)
    args = parser.parse_args()
    run_adaptive_meta_eval(
        edge_model_path=Path(args.edge_model),
        meta_filter_path=Path(args.meta_filter),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        seed=args.seed,
        target_max_windows=args.target_max_windows,
        recent_windows=args.recent_windows,
        random_windows=args.random_windows,
        regime_windows_per_regime=args.regime_windows_per_regime,
        non_overlap_windows=args.non_overlap_windows,
        gate_threshold=args.gate_threshold,
        max_hold_steps=args.max_hold_steps,
        close_edge_threshold=args.close_edge_threshold,
        take_profit_roe=args.take_profit_roe,
    )


if __name__ == "__main__":
    main()
