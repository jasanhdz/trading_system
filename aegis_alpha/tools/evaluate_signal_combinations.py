#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float  # noqa: E402
from aegis_alpha.env.risk_engine import Position, current_roe  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.build_long_edge_candidate_dataset import BASE_GUARD  # noqa: E402
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES, select_robust_windows  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_MODEL_DIR = Path("aegis_alpha/models/signals")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/signal_combination_report_v050.json")
DEFAULT_SEEDS = (8123, 9137)
WINDOW_STEPS = 4032
FULL_SIZE = 0.25
REDUCED_SIZE = 0.125
MAX_HOLD_STEPS = 24
TAKE_PROFIT_ROE = 0.06
EDGE_DETERIORATION_RATIO = 0.65


@dataclass(frozen=True)
class SignalRule:
    model_name: str
    pct: float
    high_is_good: bool = True


@dataclass(frozen=True)
class ComboSpec:
    name: str
    edge_rules: tuple[SignalRule, ...]
    risk_rules: tuple[SignalRule, ...] = ()
    allowed_regimes: tuple[str, ...] = ("mixed", "chop", "high_vol")


COMBOS: tuple[ComboSpec, ...] = (
    ComboSpec("A_long_edge_h12_top3", (SignalRule("long_edge_h12", 0.03),)),
    ComboSpec("B_long_edge_h24_top3", (SignalRule("long_edge_h24", 0.03),)),
    ComboSpec("C_long_edge_h48_top3", (SignalRule("long_edge_h48", 0.03),)),
    ComboSpec("D_long_h24_top3_risk_bottom50", (SignalRule("long_edge_h24", 0.03),), (SignalRule("long_failure_risk_h24", 0.50, high_is_good=False),)),
    ComboSpec("E_long_h24_top3_risk_bottom30", (SignalRule("long_edge_h24", 0.03),), (SignalRule("long_failure_risk_h24", 0.30, high_is_good=False),)),
    ComboSpec("F_long_h48_top3_risk_bottom50", (SignalRule("long_edge_h48", 0.03),), (SignalRule("long_failure_risk_h48", 0.50, high_is_good=False),)),
    ComboSpec("G_long_h48_top3_risk_bottom30", (SignalRule("long_edge_h48", 0.03),), (SignalRule("long_failure_risk_h48", 0.30, high_is_good=False),)),
    ComboSpec(
        "H_ensemble_h12h24_risk50",
        (SignalRule("long_edge_h12", 0.05), SignalRule("long_edge_h24", 0.05)),
        (SignalRule("long_failure_risk_h24", 0.50, high_is_good=False),),
    ),
)


def _trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def _open_position(balance: float, price: float, step: int, market: Any, fee_multiplier: float) -> tuple[float, Position, float]:
    notional = balance * market.cfg.risk.leverage * FULL_SIZE
    fee = notional * market.cfg.risk.total_fee * fee_multiplier
    if balance <= fee * 1.5:
        return balance, Position(), 0.0
    size = notional / max(price, 1e-10)
    return balance - fee, Position(side=1, size=size, entry_price=price, entry_step=step), fee


def _close_position(balance: float, position: Position, price: float, market: Any, fee_multiplier: float) -> tuple[float, float, float]:
    pnl = abs(position.size) * (price - position.entry_price)
    fee = abs(position.size) * price * market.cfg.risk.total_fee * fee_multiplier
    return max(0.0, balance + pnl - fee), pnl - fee, fee


def _load_models(model_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    required = {
        "long_edge_h12",
        "long_edge_h24",
        "long_edge_h48",
        "long_failure_risk_h24",
        "long_failure_risk_h48",
    }
    for name in required:
        bundle = load_model_bundle(model_dir / f"aegis_{name}_v050.joblib")
        estimator = bundle.get("estimator") or bundle.get("classifier") or bundle.get("regressor")
        if estimator is None:
            raise RuntimeError(f"Missing estimator for {name}")
        out[name] = {"bundle": bundle, "estimator": estimator}
    return out


def _predict_scores(market: Any, models: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    x = market.signal_features.astype(np.float32)
    for name, payload in models.items():
        est = payload["estimator"]
        if hasattr(est, "predict_proba"):
            scores = est.predict_proba(x)[:, 1]
        else:
            scores = est.predict(x)
        preds[name] = np.asarray(scores, dtype=np.float32)
    return preds


def _threshold(scores: np.ndarray, pct: float, high_is_good: bool) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    pct = min(max(float(pct), 0.0), 1.0)
    q = 1.0 - pct if high_is_good else pct
    return safe_float(np.quantile(scores, q))


def _evaluate_window(
    market: Any,
    preds: dict[str, np.ndarray],
    combo: ComboSpec,
    thresholds: dict[str, float],
    start_step: int,
    window_steps: int,
    fee_multiplier: float,
    source: str,
) -> dict[str, Any]:
    risk = market.cfg.risk
    initial_balance = risk.initial_balance
    loss_floor = initial_balance * (1.0 - BASE_GUARD["max_window_loss_pct"])
    balance = initial_balance
    position = Position()
    open_trade: dict[str, Any] | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    pause_until = -1
    consecutive_losses = 0
    exposure_steps = 0
    total_fees = 0.0
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    skipped_by_regime = 0
    skipped_by_signal = 0
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        regime = str(market.regimes[step])
        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps
            if entry_signal and regime not in combo.allowed_regimes:
                skipped_by_regime += 1
                flat_steps += 1
                continue
            if entry_signal and step < pause_until:
                flat_steps += 1
                continue
            if entry_signal and balance <= loss_floor:
                flat_steps += 1
                continue

            if entry_signal:
                for rule in combo.edge_rules:
                    score = float(preds[rule.model_name][step - market.cfg.model.window_size])
                    if rule.high_is_good:
                        if score < thresholds[rule.model_name]:
                            entry_signal = False
                            break
                    else:
                        if score > thresholds[rule.model_name]:
                            entry_signal = False
                            break
                if entry_signal:
                    for rule in combo.risk_rules:
                        score = float(preds[rule.model_name][step - market.cfg.model.window_size])
                        if rule.high_is_good:
                            if score < thresholds[rule.model_name]:
                                entry_signal = False
                                break
                        else:
                            if score > thresholds[rule.model_name]:
                                entry_signal = False
                                break

            if entry_signal:
                before = balance
                balance, position, fee = _open_position(balance, price, step, market, fee_multiplier)
                if position.side > 0:
                    total_fees += fee
                    edge_score = float(preds[combo.edge_rules[0].model_name][step - market.cfg.model.window_size])
                    risk_score = float(preds[combo.risk_rules[0].model_name][step - market.cfg.model.window_size]) if combo.risk_rules else 0.0
                    open_trade = {
                        "entry_step": step,
                        "entry_price": price,
                        "entry_balance": before,
                        "entry_notional": abs(position.size) * price,
                        "entry_edge_score": edge_score,
                        "entry_risk_score": risk_score,
                        "entry_regime": regime,
                        "entry_fee": fee,
                    }
                    hold_steps = 0
                    flat_steps = 0
                else:
                    skipped_by_signal += 1
                    flat_steps += 1
            else:
                skipped_by_signal += 1
                flat_steps += 1
        else:
            exposure_steps += 1
            edge_score = float(preds[combo.edge_rules[0].model_name][step - market.cfg.model.window_size])
            roe = current_roe(position, price, risk)
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and roe >= TAKE_PROFIT_ROE:
                close_reason = "take_profit"
            elif hold_steps >= MAX_HOLD_STEPS:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and open_trade is not None and edge_score <= open_trade["entry_edge_score"] * EDGE_DETERIORATION_RATIO:
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                balance, _, close_fee = _close_position(balance, position, price, market, fee_multiplier)
                total_fees += close_fee
                net = balance - float(open_trade["entry_balance"])
                trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
                mfe, mae = _trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
                trades.append(
                    {
                        "entry_step": int(open_trade["entry_step"]),
                        "exit_step": int(step),
                        "entry_regime": str(open_trade["entry_regime"]),
                        "return": safe_float(trade_return),
                        "net": safe_float(net),
                        "mfe": safe_float(mfe),
                        "mae": safe_float(mae),
                        "reason": close_reason,
                    }
                )
                if net < 0.0:
                    consecutive_losses += 1
                    pause_until = max(pause_until, step + BASE_GUARD["pause_after_loss_steps"])
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
        step = end_limit
        price = float(market.close[step])
        balance, _, close_fee = _close_position(balance, position, price, market, fee_multiplier)
        total_fees += close_fee
        net = balance - float(open_trade["entry_balance"])
        trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
        mfe, mae = _trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
        trades.append(
            {
                "entry_step": int(open_trade["entry_step"]),
                "exit_step": int(step),
                "entry_regime": str(open_trade["entry_regime"]),
                "return": safe_float(trade_return),
                "net": safe_float(net),
                "mfe": safe_float(mfe),
                "mae": safe_float(mae),
                "reason": "window_end",
            }
        )
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
        "skipped_by_regime": int(skipped_by_regime),
        "skipped_by_signal": int(skipped_by_signal),
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
        "full_size_trades": int(len(trades)),
        "reduced_size_trades": 0,
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
        "trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "avg_return_per_trade": safe_float(np.median(avg_returns)),
        "exposure_time": safe_float(np.median(exposure)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_regime": int(sum(w["skipped_by_regime"] for w in windows)),
        "skipped_by_signal": int(sum(w["skipped_by_signal"] for w in windows)),
    }


def evaluate_signal_combinations(
    config_path: str,
    model_dir: Path,
    report_path: Path,
    seeds: tuple[int, ...],
    allowed_regimes: tuple[str, ...],
) -> dict[str, Any]:
    market = load_signal_market(config_path)
    window_map: dict[int, set[str]] = {}
    for seed in seeds:
        selected = select_robust_windows(
            market,
            window_steps=WINDOW_STEPS,
            seed=seed,
            target_max=144,
            recent_windows=24,
            random_windows=24,
            regime_windows_per_regime=6,
            non_overlap_windows=24,
        )
        for start_step, source in selected:
            window_map.setdefault(int(start_step), set()).add(str(source))
    windows = sorted((start, "+".join(sorted(sources))) for start, sources in window_map.items())
    if len(windows) > 144:
        starts = np.asarray([item[0] for item in windows], dtype=np.int64)
        keep = np.linspace(0, len(starts) - 1, 144).round().astype(int)
        windows = [windows[idx] for idx in keep]
    if len(windows) < 100:
        raise RuntimeError(f"Signal combination selection produced only {len(windows)} windows; need >=100")
    models = _load_models(model_dir)
    preds = _predict_scores(market, models)

    thresholds: dict[tuple[str, float, bool], float] = {}
    for name, scores in preds.items():
        high_is_good = not name.startswith("long_failure_risk")
        for pct in (0.03, 0.05, 0.10, 0.20, 0.30, 0.50):
            thresholds[(name, pct, high_is_good)] = _threshold(scores, pct, high_is_good=high_is_good)

    reports: list[dict[str, Any]] = []
    for fee_multiplier in (1.0, 1.25):
        for combo in COMBOS:
            combo_windows = [
                _evaluate_window(
                    market=market,
                    preds=preds,
                    combo=ComboSpec(combo.name, combo.edge_rules, combo.risk_rules, allowed_regimes),
                    thresholds={
                        rule.model_name: thresholds[(rule.model_name, rule.pct, rule.high_is_good)]
                        for rule in (*combo.edge_rules, *combo.risk_rules)
                    },
                    start_step=int(start_step),
                    window_steps=WINDOW_STEPS,
                    fee_multiplier=fee_multiplier,
                    source=source,
                )
                for start_step, source in windows
            ]
            summary = _summary(combo_windows, market.cfg.risk.initial_balance)
            reports.append(
                {
                    "combo": combo.name,
                    "fee_multiplier": fee_multiplier,
                    "edge_rules": [rule.__dict__ for rule in combo.edge_rules],
                    "risk_rules": [rule.__dict__ for rule in combo.risk_rules],
                    "summary": summary,
                    "windows": combo_windows,
                }
            )
            print(
                f"{combo.name} fee={fee_multiplier:.2f} p25pf={summary['p25_pf']:.2f} "
                f"worst={summary['worst_balance']:.2f} dd={summary['worst_max_dd']:.1%} "
                f"prof={summary['profitable_window_pct']:.1%} trades={summary['median_trades']:.1f}"
            )

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        s = row["summary"]
        return (
            float(s["p25_pf"]),
            float(s["worst_balance"]),
            -float(s["worst_max_dd"]),
            float(s["profitable_window_pct"]),
            float(s["median_trades"]),
        )

    ranking = sorted(reports, key=rank_key, reverse=True)
    report = {
        "schema_version": "aegis_signal_combination_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config_path": config_path,
        "model_dir": str(model_dir),
        "window_count": len(windows),
        "seeds": list(seeds),
        "allowed_regimes": list(allowed_regimes),
        "thresholds": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in thresholds.items()},
        "reports": reports,
        "ranking": [
            {
                "combo": row["combo"],
                "fee_multiplier": row["fee_multiplier"],
                **row["summary"],
            }
            for row in ranking
        ],
        "best": {
            "combo": ranking[0]["combo"],
            "fee_multiplier": ranking[0]["fee_multiplier"],
            **ranking[0]["summary"],
        } if ranking else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json_dumps(report), encoding="utf-8")
    print(f"Signal combination report -> {report_path}")
    return report


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--seeds", default="8123,9137")
    parser.add_argument("--allowed-regimes", default="mixed,chop,high_vol")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    allowed_regimes = tuple(part.strip() for part in args.allowed_regimes.split(",") if part.strip())
    evaluate_signal_combinations(
        config_path=args.config,
        model_dir=Path(args.model_dir),
        report_path=Path(args.report),
        seeds=seeds,
        allowed_regimes=allowed_regimes,
    )


if __name__ == "__main__":
    main()
