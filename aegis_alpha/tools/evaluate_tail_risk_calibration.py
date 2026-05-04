#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float, write_json  # noqa: E402
from aegis_alpha.env.risk_engine import Position, current_roe  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.signals.combination_utils import predict_scores, threshold_for_rule, trade_mfe_mae  # noqa: E402
from aegis_alpha.signals.combination_utils import RuleCondition  # noqa: E402
from aegis_alpha.tools.build_long_edge_candidate_dataset import BASE_GUARD  # noqa: E402
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES, select_robust_windows  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_DATASET = Path("aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
DEFAULT_MODEL_DIR = Path("aegis_alpha/models/signals")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/tail_risk_calibration_v052.json")
DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_h12_tail_risk_candidate_v052.json")
DEFAULT_SEEDS = (9101, 9203)
WINDOW_STEPS = 4032
FEE_MULTIPLIERS = (1.0, 1.25)
TAKE_PROFIT_ROE = 0.06
MAX_HOLD_STEPS = 24
EDGE_DETERIORATION_RATIO = 0.65
TAIL_PCTS = (0.30, 0.35, 0.40, 0.45, 0.50)


@dataclass(frozen=True)
class SizeBand:
    max_pct: float
    fraction: float
    label: str


@dataclass(frozen=True)
class CalibrationConfig:
    config_id: str
    sizing_mode: str
    tail_threshold: float
    bands: tuple[SizeBand, ...]


def _configs() -> tuple[CalibrationConfig, ...]:
    hard = tuple(
        CalibrationConfig(
            config_id=f"hard_tail{int(pct * 100)}",
            sizing_mode="hard_filter",
            tail_threshold=pct,
            bands=(SizeBand(pct, 0.25, "full"),),
        )
        for pct in TAIL_PCTS
    )
    dynamic = (
        CalibrationConfig(
            "dynamic_tail30_50",
            "dynamic_tail_sizing",
            0.50,
            (SizeBand(0.30, 0.25, "full"), SizeBand(0.50, 0.125, "reduced")),
        ),
        CalibrationConfig(
            "conservative_tail35_50",
            "conservative_dynamic_sizing",
            0.50,
            (SizeBand(0.35, 0.25, "full"), SizeBand(0.50, 0.10, "reduced")),
        ),
        CalibrationConfig(
            "ultra_defensive_tail30_45",
            "ultra_defensive",
            0.45,
            (SizeBand(0.30, 0.20, "full"), SizeBand(0.45, 0.10, "reduced")),
        ),
    )
    return hard + dynamic


def _load_estimator(path: Path) -> Any:
    bundle = load_model_bundle(path)
    estimator = bundle.get("estimator") or bundle.get("classifier") or bundle.get("regressor")
    if estimator is None:
        raise RuntimeError(f"Missing estimator in {path}")
    return estimator


def _select_windows(market: Any, seeds: tuple[int, ...]) -> list[tuple[int, str]]:
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
        keep = np.linspace(0, len(windows) - 1, 144).round().astype(int)
        windows = [windows[int(idx)] for idx in keep]
    if len(windows) < 100:
        raise RuntimeError(f"Window selection produced only {len(windows)} windows; need >=100")
    return windows


def _position_size(config: CalibrationConfig, tail_score: float, tail_thresholds: dict[float, float]) -> tuple[float, str]:
    for band in config.bands:
        if tail_score <= tail_thresholds[band.max_pct]:
            return band.fraction, band.label
    return 0.0, "skip"


def _open_position(balance: float, price: float, step: int, market: Any, fee_multiplier: float, fraction: float) -> tuple[float, Position, float, float]:
    notional = balance * market.cfg.risk.leverage * fraction
    fee = notional * market.cfg.risk.total_fee * fee_multiplier
    if balance <= fee * 1.5:
        return balance, Position(), 0.0, 0.0
    size = notional / max(price, 1e-10)
    return balance - fee, Position(side=1, size=size, entry_price=price, entry_step=step), fee, notional


def _close_position(balance: float, position: Position, price: float, market: Any, fee_multiplier: float) -> tuple[float, float]:
    pnl = abs(position.size) * (price - position.entry_price)
    fee = abs(position.size) * price * market.cfg.risk.total_fee * fee_multiplier
    return max(0.0, balance + pnl - fee), fee


def _evaluate_window(
    market: Any,
    preds: dict[str, np.ndarray],
    config: CalibrationConfig,
    edge_threshold: float,
    tail_thresholds: dict[float, float],
    start_step: int,
    source: str,
    fee_multiplier: float,
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
    trades_by_day: dict[str, int] = defaultdict(int)
    skipped_by_regime = 0
    skipped_by_signal = 0
    skipped_by_tail_risk = 0
    full_size_trades = 0
    reduced_size_trades = 0
    end_limit = min(start_step + WINDOW_STEPS, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        regime = str(market.regimes[step])
        rel_idx = step - market.cfg.model.window_size
        if rel_idx < 0 or rel_idx >= len(preds["long_edge_h12"]):
            continue

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps
            day = str(market.timestamps[step])[:10]
            size_fraction = 0.0
            size_label = "skip"
            if entry_signal and regime not in ALLOWED_REGIMES:
                skipped_by_regime += 1
                entry_signal = False
            elif entry_signal and step < pause_until:
                entry_signal = False
            elif entry_signal and balance <= loss_floor:
                entry_signal = False
            elif entry_signal and trades_by_day[day] >= BASE_GUARD["max_trades_per_day"]:
                entry_signal = False
            elif entry_signal and float(preds["long_edge_h12"][rel_idx]) < edge_threshold:
                skipped_by_signal += 1
                entry_signal = False
            elif entry_signal:
                size_fraction, size_label = _position_size(config, float(preds["long_tail_risk_h12"][rel_idx]), tail_thresholds)
                if size_fraction <= 0.0:
                    skipped_by_tail_risk += 1
                    entry_signal = False

            if entry_signal:
                before = balance
                balance, position, fee, notional = _open_position(balance, price, step, market, fee_multiplier, size_fraction)
                if position.side > 0:
                    total_fees += fee
                    trades_by_day[day] += 1
                    full_size_trades += int(size_label == "full")
                    reduced_size_trades += int(size_label == "reduced")
                    open_trade = {
                        "entry_step": step,
                        "entry_price": price,
                        "entry_balance": before,
                        "entry_notional": notional,
                        "entry_edge_score": float(preds["long_edge_h12"][rel_idx]),
                        "entry_tail_score": float(preds["long_tail_risk_h12"][rel_idx]),
                        "size_fraction": size_fraction,
                        "size_label": size_label,
                    }
                    hold_steps = 0
                    flat_steps = 0
                else:
                    skipped_by_signal += 1
                    flat_steps += 1
            else:
                flat_steps += 1
        else:
            exposure_steps += 1
            edge_score = float(preds["long_edge_h12"][rel_idx])
            roe = current_roe(position, price, risk)
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and roe >= TAKE_PROFIT_ROE:
                close_reason = "take_profit"
            elif hold_steps >= MAX_HOLD_STEPS:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and open_trade is not None and edge_score <= float(open_trade["entry_edge_score"]) * EDGE_DETERIORATION_RATIO:
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                balance, close_fee = _close_position(balance, position, price, market, fee_multiplier)
                total_fees += close_fee
                net = balance - float(open_trade["entry_balance"])
                trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
                mfe, mae = trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
                trades.append(
                    {
                        "entry_step": int(open_trade["entry_step"]),
                        "exit_step": int(step),
                        "return": safe_float(trade_return),
                        "net": safe_float(net),
                        "mfe": safe_float(mfe),
                        "mae": safe_float(mae),
                        "reason": close_reason,
                        "size_fraction": safe_float(open_trade["size_fraction"]),
                        "size_label": str(open_trade["size_label"]),
                        "tail_score": safe_float(open_trade["entry_tail_score"]),
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
        balance, close_fee = _close_position(balance, position, price, market, fee_multiplier)
        total_fees += close_fee
        net = balance - float(open_trade["entry_balance"])
        trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
        mfe, mae = trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
        trades.append(
            {
                "entry_step": int(open_trade["entry_step"]),
                "exit_step": int(step),
                "return": safe_float(trade_return),
                "net": safe_float(net),
                "mfe": safe_float(mfe),
                "mae": safe_float(mae),
                "reason": "window_end",
                "size_fraction": safe_float(open_trade["size_fraction"]),
                "size_label": str(open_trade["size_label"]),
                "tail_score": safe_float(open_trade["entry_tail_score"]),
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
    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "balance": safe_float(final_balance),
        "net": safe_float(final_balance - initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "fees": safe_float(total_fees),
        "exposure_time": safe_float(exposure_steps / max(WINDOW_STEPS, 1)),
        "trades_per_month": safe_float(len(trades) / max(WINDOW_STEPS * 5.0 / 60.0 / 24.0 / 30.4375, 1e-10)),
        "full_size_trades": int(full_size_trades),
        "reduced_size_trades": int(reduced_size_trades),
        "skipped_by_regime": int(skipped_by_regime),
        "skipped_by_signal": int(skipped_by_signal),
        "skipped_by_tail_risk": int(skipped_by_tail_risk),
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
    }


def _summary(config: CalibrationConfig, fee_multiplier: float, windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    trades_month = np.asarray([w["trades_per_month"] for w in windows], dtype=np.float32)
    return {
        "config_id": config.config_id,
        "fee_multiplier": safe_float(fee_multiplier),
        "tail_threshold": safe_float(config.tail_threshold),
        "sizing_mode": config.sizing_mode,
        "median_balance": safe_float(np.median(balances)),
        "p25_balance": safe_float(np.quantile(balances, 0.25)),
        "worst_balance": safe_float(np.min(balances)),
        "median_pf": safe_float(np.median(pfs)),
        "p25_pf": safe_float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": safe_float(np.mean(balances > initial_balance)),
        "median_trades": safe_float(np.median(trades)),
        "trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "exposure_time": safe_float(np.median(exposure)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_tail_risk": int(sum(w["skipped_by_tail_risk"] for w in windows)),
        "skipped_by_regime": int(sum(w["skipped_by_regime"] for w in windows)),
        "skipped_by_signal": int(sum(w["skipped_by_signal"] for w in windows)),
    }


def _passes_success(row: dict[str, Any]) -> bool:
    return (
        float(row["median_balance"]) >= 20.25
        and float(row["p25_balance"]) >= 20.00
        and float(row["worst_balance"]) >= 19.00
        and float(row["p25_pf"]) >= 1.0
        and float(row["profitable_window_pct"]) >= 0.75
        and float(row["median_trades"]) >= 5.0
        and float(row["worst_max_dd"]) <= 0.09
    )


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["worst_balance"]),
        -float(row["worst_max_dd"]),
        float(row["p25_pf"]),
        float(row["profitable_window_pct"]),
        float(row["median_balance"]),
        float(row["median_trades"]),
    )


def _with_composite_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        ("worst_balance", True, 6.0),
        ("worst_max_dd", False, 5.0),
        ("p25_pf", True, 4.0),
        ("profitable_window_pct", True, 3.0),
        ("median_balance", True, 2.0),
        ("median_trades", True, 1.0),
    )
    scored = [dict(row) for row in rows]
    n = max(len(scored) - 1, 1)
    for key, high, weight in metrics:
        ordered = sorted(range(len(scored)), key=lambda i: float(scored[i][key]), reverse=high)
        for rank, idx in enumerate(ordered):
            scored[idx]["composite_score"] = float(scored[idx].get("composite_score", 0.0)) + weight * (1.0 - rank / n)
    return sorted(scored, key=lambda row: (_rank_key(row), float(row["composite_score"])), reverse=True)


def _best_views(ranking: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranking:
        return {}
    fee_1 = [row for row in ranking if float(row["fee_multiplier"]) == 1.0]
    pool = fee_1 or ranking
    return {
        "best_balanced": pool[0],
        "best_defensive": max(pool, key=lambda row: (float(row["worst_balance"]), -float(row["worst_max_dd"]), float(row["median_trades"]))),
        "best_profit_quality": max(pool, key=lambda row: (float(row["p25_pf"]), float(row["profitable_window_pct"]), float(row["median_balance"]))),
    }


def _write_candidate(
    candidate_path: Path,
    best: dict[str, Any] | None,
    model_dir: Path,
    sklearn_version: str,
) -> str | None:
    if best is None:
        return None
    payload = {
        "schema_version": "aegis_strategy_candidate_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "OFFLINE_CANDIDATE_NOT_LIVE",
        "model_paths": {
            "long_edge_h12": str(model_dir / "aegis_long_edge_h12_v050.joblib"),
            "long_tail_risk_h12": str(model_dir / "aegis_long_tail_risk_h12_v052.joblib"),
        },
        "sklearn_version": sklearn_version,
        "signal": "long_edge_h12_top3",
        "tail_risk_model": "long_tail_risk_h12_v052",
        "tail_threshold": best["tail_threshold"],
        "sizing_mode": best["sizing_mode"],
        "config_id": best["config_id"],
        "fee_multiplier": best["fee_multiplier"],
        "oos_metrics": best,
        "reason_not_live": [
            "needs_shadow_validation",
            "needs_fee_slippage_live_validation",
            "not_promoted_to_champion",
        ],
    }
    write_json(candidate_path, payload)
    return str(candidate_path)


def evaluate_tail_risk_calibration(
    config_path: str,
    dataset_path: Path,
    model_dir: Path,
    report_path: Path,
    candidate_path: Path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    market = load_signal_market(config_path)
    windows = _select_windows(market, seeds)
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset_meta = {
        "path": str(dataset_path),
        "samples": int(len(dataset["X"] if "X" in dataset else dataset["x"])),
        "date_start": str(dataset["timestamp"][0]),
        "date_end": str(dataset["timestamp"][-1]),
    }

    models = {
        "long_edge_h12": _load_estimator(model_dir / "aegis_long_edge_h12_v050.joblib"),
        "long_tail_risk_h12": _load_estimator(model_dir / "aegis_long_tail_risk_h12_v052.joblib"),
    }
    preds = predict_scores(market, models)
    edge_threshold = threshold_for_rule(preds["long_edge_h12"], RuleCondition("long_edge_h12", "top_pct", 0.03))
    tail_thresholds = {
        pct: threshold_for_rule(preds["long_tail_risk_h12"], RuleCondition("long_tail_risk_h12", "bottom_pct", pct))
        for pct in TAIL_PCTS
    }

    reports: list[dict[str, Any]] = []
    for fee_multiplier in FEE_MULTIPLIERS:
        for config in _configs():
            combo_windows = [
                _evaluate_window(
                    market=market,
                    preds=preds,
                    config=config,
                    edge_threshold=edge_threshold,
                    tail_thresholds=tail_thresholds,
                    start_step=int(start_step),
                    source=source,
                    fee_multiplier=fee_multiplier,
                )
                for start_step, source in windows
            ]
            summary = _summary(config, fee_multiplier, combo_windows, market.cfg.risk.initial_balance)
            summary["passes_success_criteria"] = bool(_passes_success(summary))
            reports.append(
                {
                    **summary,
                    "bands": [band.__dict__ for band in config.bands],
                    "windows": combo_windows,
                }
            )
            print(
                f"{config.config_id} fee={fee_multiplier:.2f} worst={summary['worst_balance']:.2f} "
                f"dd={summary['worst_max_dd']:.1%} p25pf={summary['p25_pf']:.2f} "
                f"prof={summary['profitable_window_pct']:.1%} trades={summary['median_trades']:.1f}"
            )

    ranking = _with_composite_scores([{k: v for k, v in row.items() if k != "windows"} for row in reports])
    passing = [row for row in ranking if row["passes_success_criteria"] and float(row["fee_multiplier"]) == 1.0]
    candidate_written = _write_candidate(candidate_path, passing[0] if passing else None, model_dir, sklearn.__version__)
    best = _best_views(ranking)

    report = {
        "schema_version": "aegis_tail_risk_calibration_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sklearn_version": sklearn.__version__,
        "config_path": config_path,
        "dataset": dataset_meta,
        "model_paths": {
            "long_edge_h12": str(model_dir / "aegis_long_edge_h12_v050.joblib"),
            "long_tail_risk_h12": str(model_dir / "aegis_long_tail_risk_h12_v052.joblib"),
        },
        "window_count": len(windows),
        "seeds": list(seeds),
        "allowed_regimes": sorted(ALLOWED_REGIMES),
        "guard": {**BASE_GUARD, "fee_multiplier": "grid"},
        "thresholds": {
            "long_edge_h12_top3": safe_float(edge_threshold),
            **{f"long_tail_risk_h12_bottom{int(pct * 100)}": safe_float(value) for pct, value in tail_thresholds.items()},
        },
        "success_criteria": {
            "median_balance": ">=20.25",
            "p25_balance": ">=20.00",
            "worst_balance": ">=19.00",
            "p25_pf": ">=1.0",
            "profitable_window_pct": ">=75%",
            "median_trades": ">=5",
            "worst_max_dd": "<=9%",
        },
        "candidate_path": candidate_written,
        "candidate_created": bool(candidate_written),
        "best": best,
        "ranking": ranking,
        "reports": reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Tail-risk calibration report -> {report_path}")
    if candidate_written:
        print(f"Offline candidate frozen -> {candidate_written}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--seeds", default="9101,9203")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    evaluate_tail_risk_calibration(
        config_path=args.config,
        dataset_path=Path(args.dataset),
        model_dir=Path(args.model_dir),
        report_path=Path(args.report),
        candidate_path=Path(args.candidate),
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
