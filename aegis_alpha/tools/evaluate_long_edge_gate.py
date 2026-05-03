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

from aegis_alpha.config import AegisConfig, RiskConfig, load_config
from aegis_alpha.edge.common import build_edge_feature_matrix, load_model_bundle, profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, close_position, current_roe, open_position
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from data.storage.database_manager import DatabaseManager


REGIME_ORDER = ("trend_up", "trend_down", "chop", "compression", "high_vol", "mixed")
GATE_PCTS = (0.02, 0.01, 0.005)


@dataclass(frozen=True)
class MarketData:
    cfg: AegisConfig
    features: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray
    regimes: np.ndarray
    expected_long_return: np.ndarray


@dataclass
class OpenTrade:
    balance_before_open: float
    entry_notional: float
    entry_step: int
    entry_price: float
    entry_score: float
    entry_fee: float


def _load_candles(cfg: AegisConfig):
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    df = db.get_ohlcv_data(symbol, cfg.timeframe)
    if df.empty and symbol != cfg.symbol:
        df = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    if df.empty:
        raise RuntimeError(f"No candles found for {cfg.symbol} {cfg.timeframe}")
    return df


def _detect_regimes(features: np.ndarray, window_size: int) -> np.ndarray:
    regimes = np.empty((len(features),), dtype="U16")
    for idx in range(len(features)):
        start = max(0, idx - window_size + 1)
        regimes[idx] = detect_regime(features[start : idx + 1]).type
    return regimes


def _dominant_regime(regimes: np.ndarray) -> str:
    if len(regimes) == 0:
        return "unknown"
    return Counter(regimes.tolist()).most_common(1)[0][0]


def _load_market(config_path: str, model_path: Path) -> MarketData:
    cfg = load_config(config_path)
    candles = _load_candles(cfg)
    frame = build_feature_frame(candles)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = frame["close"].values.astype(np.float32)
    timestamps = frame.index.astype(str).values
    regimes = _detect_regimes(features, cfg.model.window_size)

    bundle = load_model_bundle(model_path)
    edge_x = build_edge_feature_matrix(features, cfg.model.window_size)
    predictions = bundle["long_return_regressor"].predict(edge_x).astype(np.float32)
    expected_long_return = np.full((len(features),), -np.inf, dtype=np.float32)
    expected_long_return[cfg.model.window_size :] = predictions
    return MarketData(
        cfg=cfg,
        features=features,
        close=close,
        timestamps=timestamps,
        regimes=regimes,
        expected_long_return=expected_long_return,
    )


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
        idxs = np.flatnonzero(market.regimes[min_start : max_start + 1] == regime) + min_start
        if len(idxs) == 0:
            continue
        picks = rng.choice(idxs, size=min(regime_windows_per_regime, len(idxs)), replace=False)
        for pick in picks:
            _add_window(windows, int(pick), f"regime:{regime}", min_start, max_start)

    return [(start, "+".join(sorted(sources))) for start, sources in sorted(windows.items())]


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
        "entry_timestamp": str(market.timestamps[trade.entry_step]),
        "exit_timestamp": str(market.timestamps[step]),
        "entry_price": safe_float(trade.entry_price),
        "exit_price": safe_float(price),
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


def _evaluate_window(
    market: MarketData,
    gate_name: str,
    gate_threshold: float,
    start_step: int,
    window_steps: int,
    source: str,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> dict[str, Any]:
    risk: RiskConfig = market.cfg.risk
    balance = risk.initial_balance
    position = Position()
    open_trade: OpenTrade | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    equity_curve: list[float] = []
    exposure_steps = 0
    total_fees = 0.0
    trades: list[dict[str, Any]] = []
    action_counts = Counter()
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
                    )
                    hold_steps = 0
                    flat_steps = 0
                    action_counts["LONG"] += 1
                else:
                    action_counts["IDLE"] += 1
                    flat_steps += 1
            else:
                action_counts["IDLE"] += 1
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
                action_counts["CLOSE"] += 1
            else:
                hold_steps += 1
                flat_steps = 0
                action_counts["IDLE"] += 1

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
        position = Position()
        action_counts["CLOSE"] += 1
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
        "gate": gate_name,
        "gate_threshold": safe_float(gate_threshold),
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "start_timestamp": str(market.timestamps[start_step]),
        "end_timestamp": str(market.timestamps[end_limit]),
        "regime_dominant": _dominant_regime(window_regimes),
        "regime_distribution": dict(Counter(window_regimes.tolist())),
        "balance": safe_float(final_balance),
        "net": safe_float(final_balance - risk.initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(len(wins) / max(len(returns), 1)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "fees": safe_float(total_fees),
        "fees_per_trade": safe_float(total_fees / max(len(trades), 1)),
        "exposure_time": safe_float(exposure_steps / max(window_steps, 1)),
        "mfe_avg": safe_float(np.mean([trade["mfe"] for trade in trades])) if trades else 0.0,
        "mae_avg": safe_float(np.mean([trade["mae"] for trade in trades])) if trades else 0.0,
        "mfe_median": safe_float(np.median([trade["mfe"] for trade in trades])) if trades else 0.0,
        "mae_median": safe_float(np.median([trade["mae"] for trade in trades])) if trades else 0.0,
        "action_counts": dict(action_counts),
        "close_reasons": dict(Counter(trade["reason"] for trade in trades)),
    }


def _summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    fees = np.asarray([w["fees"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    return {
        "window_count": int(len(windows)),
        "median_balance": safe_float(np.median(balances)),
        "p25_balance": safe_float(np.quantile(balances, 0.25)),
        "worst_balance": safe_float(np.min(balances)),
        "median_pf": safe_float(np.median(pfs)),
        "p25_pf": safe_float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": safe_float(np.mean(balances > initial_balance)),
        "median_trades": safe_float(np.median(trades)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_fees": safe_float(np.median(fees)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "median_exposure_time": safe_float(np.median(exposure)),
    }


def _comparison(gate_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for gate_name, report in gate_reports.items():
        summary = report["summary"]
        rows.append(
            {
                "gate": gate_name,
                "median_balance": summary["median_balance"],
                "p25_balance": summary["p25_balance"],
                "worst_balance": summary["worst_balance"],
                "median_pf": summary["median_pf"],
                "p25_pf": summary["p25_pf"],
                "profitable_window_pct": summary["profitable_window_pct"],
                "median_trades": summary["median_trades"],
                "worst_max_dd": summary["worst_max_dd"],
                "median_avg_return_per_trade": summary["median_avg_return_per_trade"],
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["median_balance"]),
            float(row["p25_balance"]),
            float(row["worst_balance"]),
            float(row["median_pf"]),
            -float(row["worst_max_dd"]),
            float(row["median_trades"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_validation(
    model_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    recent_windows: int,
    random_windows: int,
    regime_windows_per_regime: int,
    seed: int,
    gate_pcts: tuple[float, ...],
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Edge model not found: {model_path}")

    market = _load_market(config_path, model_path)
    valid_scores = market.expected_long_return[np.isfinite(market.expected_long_return)]
    thresholds = {f"top_{pct * 100:g}pct": float(np.quantile(valid_scores, 1.0 - pct)) for pct in gate_pcts}
    selected = select_windows(
        market,
        window_steps=window_steps,
        recent_windows=recent_windows,
        random_windows=random_windows,
        regime_windows_per_regime=regime_windows_per_regime,
        seed=seed,
    )
    print(f"Selected windows: {len(selected)}")
    print(f"Gate thresholds: {thresholds}")

    gate_reports: dict[str, dict[str, Any]] = {}
    for gate_name, threshold in thresholds.items():
        windows = []
        for idx, (start_step, source) in enumerate(selected, start=1):
            metrics = _evaluate_window(
                market,
                gate_name,
                threshold,
                start_step,
                window_steps,
                source,
                max_hold_steps,
                close_edge_threshold,
                take_profit_roe,
            )
            windows.append(metrics)
            print(
                f"[{gate_name} {idx:02d}/{len(selected):02d}] {metrics['start_timestamp']} -> "
                f"{metrics['end_timestamp']} balance={metrics['balance']:.2f} "
                f"trades={metrics['trades']} pf={metrics['profit_factor']:.2f} "
                f"dd={metrics['max_dd']:.2%}"
            )
        gate_reports[gate_name] = {
            "gate_threshold": threshold,
            "summary": _summary(windows, market.cfg.risk.initial_balance),
            "windows": windows,
        }

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_gate_validation_v1",
        "created_at": created_at,
        "model_path": str(model_path),
        "config_path": config_path,
        "symbol": market.cfg.symbol,
        "timeframe": market.cfg.timeframe,
        "risk_budget": {
            "initial_balance": market.cfg.risk.initial_balance,
            "leverage": market.cfg.risk.leverage,
            "position_fraction": market.cfg.risk.position_fraction,
            "hard_stop_roe": market.cfg.risk.hard_stop_roe,
            "min_hold_steps": market.cfg.risk.min_hold_steps,
            "min_flat_steps": market.cfg.risk.min_flat_steps,
            "commission_rate": market.cfg.risk.commission_rate,
            "slippage": market.cfg.risk.slippage,
        },
        "policy": {
            "side": "LONG_ONLY",
            "entry": "flat and expected_return_long >= gate_threshold",
            "exit": "hard_stop, take_profit, max_hold, edge_deterioration, or window_end",
            "max_hold_steps": max_hold_steps,
            "close_edge_threshold": close_edge_threshold,
            "take_profit_roe": take_profit_roe,
            "short_entries": False,
        },
        "window_steps": window_steps,
        "selection": {
            "recent_windows": recent_windows,
            "random_windows": random_windows,
            "regime_windows_per_regime": regime_windows_per_regime,
            "seed": seed,
        },
        "gate_thresholds": thresholds,
        "gates": gate_reports,
        "comparison": _comparison(gate_reports),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"long_edge_gate_validation_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    return output_path


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
    parser.add_argument("--gate-pcts", type=float, nargs="+", default=list(GATE_PCTS))
    parser.add_argument("--max-hold-steps", type=int, default=24)
    parser.add_argument("--close-edge-threshold", type=float, default=0.0)
    parser.add_argument("--take-profit-roe", type=float, default=0.06)
    args = parser.parse_args()

    run_validation(
        model_path=Path(args.model),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        recent_windows=args.recent_windows,
        random_windows=args.random_windows,
        regime_windows_per_regime=args.regime_windows_per_regime,
        seed=args.seed,
        gate_pcts=tuple(args.gate_pcts),
        max_hold_steps=args.max_hold_steps,
        close_edge_threshold=args.close_edge_threshold,
        take_profit_roe=args.take_profit_roe,
    )


if __name__ == "__main__":
    main()
