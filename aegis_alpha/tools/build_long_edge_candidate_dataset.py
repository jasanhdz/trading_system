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

from aegis_alpha.config import AegisConfig, load_config
from aegis_alpha.edge.common import build_edge_feature_matrix, load_model_bundle, safe_float
from aegis_alpha.env.risk_engine import Position, current_roe, position_notional
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES, select_robust_windows
from data.storage.database_manager import DatabaseManager


REGIME_ORDER = ("trend_up", "trend_down", "chop", "compression", "high_vol", "mixed")
BASE_GUARD = {
    "max_window_loss_pct": 0.07,
    "pause_after_loss_steps": 48,
    "pause_after_2_losses_steps": 48,
    "max_trades_per_day": 3,
    "fee_multiplier": 1.0,
}
COMPACT_BASE_COLUMNS = (
    "log_ret",
    "vol_norm",
    "rsi_norm",
    "ema_9_norm",
    "ema_21_norm",
    "ema_200_norm",
    "cvd_z",
    "cvd_roc",
    "ema_1h_slope",
    "ema_4h_slope",
    "vol_z",
    "cvd_div",
    "ema_1h_accel",
    "ema_4h_accel",
    "cvd_accel",
    "adx_norm",
    "trend_efficiency",
    "vol_regime",
)


@dataclass(frozen=True)
class MetaMarketData:
    cfg: AegisConfig
    features: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray
    regimes: np.ndarray
    expected_long_return: np.ndarray
    expected_short_return: np.ndarray
    long_success_prob: np.ndarray
    short_success_prob: np.ndarray


@dataclass
class OpenTrade:
    balance_before_open: float
    entry_notional: float
    entry_step: int
    entry_price: float
    entry_score: float
    entry_fee: float
    entry_regime: str
    entry_features: np.ndarray
    entry_meta: dict[str, Any]


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


def load_meta_market(config_path: str, edge_model_path: Path) -> MetaMarketData:
    cfg = load_config(config_path)
    candles = _load_candles(cfg)
    frame = build_feature_frame(candles)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = frame["close"].values.astype(np.float32)
    timestamps = frame.index.astype(str).values
    regimes = _detect_regimes(features, cfg.model.window_size)

    bundle = load_model_bundle(edge_model_path)
    edge_x = build_edge_feature_matrix(features, cfg.model.window_size)
    expected_long = bundle["long_return_regressor"].predict(edge_x).astype(np.float32)
    expected_short = bundle["short_return_regressor"].predict(edge_x).astype(np.float32)
    long_prob = bundle["long_classifier"].predict_proba(edge_x)[:, 1].astype(np.float32)
    short_prob = bundle["short_classifier"].predict_proba(edge_x)[:, 1].astype(np.float32)

    expected_long_return = np.full((len(features),), -np.inf, dtype=np.float32)
    expected_short_return = np.full((len(features),), -np.inf, dtype=np.float32)
    long_success_prob = np.zeros((len(features),), dtype=np.float32)
    short_success_prob = np.zeros((len(features),), dtype=np.float32)
    start = cfg.model.window_size
    expected_long_return[start:] = expected_long
    expected_short_return[start:] = expected_short
    long_success_prob[start:] = long_prob
    short_success_prob[start:] = short_prob
    return MetaMarketData(
        cfg=cfg,
        features=features,
        close=close,
        timestamps=timestamps,
        regimes=regimes,
        expected_long_return=expected_long_return,
        expected_short_return=expected_short_return,
        long_success_prob=long_success_prob,
        short_success_prob=short_success_prob,
    )


def compact_feature_names() -> list[str]:
    names = [
        "expected_return_long",
        "long_success_prob",
        "expected_return_short",
        "short_success_prob",
        "edge_gap",
        "success_prob_gap",
        "gate_distance",
        "abs_expected_short_return",
    ]
    for prefix in ("last", "mean6", "mean12", "std12", "delta6", "delta12"):
        names.extend(f"{prefix}_{name}" for name in COMPACT_BASE_COLUMNS)
    names.extend(f"regime_{regime}" for regime in REGIME_ORDER)
    return names


def candidate_features(market: MetaMarketData, step: int, gate_threshold: float) -> np.ndarray:
    idxs = [FEATURE_COLUMNS.index(name) for name in COMPACT_BASE_COLUMNS]
    features = market.features
    row = features[step, idxs]
    window_start = max(0, step - market.cfg.model.window_size)
    window = features[window_start:step, idxs]
    if len(window) == 0:
        window = features[step : step + 1, idxs]
    mean6 = window[-6:].mean(axis=0)
    mean12 = window[-12:].mean(axis=0)
    std12 = window[-12:].std(axis=0)
    prev6 = features[max(0, step - 6), idxs]
    prev12 = features[max(0, step - 12), idxs]
    expected_long = float(market.expected_long_return[step])
    expected_short = float(market.expected_short_return[step])
    long_prob = float(market.long_success_prob[step])
    short_prob = float(market.short_success_prob[step])
    regime = str(market.regimes[step])
    regime_one_hot = np.asarray([1.0 if regime == item else 0.0 for item in REGIME_ORDER], dtype=np.float32)
    head = np.asarray(
        [
            expected_long,
            long_prob,
            expected_short,
            short_prob,
            expected_long - expected_short,
            long_prob - short_prob,
            expected_long - gate_threshold,
            abs(expected_short),
        ],
        dtype=np.float32,
    )
    out = np.concatenate((head, row, mean6, mean12, std12, row - prev6, row - prev12, regime_one_hot))
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0).astype(np.float32)


def _open_position(
    balance: float,
    price: float,
    step: int,
    market: MetaMarketData,
    fee_multiplier: float,
) -> tuple[float, Position, float]:
    notional = position_notional(balance, market.cfg.risk)
    fee = notional * market.cfg.risk.total_fee * fee_multiplier
    if balance <= fee * 1.5:
        return balance, Position(), 0.0
    return balance - fee, Position(side=1, size=notional / max(price, 1e-10), entry_price=price, entry_step=step), fee


def _close_position(
    balance: float,
    position: Position,
    price: float,
    market: MetaMarketData,
    fee_multiplier: float,
) -> tuple[float, float, float]:
    pnl = abs(position.size) * (price - position.entry_price)
    fee = abs(position.size) * price * market.cfg.risk.total_fee * fee_multiplier
    return max(0.0, balance + pnl - fee), pnl - fee, fee


def _trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def _day_key(step: int) -> int:
    return step // 288


def _candidate_row(
    market: MetaMarketData,
    trade: OpenTrade,
    position: Position,
    balance: float,
    exit_step: int,
    reason: str,
    close_fee: float,
) -> dict[str, Any]:
    net = balance - trade.balance_before_open
    trade_return = net / max(trade.entry_notional, 1e-10)
    mfe, mae = _trade_mfe_mae(market.close, trade.entry_step, exit_step)
    return {
        **trade.entry_meta,
        "exit_step": int(exit_step),
        "exit_timestamp": str(market.timestamps[exit_step]),
        "exit_price": safe_float(market.close[exit_step]),
        "exit_score": safe_float(market.expected_long_return[exit_step]),
        "hold_steps": int(exit_step - trade.entry_step),
        "simulated_trade_return": safe_float(trade_return),
        "simulated_trade_net": safe_float(net),
        "win": int(trade_return > 0.0),
        "mfe": safe_float(mfe),
        "mae": safe_float(mae),
        "fees": safe_float(trade.entry_fee + close_fee),
        "reason": reason,
        "entry_notional": safe_float(trade.entry_notional),
        "position_size": safe_float(position.size),
    }


def collect_candidates(
    market: MetaMarketData,
    windows: list[tuple[int, str]],
    gate_threshold: float,
    window_steps: int,
    max_hold_steps: int,
    close_edge_threshold: float,
    take_profit_roe: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    risk = market.cfg.risk
    x_rows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for window_id, (start_step, source) in enumerate(windows):
        initial_balance = risk.initial_balance
        loss_floor = initial_balance * (1.0 - BASE_GUARD["max_window_loss_pct"])
        balance = initial_balance
        position = Position()
        open_trade: OpenTrade | None = None
        hold_steps = 0
        flat_steps = risk.min_flat_steps
        pause_until = -1
        consecutive_losses = 0
        trades_by_day: Counter[int] = Counter()
        end_limit = min(start_step + window_steps, len(market.close) - 1)
        for step in range(start_step, end_limit):
            price = float(market.close[step])
            score = float(market.expected_long_return[step])
            regime = str(market.regimes[step])
            day = _day_key(step)
            if position.side == 0:
                entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
                if entry_signal and balance <= loss_floor:
                    counts["guard_max_window_loss"] += 1
                    flat_steps += 1
                elif entry_signal and step < pause_until:
                    counts["guard_pause"] += 1
                    flat_steps += 1
                elif entry_signal and trades_by_day[day] >= BASE_GUARD["max_trades_per_day"]:
                    counts["guard_max_trades_per_day"] += 1
                    flat_steps += 1
                elif entry_signal and regime not in ALLOWED_REGIMES:
                    counts[f"blocked_regime_{regime}"] += 1
                    flat_steps += 1
                elif entry_signal:
                    before = balance
                    balance, position, fee = _open_position(balance, price, step, market, BASE_GUARD["fee_multiplier"])
                    if position.side > 0:
                        features = candidate_features(market, step, gate_threshold)
                        meta = {
                            "window_id": int(window_id),
                            "window_start_step": int(start_step),
                            "window_source": source,
                            "entry_step": int(step),
                            "entry_timestamp": str(market.timestamps[step]),
                            "entry_price": safe_float(price),
                            "regime": regime,
                            "expected_return_long": safe_float(score),
                            "long_success_prob": safe_float(market.long_success_prob[step]),
                            "expected_return_short": safe_float(market.expected_short_return[step]),
                            "short_success_prob": safe_float(market.short_success_prob[step]),
                            "edge_gap": safe_float(score - float(market.expected_short_return[step])),
                        }
                        open_trade = OpenTrade(
                            balance_before_open=before,
                            entry_notional=abs(position.size) * position.entry_price,
                            entry_step=step,
                            entry_price=price,
                            entry_score=score,
                            entry_fee=fee,
                            entry_regime=regime,
                            entry_features=features,
                            entry_meta=meta,
                        )
                        trades_by_day[day] += 1
                        hold_steps = 0
                        flat_steps = 0
                    else:
                        counts["open_failed"] += 1
                        flat_steps += 1
                else:
                    flat_steps += 1
            else:
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
                    balance, _, close_fee = _close_position(balance, position, price, market, BASE_GUARD["fee_multiplier"])
                    row = _candidate_row(market, open_trade, position, balance, step, close_reason, close_fee)
                    x_rows.append(open_trade.entry_features)
                    rows.append(row)
                    if float(row["simulated_trade_net"]) < 0.0:
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

        if position.side != 0 and open_trade is not None:
            exit_step = end_limit
            price = float(market.close[exit_step])
            balance, _, close_fee = _close_position(balance, position, price, market, BASE_GUARD["fee_multiplier"])
            row = _candidate_row(market, open_trade, position, balance, exit_step, "window_end", close_fee)
            x_rows.append(open_trade.entry_features)
            rows.append(row)

    if not x_rows:
        raise RuntimeError("No LONG edge candidates were collected")
    return np.vstack(x_rows).astype(np.float32), rows, dict(counts)


def build_candidate_dataset(
    edge_model_path: Path,
    config_path: str,
    output_path: Path,
    report_path: Path,
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
) -> tuple[Path, Path]:
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
    x, rows, guard_counts = collect_candidates(
        market=market,
        windows=windows,
        gate_threshold=threshold,
        window_steps=window_steps,
        max_hold_steps=max_hold_steps,
        close_edge_threshold=close_edge_threshold,
        take_profit_roe=take_profit_roe,
    )
    y = np.asarray([row["win"] for row in rows], dtype=np.int8)
    returns = np.asarray([row["simulated_trade_return"] for row in rows], dtype=np.float32)
    mfe = np.asarray([row["mfe"] for row in rows], dtype=np.float32)
    mae = np.asarray([row["mae"] for row in rows], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        simulated_trade_return=returns,
        win=y,
        mfe=mfe,
        mae=mae,
        step=np.asarray([row["entry_step"] for row in rows], dtype=np.int64),
        timestamp=np.asarray([row["entry_timestamp"] for row in rows]),
        price=np.asarray([row["entry_price"] for row in rows], dtype=np.float32),
        regime=np.asarray([row["regime"] for row in rows]),
        expected_return_long=np.asarray([row["expected_return_long"] for row in rows], dtype=np.float32),
        long_success_prob=np.asarray([row["long_success_prob"] for row in rows], dtype=np.float32),
        expected_return_short=np.asarray([row["expected_return_short"] for row in rows], dtype=np.float32),
        short_success_prob=np.asarray([row["short_success_prob"] for row in rows], dtype=np.float32),
        edge_gap=np.asarray([row["edge_gap"] for row in rows], dtype=np.float32),
        window_id=np.asarray([row["window_id"] for row in rows], dtype=np.int32),
        window_start_step=np.asarray([row["window_start_step"] for row in rows], dtype=np.int64),
        window_source=np.asarray([row["window_source"] for row in rows]),
        hold_steps=np.asarray([row["hold_steps"] for row in rows], dtype=np.int32),
        reason=np.asarray([row["reason"] for row in rows]),
        fees=np.asarray([row["fees"] for row in rows], dtype=np.float32),
        feature_names=np.asarray(compact_feature_names()),
        compact_base_columns=np.asarray(COMPACT_BASE_COLUMNS),
        policy=np.asarray(
            json.dumps(
                {
                    "side": "LONG_ONLY",
                    "entry_gate": "top_3pct_expected_return_long",
                    "gate_threshold": threshold,
                    "allowed_regimes": sorted(ALLOWED_REGIMES),
                    "guard": BASE_GUARD,
                    "max_hold_steps": max_hold_steps,
                    "close_edge_threshold": close_edge_threshold,
                    "take_profit_roe": take_profit_roe,
                },
                sort_keys=True,
            )
        ),
    )

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_candidate_dataset_v1",
        "created_at": created_at,
        "dataset_path": str(output_path),
        "edge_model_path": str(edge_model_path),
        "config_path": config_path,
        "candidate_count": int(len(rows)),
        "feature_count": int(x.shape[1]),
        "window_count": int(len(windows)),
        "window_steps": int(window_steps),
        "gate_threshold": safe_float(threshold),
        "allowed_regimes": sorted(ALLOWED_REGIMES),
        "guard": BASE_GUARD,
        "summary": {
            "win_rate": safe_float(np.mean(y)),
            "avg_return": safe_float(np.mean(returns)),
            "median_return": safe_float(np.median(returns)),
            "avg_mfe": safe_float(np.mean(mfe)),
            "avg_mae": safe_float(np.mean(mae)),
            "regime_counts": dict(Counter(row["regime"] for row in rows)),
            "close_reasons": dict(Counter(row["reason"] for row in rows)),
            "guard_counts": guard_counts,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Candidates: {len(rows):,} win_rate={np.mean(y):.2%} avg_return={np.mean(returns):.4%}")
    print(f"Dataset saved -> {output_path}")
    print(f"Report saved -> {report_path}")
    return output_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output", default="aegis_alpha/data/processed/long_edge_candidates_v040.npz")
    parser.add_argument("--report", default="aegis_alpha/logs/edge/long_edge_candidate_dataset_v040.json")
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
    build_candidate_dataset(
        edge_model_path=Path(args.edge_model),
        config_path=args.config,
        output_path=Path(args.output),
        report_path=Path(args.report),
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
