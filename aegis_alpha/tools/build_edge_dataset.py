#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aegis_alpha.config import AegisConfig, load_config
from aegis_alpha.edge.common import build_edge_feature_matrix, edge_feature_names
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from data.storage.database_manager import DatabaseManager


@dataclass(frozen=True)
class EdgeDatasetConfig:
    window_size: int = 64
    horizon: int = 12
    eval_horizon: int = 24
    profit_threshold: float = 0.0030
    risk_threshold: float = 0.0030


def load_candles(cfg: AegisConfig):
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    df = db.get_ohlcv_data(symbol, cfg.timeframe)
    if df.empty and symbol != cfg.symbol:
        df = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    if df.empty:
        raise RuntimeError(f"No candles found for {cfg.symbol} {cfg.timeframe}")
    return df


def _future_stats(close: np.ndarray, idx: int, horizon: int) -> tuple[float, float, float]:
    price = float(close[idx])
    end = min(idx + horizon, len(close) - 1)
    future = close[idx + 1 : end + 1]
    if len(future) == 0 or price <= 0:
        return 0.0, 0.0, 0.0
    returns = future / price - 1.0
    close_return = float(close[end] / price - 1.0)
    mfe = float(np.max(returns))
    mae = float(np.min(returns))
    return close_return, mfe, mae


def build_edge_dataset(output: Path, cfg: AegisConfig, ds_cfg: EdgeDatasetConfig, max_samples: int | None) -> dict[str, float]:
    candles = load_candles(cfg)
    frame = build_feature_frame(candles)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = frame["close"].values.astype(np.float32)
    timestamps = frame.index.astype(str).values

    max_horizon = max(ds_cfg.horizon, ds_cfg.eval_horizon)
    last_step = len(features) - max_horizon - 1
    if last_step <= ds_cfg.window_size:
        raise RuntimeError(f"Not enough rows for window={ds_cfg.window_size}, horizon={max_horizon}")

    all_x = build_edge_feature_matrix(features[: last_step + 1], ds_cfg.window_size)
    steps = np.arange(ds_cfg.window_size, last_step + 1, dtype=np.int64)
    if len(all_x) != len(steps):
        raise RuntimeError(f"feature/step length mismatch: {len(all_x)} != {len(steps)}")

    fee_round_trip = 2.0 * cfg.risk.total_fee
    n = len(steps)
    long_return = np.empty((n,), dtype=np.float32)
    short_return = np.empty((n,), dtype=np.float32)
    long_mfe = np.empty((n,), dtype=np.float32)
    long_mae = np.empty((n,), dtype=np.float32)
    short_mfe = np.empty((n,), dtype=np.float32)
    short_mae = np.empty((n,), dtype=np.float32)
    eval_long_return = np.empty((n,), dtype=np.float32)
    eval_short_return = np.empty((n,), dtype=np.float32)

    for out_idx, step in enumerate(steps):
        ret_h, mfe_h, mae_h = _future_stats(close, int(step), ds_cfg.horizon)
        ret_eval, _, _ = _future_stats(close, int(step), ds_cfg.eval_horizon)
        long_return[out_idx] = ret_h - fee_round_trip
        short_return[out_idx] = -ret_h - fee_round_trip
        long_mfe[out_idx] = mfe_h
        long_mae[out_idx] = max(0.0, -mae_h)
        short_mfe[out_idx] = max(0.0, -mae_h)
        short_mae[out_idx] = max(0.0, mfe_h)
        eval_long_return[out_idx] = ret_eval - fee_round_trip
        eval_short_return[out_idx] = -ret_eval - fee_round_trip

    long_good = (
        (long_mfe >= ds_cfg.profit_threshold)
        & (long_mae <= ds_cfg.risk_threshold)
        & (long_return > 0.0)
    ).astype(np.int8)
    short_good = (
        (short_mfe >= ds_cfg.profit_threshold)
        & (short_mae <= ds_cfg.risk_threshold)
        & (short_return > 0.0)
    ).astype(np.int8)
    no_trade = ((long_good == 0) & (short_good == 0)).astype(np.int8)

    if max_samples and n > max_samples:
        rng = np.random.default_rng(4667)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        all_x = all_x[idx]
        steps = steps[idx]
        long_return = long_return[idx]
        short_return = short_return[idx]
        long_mfe = long_mfe[idx]
        long_mae = long_mae[idx]
        short_mfe = short_mfe[idx]
        short_mae = short_mae[idx]
        eval_long_return = eval_long_return[idx]
        eval_short_return = eval_short_return[idx]
        long_good = long_good[idx]
        short_good = short_good[idx]
        no_trade = no_trade[idx]
        n = len(idx)

    ts = timestamps[steps]
    price = close[steps]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=all_x.astype(np.float32),
        timestamp=ts,
        step=steps,
        price=price,
        long_good=long_good,
        short_good=short_good,
        no_trade=no_trade,
        long_return=long_return,
        short_return=short_return,
        long_mfe=long_mfe,
        long_mae=long_mae,
        short_mfe=short_mfe,
        short_mae=short_mae,
        eval_long_return=eval_long_return,
        eval_short_return=eval_short_return,
        feature_names=np.array(edge_feature_names(FEATURE_COLUMNS)),
        base_feature_columns=np.array(FEATURE_COLUMNS),
        config=np.array(str(asdict(ds_cfg))),
        fee_round_trip=np.array(fee_round_trip, dtype=np.float32),
    )

    summary = {
        "samples": float(n),
        "long_good_pct": float(long_good.mean()),
        "short_good_pct": float(short_good.mean()),
        "no_trade_pct": float(no_trade.mean()),
        "avg_long_return": float(long_return.mean()),
        "avg_short_return": float(short_return.mean()),
    }
    print(f"Samples: {n:,}")
    print(f"LONG good: {int(long_good.sum()):,} ({summary['long_good_pct']:.2%})")
    print(f"SHORT good: {int(short_good.sum()):,} ({summary['short_good_pct']:.2%})")
    print(f"NO TRADE: {int(no_trade.sum()):,} ({summary['no_trade_pct']:.2%})")
    print(f"Avg net returns: long={summary['avg_long_return']:.4%} short={summary['avg_short_return']:.4%}")
    print(f"Saved -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output", default="aegis_alpha/data/processed/edge_dataset_v030.npz")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--eval-horizon", type=int, default=24)
    parser.add_argument("--profit-threshold", type=float, default=0.0030)
    parser.add_argument("--risk-threshold", type=float, default=0.0030)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    build_edge_dataset(
        Path(args.output),
        load_config(args.config),
        EdgeDatasetConfig(
            window_size=args.window_size,
            horizon=args.horizon,
            eval_horizon=args.eval_horizon,
            profit_threshold=args.profit_threshold,
            risk_threshold=args.risk_threshold,
        ),
        args.max_samples,
    )


if __name__ == "__main__":
    main()
