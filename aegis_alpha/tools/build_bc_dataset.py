#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from aegis_alpha.bc.labeler import LabelerConfig, label_prudent_action
from aegis_alpha.config import AegisConfig, load_config
from aegis_alpha.env.action_mask import CLOSE, LONG, SHORT
from aegis_alpha.env.risk_engine import Position, close_position, current_roe, open_position
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from data.storage.database_manager import DatabaseManager

WINDOW_SIZE = 64
ACTION_NAMES = {0: "IDLE", 1: "LONG", 2: "SHORT", 3: "CLOSE"}


def _future_stats(close: np.ndarray, idx: int, horizon: int = 12) -> tuple[float, float, float, float, float]:
    price = float(close[idx])
    def ret(steps: int) -> float:
        j = min(idx + steps, len(close) - 1)
        return float(close[j] / price - 1.0)

    end = min(idx + horizon, len(close) - 1)
    future = close[idx + 1 : end + 1]
    if len(future) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mfe = float(np.max(future / price - 1.0))
    mae = float(np.min(future / price - 1.0))
    return ret(3), ret(6), ret(12), mfe, mae


def _account_obs(
    cfg: AegisConfig,
    balance: float,
    position: Position,
    step: int,
    price: float,
    hold_steps: int,
    flat_steps: int,
) -> np.ndarray:
    roe = current_roe(position, price, cfg.risk)
    if position.side == 0:
        equity = balance
        signed_exposure = 0.0
    else:
        pnl = abs(position.size) * (price - position.entry_price if position.side > 0 else position.entry_price - price)
        equity = balance + pnl
        signed_exposure = position.side * abs(position.size) * price / max(equity, 1e-10)
    return np.array(
        [
            equity / cfg.risk.initial_balance,
            signed_exposure,
            roe,
            float(position.side),
            hold_steps / 288.0,
            flat_steps / 288.0,
        ],
        dtype=np.float32,
    )


def load_candles(cfg: AegisConfig):
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    df = db.get_ohlcv_data(symbol, cfg.timeframe)
    if df.empty and symbol != cfg.symbol:
        df = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    if df.empty:
        raise RuntimeError(f"No candles found for {cfg.symbol} {cfg.timeframe}")
    return df


def _balance_idle_refs(
    refs: list[tuple[int, np.ndarray, int, str, float, float, float, float, float]],
    target_idle_pct: float | None,
) -> list[tuple[int, np.ndarray, int, str, float, float, float, float, float]]:
    if target_idle_pct is None or not 0.0 < target_idle_pct < 1.0:
        return refs
    idle_refs = [ref for ref in refs if ref[2] == 0]
    active_refs = [ref for ref in refs if ref[2] != 0]
    if not active_refs:
        return refs
    max_idle = int(round(len(active_refs) * target_idle_pct / (1.0 - target_idle_pct)))
    if len(idle_refs) <= max_idle:
        return refs
    rng = np.random.default_rng(4667)
    keep_idx = np.sort(rng.choice(len(idle_refs), size=max_idle, replace=False))
    balanced = [idle_refs[i] for i in keep_idx] + active_refs
    return sorted(balanced, key=lambda item: item[0])


def build_dataset(
    output: Path,
    max_samples: int | None,
    cfg: AegisConfig,
    label_cfg: LabelerConfig,
    target_idle_pct: float | None,
) -> dict[str, float]:
    candles = load_candles(cfg)
    feature_frame = build_feature_frame(candles)
    features = feature_frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = feature_frame["close"].values.astype(np.float32)
    timestamps = feature_frame.index.astype(str).values

    balance = cfg.risk.initial_balance
    position = Position()
    hold_steps = 0
    flat_steps = cfg.risk.min_flat_steps

    refs: list[tuple[int, np.ndarray, int, str, float, float, float, float, float]] = []
    for step in range(WINDOW_SIZE, len(features) - 13):
        price = float(close[step])
        row = features[step]
        roe = current_roe(position, price, cfg.risk)
        account = _account_obs(cfg, balance, position, step, price, hold_steps, flat_steps)
        action = label_prudent_action(row, position.side, hold_steps, flat_steps, roe=roe, cfg=label_cfg)
        regime = detect_regime(features[max(0, step - WINDOW_SIZE) : step + 1]).type
        f3, f6, f12, mfe12, mae12 = _future_stats(close, step)
        refs.append((step, account, action, regime, f3, f6, f12, mfe12, mae12))

        if action in (LONG, SHORT) and position.side == 0:
            side = 1 if action == LONG else -1
            balance, position, _ = open_position(balance, side, price, step, cfg.risk)
            hold_steps = 0
            flat_steps = 0
        elif action == CLOSE and position.side != 0:
            balance, _, _ = close_position(balance, position, price, cfg.risk)
            position = Position()
            hold_steps = 0
            flat_steps = 0
        else:
            if position.side == 0:
                flat_steps += 1
            else:
                hold_steps += 1

    refs = _balance_idle_refs(refs, target_idle_pct)

    if max_samples and len(refs) > max_samples:
        rng = np.random.default_rng(4667)
        indices = np.sort(rng.choice(len(refs), size=max_samples, replace=False))
        refs = [refs[i] for i in indices]

    n = len(refs)
    market = np.empty((n, WINDOW_SIZE, features.shape[1]), dtype=np.float16)
    account = np.empty((n, 6), dtype=np.float16)
    actions = np.empty((n,), dtype=np.int64)
    ts = np.empty((n,), dtype="U32")
    price_arr = np.empty((n,), dtype=np.float32)
    regimes = np.empty((n,), dtype="U16")
    future_3 = np.empty((n,), dtype=np.float32)
    future_6 = np.empty((n,), dtype=np.float32)
    future_12 = np.empty((n,), dtype=np.float32)
    mfe_12 = np.empty((n,), dtype=np.float32)
    mae_12 = np.empty((n,), dtype=np.float32)

    for out_idx, (step, account_obs, action, regime, f3, f6, f12, mfe12, mae12) in enumerate(refs):
        market[out_idx] = features[step - WINDOW_SIZE : step].astype(np.float16)
        account[out_idx] = account_obs.astype(np.float16)
        actions[out_idx] = action
        ts[out_idx] = timestamps[step]
        price_arr[out_idx] = close[step]
        regimes[out_idx] = regime
        future_3[out_idx] = f3
        future_6[out_idx] = f6
        future_12[out_idx] = f12
        mfe_12[out_idx] = mfe12
        mae_12[out_idx] = mae12

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        market=market,
        account=account,
        actions=actions,
        timestamp=ts,
        price=price_arr,
        regime=regimes,
        future_return_3=future_3,
        future_return_6=future_6,
        future_return_12=future_12,
        mfe_12=mfe_12,
        mae_12=mae_12,
        feature_columns=np.array(FEATURE_COLUMNS),
        action_names=np.array([ACTION_NAMES[i] for i in range(4)]),
    )

    counts = Counter(actions.tolist())
    regime_counts = Counter(regimes.tolist())
    print(f"Samples: {n:,}")
    for action in range(4):
        pct = counts[action] / max(n, 1) * 100
        print(f"{ACTION_NAMES[action]}: {counts[action]:,} ({pct:.1f}%)")
    print("Regimes:")
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime}: {count:,} ({count / max(n, 1) * 100:.1f}%)")
    print(f"Saved -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return {ACTION_NAMES[action].lower(): counts[action] / max(n, 1) for action in range(4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output", default="aegis_alpha/data/processed/bc_prudent_dataset.npz")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-idle-pct", type=float, default=0.84)
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_dataset(Path(args.output), args.max_samples, cfg, LabelerConfig(), args.target_idle_pct)


if __name__ == "__main__":
    main()
