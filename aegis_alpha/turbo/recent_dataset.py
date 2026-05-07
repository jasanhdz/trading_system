#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import safe_float  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import save_npz_atomic, turbo_snapshot_path  # noqa: E402


MAX_TARGET_HORIZON = 24


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_datetime64_seconds(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[s]")


def _target_stats(close: np.ndarray, idx: int, horizon: int, fee_round_trip: float) -> dict[str, float]:
    entry = float(close[idx])
    future = close[idx + 1 : idx + horizon + 1].astype(np.float32)
    if entry <= 0.0 or len(future) < horizon:
        return {
            "future_return": 0.0,
            "long_net": 0.0,
            "short_net": 0.0,
            "long_mfe": 0.0,
            "long_mae": 0.0,
            "short_mfe": 0.0,
            "short_mae": 0.0,
        }
    long_path = future / entry - 1.0
    short_path = entry / np.maximum(future, 1e-10) - 1.0
    future_return = float(close[idx + horizon] / entry - 1.0)
    return {
        "future_return": safe_float(future_return),
        "long_net": safe_float(future_return - fee_round_trip),
        "short_net": safe_float((-future_return) - fee_round_trip),
        "long_mfe": safe_float(np.max(long_path)),
        "long_mae": safe_float(max(0.0, -np.min(long_path))),
        "short_mfe": safe_float(np.max(short_path)),
        "short_mae": safe_float(max(0.0, -np.min(short_path))),
    }


def build_recent_dataset(symbol: str, lookback_days: int, save: bool = True) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    symbol = symbol.replace("/", "").upper()
    market = load_signal_market(cfg.config_path, symbol_override=symbol)

    valid_mask = market.steps + MAX_TARGET_HORIZON < len(market.close)
    steps = market.steps[valid_mask]
    x = market.signal_features[valid_mask].astype(np.float32)
    timestamps = market.timestamps[steps].astype(str)
    timestamp_dt = _to_datetime64_seconds(timestamps)
    end_dt = timestamp_dt[-1]
    start_dt = end_dt - np.timedelta64(int(lookback_days), "D")
    recent_mask = timestamp_dt >= start_dt
    steps = steps[recent_mask]
    x = x[recent_mask]
    timestamps = timestamps[recent_mask]

    fee_round_trip = market.cfg.risk.total_fee * 2.0
    live_feature = market.signal_features[-1:].astype(np.float32)
    live_timestamp = str(market.timestamps[-1]) if len(market.timestamps) else None
    target_rows = [_target_stats(market.close, int(step), MAX_TARGET_HORIZON, fee_round_trip) for step in steps]
    future_return_6 = np.asarray([_target_stats(market.close, int(step), 6, fee_round_trip)["future_return"] for step in steps], dtype=np.float32)
    t12 = [_target_stats(market.close, int(step), 12, fee_round_trip) for step in steps]
    future_return_12 = np.asarray([row["future_return"] for row in t12], dtype=np.float32)
    future_return_24 = np.asarray([row["future_return"] for row in target_rows], dtype=np.float32)
    mfe_12 = np.asarray([row["long_mfe"] for row in t12], dtype=np.float32)
    mae_12 = np.asarray([row["long_mae"] for row in t12], dtype=np.float32)
    mfe_24 = np.asarray([row["long_mfe"] for row in target_rows], dtype=np.float32)
    mae_24 = np.asarray([row["long_mae"] for row in target_rows], dtype=np.float32)
    long_net_return_12 = np.asarray([row["long_net"] for row in t12], dtype=np.float32)
    short_net_return_12 = np.asarray([row["short_net"] for row in t12], dtype=np.float32)
    long_good_12 = (long_net_return_12 > 0.0).astype(np.int8)
    short_good_12 = (short_net_return_12 > 0.0).astype(np.int8)

    dataset = {
        "schema_version": "aegis_turbo_recent_dataset_v1",
        "symbol": symbol,
        "lookback_days": int(lookback_days),
        "X": x,
        "step": steps.astype(np.int64),
        "timestamp": timestamps,
        "feature_names": np.asarray(market.feature_names),
        "future_return_6": future_return_6,
        "future_return_12": future_return_12,
        "future_return_24": future_return_24,
        "mfe_12": mfe_12,
        "mae_12": mae_12,
        "mfe_24": mfe_24,
        "mae_24": mae_24,
        "long_net_return_12": long_net_return_12,
        "short_net_return_12": short_net_return_12,
        "long_good_12": long_good_12,
        "short_good_12": short_good_12,
        "regime": market.regimes[steps],
        "close": market.close[steps].astype(np.float32),
        "live_X": live_feature,
        "feature_timestamp": live_timestamp,
    }

    report = {
        "schema_version": "aegis_turbo_recent_dataset_report_v1",
        "created_at": _utc_stamp(),
        "symbol": symbol,
        "lookback_days": int(lookback_days),
        "sample_count": int(len(x)),
        "date_start": str(timestamps[0]) if len(timestamps) else None,
        "date_end": str(timestamps[-1]) if len(timestamps) else None,
        "feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
        "feature_timestamp": live_timestamp,
        "long_good_rate": safe_float(np.mean(long_good_12)) if len(long_good_12) else 0.0,
        "short_good_rate": safe_float(np.mean(short_good_12)) if len(short_good_12) else 0.0,
        "avg_long_return": safe_float(np.mean(long_net_return_12)) if len(long_net_return_12) else 0.0,
        "avg_short_return": safe_float(np.mean(short_net_return_12)) if len(short_net_return_12) else 0.0,
        "last_timestamp": str(timestamps[-1]) if len(timestamps) else None,
    }

    if save:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = turbo_snapshot_path(lookback_days, symbol)
        save_npz_atomic(dataset_path, **dataset)
        report_path = cfg.log_dir / f"turbo_recent_dataset_{_utc_stamp()}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["dataset_path"] = str(dataset_path)
        report["report_path"] = str(report_path)
    return {"dataset": dataset, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--lookback-days", type=int, action="append")
    args = parser.parse_args()
    days = args.lookback_days or list(DEFAULT_TURBO_CONFIG.lookback_days)
    reports = [build_recent_dataset(args.symbol, day)["report"] for day in days]
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
