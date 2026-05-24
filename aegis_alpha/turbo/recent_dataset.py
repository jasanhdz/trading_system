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
OPERABLE_TARGET_SCHEMA_VERSION = "aegis_turbo_operable_targets_v2"
OPERABLE_HIT_RULES: dict[str, tuple[float, float]] = {
    "hit5_before_minus5": (0.005, 0.005),
    "hit8_before_minus5": (0.008, 0.005),
    "hit10_before_minus8": (0.010, 0.008),
    "hit15_before_minus10": (0.015, 0.010),
}
OPERABLE_TARGET_NAMES: tuple[str, ...] = tuple(
    f"{side}_{name}_{horizon}"
    for horizon in (12, 24)
    for side in ("long", "short")
    for name in (
        *OPERABLE_HIT_RULES.keys(),
        "mfe",
        "mae",
        "mae_danger",
        "mae_severe",
        "mfe_mae_ratio",
        "trade_quality",
        "time_to_hit8",
        "time_to_minus5",
        "ambiguous_hit_stop",
    )
)
TRADE_QUALITY_FORMULA = (
    "clip((1.0 if hit8_before_minus5 else 0.5 if hit5_before_minus5 else 0.0)"
    " - min(mae / 0.01, 1.0) * 0.7"
    " + min(mfe / 0.015, 1.0) * 0.3"
    " - fee_round_trip, -1.0, 1.0)"
)


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


def compute_path_outcome(
    entry: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    side: str,
    target_return: float,
    stop_return: float,
) -> dict[str, float | int | bool]:
    """Compute a side-aware target/stop path; a same-candle conflict counts as stop first."""
    if entry <= 0.0 or len(future_high) != len(future_low):
        return {
            "hit_before_stop": False,
            "ambiguous_same_candle": False,
            "time_to_target": -1,
            "time_to_stop": -1,
            "mfe": 0.0,
            "mae": 0.0,
        }
    normalized_side = side.lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError(f"unsupported side: {side}")
    highs = np.asarray(future_high, dtype=np.float32)
    lows = np.asarray(future_low, dtype=np.float32)
    if normalized_side == "long":
        favorable = highs / entry - 1.0
        adverse = 1.0 - lows / entry
        targets = highs >= entry * (1.0 + target_return)
        stops = lows <= entry * (1.0 - stop_return)
    else:
        favorable = 1.0 - lows / entry
        adverse = highs / entry - 1.0
        targets = lows <= entry * (1.0 - target_return)
        stops = highs >= entry * (1.0 + stop_return)
    target_indices = np.flatnonzero(targets)
    stop_indices = np.flatnonzero(stops)
    time_to_target = int(target_indices[0] + 1) if len(target_indices) else -1
    time_to_stop = int(stop_indices[0] + 1) if len(stop_indices) else -1
    ambiguous = bool(np.any(targets & stops))
    hit_before_stop = bool(
        time_to_target >= 0
        and (time_to_stop < 0 or time_to_target < time_to_stop)
    )
    return {
        "hit_before_stop": bool(hit_before_stop),
        "ambiguous_same_candle": bool(ambiguous),
        "time_to_target": int(time_to_target),
        "time_to_stop": int(time_to_stop),
        "mfe": safe_float(max(0.0, float(np.max(favorable))) if len(favorable) else 0.0),
        "mae": safe_float(max(0.0, float(np.max(adverse))) if len(adverse) else 0.0),
    }


def compute_trade_quality(
    hit5_before_minus5: bool,
    hit8_before_minus5: bool,
    mfe: float,
    mae: float,
    fee_round_trip: float,
) -> float:
    base_reward = 1.0 if hit8_before_minus5 else (0.5 if hit5_before_minus5 else 0.0)
    mae_penalty = min(max(float(mae), 0.0) / 0.01, 1.0) * 0.7
    mfe_reward = min(max(float(mfe), 0.0) / 0.015, 1.0) * 0.3
    value = base_reward - mae_penalty + mfe_reward - max(float(fee_round_trip), 0.0)
    return safe_float(np.clip(value, -1.0, 1.0))


def compute_long_short_targets(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    horizon: int,
    fee_round_trip: float,
) -> dict[str, float | int]:
    entry = float(close[idx])
    future_high = np.asarray(high[idx + 1 : idx + horizon + 1], dtype=np.float32)
    future_low = np.asarray(low[idx + 1 : idx + horizon + 1], dtype=np.float32)
    result: dict[str, float | int] = {}
    for side in ("long", "short"):
        outcomes = {
            name: compute_path_outcome(entry, future_high, future_low, side, target, stop)
            for name, (target, stop) in OPERABLE_HIT_RULES.items()
        }
        canonical = outcomes["hit8_before_minus5"]
        mfe = float(canonical["mfe"])
        mae = float(canonical["mae"])
        for name, outcome in outcomes.items():
            result[f"{side}_{name}"] = int(bool(outcome["hit_before_stop"]))
        result[f"{side}_mfe"] = safe_float(mfe)
        result[f"{side}_mae"] = safe_float(mae)
        result[f"{side}_mae_danger"] = int(mae >= 0.005)
        result[f"{side}_mae_severe"] = int(mae >= 0.010)
        result[f"{side}_mfe_mae_ratio"] = safe_float(mfe / max(mae, 1e-6))
        result[f"{side}_trade_quality"] = compute_trade_quality(
            bool(outcomes["hit5_before_minus5"]["hit_before_stop"]),
            bool(canonical["hit_before_stop"]),
            mfe,
            mae,
            fee_round_trip,
        )
        result[f"{side}_time_to_hit8"] = int(canonical["time_to_target"])
        result[f"{side}_time_to_minus5"] = int(canonical["time_to_stop"])
        result[f"{side}_ambiguous_hit_stop"] = int(bool(canonical["ambiguous_same_candle"]))
    return result


def build_recent_dataset(symbol: str, lookback_days: int, save: bool = True, market: Any | None = None) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    symbol = symbol.replace("/", "").upper()
    market = market or load_signal_market(cfg.config_path, symbol_override=symbol)

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
    operable_rows = {
        horizon: [
            compute_long_short_targets(market.high, market.low, market.close, int(step), horizon, fee_round_trip)
            for step in steps
        ]
        for horizon in (12, 24)
    }
    operable_targets: dict[str, np.ndarray] = {}
    for horizon, rows in operable_rows.items():
        for side in ("long", "short"):
            for key in (
                *OPERABLE_HIT_RULES.keys(),
                "mfe",
                "mae",
                "mae_danger",
                "mae_severe",
                "mfe_mae_ratio",
                "trade_quality",
                "time_to_hit8",
                "time_to_minus5",
                "ambiguous_hit_stop",
            ):
                name = f"{side}_{key}_{horizon}"
                dtype = np.int16 if key.startswith("time_to_") else (np.int8 if key in OPERABLE_HIT_RULES or key.startswith("mae_") or key == "ambiguous_hit_stop" else np.float32)
                operable_targets[name] = np.asarray([row[f"{side}_{key}"] for row in rows], dtype=dtype)

    dataset = {
        "schema_version": "aegis_turbo_recent_dataset_v1",
        "operable_targets_schema_version": OPERABLE_TARGET_SCHEMA_VERSION,
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
        **operable_targets,
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
        "operable_targets_schema_version": OPERABLE_TARGET_SCHEMA_VERSION,
        "operable_target_names": list(OPERABLE_TARGET_NAMES),
        "trade_quality_formula": TRADE_QUALITY_FORMULA,
        "long_hit8_before_minus5_12_rate": safe_float(np.mean(operable_targets["long_hit8_before_minus5_12"])) if len(x) else 0.0,
        "short_hit8_before_minus5_12_rate": safe_float(np.mean(operable_targets["short_hit8_before_minus5_12"])) if len(x) else 0.0,
        "long_trade_quality_12_avg": safe_float(np.mean(operable_targets["long_trade_quality_12"])) if len(x) else 0.0,
        "short_trade_quality_12_avg": safe_float(np.mean(operable_targets["short_trade_quality_12"])) if len(x) else 0.0,
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
