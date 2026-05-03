from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HORIZONS: tuple[int, ...] = (6, 12, 24, 48)


@dataclass(frozen=True)
class HorizonTargetSpec:
    horizon: int
    profit_threshold: float = 0.0030
    risk_threshold: float = 0.0030


def _future_slice(close: np.ndarray, idx: int, horizon: int) -> tuple[np.ndarray, int]:
    end = min(idx + horizon, len(close) - 1)
    return close[idx + 1 : end + 1], end


def horizon_target_row(
    close: np.ndarray,
    idx: int,
    horizon: int,
    total_fee: float,
    profit_threshold: float = 0.0030,
    risk_threshold: float = 0.0030,
) -> dict[str, Any]:
    entry = float(close[idx])
    future, end = _future_slice(close, idx, horizon)
    if len(future) == 0 or entry <= 0.0:
        path = np.asarray([0.0], dtype=np.float32)
    else:
        path = future / entry - 1.0

    future_return = float(close[end] / entry - 1.0) if entry > 0.0 else 0.0
    long_mfe = float(np.max(path))
    long_mae = float(max(0.0, -np.min(path)))
    short_path = -path
    short_mfe = float(np.max(short_path))
    short_mae = float(max(0.0, -np.min(short_path)))
    fee_round_trip = float(total_fee * 2.0)
    long_net_return = float(future_return - fee_round_trip)
    short_net_return = float(-future_return - fee_round_trip)
    long_good = int(long_net_return > 0.0 and long_mfe >= profit_threshold and long_mae <= risk_threshold)
    short_good = int(short_net_return > 0.0 and short_mfe >= profit_threshold and short_mae <= risk_threshold)
    no_trade = int((long_good == 0) and (short_good == 0))
    failure_bad_long = int((long_net_return <= 0.0) or (long_mae > risk_threshold))
    failure_bad_short = int((short_net_return <= 0.0) or (short_mae > risk_threshold))
    return {
        "future_return": float(future_return),
        "mfe": float(long_mfe),
        "mae": float(long_mae),
        "long_net_return": float(long_net_return),
        "short_net_return": float(short_net_return),
        "long_good": long_good,
        "short_good": short_good,
        "no_trade": no_trade,
        "failure_bad_long": failure_bad_long,
        "failure_bad_short": failure_bad_short,
    }


def build_horizon_targets(
    close: np.ndarray,
    steps: np.ndarray,
    total_fee: float,
    horizons: tuple[int, ...] = HORIZONS,
    profit_threshold: float = 0.0030,
    risk_threshold: float = 0.0030,
) -> dict[str, np.ndarray]:
    close = np.asarray(close, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.int64)
    out: dict[str, list[Any]] = {
        f"h{h}_future_return": [] for h in horizons
    }
    for h in horizons:
        out.update({f"h{h}_mfe": [], f"h{h}_mae": [], f"h{h}_long_net_return": [], f"h{h}_short_net_return": [], f"h{h}_long_good": [], f"h{h}_short_good": [], f"h{h}_no_trade": [], f"h{h}_failure_bad_long": [], f"h{h}_failure_bad_short": []})

    for step in steps:
        for h in horizons:
            row = horizon_target_row(
                close=close,
                idx=int(step),
                horizon=h,
                total_fee=total_fee,
                profit_threshold=profit_threshold,
                risk_threshold=risk_threshold,
            )
            out[f"h{h}_future_return"].append(row["future_return"])
            out[f"h{h}_mfe"].append(row["mfe"])
            out[f"h{h}_mae"].append(row["mae"])
            out[f"h{h}_long_net_return"].append(row["long_net_return"])
            out[f"h{h}_short_net_return"].append(row["short_net_return"])
            out[f"h{h}_long_good"].append(row["long_good"])
            out[f"h{h}_short_good"].append(row["short_good"])
            out[f"h{h}_no_trade"].append(row["no_trade"])
            out[f"h{h}_failure_bad_long"].append(row["failure_bad_long"])
            out[f"h{h}_failure_bad_short"].append(row["failure_bad_short"])

    return {
        key: np.asarray(values, dtype=np.float32 if "return" in key or key.endswith(("mfe", "mae")) else np.int8)
        for key, values in out.items()
    }

