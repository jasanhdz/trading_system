"""Causal primitives for W7 opportunity and frozen-direction decomposition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def stable_signal_id(symbol: str, timestamp: str, direction: str) -> str:
    if not symbol or not timestamp or direction not in {"LONG", "SHORT"}:
        raise ValueError("AEGIS_W7_SIGNAL_ID_INVALID")
    value = f"W7|{symbol}|{timestamp}|{direction}".encode()
    return "W7-" + hashlib.sha256(value).hexdigest()


def validate_opportunity_features(names: Sequence[str]) -> None:
    if not names or len(names) != len(set(names)):
        raise ValueError("AEGIS_W7_FEATURE_CONTRACT_INVALID")
    forbidden = [
        name for name in names
        if name.startswith(("target_", "future_", "outcome_", "side_"))
        or name in {"direction", "entry_brain_action", "long_terminal_return", "short_terminal_return"}
    ]
    if forbidden:
        raise ValueError(f"AEGIS_W7_DIRECTION_OR_FUTURE_FEATURE_PROHIBITED:{','.join(forbidden)}")


def opportunity_path_outcomes(
    *, entry: float, highs: Sequence[float], lows: Sequence[float],
    closes: Sequence[float], frozen_direction: str, cost_bps: float,
) -> Mapping[str, float]:
    """Compute direction-neutral magnitude and separate frozen-side economics."""

    high = np.asarray(highs, dtype=float)
    low = np.asarray(lows, dtype=float)
    close = np.asarray(closes, dtype=float)
    if (
        entry <= 0 or frozen_direction not in {"LONG", "SHORT"}
        or not len(close) or not (len(high) == len(low) == len(close))
        or not np.isfinite(np.concatenate((high, low, close))).all()
    ):
        raise ValueError("AEGIS_W7_PATH_INPUT_INVALID")
    upward = max(0.0, float(high.max(initial=entry) / entry - 1.0))
    downward = max(0.0, float(1.0 - low.min(initial=entry) / entry))
    magnitude = max(upward, downward)
    sign = 1.0 if frozen_direction == "LONG" else -1.0
    directional_terminal = sign * (float(close[-1]) / entry - 1.0)
    favorable = upward if frozen_direction == "LONG" else downward
    adverse = downward if frozen_direction == "LONG" else upward
    return {
        "opportunity_magnitude_bps": magnitude * 10_000.0,
        "upward_excursion_bps": upward * 10_000.0,
        "downward_excursion_bps": downward * 10_000.0,
        "directional_gross_return_bps": directional_terminal * 10_000.0,
        "directional_net_return_bps": directional_terminal * 10_000.0 - cost_bps,
        "directional_mfe_bps": favorable * 10_000.0,
        "directional_mae_bps": adverse * 10_000.0,
        "mfe_mae_ratio": favorable / max(adverse, 1e-12),
    }


def day_block_bootstrap(
    frame: pd.DataFrame, *, value_column: str, repetitions: int, seed: int,
) -> tuple[list[float], float]:
    daily = frame.groupby("utc_day", observed=True)[value_column].mean().to_numpy(float)
    if not len(daily):
        return [float("nan"), float("nan")], float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(repetitions, len(daily)), replace=True).mean(1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])], float((samples <= 0).mean())


def economic_summary(frame: pd.DataFrame, return_column: str) -> Mapping[str, float | int]:
    values = frame[return_column].to_numpy(float)
    positive = float(values[values > 0].sum())
    negative = float(-values[values < 0].sum())
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    drawdown = peak - equity
    downside = values[values < 0]
    return {
        "episodes": int(len(frame)),
        "net_expectancy_bps": float(values.mean()) if len(values) else 0.0,
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "profit_factor": positive / negative if negative else 1_000_000_000.0,
        "maximum_drawdown_bps_additive": float(drawdown.max(initial=0.0)),
        "sortino_episode": float(values.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else 0.0,
        "median_mfe_bps": float(frame.directional_mfe_bps.median()) if len(frame) else 0.0,
        "median_mae_bps": float(frame.directional_mae_bps.median()) if len(frame) else 0.0,
        "median_mfe_mae_ratio": float(frame.mfe_mae_ratio.median()) if len(frame) else 0.0,
    }


def benjamini_hochberg(pvalues: Mapping[str, float], alpha: float = 0.05) -> Mapping[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    accepted = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / max(len(ordered), 1):
            accepted = rank
    names = {name for name, _ in ordered[:accepted]}
    return {name: name in names for name in pvalues}


def partition(timestamp_ms: int, config: Mapping[str, Any]) -> str:
    for name in ("train", "validation", "final_holdout"):
        start, end = config["W7A"]["partitions"][name]
        if int(pd.Timestamp(start).timestamp() * 1_000) <= timestamp_ms < int(pd.Timestamp(end).timestamp() * 1_000):
            return name.upper()
    return "OUT_OF_SCOPE"
