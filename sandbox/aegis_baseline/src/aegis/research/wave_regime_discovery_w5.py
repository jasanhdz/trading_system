"""Causal GOOD/NEUTRAL/BAD wave regime research primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def stable_wave_id(symbol: str, side: str, event_timestamp_ms: int) -> str:
    if side not in {"LONG", "SHORT"} or event_timestamp_ms <= 0:
        raise ValueError("AEGIS_W5_WAVE_ID_INVALID")
    raw = f"W5|{symbol}|{side}|{event_timestamp_ms}".encode()
    return "W5-" + hashlib.sha256(raw).hexdigest()


def resolve_wave_path(
    *, entry: float, atr: float, direction: int, highs: Sequence[float],
    lows: Sequence[float], closes: Sequence[float], favorable_atr: float = 0.50,
    adverse_atr: float = 0.25, cost_bps: float = 14.0,
) -> dict[str, float | int | str]:
    if entry <= 0 or atr <= 0 or direction not in {-1, 1}:
        raise ValueError("AEGIS_W5_PATH_INPUT_INVALID")
    high = np.asarray(highs, dtype=float)
    low = np.asarray(lows, dtype=float)
    close = np.asarray(closes, dtype=float)
    if not (len(high) == len(low) == len(close) and len(close) > 0):
        raise ValueError("AEGIS_W5_PATH_LENGTH_INVALID")
    favorable = (high / entry - 1) if direction > 0 else (1 - low / entry)
    adverse = (1 - low / entry) if direction > 0 else (high / entry - 1)
    fav_fraction, adv_fraction = favorable_atr * atr / entry, adverse_atr * atr / entry
    outcome, exit_index = "TIME", len(close) - 1
    gross = direction * (close[-1] / entry - 1)
    for index, (fav, adv) in enumerate(zip(favorable, adverse, strict=True)):
        if adv >= adv_fraction:
            outcome, exit_index, gross = "ADVERSE", index, -adv_fraction
            break
        if fav >= fav_fraction:
            outcome, exit_index, gross = "FAVORABLE", index, fav_fraction
            break
    directional_closes = direction * (close[: exit_index + 1] / entry - 1)
    path = np.diff(np.concatenate(([0.0], directional_closes)))
    efficiency = abs(float(directional_closes[-1])) / max(float(np.abs(path).sum()), 1e-12)
    mfe = float(favorable[: exit_index + 1].max())
    mae = float(adverse[: exit_index + 1].max())
    mae_atr = mae * entry / atr
    mfe_atr = mfe * entry / atr
    net_bps = gross * 10_000 - cost_bps
    ratio = mfe_atr / max(mae_atr, 1e-9)
    if outcome == "FAVORABLE" and net_bps > 0 and mae_atr <= 0.25 and ratio >= 1.5 and efficiency >= 0.25:
        label = "GOOD_WAVE"
    elif outcome == "ADVERSE" or net_bps <= -14.0:
        label = "BAD_WAVE"
    else:
        label = "NEUTRAL_WAVE"
    return {
        "barrier_outcome": outcome, "exit_minute": exit_index + 1,
        "gross_return_bps": gross * 10_000, "net_return_bps": net_bps,
        "mfe_atr": mfe_atr, "mae_atr": mae_atr, "mfe_mae_ratio": ratio,
        "path_efficiency": efficiency, "wave_label": label,
    }


def correlation_cluster_id(timestamp_ms: int) -> int:
    return timestamp_ms // (15 * 60 * 1000)


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    accepted = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / max(len(ordered), 1):
            accepted = rank
    names = {name for name, _ in ordered[:accepted]}
    return {name: name in names for name in pvalues}


def economic_summary(frame: pd.DataFrame, cost_delta_bps: float = 0.0) -> dict[str, Any]:
    returns = frame["net_return_bps"].to_numpy(float) - cost_delta_bps
    positive = returns[returns > 0].sum()
    negative = -returns[returns < 0].sum()
    return {
        "episodes": int(len(frame)),
        "independent_clusters": int(frame["correlation_cluster_id"].nunique()),
        "net_expectancy_bps": float(returns.mean()) if len(returns) else 0.0,
        "profit_factor": float(positive / negative) if negative else 1_000_000_000.0,
        "good_rate": float(frame["wave_label"].eq("GOOD_WAVE").mean()) if len(frame) else 0.0,
        "bad_rate": float(frame["wave_label"].eq("BAD_WAVE").mean()) if len(frame) else 0.0,
        "median_mfe_atr": float(frame["mfe_atr"].median()) if len(frame) else 0.0,
        "median_mae_atr": float(frame["mae_atr"].median()) if len(frame) else 0.0,
        "maximum_symbol_share": float(frame["symbol"].value_counts(normalize=True).max()) if len(frame) else 0.0,
    }
