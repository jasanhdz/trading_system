"""Causal, symmetric primitives for W8 conditional direction research."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


CLASSES = ("LONG", "SHORT", "SKIP")


def causal_previous_close(values: Sequence[float]) -> np.ndarray:
    """Align a completed close to the next candle-open decision timestamp."""

    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("AEGIS_W8_CLOSE_SERIES_INVALID")
    return np.concatenate(([np.nan], array[:-1]))


def stable_opportunity_id(symbol: str, timestamp: str) -> str:
    if not symbol or not timestamp:
        raise ValueError("AEGIS_W8_EPISODE_ID_INVALID")
    return "W8-" + hashlib.sha256(f"W8|{symbol}|{timestamp}".encode()).hexdigest()


def validate_direction_features(names: Sequence[str]) -> None:
    if not names or len(names) != len(set(names)):
        raise ValueError("AEGIS_W8_FEATURE_CONTRACT_INVALID")
    forbidden = [
        name for name in names
        if name.startswith(("future_", "target_", "outcome_", "side_"))
        or name in {
            "economic_label", "utility_long", "utility_short",
            "directional_advantage", "entry_brain_action",
        }
    ]
    if forbidden:
        raise ValueError("AEGIS_W8_FUTURE_OR_SIDE_FEATURE_PROHIBITED:" + ",".join(forbidden))


def _barrier_utility(
    *, entry: float, highs: np.ndarray, lows: np.ndarray, side: str,
    favorable_bps: float, adverse_bps: float, cost_bps: float,
) -> tuple[float, str]:
    sign = 1.0 if side == "LONG" else -1.0
    for high, low in zip(highs, lows, strict=True):
        favorable = (high / entry - 1.0) * 10_000.0 if side == "LONG" else (1.0 - low / entry) * 10_000.0
        adverse = (1.0 - low / entry) * 10_000.0 if side == "LONG" else (high / entry - 1.0) * 10_000.0
        if adverse >= adverse_bps:
            return -adverse_bps - cost_bps, "ADVERSE_FIRST"
        if favorable >= favorable_bps:
            return favorable_bps - cost_bps, "FAVORABLE_FIRST"
    return sign * 0.0, "NEITHER"


def symmetric_path_outcome(
    *, entry: float, highs: Sequence[float], lows: Sequence[float],
    closes: Sequence[float], favorable_bps: float, adverse_bps: float,
    cost_bps: float, minimum_utility_bps: float,
    minimum_advantage_bps: float,
) -> Mapping[str, Any]:
    high = np.asarray(highs, dtype=float)
    low = np.asarray(lows, dtype=float)
    close = np.asarray(closes, dtype=float)
    if (
        entry <= 0 or not len(close) or len(high) != len(low) or len(low) != len(close)
        or not np.isfinite(np.concatenate((high, low, close))).all()
        or min(favorable_bps, adverse_bps, cost_bps, minimum_utility_bps, minimum_advantage_bps) < 0
    ):
        raise ValueError("AEGIS_W8_PATH_INPUT_INVALID")
    terminal_bps = (float(close[-1]) / entry - 1.0) * 10_000.0
    up = max(0.0, (float(high.max()) / entry - 1.0) * 10_000.0)
    down = max(0.0, (1.0 - float(low.min()) / entry) * 10_000.0)
    long_utility, long_barrier = _barrier_utility(
        entry=entry, highs=high, lows=low, side="LONG",
        favorable_bps=favorable_bps, adverse_bps=adverse_bps, cost_bps=cost_bps,
    )
    short_utility, short_barrier = _barrier_utility(
        entry=entry, highs=high, lows=low, side="SHORT",
        favorable_bps=favorable_bps, adverse_bps=adverse_bps, cost_bps=cost_bps,
    )
    if long_barrier == "NEITHER":
        long_utility = terminal_bps - cost_bps
    if short_barrier == "NEITHER":
        short_utility = -terminal_bps - cost_bps
    advantage = long_utility - short_utility
    best_side = "LONG" if long_utility >= short_utility else "SHORT"
    best_utility = max(long_utility, short_utility)
    label = (
        best_side
        if best_utility >= minimum_utility_bps and abs(advantage) >= minimum_advantage_bps
        else "SKIP"
    )
    path = np.diff(np.concatenate(([entry], close)))
    efficiency = abs(float(close[-1]) - entry) / max(float(np.abs(path).sum()), 1e-12)
    return {
        "utility_long_bps": float(long_utility),
        "utility_short_bps": float(short_utility),
        "directional_advantage_bps": float(advantage),
        "economic_label": label,
        "long_barrier": long_barrier,
        "short_barrier": short_barrier,
        "terminal_return_bps": float(terminal_bps),
        "long_mfe_bps": float(up), "long_mae_bps": float(down),
        "short_mfe_bps": float(down), "short_mae_bps": float(up),
        "path_efficiency": float(efficiency),
    }


def policy_actions(
    family: str, predictions: Mapping[str, np.ndarray], *,
    probability_threshold: float, utility_threshold: float,
    advantage_threshold: float, absolute_advantage_threshold: float,
) -> np.ndarray:
    if family == "A_MULTICLASS_LOGISTIC":
        probabilities = np.asarray(predictions["probabilities"], dtype=float)
        classes = np.asarray(predictions["classes"], dtype=object)
        indices = probabilities.argmax(axis=1)
        labels = classes[indices].astype(object)
        labels[probabilities[np.arange(len(indices)), indices] < probability_threshold] = "SKIP"
        return labels
    if family == "B_DUAL_UTILITY_RIDGE":
        long = np.asarray(predictions["long"], dtype=float)
        short = np.asarray(predictions["short"], dtype=float)
        best = np.maximum(long, short)
        gap = np.abs(long - short)
        labels = np.where(long >= short, "LONG", "SHORT").astype(object)
        labels[(best < utility_threshold) | (gap < advantage_threshold)] = "SKIP"
        return labels
    if family == "C_ADVANTAGE_RIDGE":
        advantage = np.asarray(predictions["advantage"], dtype=float)
        labels = np.where(advantage >= 0, "LONG", "SHORT").astype(object)
        labels[np.abs(advantage) < absolute_advantage_threshold] = "SKIP"
        return labels
    raise ValueError("AEGIS_W8_MODEL_FAMILY_INVALID")


def realized_policy_returns(frame: pd.DataFrame, actions: Sequence[str], horizon: int) -> np.ndarray:
    values = np.zeros(len(frame), dtype=float)
    actions_array = np.asarray(actions, dtype=object)
    values[actions_array == "LONG"] = frame.loc[actions_array == "LONG", f"h{horizon}_utility_long_bps"]
    values[actions_array == "SHORT"] = frame.loc[actions_array == "SHORT", f"h{horizon}_utility_short_bps"]
    return values


def day_block_bootstrap(values: pd.DataFrame, repetitions: int, seed: int) -> tuple[list[float], float]:
    daily = values.groupby("utc_day", observed=True)["policy_return_bps"].mean().to_numpy(float)
    if not len(daily):
        return [float("nan"), float("nan")], 1.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(repetitions, len(daily)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, (0.025, 0.975))], float((samples <= 0).mean())


def benjamini_hochberg(pvalues: Mapping[str, float], alpha: float) -> Mapping[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    accepted = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / max(len(ordered), 1):
            accepted = rank
    names = {name for name, _ in ordered[:accepted]}
    return {name: name in names for name in pvalues}
