"""Causal intrabar wave-state research primitives for W3."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


W3_FEATURE_COLUMNS = (
    "offset_minutes",
    "impulse_size_atr",
    "impulse_volume_ratio",
    "impulse_volume_zscore",
    "impulse_body_ratio",
    "impulse_directional_taker_imbalance",
    "directional_return_1m",
    "range_atr_1m",
    "body_atr_1m",
    "body_ratio_1m",
    "clv_directional_1m",
    "volume_ratio_1m",
    "volume_zscore_1m",
    "directional_taker_imbalance",
    "trade_count_ratio_1m",
    "pullback_size_atr",
    "pullback_fraction",
    "pullback_duration_minutes",
    "pullback_volume_vs_impulse",
    "velocity_1",
    "velocity_mean_3",
    "velocity_slope",
    "velocity_acceleration",
    "taker_slope",
    "taker_acceleration",
    "taker_recovery",
    "micro_structure_aligned",
    "micro_structure_opposed",
    "break_of_impulse_extreme",
    "distance_recent_favorable_atr",
    "distance_recent_adverse_atr",
    "trend_5m",
    "velocity_5m",
    "rsi_5m_directional",
    "trend_15m",
    "velocity_15m",
    "rsi_15m_directional",
    "btc_directional_return_1m",
    "btc_directional_return_5m",
    "btc_directional_return_15m",
    "btc_directional_alignment",
)


def stable_wave_episode_id(symbol: str, side: str, impulse_close_ms: int) -> str:
    payload = f"W3|{symbol}|{side}|{impulse_close_ms}".encode()
    return "W3-" + hashlib.sha256(payload).hexdigest()


def directional_clv(open_: float, high: float, low: float, close: float, direction: int) -> float:
    span = high - low
    if span <= 0.0:
        return 0.5
    raw = (close - low) / span
    return raw if direction > 0 else 1.0 - raw


def directional_excursions(
    entry: float, highs: Sequence[float], lows: Sequence[float], direction: int
) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(highs, dtype=float)
    low = np.asarray(lows, dtype=float)
    if direction > 0:
        favorable = high / entry - 1.0
        adverse = 1.0 - low / entry
    else:
        favorable = 1.0 - low / entry
        adverse = high / entry - 1.0
    return favorable, adverse


def barrier_return(
    entry: float,
    atr: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    direction: int,
    favorable_atr: float,
    adverse_atr: float,
) -> tuple[float, int, float, float]:
    """Resolve a triple barrier adverse-first and return gross underlying return."""

    favorable, adverse = directional_excursions(entry, highs, lows, direction)
    close_values = np.asarray(closes, dtype=float)
    favorable_fraction = favorable_atr * atr / entry
    adverse_fraction = adverse_atr * atr / entry
    outcome = 0
    gross = direction * (float(close_values[-1]) / entry - 1.0)
    for fav, adv in zip(favorable, adverse, strict=True):
        if adv >= adverse_fraction:
            outcome = -1
            gross = -adverse_fraction
            break
        if fav >= favorable_fraction:
            outcome = 1
            gross = favorable_fraction
            break
    return gross, outcome, float(np.max(favorable)), float(np.max(adverse))


def future_giveback_before_new_extreme(
    *,
    peak_favorable: float,
    current_favorable: float,
    future_favorable_highs: Sequence[float],
    future_favorable_lows: Sequence[float],
    atr_fraction: float,
    giveback_atr: float = 0.25,
    new_extreme_atr: float = 0.25,
) -> int:
    giveback_level = peak_favorable - giveback_atr * atr_fraction
    new_extreme_level = peak_favorable + new_extreme_atr * atr_fraction
    for favorable_high, favorable_low in zip(
        future_favorable_highs, future_favorable_lows, strict=True
    ):
        if favorable_low <= giveback_level:
            return 1
        if favorable_high >= new_extreme_level:
            return 0
    return int(current_favorable <= giveback_level)


def episode_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("wave_episode_id")["wave_episode_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def finite_frame(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    values = frame[list(columns)].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def profit_capture_ratio(realized_return: float, peak_mfe: float) -> float:
    if peak_mfe <= 0.0:
        return 0.0
    return float(np.clip(max(0.0, realized_return) / peak_mfe, 0.0, 1.0))


def summarize_returns(frame: pd.DataFrame) -> dict[str, Any]:
    returns = frame["net_return"].to_numpy(dtype=float)
    positive = returns[returns > 0.0].sum()
    negative = -returns[returns < 0.0].sum()
    return {
        "episodes": int(len(frame)),
        "trades": int(frame["traded"].sum()),
        "net_expectancy": float(returns.mean()) if len(returns) else 0.0,
        "profit_factor": float(positive / negative) if negative > 0.0 else 1_000_000_000.0,
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "mean_mae_atr": float(frame["mae_atr"].mean()) if len(frame) else 0.0,
        "median_profit_capture_ratio": float(
            frame["profit_capture_ratio"].median()
        ) if "profit_capture_ratio" in frame and len(frame) else 0.0,
        "maximum_symbol_share": float(
            frame["symbol"].value_counts(normalize=True).max()
        ) if len(frame) else 0.0,
    }


def assert_probability_contract(values: np.ndarray) -> None:
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("AEGIS_W3_PROBABILITY_CONTRACT_INVALID")


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default
