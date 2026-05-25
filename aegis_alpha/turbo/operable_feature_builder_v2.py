#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OPERABLE_FEATURE_SCHEMA_VERSION = "aegis_turbo_operable_features_v2_research_v1"
OPERABLE_FEATURE_NAMES = (
    "close_location_12",
    "close_location_24",
    "close_location_64",
    "range_position_12",
    "range_position_24",
    "range_position_64",
    "distance_to_high_12",
    "distance_to_low_12",
    "distance_to_high_24",
    "distance_to_low_24",
    "distance_ema9",
    "distance_ema21",
    "distance_ema200",
    "ema9_ema21_spread",
    "ema21_ema200_spread",
    "ema_stack_bullish",
    "ema_stack_bearish",
    "ema9_slope_6",
    "ema21_slope_12",
    "ema200_slope_64",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "adverse_wick_long",
    "adverse_wick_short",
    "rejection_wick_long",
    "rejection_wick_short",
    "volume_ratio_6",
    "volume_ratio_12",
    "volume_ratio_24",
    "volume_persistence_3",
    "volume_persistence_6",
    "volume_spike_z",
    "volume_dryup",
    "atr_ratio_14",
    "atr_percentile_64",
    "realized_vol_12",
    "realized_vol_24",
    "realized_vol_64",
    "range_expansion_12",
    "range_expansion_24",
    "return_15m",
    "return_30m",
    "return_60m",
    "return_120m",
    "consecutive_green",
    "consecutive_red",
    "momentum_acceleration_12",
    "trend_efficiency_12",
    "trend_efficiency_24",
    "trend_efficiency_64",
    "breakout_up_strength_12",
    "breakout_down_strength_12",
    "failed_breakout_up_risk",
    "failed_breakdown_risk",
    "close_back_inside_range",
    "sweep_high_reversal",
    "sweep_low_reversal",
)
BASE_FEATURE_INDEX = {
    "vol_norm": 3,
}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def _window(values: np.ndarray, idx: int, length: int) -> np.ndarray:
    return values[max(0, idx - length + 1) : idx + 1]


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().values.astype(np.float64)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    previous = np.roll(close, 1)
    previous[0] = close[0]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(true_range).ewm(alpha=1 / period, min_periods=1, adjust=False).mean().values.astype(np.float64)


def _consecutive(direction: np.ndarray, idx: int, expected: bool) -> float:
    count = 0
    for value in direction[: idx + 1][::-1]:
        if bool(value) is not expected:
            break
        count += 1
    return float(count)


def build_operable_feature_matrix_v2(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    steps: np.ndarray,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    base_features: np.ndarray | None = None,
) -> dict[str, Any]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low and close must have equal lengths")
    if np.any(steps < 0) or np.any(steps >= len(close)):
        raise ValueError("steps outside price arrays")
    inferred_open = open_ is None
    open_values = np.asarray(open_, dtype=np.float64) if open_ is not None else np.concatenate((close[:1], close[:-1]))
    if len(open_values) != len(close):
        raise ValueError("open must align with close")
    inferred_volume = volume is None
    if volume is not None:
        volume_values = np.asarray(volume, dtype=np.float64)
    elif base_features is not None and np.asarray(base_features).shape[1] > BASE_FEATURE_INDEX["vol_norm"]:
        volume_values = np.asarray(base_features, dtype=np.float64)[:, BASE_FEATURE_INDEX["vol_norm"]]
    else:
        volume_values = np.ones(len(close), dtype=np.float64)
    if len(volume_values) != len(close):
        raise ValueError("volume proxy must align with close")

    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema200 = _ema(close, 200)
    atr14 = _atr(high, low, close)
    returns = np.zeros(len(close), dtype=np.float64)
    returns[1:] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    green = close >= open_values
    red = close < open_values
    rows: list[list[float]] = []
    for idx in steps:
        price = max(float(close[idx]), 1e-12)
        row: list[float] = []
        for size in (12, 24, 64):
            window_high = _window(high, int(idx), size)
            window_low = _window(low, int(idx), size)
            row.append(np.clip(_safe_div(close[idx] - np.min(window_low), np.max(window_high) - np.min(window_low)), 0.0, 1.0))
        for size in (12, 24, 64):
            window_close = _window(close, int(idx), size)
            row.append(np.clip(_safe_div(close[idx] - np.min(window_close), np.max(window_close) - np.min(window_close)), 0.0, 1.0))
        for size in (12, 24):
            row.append(_safe_div(np.max(_window(high, int(idx), size)) - close[idx], price))
            row.append(_safe_div(close[idx] - np.min(_window(low, int(idx), size)), price))
        d9 = _safe_div(close[idx] - ema9[idx], price)
        d21 = _safe_div(close[idx] - ema21[idx], price)
        d200 = _safe_div(close[idx] - ema200[idx], price)
        row.extend((d9, d21, d200, _safe_div(ema9[idx] - ema21[idx], price), _safe_div(ema21[idx] - ema200[idx], price)))
        row.extend((float(ema9[idx] > ema21[idx] > ema200[idx]), float(ema9[idx] < ema21[idx] < ema200[idx])))
        row.extend((
            _safe_div(ema9[idx] - ema9[max(0, idx - 6)], price),
            _safe_div(ema21[idx] - ema21[max(0, idx - 12)], price),
            _safe_div(ema200[idx] - ema200[max(0, idx - 64)], price),
        ))
        candle_range = max(float(high[idx] - low[idx]), 1e-12)
        body_high = max(float(open_values[idx]), float(close[idx]))
        body_low = min(float(open_values[idx]), float(close[idx]))
        body = abs(float(close[idx] - open_values[idx]))
        upper_wick = max(0.0, float(high[idx]) - body_high)
        lower_wick = max(0.0, body_low - float(low[idx]))
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range
        row.extend((
            body / candle_range,
            upper_ratio,
            lower_ratio,
            upper_ratio,
            lower_ratio,
            upper_ratio * float(close[idx] < open_values[idx]),
            lower_ratio * float(close[idx] > open_values[idx]),
        ))
        current_volume = max(float(volume_values[idx]), 0.0)
        volume_ratios: list[float] = []
        for size in (6, 12, 24):
            average = float(np.mean(_window(volume_values, int(idx), size)))
            volume_ratios.append(_safe_div(current_volume, average))
        volume_window = _window(volume_values, int(idx), 24)
        volume_std = float(np.std(volume_window))
        volume_z = _safe_div(current_volume - float(np.mean(volume_window)), volume_std)
        row.extend((
            *volume_ratios,
            float(np.mean(_window(volume_values, int(idx), 3) > np.mean(_window(volume_values, int(idx), 12)))),
            float(np.mean(_window(volume_values, int(idx), 6) > np.mean(_window(volume_values, int(idx), 24)))),
            volume_z,
            float(volume_ratios[-1] < 0.70),
        ))
        atr_ratio = _safe_div(atr14[idx], price)
        atr_history = _window(atr14 / np.maximum(close, 1e-12), int(idx), 64)
        row.extend((
            atr_ratio,
            float(np.mean(atr_history <= atr_ratio)),
            float(np.std(_window(returns, int(idx), 12))),
            float(np.std(_window(returns, int(idx), 24))),
            float(np.std(_window(returns, int(idx), 64))),
            _safe_div(float(np.mean(_window(high - low, int(idx), 12))), float(np.mean(_window(high - low, int(idx), 64)))),
            _safe_div(float(np.mean(_window(high - low, int(idx), 24))), float(np.mean(_window(high - low, int(idx), 64)))),
        ))
        momentum_returns = [
            _safe_div(close[idx] - close[max(0, idx - bars)], close[max(0, idx - bars)])
            for bars in (3, 6, 12, 24)
        ]
        row.extend((*momentum_returns, _consecutive(green, int(idx), True), _consecutive(red, int(idx), True)))
        row.append(momentum_returns[1] - momentum_returns[2])
        for size in (12, 24, 64):
            price_path = _window(close, int(idx), size)
            path_move = abs(float(price_path[-1] - price_path[0]))
            path_travel = float(np.sum(np.abs(np.diff(price_path))))
            row.append(_safe_div(path_move, path_travel))
        prior_high = np.max(_window(high, max(0, int(idx) - 1), 12))
        prior_low = np.min(_window(low, max(0, int(idx) - 1), 12))
        up_strength = max(0.0, _safe_div(close[idx] - prior_high, price))
        down_strength = max(0.0, _safe_div(prior_low - close[idx], price))
        swept_high = high[idx] > prior_high and close[idx] <= prior_high
        swept_low = low[idx] < prior_low and close[idx] >= prior_low
        row.extend((
            up_strength,
            down_strength,
            float(swept_high),
            float(swept_low),
            float((low[idx] <= prior_high <= high[idx]) or (low[idx] <= prior_low <= high[idx])),
            float(swept_high and close[idx] < open_values[idx]),
            float(swept_low and close[idx] > open_values[idx]),
        ))
        rows.append(row)
    x = np.asarray(rows, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)
    return {
        "X_v2": x,
        "feature_names_v2": np.asarray(OPERABLE_FEATURE_NAMES),
        "diagnostics": {
            "schema_version": OPERABLE_FEATURE_SCHEMA_VERSION,
            "feature_count": int(len(OPERABLE_FEATURE_NAMES)),
            "causal_only": True,
            "open_source": "previous_close_proxy" if inferred_open else "provided_ohlcv_open",
            "volume_source": "base_vol_norm_proxy" if inferred_volume and base_features is not None else ("unit_proxy" if inferred_volume else "provided_ohlcv_volume"),
            "btc_eth_context_included": False,
            "btc_eth_context_pending_reason": "SignalMarket does not expose aligned context markets in research dataset.",
        },
    }


def apply_feature_set(dataset: dict[str, Any], market: Any, feature_set: str = "base") -> dict[str, Any]:
    normalized = feature_set.lower()
    if normalized not in {"base", "operable_v2", "combined"}:
        raise ValueError(f"unsupported feature_set: {feature_set}")
    base_x = np.asarray(dataset["X"], dtype=np.float32)
    base_names = np.asarray(dataset["feature_names"]).astype(str)
    built = build_operable_feature_matrix_v2(
        high=market.high,
        low=market.low,
        close=market.close,
        steps=np.asarray(dataset["step"], dtype=np.int64),
        base_features=getattr(market, "features", None),
    )
    new_x = np.asarray(built["X_v2"], dtype=np.float32)
    new_names = np.asarray(built["feature_names_v2"]).astype(str)
    if normalized == "base":
        selected_x, selected_names = base_x, base_names
    elif normalized == "operable_v2":
        selected_x, selected_names = new_x, new_names
    else:
        selected_x = np.concatenate((base_x, new_x), axis=1)
        selected_names = np.concatenate((base_names, new_names))
    if len(np.unique(selected_names)) != len(selected_names):
        raise ValueError("feature_names must remain unique")
    updated = dict(dataset)
    updated["X"] = np.nan_to_num(selected_x, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    updated["feature_names"] = selected_names
    updated["feature_set"] = normalized
    updated["base_feature_count"] = int(base_x.shape[1])
    updated["new_feature_count"] = int(new_x.shape[1])
    updated["feature_diagnostics"] = built["diagnostics"]
    return updated
