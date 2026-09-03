"""Causal entry timing and multi-timeframe feature contract for M1C."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .market_event_economic_path_m1b import FEATURE_NAMES, M1BContractError, feature_row


ADDED_FEATURE_NAMES = (
    "side_return_5m",
    "side_return_15m",
    "side_return_240m",
    "side_taker_flow_5m",
    "side_taker_flow_15m",
    "realized_volatility_5m",
    "realized_volatility_15m",
    "realized_volatility_60m",
    "realized_volatility_240m",
    "favorable_range_location_15m",
    "favorable_range_location_60m",
    "favorable_range_location_240m",
    "side_trend_distance_sma_15m",
    "side_trend_distance_sma_60m",
    "side_trend_distance_sma_240m",
)
M1C_FEATURE_NAMES = FEATURE_NAMES + ADDED_FEATURE_NAMES


def add_multitimeframe_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"]
    returns = close.pct_change()
    for bars in (5, 15, 240):
        result[f"ret_{bars}"] = close / close.shift(bars) - 1.0
    for bars in (5, 15):
        result[f"flow_{bars}"] = result["flow_1"].rolling(bars, min_periods=bars).mean()
    for bars in (5, 15, 60, 240):
        result[f"realized_volatility_{bars}"] = returns.rolling(
            bars, min_periods=bars
        ).std(ddof=0)
    for bars in (15, 60, 240):
        low = result["low"].rolling(bars, min_periods=bars).min()
        high = result["high"].rolling(bars, min_periods=bars).max()
        result[f"range_location_{bars}"] = (close - low) / (high - low).replace(
            0.0, np.nan
        )
        average = close.rolling(bars, min_periods=bars).mean()
        result[f"trend_distance_sma_{bars}"] = close / average - 1.0
    return result


def m1c_feature_row(row: Mapping[str, Any], side: str = "LONG") -> tuple[float, ...]:
    sign = 1.0 if side == "LONG" else -1.0
    base = feature_row(row, side)
    added = (
        sign * float(row["ret_5"]),
        sign * float(row["ret_15"]),
        sign * float(row["ret_240"]),
        sign * float(row["flow_5"]),
        sign * float(row["flow_15"]),
        float(row["realized_volatility_5"]),
        float(row["realized_volatility_15"]),
        float(row["realized_volatility_60"]),
        float(row["realized_volatility_240"]),
        float(row["range_location_15"]),
        float(row["range_location_60"]),
        float(row["range_location_240"]),
        sign * float(row["trend_distance_sma_15"]),
        sign * float(row["trend_distance_sma_60"]),
        sign * float(row["trend_distance_sma_240"]),
    )
    values = base + added
    if len(values) != len(M1C_FEATURE_NAMES) or not np.isfinite(values).all():
        raise M1BContractError("AEGIS_M1C_FEATURE_CONTRACT_INVALID")
    return values


def pullback_reclaim_confirmation(
    frame: pd.DataFrame,
    *,
    event_open_time: int,
    observation_bars: int = 5,
) -> tuple[int, int] | None:
    """Return confirmation candle and next-bar entry timestamps."""

    indexed = frame.set_index("open_time", drop=False)
    if event_open_time not in indexed.index:
        raise M1BContractError("AEGIS_M1C_EVENT_CANDLE_MISSING")
    event = indexed.loc[event_open_time]
    event_close = float(event["close"])
    prior_high = float(event["prior_high"])
    for offset in range(1, observation_bars + 1):
        confirmation_time = event_open_time + offset * 60_000
        entry_time = confirmation_time + 60_000
        if confirmation_time not in indexed.index or entry_time not in indexed.index:
            return None
        candle = indexed.loc[confirmation_time]
        if (
            float(candle["low"]) <= event_close
            and float(candle["close"]) >= prior_high
            and float(candle["close"]) > float(candle["open"])
        ):
            return confirmation_time, entry_time
    return None


def validate_feature_order(names: Sequence[str]) -> None:
    if tuple(names) != M1C_FEATURE_NAMES:
        raise M1BContractError("AEGIS_M1C_FEATURE_ORDER_INVALID")
