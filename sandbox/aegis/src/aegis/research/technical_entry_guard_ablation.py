"""Frozen factor definitions for the technical entry guard ablation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def factor_state(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, bool]:
    rsi_room = [_number(row, f"dir{tf}m__rsi6_remaining_room") for tf in (5, 15)]
    ema = [_number(row, f"dir{tf}m__ema25_extension_atr") for tf in (5, 15)]
    prior = [_number(row, f"dir{tf}m__prior_move_6_atr") for tf in (5, 15)]
    space = [_number(row, f"dir{tf}m__favorable_space_atr") for tf in (5, 15)]
    rsi_extreme = any(value <= thresholds["rsi_remaining_room"] for value in rsi_room)
    ema_opposed = all(value < thresholds["ema25_opposed_atr"] for value in ema)
    ema_extended = any(value >= thresholds["ema25_extended_atr"] for value in ema)
    mature = any(value >= thresholds["prior_move_atr"] for value in prior)
    no_space = any(value <= thresholds["favorable_space_atr"] for value in space)
    price_opposed = (
        _number(row, "dir1m__return_3_bps") <= thresholds["price_return_1m_bps"]
        and _number(row, "dir5m__return_1_bps") <= thresholds["price_return_5m_bps"]
    )
    flow_opposed = (
        _number(row, "dir1m__taker_imbalance") <= thresholds["taker_imbalance"]
        and _number(row, "dir5m__taker_imbalance") <= thresholds["taker_imbalance"]
    )
    shock = (
        max(_number(row, "tf5m__atr_percentile_96"), _number(row, "tf15m__atr_percentile_96")) >= thresholds["atr_percentile"]
        and max(_number(row, "tf5m__volume_ratio20"), _number(row, "tf15m__volume_ratio20")) >= thresholds["volume_ratio"]
        and min(_number(row, "tf5m__path_efficiency_6"), _number(row, "tf15m__path_efficiency_6")) <= thresholds["path_efficiency"]
    )
    return {
        "rsi_extreme": rsi_extreme,
        "ema_opposed": ema_opposed,
        "ema_extended": ema_extended,
        "mature": mature,
        "no_space": no_space,
        "price_opposed": price_opposed,
        "flow_opposed": flow_opposed,
        "shock": shock,
    }


def policy_skips(state: Mapping[str, bool]) -> dict[str, bool]:
    late = state["rsi_extreme"] or state["ema_extended"] or state["mature"]
    return {
        "ema_only": state["ema_opposed"],
        "rsi_only": state["rsi_extreme"],
        "structure_only": state["no_space"],
        "rsi_and_structure": state["rsi_extreme"] and state["no_space"],
        "rsi_and_extension": state["rsi_extreme"] and (state["ema_extended"] or state["mature"]),
        "ema_and_structure": state["ema_opposed"] and state["no_space"],
        "ema_and_price": state["ema_opposed"] and state["price_opposed"],
        "ema_and_flow": state["ema_opposed"] and state["flow_opposed"],
        "price_and_flow": state["price_opposed"] and state["flow_opposed"],
        "ema_price_flow": state["ema_opposed"] and state["price_opposed"] and state["flow_opposed"],
        "late_and_structure": late and state["no_space"],
        "volatility_only": state["shock"],
    }


def add_policy_columns(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict(orient="records"):
        state = factor_state(row, thresholds)
        rows.append({**{f"factor__{key}": value for key, value in state.items()}, **policy_skips(state)})
    return pd.concat([frame.copy(), pd.DataFrame(rows, index=frame.index)], axis=1)

