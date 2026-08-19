"""Causal flow-effectiveness and cross-market propagation features."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from aegis_strategy_router.domain.serialization import canonical_json_bytes


FORBIDDEN = ("target__", "future", "mfe", "mae", "pnl", "outcome", "aegis", "candidate_strategy")


def add_directional_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    result = frame.copy()
    groups: dict[str, str] = {}
    timeframes = ("1m", "5m", "15m", "1h")
    flow_columns = []
    response_columns = []
    for timeframe in timeframes:
        flow = f"feature__tf{timeframe}__directional_taker_imbalance"
        response = f"feature__tf{timeframe}__directional_return_1_bps"
        flow_columns.append(flow)
        response_columns.append(response)
        for source, suffix in (
            (flow, "flow"),
            (response, "price_response"),
            (f"feature__tf{timeframe}__flow_price_response_product", "flow_response_product"),
            (f"feature__tf{timeframe}__price_response_per_abs_flow", "response_per_abs_flow"),
        ):
            key = f"feature__directional_flow__tf{timeframe}__{suffix}"
            result[key] = result[source]
            groups[key] = "FLOW"
        product = result[flow] * result[response]
        absorption = np.maximum(0.0, np.abs(result[flow]) - np.abs(result[response]) / 10.0)
        resistance = np.where(result[flow] < 0.0, np.maximum(0.0, result[response]), 0.0)
        for suffix, value in (
            ("effectiveness", product), ("flow_without_response", absorption),
            ("adverse_flow_resistance", resistance),
        ):
            key = f"feature__directional_flow__tf{timeframe}__{suffix}"
            result[key] = value
            groups[key] = "FLOW"
    flow_matrix = result[flow_columns].to_numpy(float)
    response_matrix = result[response_columns].to_numpy(float)
    sequence = {
        "flow_short_minus_medium": flow_matrix[:, 0] - flow_matrix[:, 1],
        "flow_medium_minus_long": flow_matrix[:, 1] - flow_matrix[:, 2],
        "flow_acceleration": (flow_matrix[:, 0] - flow_matrix[:, 1]) - (flow_matrix[:, 1] - flow_matrix[:, 2]),
        "flow_persistence": np.mean(flow_matrix > 0.0, axis=1),
        "flow_reversal": np.sign(flow_matrix[:, 0]) != np.sign(flow_matrix[:, 2]),
        "response_short_minus_medium": response_matrix[:, 0] - response_matrix[:, 1],
        "response_acceleration": (response_matrix[:, 0] - response_matrix[:, 1]) - (response_matrix[:, 1] - response_matrix[:, 2]),
        "effectiveness_decay": flow_matrix[:, 0] * response_matrix[:, 0] - flow_matrix[:, 2] * response_matrix[:, 2],
    }
    for suffix, value in sequence.items():
        key = f"feature__directional_flow__sequence__{suffix}"
        result[key] = np.asarray(value, dtype=float)
        groups[key] = "FLOW"

    long_rows = result.side.eq("LONG")
    neutral = result.loc[long_rows].copy()
    state_columns = [
        f"feature__tf{tf}__directional_return_3_bps" for tf in ("1m", "5m", "15m", "1h", "4h")
    ]
    neutral_state = neutral[["decision_at", "symbol", *state_columns]].copy()
    for column in state_columns:
        suffix = column.split("__tf", 1)[1].replace("__directional_", "__")
        pivot = neutral_state.pivot(index="decision_at", columns="symbol", values=column)
        breadth = pivot.gt(0.0).mean(axis=1)
        dispersion = pivot.std(axis=1, ddof=0)
        rank = neutral_state.groupby("decision_at")[column].rank(pct=True, method="average")
        lookup = neutral_state[["decision_at", "symbol"]].copy()
        lookup[f"raw_breadth__{suffix}"] = lookup.decision_at.map(breadth)
        lookup[f"dispersion__{suffix}"] = lookup.decision_at.map(dispersion)
        lookup[f"rank__{suffix}"] = rank.to_numpy(float)
        result = result.merge(lookup, on=["decision_at", "symbol"], how="left", validate="many_to_one")
        current_sign = result["feature__context__side_sign"]
        for raw_name, oriented_name in (
            (f"raw_breadth__{suffix}", f"feature__propagation__directional_breadth__{suffix}"),
            (f"dispersion__{suffix}", f"feature__propagation__dispersion__{suffix}"),
            (f"rank__{suffix}", f"feature__propagation__directional_rank__{suffix}"),
        ):
            if raw_name.startswith("raw_breadth") or raw_name.startswith("rank"):
                result[oriented_name] = np.where(current_sign > 0, result[raw_name], 1.0 - result[raw_name])
            else:
                result[oriented_name] = result[raw_name]
            groups[oriented_name] = "CROSS_MARKET"
            result = result.drop(columns=raw_name)
    sign = result["feature__context__side_sign"]
    btc_1h = result["feature__cross__btcusdt__tf1h__directional_return_3_bps"] * sign
    eth_1h = result["feature__cross__ethusdt__tf1h__directional_return_3_bps"] * sign
    own_1h = result["feature__tf1h__directional_return_3_bps"]
    propagation = {
        "btc_eth_agreement": np.sign(btc_1h) == np.sign(eth_1h),
        "btc_eth_directional_mean": (btc_1h + eth_1h) / 2.0,
        "btc_directional_acceleration": sign * (
            result["feature__cross__btcusdt__tf15m__directional_return_3_bps"]
            - result["feature__cross__btcusdt__tf1h__directional_return_3_bps"] / 4.0
        ),
        "eth_directional_acceleration": sign * (
            result["feature__cross__ethusdt__tf15m__directional_return_3_bps"]
            - result["feature__cross__ethusdt__tf1h__directional_return_3_bps"] / 4.0
        ),
        "asset_residual_vs_btc_1h": own_1h - btc_1h,
        "asset_lag_vs_btc_1h": btc_1h - own_1h,
    }
    for suffix, value in propagation.items():
        key = f"feature__propagation__{suffix}"
        result[key] = np.asarray(value, dtype=float)
        groups[key] = "CROSS_MARKET"
    breadth_1m = result["feature__propagation__directional_breadth__1m__return_3_bps"]
    breadth_15m = result["feature__propagation__directional_breadth__15m__return_3_bps"]
    result["feature__propagation__breadth_acceleration"] = breadth_1m - breadth_15m
    groups["feature__propagation__breadth_acceleration"] = "CROSS_MARKET"
    feature_columns = sorted(groups)
    if result[feature_columns].isna().any().any() or not np.isfinite(result[feature_columns].to_numpy(float)).all():
        raise ValueError("DIRECTIONAL_FEATURES_NON_FINITE")
    assert_allowlist(feature_columns)
    return result, groups


def assert_allowlist(columns: list[str]) -> None:
    leaked = [column for column in columns if any(token in column.lower() for token in FORBIDDEN)]
    if leaked:
        raise ValueError(f"DIRECTIONAL_FEATURE_LEAKAGE:{leaked}")
    if any(not column.startswith("feature__") for column in columns):
        raise ValueError("DIRECTIONAL_NON_FEATURE_COLUMN")


def feature_hash(row: dict[str, Any], columns: list[str]) -> str:
    payload = {key: row[key] for key in sorted(columns)}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def dictionary_payload(groups: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "directional-alpha-v1-feature-dictionary",
        "features": [
            {"name": name, "family": family, "availability": "at_or_before_decision_at", "dtype": "float64"}
            for name, family in sorted(groups.items())
        ],
        "forbidden_tokens": list(FORBIDDEN),
    }
