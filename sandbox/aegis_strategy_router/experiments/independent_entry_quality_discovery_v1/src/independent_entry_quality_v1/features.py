"""Frozen causal feature allowlist and snapshot extraction."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from aegis_strategy_router.domain.serialization import canonical_json_bytes
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, Side, Timeframe


TIMEFRAMES = (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
SIGNED = (
    "return_1_bps", "return_3_bps", "return_6_bps",
    "ema7_extension_atr", "ema25_extension_atr", "ema99_extension_atr",
    "ema7_slope_atr", "ema25_slope_atr", "trend_age", "prior_move_6_atr",
    "taker_imbalance",
)
PRICE_PATH = ("return_1_bps", "return_3_bps", "return_6_bps", "path_efficiency_6", "body_ratio", "clv")
TREND = (
    "ema7_extension_atr", "ema25_extension_atr", "ema99_extension_atr",
    "ema7_slope_atr", "ema25_slope_atr", "trend_age", "prior_move_6_atr",
    "rsi6", "rsi12", "rsi24", "breakout_up", "breakout_down",
)
VOL_VOLUME = ("atr_pct_bps", "atr_percentile_96", "volume_ratio20", "volume_z50", "range_48_atr")


def extract_features(snapshot: MarketSnapshot, side: Side) -> tuple[dict[str, float], dict[str, str], str]:
    direction = 1.0 if side is Side.LONG else -1.0
    values: dict[str, float] = {"context__side_sign": direction}
    groups: dict[str, str] = {"context__side_sign": "PRICE_PATH"}
    availability = []
    for state in snapshot.timeframes:
        if state.status is not DataStatus.AVAILABLE:
            raise ValueError(f"TIMEFRAME_NOT_AVAILABLE:{state.timeframe.value}")
        raw = {}
        for observation in state.features.observations:
            if observation.status is not DataStatus.AVAILABLE or observation.value is None:
                raise ValueError(f"FEATURE_NOT_AVAILABLE:{observation.name}")
            if observation.available_at is None or observation.available_at > snapshot.decision_at:
                raise ValueError(f"FEATURE_FUTURE_AVAILABILITY:{observation.name}")
            raw[observation.name.split("__", 1)[1]] = float(observation.value)
            availability.append(observation.available_at)
        prefix = f"tf{state.timeframe.value}__"
        for name in PRICE_PATH:
            if name in SIGNED:
                key, value = prefix + "directional_" + name, direction * raw[name]
            elif name == "clv":
                key, value = prefix + "directional_clv", direction * (2.0 * raw[name] - 1.0)
            else:
                key, value = prefix + name, raw[name]
            values[key], groups[key] = value, "PRICE_PATH"
        velocity = direction * raw["return_1_bps"]
        acceleration = velocity - direction * raw["return_3_bps"] / 3.0
        values[prefix + "directional_velocity_1"] = velocity
        values[prefix + "directional_acceleration_proxy"] = acceleration
        groups[prefix + "directional_velocity_1"] = groups[prefix + "directional_acceleration_proxy"] = "PRICE_PATH"
        for name in TREND:
            if name in SIGNED:
                key, value = prefix + "directional_" + name, direction * raw[name]
                values[key], groups[key] = value, "TREND_EXTENSION"
            elif name.startswith("rsi"):
                extension = prefix + name + "_directional_extension"
                room = prefix + name + "_remaining_room"
                values[extension] = direction * (raw[name] - 50.0)
                values[room] = 100.0 - raw[name] if side is Side.LONG else raw[name]
                groups[extension] = groups[room] = "TREND_EXTENSION"
            elif name == "breakout_up":
                aligned = prefix + "aligned_breakout"
                values[aligned] = raw["breakout_up"] if side is Side.LONG else raw["breakout_down"]
                groups[aligned] = "TREND_EXTENSION"
            elif name == "breakout_down":
                opposed = prefix + "opposed_breakout"
                values[opposed] = raw["breakout_down"] if side is Side.LONG else raw["breakout_up"]
                groups[opposed] = "TREND_EXTENSION"
        for name in VOL_VOLUME:
            key = prefix + name
            values[key], groups[key] = raw[name], "VOLATILITY_VOLUME"
        favorable_space = raw["distance_recent_high_atr"] if side is Side.LONG else raw["distance_recent_low_atr"]
        adverse_space = raw["distance_recent_low_atr"] if side is Side.LONG else raw["distance_recent_high_atr"]
        values[prefix + "favorable_recent_space_atr"] = favorable_space
        values[prefix + "adverse_recent_space_atr"] = adverse_space
        groups[prefix + "favorable_recent_space_atr"] = groups[prefix + "adverse_recent_space_atr"] = "STRUCTURE_LOCATION"
        flow = direction * raw["taker_imbalance"]
        response = direction * raw["return_1_bps"]
        for key, value in (
            (prefix + "directional_taker_imbalance", flow),
            (prefix + "flow_price_response_product", flow * response),
            (prefix + "price_response_per_abs_flow", response / (abs(flow) + 1.0e-6)),
        ):
            values[key], groups[key] = value, "FLOW_RESPONSE"
        values[prefix + "warmup_margin_bars"] = float(state.candle_count - state.required_warmup_bars)
        groups[prefix + "warmup_margin_bars"] = "QUALITY_SUPPORT"
        structural = state.structural
        if structural is not None:
            if structural.status is not DataStatus.AVAILABLE:
                raise ValueError(f"STRUCTURE_NOT_AVAILABLE:{state.timeframe.value}")
            favorable = structural.nearest_above if side is Side.LONG else structural.nearest_below
            adverse = structural.nearest_below if side is Side.LONG else structural.nearest_above
            structural_values = {
                "structural_atr_bps": float(structural.atr14 / snapshot.reference_price * 10_000.0),
                "structural_level_count": float(len(structural.levels)),
                "structural_pivot_count": float(len(structural.pivots)),
                "favorable_level_distance_atr": float(favorable.distance_atr) if favorable else 99.0,
                "favorable_level_distance_bps": float(favorable.distance_bps) if favorable else 99_999.0,
                "adverse_level_distance_atr": float(adverse.distance_atr) if adverse else 99.0,
                "adverse_level_distance_bps": float(adverse.distance_bps) if adverse else 99_999.0,
            }
            for name, value in structural_values.items():
                key = prefix + name
                values[key], groups[key] = value, "STRUCTURE_LOCATION"
            availability.extend(pivot.available_at for pivot in structural.pivots)
            availability.extend(level.available_at for level in structural.levels)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("NON_FINITE_ALLOWLIST_FEATURE")
    latest = max(availability).isoformat() if availability else snapshot.decision_at.isoformat()
    return values, groups, latest


def add_cross_market_features(frame: Any) -> tuple[Any, dict[str, str]]:
    """Join neutral BTC/ETH state at the same causal timestamp."""
    import pandas as pd

    result = frame.copy()
    groups: dict[str, str] = {}
    source_columns = [
        column for column in result.columns
        if column.startswith("feature__tf") and any(
            token in column for token in ("directional_return_3_bps", "atr_percentile_96", "volume_ratio20")
        )
    ]
    # LONG rows preserve raw directional sign; convert reference returns back to raw.
    neutral = result.loc[result.side.eq("LONG"), ["decision_at", "symbol", *source_columns]].copy()
    for reference in ("BTCUSDT", "ETHUSDT"):
        reference_frame = neutral.loc[neutral.symbol.eq(reference)].drop(columns="symbol")
        rename = {column: f"feature__cross__{reference.lower()}__{column.removeprefix('feature__')}" for column in source_columns}
        reference_frame = reference_frame.rename(columns=rename)
        result = result.merge(reference_frame, on="decision_at", how="left", validate="many_to_one")
        for name in rename.values():
            groups[name.removeprefix("feature__")] = "CROSS_MARKET"
    directional_return_columns = [
        column for column in source_columns if "directional_return_3_bps" in column
    ]
    for own in directional_return_columns:
        suffix = own.removeprefix("feature__")
        btc = f"feature__cross__btcusdt__{suffix}"
        key = f"feature__cross__relative_vs_btc__{suffix}"
        # Reference is raw LONG-oriented; orient it to the row side before subtraction.
        result[key] = result[own] - result["feature__context__side_sign"] * result[btc]
        groups[key.removeprefix("feature__")] = "CROSS_MARKET"
    breadth_base = neutral.loc[:, ["decision_at", *directional_return_columns]].copy()
    breadth_base["positive"] = breadth_base[directional_return_columns[0]].gt(0).astype(float)
    breadth = breadth_base.groupby("decision_at", as_index=False).positive.mean().rename(columns={"positive": "raw_breadth"})
    result = result.merge(breadth, on="decision_at", how="left", validate="many_to_one")
    result["feature__cross__directional_breadth"] = result["raw_breadth"].where(
        result.side.eq("LONG"), 1.0 - result["raw_breadth"]
    )
    result = result.drop(columns="raw_breadth")
    groups["cross__directional_breadth"] = "CROSS_MARKET"
    return result, groups


def feature_hash(row: dict[str, Any]) -> str:
    payload = {key: row[key] for key in sorted(row) if key.startswith("feature__")}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


FORBIDDEN_INPUT_TOKENS = (
    "target__", "future", "mfe", "mae", "outcome", "label", "pnl", "realized", "barrier_first"
)


def assert_feature_allowlist(columns: list[str]) -> None:
    leaked = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if leaked:
        raise ValueError(f"LEAKAGE_FEATURES:{sorted(leaked)}")
    if any(not column.startswith("feature__") for column in columns):
        raise ValueError("NON_ALLOWLIST_FEATURE_COLUMN")


def dictionary_payload(groups: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "independent-entry-quality-feature-dictionary-v1",
        "features": [
            {"name": f"feature__{name}", "family": family, "availability": "at_or_before_decision_at", "dtype": "float64"}
            for name, family in sorted(groups.items())
        ],
        "forbidden_tokens": list(FORBIDDEN_INPUT_TOKENS),
    }
