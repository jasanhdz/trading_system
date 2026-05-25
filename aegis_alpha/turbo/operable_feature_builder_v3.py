#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from aegis_alpha.turbo.operable_feature_builder_v2 import (
    OPERABLE_FEATURE_NAMES,
    _atr,
    _ema,
    _safe_div,
    _window,
    apply_feature_set as apply_feature_set_v2,
    build_operable_feature_matrix_v2,
)


OPERABLE_FEATURE_SCHEMA_VERSION_V3 = "aegis_turbo_operable_features_v3_short_research_v1"
OPERABLE_FEATURE_NAMES_V3 = (
    "short_breakdown_strength_12",
    "short_breakdown_strength_24",
    "short_breakdown_close_beyond_range_12",
    "short_breakdown_close_beyond_range_24",
    "short_breakdown_followthrough_3",
    "short_breakdown_followthrough_6",
    "short_breakdown_volume_confirmed",
    "short_breakdown_body_confirmed",
    "short_failed_breakdown_risk_12",
    "short_failed_breakdown_risk_24",
    "short_lower_wick_sweep_risk",
    "short_close_back_inside_range",
    "short_reclaim_range_risk",
    "short_reversal_after_low_sweep",
    "short_absorption_risk",
    "short_extension_below_ema21",
    "short_extension_below_ema200",
    "short_distance_from_ema_stack",
    "short_downmove_age",
    "short_consecutive_red_exhaustion",
    "short_volume_climax_risk",
    "short_volatility_exhaustion_risk",
    "short_breakdown_retest_distance",
    "short_retest_failed",
    "short_retest_success",
    "short_retest_volume_dryup",
    "short_retest_rejection_wick",
    "short_distance_to_recent_low_12",
    "short_distance_to_recent_low_24",
    "short_distance_to_recent_high_12",
    "short_distance_to_recent_high_24",
    "short_room_to_fall_12",
    "short_room_to_fall_24",
    "short_overhead_risk_12",
    "short_overhead_risk_24",
    "local_chop_score",
    "local_trend_down_score",
    "local_momentum_down_score",
    "local_breakdown_score",
    "local_exhaustion_score",
    "local_high_vol_risk",
    "local_transition_risk",
    "btc_return_15m",
    "btc_return_30m",
    "btc_return_60m",
    "btc_return_120m",
    "eth_return_15m",
    "eth_return_30m",
    "eth_return_60m",
    "eth_return_120m",
    "btc_ema_trend_short",
    "eth_ema_trend_short",
    "btc_eth_short_agreement",
    "btc_eth_long_contradiction",
    "btc_eth_mixed_context",
    "symbol_vs_btc_relative_strength_30m",
    "symbol_vs_eth_relative_strength_30m",
    "symbol_underperforming_btc_60m",
    "symbol_underperforming_eth_60m",
)
FEATURE_SETS = ("base", "operable_v2", "operable_v3", "combined", "combined_v3")


def _feature_hash(names: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(name) for name in names.tolist()).encode("utf-8")).hexdigest()


def _return(close: np.ndarray, idx: int, bars: int) -> float:
    return _safe_div(close[idx] - close[max(0, idx - bars)], close[max(0, idx - bars)])


def build_operable_feature_matrix_v3(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    steps: np.ndarray,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    base_features: np.ndarray | None = None,
    context_markets: dict[str, Any] | None = None,
    include_market_breadth: bool = False,
) -> dict[str, Any]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    timestamps = np.asarray(timestamps).astype(str)
    steps = np.asarray(steps, dtype=np.int64)
    if not (len(high) == len(low) == len(close) == len(timestamps)):
        raise ValueError("OHLC timestamps must align")
    if np.any(steps < 0) or np.any(steps >= len(close)):
        raise ValueError("steps outside price arrays")
    open_values = np.asarray(open_, dtype=np.float64) if open_ is not None else np.concatenate((close[:1], close[:-1]))
    if volume is not None:
        volumes = np.asarray(volume, dtype=np.float64)
    elif base_features is not None and np.asarray(base_features).shape[1] > 3:
        volumes = np.asarray(base_features, dtype=np.float64)[:, 3]
    else:
        volumes = np.ones(len(close), dtype=np.float64)
    ema21 = _ema(close, 21)
    ema200 = _ema(close, 200)
    atr = _atr(high, low, close)
    returns = np.zeros(len(close), dtype=np.float64)
    returns[1:] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    context_markets = context_markets or {}
    context_data: dict[str, dict[str, Any]] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        context_market = context_markets.get(symbol)
        if context_market is not None:
            context_close = np.asarray(context_market.close, dtype=np.float64)
            context_data[symbol] = {
                "close": context_close,
                "index": {
                    str(timestamp): position
                    for position, timestamp in enumerate(np.asarray(context_market.timestamps).astype(str))
                },
                "ema9": _ema(context_close, 9),
                "ema21": _ema(context_close, 21),
            }
    rows: list[list[float]] = []
    aligned_context_rows = 0

    for idx in steps:
        idx = int(idx)
        price = max(float(close[idx]), 1e-12)
        prior_idx = max(0, idx - 1)
        prior_high_12 = float(np.max(_window(high, prior_idx, 12)))
        prior_high_24 = float(np.max(_window(high, prior_idx, 24)))
        prior_low_12 = float(np.min(_window(low, prior_idx, 12)))
        prior_low_24 = float(np.min(_window(low, prior_idx, 24)))
        range_12 = max(prior_high_12 - prior_low_12, 1e-12)
        range_24 = max(prior_high_24 - prior_low_24, 1e-12)
        candle_range = max(float(high[idx] - low[idx]), 1e-12)
        body = abs(float(close[idx] - open_values[idx]))
        lower_wick = max(0.0, min(float(open_values[idx]), float(close[idx])) - float(low[idx]))
        upper_wick = max(0.0, float(high[idx]) - max(float(open_values[idx]), float(close[idx])))
        lower_wick_ratio = lower_wick / candle_range
        upper_wick_ratio = upper_wick / candle_range
        break_12 = max(0.0, _safe_div(prior_low_12 - close[idx], price))
        break_24 = max(0.0, _safe_div(prior_low_24 - close[idx], price))
        swept_12 = low[idx] < prior_low_12 and close[idx] >= prior_low_12
        swept_24 = low[idx] < prior_low_24 and close[idx] >= prior_low_24
        returns_3 = _return(close, idx, 3)
        returns_6 = _return(close, idx, 6)
        returns_12 = _return(close, idx, 12)
        returns_24 = _return(close, idx, 24)
        vol_window = _window(volumes, idx, 24)
        current_volume_ratio = _safe_div(volumes[idx], float(np.mean(vol_window)))
        vol_std = float(np.std(vol_window))
        volume_z = _safe_div(volumes[idx] - float(np.mean(vol_window)), vol_std)
        atr_ratio = _safe_div(atr[idx], price)
        atr_history = _window(atr / np.maximum(close, 1e-12), idx, 64)
        atr_percentile = float(np.mean(atr_history <= atr_ratio))
        recent_red = float(np.sum(np.diff(_window(close, idx, 8)) < 0.0))
        retest_distance = _safe_div(close[idx] - prior_low_12, price)
        breakdown_present = break_12 > 0.0 or break_24 > 0.0
        recent_low_distance_12 = _safe_div(close[idx] - prior_low_12, price)
        recent_low_distance_24 = _safe_div(close[idx] - prior_low_24, price)
        recent_high_distance_12 = _safe_div(prior_high_12 - close[idx], price)
        recent_high_distance_24 = _safe_div(prior_high_24 - close[idx], price)
        trend_efficiency = _safe_div(
            abs(float(close[idx] - close[max(0, idx - 12)])),
            float(np.sum(np.abs(np.diff(_window(close, idx, 12))))),
        )
        chop = np.clip(1.0 - trend_efficiency, 0.0, 1.0)
        trend_down = np.clip((-returns_12 / max(atr_ratio, 1e-6)) * 0.25, 0.0, 1.0)
        momentum_down = np.clip((-returns_6 - returns_12) / max(atr_ratio, 1e-6) * 0.20, 0.0, 1.0)
        exhaustion = np.clip(
            max(0.0, -returns_12) / max(atr_ratio, 1e-6) * 0.12
            + lower_wick_ratio * 0.35
            + max(0.0, volume_z) * 0.08,
            0.0,
            1.0,
        )
        transition = np.clip(chop * 0.45 + float(swept_12 or swept_24) * 0.35 + lower_wick_ratio * 0.20, 0.0, 1.0)
        local = [
            break_12,
            break_24,
            float(close[idx] < prior_low_12),
            float(close[idx] < prior_low_24),
            float(returns_3 < 0.0 and break_12 > 0.0),
            float(returns_6 < 0.0 and break_12 > 0.0),
            float(breakdown_present and current_volume_ratio > 1.0),
            float(breakdown_present and close[idx] < open_values[idx] and body / candle_range > 0.50),
            float(swept_12),
            float(swept_24),
            lower_wick_ratio * float(low[idx] < prior_low_12),
            float((swept_12 or swept_24) and close[idx] >= prior_low_12),
            float(close[idx] > prior_low_12 and low[idx] < prior_low_12),
            float(swept_12 and close[idx] > open_values[idx]),
            lower_wick_ratio * max(0.0, current_volume_ratio),
            max(0.0, _safe_div(ema21[idx] - close[idx], price)),
            max(0.0, _safe_div(ema200[idx] - close[idx], price)),
            max(0.0, _safe_div(min(ema21[idx], ema200[idx]) - close[idx], price)),
            min(recent_red / 7.0, 1.0),
            float(recent_red >= 5 and lower_wick_ratio > 0.25),
            float(volume_z > 1.5 and returns_3 < 0.0),
            float(atr_percentile > 0.85 and returns_12 < 0.0),
            retest_distance,
            float(breakdown_present and close[idx] <= prior_low_12 and upper_wick_ratio > lower_wick_ratio),
            float(swept_12),
            float(breakdown_present and current_volume_ratio < 0.8),
            upper_wick_ratio * float(close[idx] < open_values[idx]),
            recent_low_distance_12,
            recent_low_distance_24,
            recent_high_distance_12,
            recent_high_distance_24,
            max(0.0, recent_low_distance_12),
            max(0.0, recent_low_distance_24),
            max(0.0, -recent_high_distance_12),
            max(0.0, -recent_high_distance_24),
            chop,
            trend_down,
            momentum_down,
            np.clip(max(break_12, break_24) / max(atr_ratio, 1e-6), 0.0, 1.0),
            exhaustion,
            float(atr_percentile > 0.85),
            transition,
        ]
        btc = context_data.get("BTCUSDT")
        eth = context_data.get("ETHUSDT")
        btc_idx = btc["index"].get(timestamps[idx]) if btc else None
        eth_idx = eth["index"].get(timestamps[idx]) if eth else None
        context = [0.0] * 17
        if btc is not None and btc_idx is not None and eth is not None and eth_idx is not None:
            btc_close = btc["close"]
            eth_close = eth["close"]
            btc_returns = [_return(btc_close, btc_idx, bars) for bars in (3, 6, 12, 24)]
            eth_returns = [_return(eth_close, eth_idx, bars) for bars in (3, 6, 12, 24)]
            btc_trend = float(btc["ema9"][btc_idx] < btc["ema21"][btc_idx])
            eth_trend = float(eth["ema9"][eth_idx] < eth["ema21"][eth_idx])
            symbol_30 = returns_6
            symbol_60 = returns_12
            context = [
                *btc_returns,
                *eth_returns,
                btc_trend,
                eth_trend,
                float(btc_returns[2] < 0.0 and eth_returns[2] < 0.0),
                float(btc_returns[2] > 0.0 and eth_returns[2] > 0.0),
                float((btc_returns[2] < 0.0) != (eth_returns[2] < 0.0)),
                symbol_30 - btc_returns[1],
                symbol_30 - eth_returns[1],
                float(symbol_60 < btc_returns[2]),
                float(symbol_60 < eth_returns[2]),
            ]
            aligned_context_rows += 1
        rows.append(local + context)
    x = np.asarray(rows, dtype=np.float32)
    raw_nan = int(np.isnan(x).sum())
    raw_inf = int(np.isinf(x).sum())
    clipped_count = int(np.sum(np.abs(np.nan_to_num(x, nan=0.0)) > 10.0))
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)
    return {
        "X_v3": x,
        "feature_names_v3": np.asarray(OPERABLE_FEATURE_NAMES_V3),
        "diagnostics": {
            "schema_version": OPERABLE_FEATURE_SCHEMA_VERSION_V3,
            "feature_count": len(OPERABLE_FEATURE_NAMES_V3),
            "causal_only": True,
            "nan_count": raw_nan,
            "inf_count": raw_inf,
            "clipped_count": clipped_count,
            "cross_symbol_context_available": aligned_context_rows == len(steps) and len(steps) > 0,
            "cross_symbol_aligned_rows": aligned_context_rows,
            "cross_symbol_pending_reason": None if aligned_context_rows else "cross_symbol_alignment_unavailable",
            "market_breadth_available": False,
            "market_breadth_pending_reason": (
                "market_breadth_disabled_to_limit_research_runtime"
                if not include_market_breadth else "market_breadth_not_implemented_in_v3_initial_pass"
            ),
        },
    }


def apply_feature_set(
    dataset: dict[str, Any],
    market: Any,
    feature_set: str = "base",
    *,
    context_markets: dict[str, Any] | None = None,
    include_market_breadth: bool = False,
) -> dict[str, Any]:
    normalized = feature_set.lower()
    if normalized not in FEATURE_SETS:
        raise ValueError(f"unsupported feature_set: {feature_set}")
    if normalized in {"base", "operable_v2", "combined"}:
        return apply_feature_set_v2(dataset, market, normalized)
    base_x = np.asarray(dataset["X"], dtype=np.float32)
    base_names = np.asarray(dataset["feature_names"]).astype(str)
    built_v2 = build_operable_feature_matrix_v2(
        high=market.high,
        low=market.low,
        close=market.close,
        steps=np.asarray(dataset["step"], dtype=np.int64),
        base_features=getattr(market, "features", None),
    )
    built_v3 = build_operable_feature_matrix_v3(
        high=market.high,
        low=market.low,
        close=market.close,
        timestamps=market.timestamps,
        steps=np.asarray(dataset["step"], dtype=np.int64),
        base_features=getattr(market, "features", None),
        context_markets=context_markets,
        include_market_breadth=include_market_breadth,
    )
    v2_x = np.asarray(built_v2["X_v2"], dtype=np.float32)
    v2_names = np.asarray(built_v2["feature_names_v2"]).astype(str)
    v3_x = np.asarray(built_v3["X_v3"], dtype=np.float32)
    v3_names = np.asarray(built_v3["feature_names_v3"]).astype(str)
    if normalized == "base":
        selected_x, selected_names = base_x, base_names
    elif normalized == "operable_v2":
        selected_x, selected_names = v2_x, v2_names
    elif normalized == "operable_v3":
        selected_x, selected_names = v3_x, v3_names
    elif normalized == "combined":
        selected_x, selected_names = np.concatenate((base_x, v2_x), axis=1), np.concatenate((base_names, v2_names))
    else:
        selected_x = np.concatenate((base_x, v2_x, v3_x), axis=1)
        selected_names = np.concatenate((base_names, v2_names, v3_names))
    if len(np.unique(selected_names)) != len(selected_names):
        raise ValueError("feature_names must remain unique")
    updated = dict(dataset)
    updated["X"] = np.nan_to_num(selected_x, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0).astype(np.float32)
    updated["feature_names"] = selected_names
    updated["feature_set"] = normalized
    updated["base_feature_count"] = int(base_x.shape[1])
    updated["new_feature_count"] = int(selected_x.shape[1] - base_x.shape[1]) if normalized.startswith("combined") else int(selected_x.shape[1])
    updated["operable_v2_feature_count"] = int(v2_x.shape[1])
    updated["operable_v3_feature_count"] = int(v3_x.shape[1])
    updated["feature_schema_hash"] = _feature_hash(selected_names)
    updated["feature_diagnostics"] = {
        "v2": built_v2["diagnostics"],
        "v3": built_v3["diagnostics"],
        "feature_set": normalized,
    }
    return updated
