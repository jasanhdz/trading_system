#!/usr/bin/env python3
"""Research-only path-aware SHORT quality labels for operable_short_quality_v4."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np


SHORT_V4_SCHEMA_VERSION = "operable_short_quality_v4_labels_research_v1"
DEFAULT_LEVERAGE = 20.0


@dataclass(frozen=True)
class ShortV4Config:
    horizon: int = 12
    leverage: float = DEFAULT_LEVERAGE
    fee_bps: float = 4.0
    slippage_bps: float = 1.0
    clean_mfe_roe: float = 0.08
    clean_mae_roe: float = 0.055
    clean_mfe_mae_ratio: float = 1.25
    clean_time_fraction: float = 0.60
    initial_adverse_3_roe: float = 0.045
    bad_mae_roe: float = 0.06
    bad_low_mfe_roe: float = 0.04
    bad_initial_adverse_3_roe: float = 0.05
    premium_mfe_roe: float = 0.12
    premium_mae_roe: float = 0.04
    premium_mfe_mae_ratio: float = 1.75
    premium_time_fraction: float = 0.50
    premium_symbols: tuple[str, ...] = ("ADAUSDT", "SUIUSDT", "SOLUSDT", "AVAXUSDT")


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def round_trip_cost_roe(config: ShortV4Config) -> float:
    return 2.0 * (config.fee_bps + config.slippage_bps) / 10_000.0 * config.leverage


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _future_window(high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = idx + 1
    end = min(len(close), idx + horizon + 1)
    return high[start:end], low[start:end], close[start:end]


def _hit_before_stop(entry: float, highs: np.ndarray, lows: np.ndarray, target_move: float, stop_move: float) -> dict[str, Any]:
    target_price = entry * (1.0 - target_move)
    stop_price = entry * (1.0 + stop_move)
    for offset, (high, low) in enumerate(zip(highs, lows), start=1):
        hit = float(low) <= target_price
        stop = float(high) >= stop_price
        if hit and stop:
            return {"hit": False, "stopped": True, "ambiguous_hit_stop": True, "time_to_hit": None, "time_to_stop": offset}
        if stop:
            return {"hit": False, "stopped": True, "ambiguous_hit_stop": False, "time_to_hit": None, "time_to_stop": offset}
        if hit:
            return {"hit": True, "stopped": False, "ambiguous_hit_stop": False, "time_to_hit": offset, "time_to_stop": None}
    return {"hit": False, "stopped": False, "ambiguous_hit_stop": False, "time_to_hit": None, "time_to_stop": None}


def compute_short_path_metrics_v4(
    *,
    high: Any,
    low: Any,
    close: Any,
    entry_index: int,
    horizon: int = 12,
    config: ShortV4Config | None = None,
) -> dict[str, Any]:
    """Compute causal SHORT path metrics from candles after entry_index."""
    cfg = config or ShortV4Config(horizon=horizon)
    high_arr = _as_float_array(high)
    low_arr = _as_float_array(low)
    close_arr = _as_float_array(close)
    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("high, low and close must have equal lengths")
    if entry_index < 0 or entry_index >= len(close_arr):
        raise ValueError("entry_index outside price arrays")
    entry = float(close_arr[entry_index])
    highs, lows, closes = _future_window(high_arr, low_arr, close_arr, entry_index, horizon)
    if entry <= 0 or len(closes) == 0:
        return {"entry_price": entry, "horizon": horizon, "sample_complete": False}

    fav_moves = np.maximum(0.0, (entry - lows) / entry)
    adv_moves = np.maximum(0.0, (highs - entry) / entry)
    mfe_idx = int(np.argmax(fav_moves))
    mae_idx = int(np.argmax(adv_moves))
    mfe_price_move = float(fav_moves[mfe_idx])
    mae_price_move = float(adv_moves[mae_idx])
    mfe_roe = mfe_price_move * cfg.leverage
    mae_roe = mae_price_move * cfg.leverage
    time_to_mfe = mfe_idx + 1 if mfe_price_move > 0 else None
    time_to_mae = mae_idx + 1 if mae_price_move > 0 else None
    mfe_before_mae = bool(time_to_mfe is not None and (time_to_mae is None or time_to_mfe < time_to_mae))
    mae_before_mfe = bool(time_to_mae is not None and (time_to_mfe is None or time_to_mae <= time_to_mfe))
    ratio = safe_div(mfe_roe, mae_roe) if mae_roe > 0 else (math.inf if mfe_roe > 0 else 0.0)
    costs = round_trip_cost_roe(cfg)
    net_quality = mfe_roe - mae_roe - costs

    first_n = min(3, len(closes))
    initial_closes = closes[:first_n]
    initial_highs = highs[:first_n]
    initial_lows = lows[:first_n]
    initial_return_1 = (entry - float(closes[0])) / entry * cfg.leverage
    initial_return_3 = (entry - float(initial_closes[-1])) / entry * cfg.leverage
    initial_adverse_3 = max(0.0, (float(np.max(initial_highs)) - entry) / entry * cfg.leverage)
    initial_favorable_3 = max(0.0, (entry - float(np.min(initial_lows))) / entry * cfg.leverage)

    hit_specs = {
        "hit3_before_minus2": (0.03, 0.02),
        "hit5_before_minus3": (0.05, 0.03),
        "hit8_before_minus5": (0.08, 0.05),
        "hit10_before_minus8": (0.10, 0.08),
        "roe6_before_minus4": (0.06 / cfg.leverage, 0.04 / cfg.leverage),
        "roe8_before_minus5": (0.08 / cfg.leverage, 0.05 / cfg.leverage),
        "roe10_before_minus6": (0.10 / cfg.leverage, 0.06 / cfg.leverage),
        "roe12_before_minus8": (0.12 / cfg.leverage, 0.08 / cfg.leverage),
    }
    hit_rows: dict[str, Any] = {}
    ambiguous = False
    for name, (target_move, stop_move) in hit_specs.items():
        result = _hit_before_stop(entry, highs, lows, target_move, stop_move)
        hit_rows[name] = bool(result["hit"])
        hit_rows[f"{name}_stopped"] = bool(result["stopped"])
        hit_rows[f"{name}_time_to_hit"] = result["time_to_hit"]
        if result["ambiguous_hit_stop"]:
            ambiguous = True

    return {
        "entry_price": entry,
        "horizon": horizon,
        "sample_complete": len(closes) >= horizon,
        "future_min_low": float(np.min(lows)),
        "future_max_high": float(np.max(highs)),
        "mfe_price_move": mfe_price_move,
        "mae_price_move": mae_price_move,
        "mfe_roe_proxy": mfe_roe,
        "mae_roe_proxy": mae_roe,
        "time_to_mfe": time_to_mfe,
        "time_to_mae": time_to_mae,
        "mfe_before_mae": mfe_before_mae,
        "mae_before_mfe": mae_before_mfe,
        "ambiguous_hit_stop": ambiguous,
        "net_quality_after_costs": net_quality,
        "round_trip_cost_roe": costs,
        "mfe_mae_ratio": ratio,
        "initial_return_1_candle": initial_return_1,
        "initial_return_3_candles": initial_return_3,
        "initial_adverse_3_candles": initial_adverse_3,
        "initial_favorable_3_candles": initial_favorable_3,
        "time_efficiency_score": 1.0 - ((time_to_mfe or horizon) / max(horizon, 1)),
        **hit_rows,
    }


def classify_short_clean_entry_v4(metrics: dict[str, Any], config: ShortV4Config | None = None) -> int:
    cfg = config or ShortV4Config(horizon=int(metrics.get("horizon") or 12))
    if metrics.get("ambiguous_hit_stop"):
        return 0
    time_to_mfe = metrics.get("time_to_mfe")
    return int(
        bool(metrics.get("mfe_before_mae"))
        and (metrics.get("mfe_roe_proxy") or 0.0) >= cfg.clean_mfe_roe
        and (metrics.get("mae_roe_proxy") or 0.0) <= cfg.clean_mae_roe
        and (metrics.get("mfe_mae_ratio") or 0.0) >= cfg.clean_mfe_mae_ratio
        and time_to_mfe is not None
        and time_to_mfe <= cfg.horizon * cfg.clean_time_fraction
        and (metrics.get("net_quality_after_costs") or 0.0) > 0.0
        and (metrics.get("initial_adverse_3_candles") or 0.0) <= cfg.initial_adverse_3_roe
    )


def classify_short_bad_entry_v4(metrics: dict[str, Any], config: ShortV4Config | None = None) -> int:
    cfg = config or ShortV4Config(horizon=int(metrics.get("horizon") or 12))
    mae = metrics.get("mae_roe_proxy") or 0.0
    mfe = metrics.get("mfe_roe_proxy") or 0.0
    initial_adverse = metrics.get("initial_adverse_3_candles") or 0.0
    net_quality = metrics.get("net_quality_after_costs") or 0.0
    ratio = metrics.get("mfe_mae_ratio") or 0.0
    mae_early = bool(metrics.get("mae_before_mfe")) and mae >= cfg.bad_mae_roe
    low_mfe_high_mae = mfe < cfg.bad_low_mfe_roe and mae >= 0.04
    management_dependency = ratio < 1.0 and mae >= 0.04
    return int(mae_early or low_mfe_high_mae or initial_adverse >= cfg.bad_initial_adverse_3_roe or net_quality < 0.0 or management_dependency)


def classify_short_management_dependent_v4(metrics: dict[str, Any], config: ShortV4Config | None = None) -> int:
    cfg = config or ShortV4Config(horizon=int(metrics.get("horizon") or 12))
    mfe = metrics.get("mfe_roe_proxy") or 0.0
    mae = metrics.get("mae_roe_proxy") or 0.0
    time_to_mfe = metrics.get("time_to_mfe") or cfg.horizon
    return int(
        mae > 0.04
        and (mfe < cfg.clean_mfe_roe or time_to_mfe > cfg.horizon * cfg.clean_time_fraction)
        and (metrics.get("mfe_mae_ratio") or 0.0) < 1.25
    )


def classify_short_premium_allowed_v4(symbol: str, metrics: dict[str, Any], config: ShortV4Config | None = None) -> int:
    cfg = config or ShortV4Config(horizon=int(metrics.get("horizon") or 12))
    time_to_mfe = metrics.get("time_to_mfe")
    return int(
        symbol in cfg.premium_symbols
        and classify_short_clean_entry_v4(metrics, cfg) == 1
        and (metrics.get("mfe_roe_proxy") or 0.0) >= cfg.premium_mfe_roe
        and (metrics.get("mae_roe_proxy") or 0.0) <= cfg.premium_mae_roe
        and (metrics.get("mfe_mae_ratio") or 0.0) >= cfg.premium_mfe_mae_ratio
        and time_to_mfe is not None
        and time_to_mfe <= cfg.horizon * cfg.premium_time_fraction
    )


def classify_short_no_trade_v4(symbol: str, metrics: dict[str, Any], config: ShortV4Config | None = None) -> int:
    cfg = config or ShortV4Config(horizon=int(metrics.get("horizon") or 12))
    return int(
        classify_short_clean_entry_v4(metrics, cfg) == 0
        and classify_short_premium_allowed_v4(symbol, metrics, cfg) == 0
        and (classify_short_bad_entry_v4(metrics, cfg) == 1 or abs(metrics.get("net_quality_after_costs") or 0.0) < 0.02)
    )


def build_operable_short_quality_v4_labels(
    *,
    symbol: str,
    high: Any,
    low: Any,
    close: Any,
    steps: Any | None = None,
    horizon: int = 12,
    config: ShortV4Config | None = None,
) -> list[dict[str, Any]]:
    cfg = config or ShortV4Config(horizon=horizon)
    high_arr = _as_float_array(high)
    low_arr = _as_float_array(low)
    close_arr = _as_float_array(close)
    if steps is None:
        steps_arr = np.arange(0, max(0, len(close_arr) - horizon), dtype=np.int64)
    else:
        steps_arr = np.asarray(steps, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for idx in steps_arr:
        if idx < 0 or idx >= len(close_arr) - 1:
            continue
        metrics = compute_short_path_metrics_v4(high=high_arr, low=low_arr, close=close_arr, entry_index=int(idx), horizon=horizon, config=cfg)
        row = {
            "symbol": symbol,
            "entry_index": int(idx),
            **metrics,
        }
        row["short_clean_entry_v4"] = classify_short_clean_entry_v4(row, cfg)
        row["short_bad_entry_v4"] = classify_short_bad_entry_v4(row, cfg)
        row["short_premium_allowed_v4"] = classify_short_premium_allowed_v4(symbol, row, cfg)
        row["short_management_dependent_v4"] = classify_short_management_dependent_v4(row, cfg)
        row["short_no_trade_v4"] = classify_short_no_trade_v4(symbol, row, cfg)
        rows.append(row)
    return rows


def summarize_short_v4_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"sample_count": 0}

    def avg(key: str, selected: list[dict[str, Any]] | None = None) -> float | None:
        data = selected if selected is not None else rows
        vals = [float(r[key]) for r in data if r.get(key) is not None and not math.isinf(float(r[key])) and not math.isnan(float(r[key]))]
        return mean(vals) if vals else None

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / count

    clean = [r for r in rows if r.get("short_clean_entry_v4")]
    bad = [r for r in rows if r.get("short_bad_entry_v4")]
    premium = [r for r in rows if r.get("short_premium_allowed_v4")]
    return {
        "sample_count": count,
        "clean_rate": rate("short_clean_entry_v4"),
        "bad_rate": rate("short_bad_entry_v4"),
        "premium_allowed_rate": rate("short_premium_allowed_v4"),
        "management_dependent_rate": rate("short_management_dependent_v4"),
        "no_trade_rate": rate("short_no_trade_v4"),
        "baseline_net_quality": avg("net_quality_after_costs"),
        "clean_net_quality": avg("net_quality_after_costs", clean),
        "bad_net_quality": avg("net_quality_after_costs", bad),
        "premium_net_quality": avg("net_quality_after_costs", premium),
        "baseline_mfe_roe": avg("mfe_roe_proxy"),
        "clean_mfe_roe": avg("mfe_roe_proxy", clean),
        "baseline_mae_roe": avg("mae_roe_proxy"),
        "clean_mae_roe": avg("mae_roe_proxy", clean),
        "baseline_mfe_mae_ratio": avg("mfe_mae_ratio"),
        "clean_mfe_mae_ratio": avg("mfe_mae_ratio", clean),
        "clean_time_to_mfe": avg("time_to_mfe", clean),
        "premium_count": len(premium),
        "clean_count": len(clean),
        "bad_count": len(bad),
    }
