#!/usr/bin/env python3
"""Research-only profiler for LONG alpha families by symbol.

This tool reads local OHLCV candles, computes heuristic family scores, and evaluates
path-aware LONG targets. It does not train models, modify manifests, YAML, PM2, or live
inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT",
)
MOMENTUM_FAMILIES = (
    "momentum_burst_long",
    "momentum_ride_long",
    "breakout_momentum_long",
    "reclaim_momentum_long",
    "micro_roe_momentum_long",
)
ENTRY_FAMILIES = (
    "momentum_burst_long",
    "momentum_ride_long",
    "breakout_momentum_long",
    "pullback_continuation_long",
    "reclaim_momentum_long",
    "capitulation_bounce_long",
    "slow_trend_long",
    "micro_roe_momentum_long",
)
AVOID_FAMILIES = ("avoid_only_bad_long_filter",)
ALL_FAMILIES = ENTRY_FAMILIES + AVOID_FAMILIES
CLASSIC_TARGETS = (
    ("long_hit3_before_minus2", 0.03, 0.02),
    ("long_hit5_before_minus3", 0.05, 0.03),
    ("long_hit6_before_minus4", 0.06, 0.04),
    ("long_hit8_before_minus5", 0.08, 0.05),
    ("long_hit10_before_minus8", 0.10, 0.08),
)
MICRO_TARGETS = (
    ("long_roe6_before_minus4", 0.06 / 20.0, 0.04 / 20.0),
    ("long_roe8_before_minus5", 0.08 / 20.0, 0.05 / 20.0),
    ("long_roe10_before_minus6", 0.10 / 20.0, 0.06 / 20.0),
    ("long_roe12_before_minus8", 0.12 / 20.0, 0.08 / 20.0),
)
HORIZONS = (6, 12, 24)
MOMENTUM_SET = set(MOMENTUM_FAMILIES)
SCHEMA_VERSION = "aegis_long_alpha_family_profile_a_v1"

INITIAL_HYPOTHESIS = {
    "BTCUSDT": {"primary": ["slow_trend_long", "momentum_ride_long"], "secondary": ["pullback_continuation_long"], "avoid": ["late_entry_exhaustion_avoid"]},
    "ETHUSDT": {"primary": ["momentum_ride_long", "pullback_continuation_long"], "secondary": ["breakout_momentum_long"], "avoid": ["upper_wick_rejection_avoid"]},
    "SOLUSDT": {"primary": ["momentum_burst_long", "momentum_ride_long"], "secondary": ["breakout_momentum_long"], "avoid": ["overextension_avoid"]},
    "SUIUSDT": {"primary": ["momentum_burst_long", "reclaim_momentum_long"], "secondary": ["capitulation_bounce_long"], "avoid": ["mae_danger_avoid", "overextension_avoid"]},
    "DOGEUSDT": {"primary": ["momentum_burst_long"], "secondary": ["breakout_momentum_long", "micro_roe_momentum_long"], "avoid": ["volume_climax_avoid"]},
    "AVAXUSDT": {"primary": ["breakout_momentum_long", "pullback_continuation_long"], "secondary": ["reclaim_momentum_long"], "avoid": ["fake_breakout_avoid"]},
    "ADAUSDT": {"primary": ["reclaim_momentum_long", "pullback_continuation_long"], "secondary": ["micro_roe_momentum_long"], "avoid": ["range_chop_avoid"]},
    "XRPUSDT": {"primary": ["reclaim_momentum_long", "momentum_burst_long"], "secondary": ["micro_roe_momentum_long"], "avoid": ["fake_pump_avoid"]},
    "BNBUSDT": {"primary": ["slow_trend_long", "pullback_continuation_long"], "secondary": ["momentum_ride_long"], "avoid": ["low_volume_breakout_avoid"]},
    "LTCUSDT": {"primary": ["slow_trend_long", "breakout_momentum_long"], "secondary": ["pullback_continuation_long"], "avoid": ["late_breakout_avoid"]},
    "LINKUSDT": {"primary": ["reclaim_momentum_long", "pullback_continuation_long"], "secondary": ["slow_trend_long"], "avoid": ["avoid_only_bad_long_filter"]},
}

CSV_COLUMNS = (
    "symbol", "side", "alpha_family", "target_name", "horizon_candles", "sample_count",
    "selected_fraction", "selected_count", "baseline_hit_rate", "selected_hit_rate",
    "hit_lift", "baseline_trade_quality", "selected_trade_quality", "quality_lift",
    "net_quality_lift_after_costs", "stop_rate", "stop_delta", "p90_mae", "p90_mae_delta",
    "avg_mfe", "avg_mae", "mfe_mae_ratio", "time_to_target_avg", "time_to_target_p50",
    "time_to_stop_avg", "ambiguous_rate", "btc_eth_agreement_rate", "exhaustion_rate",
    "late_entry_risk", "fake_breakout_rate", "upper_wick_rejection_rate", "overextension_rate",
    "avoid_selected_fraction", "avoid_hit_rate_delta", "avoid_quality_delta", "avoid_stop_rate_delta",
    "avoid_p90_mae_delta", "avoid_usefulness_score", "family_status", "family_reason",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def mean(values: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else None


def quantile(values: Any, q: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else None


def percentile_rank(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(len(arr), 0.5, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.sum() <= 1:
        return out
    order = np.argsort(arr[valid], kind="mergesort")
    ranks = np.empty(valid.sum(), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, valid.sum())
    out[valid] = ranks
    return out


def top_fraction_mask(score: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    idx = np.flatnonzero(valid & np.isfinite(score))
    if len(idx) == 0:
        return mask
    count = max(1, int(round(len(idx) * fraction)))
    picks = idx[np.argsort(score[idx])[-count:]]
    mask[picks] = True
    return mask


@dataclass(frozen=True)
class TargetResult:
    hit: np.ndarray
    stop: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray
    quality: np.ndarray
    time_to_target: np.ndarray
    time_to_stop: np.ndarray
    ambiguous: np.ndarray


def compute_long_target_arrays(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    target_move: float,
    stop_move: float,
    horizon: int,
    *,
    cost_price: float = 0.0011,
) -> TargetResult:
    n = len(close)
    hit = np.zeros(n, dtype=np.int8)
    stop = np.zeros(n, dtype=np.int8)
    mfe = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    quality = np.full(n, np.nan, dtype=np.float64)
    time_to_target = np.full(n, -1.0, dtype=np.float64)
    time_to_stop = np.full(n, -1.0, dtype=np.float64)
    ambiguous = np.zeros(n, dtype=np.int8)
    for i in range(0, max(0, n - horizon)):
        entry = float(close[i])
        if entry <= 0 or not math.isfinite(entry):
            continue
        highs = high[i + 1 : i + horizon + 1]
        lows = low[i + 1 : i + horizon + 1]
        if len(highs) != horizon or len(lows) != horizon:
            continue
        up = highs / entry - 1.0
        down = 1.0 - lows / entry
        mfe[i] = float(np.nanmax(up))
        mae[i] = float(max(0.0, np.nanmax(down)))
        target_hits = up >= target_move
        stop_hits = down >= stop_move
        t_idx = int(np.argmax(target_hits)) if bool(np.any(target_hits)) else -1
        s_idx = int(np.argmax(stop_hits)) if bool(np.any(stop_hits)) else -1
        if t_idx >= 0 and s_idx >= 0 and t_idx == s_idx:
            ambiguous[i] = 1
            stop[i] = 1
            time_to_stop[i] = float(s_idx + 1)
        elif s_idx >= 0 and (t_idx < 0 or s_idx < t_idx):
            stop[i] = 1
            time_to_stop[i] = float(s_idx + 1)
        elif t_idx >= 0:
            hit[i] = 1
            time_to_target[i] = float(t_idx + 1)
        quality[i] = (
            (target_move if hit[i] else 0.0)
            - (stop_move if stop[i] else 0.0)
            + min(float(mfe[i]), target_move) * 0.25
            - min(float(mae[i]), stop_move * 2.0) * 0.50
            - cost_price
        )
    return TargetResult(hit, stop, mfe, mae, quality, time_to_target, time_to_stop, ambiguous)


def compute_micro_roe_long_targets(market: dict[str, np.ndarray], leverage: float, target_roe: float, stop_roe: float, horizon: int) -> TargetResult:
    return compute_long_target_arrays(
        market["close"], market["high"], market["low"],
        float(target_roe) / max(float(leverage), 1e-12),
        float(stop_roe) / max(float(leverage), 1e-12),
        int(horizon),
        cost_price=(8.0 + 3.0) / 10000.0,
    )


def classify_long_family_candidate(row: dict[str, Any]) -> str:
    if int(row.get("sample_count") or 0) < 200 or int(row.get("selected_count") or 0) < 20:
        return "INSUFFICIENT_DATA"
    family = str(row.get("alpha_family") or "")
    sf = float(row.get("selected_fraction") or row.get("avoid_selected_fraction") or 0.0)
    if family == "avoid_only_bad_long_filter":
        useful = (
            0.05 <= sf <= 0.30
            and float(row.get("avoid_quality_delta") or 0.0) < 0.0
            and float(row.get("avoid_stop_rate_delta") or 0.0) > 0.0
            and float(row.get("avoid_p90_mae_delta") or 0.0) > 0.0
            and float(row.get("avoid_hit_rate_delta") or 0.0) <= 0.0
            and float(row.get("avoid_usefulness_score") or 0.0) > 0.0
        )
        return "LONG_FAMILY_AVOID_ONLY" if useful else "LONG_FAMILY_WEAK"
    hit_lift = float(row.get("hit_lift") or 0.0)
    quality_lift = float(row.get("quality_lift") or 0.0)
    net = float(row.get("net_quality_lift_after_costs") or 0.0)
    p90_delta = float(row.get("p90_mae_delta") or 0.0)
    stop_delta = float(row.get("stop_delta") or 0.0)
    ttt = row.get("time_to_target_avg")
    horizon = float(row.get("horizon_candles") or 1.0)
    fast_ok = True
    if family in MOMENTUM_SET and ttt is not None:
        fast_ok = float(ttt) <= horizon * 0.65
    if hit_lift > 0 and quality_lift > 0 and net > 0 and p90_delta <= 0.15 and stop_delta <= 0.05 and 0.05 <= sf <= 0.25 and fast_ok:
        return "LONG_FAMILY_PROMISING"
    if net < 0 or hit_lift < 0 or stop_delta > 0.08 or p90_delta > 0.30:
        return "LONG_FAMILY_FAILED"
    if hit_lift > 0 or quality_lift > 0 or net > 0:
        return "LONG_FAMILY_MIXED"
    return "LONG_FAMILY_WEAK"


def _status_reason(row: dict[str, Any]) -> str:
    status = row.get("family_status")
    if status == "LONG_FAMILY_PROMISING":
        return "hit/quality/net lift positive with controlled MAE and stop risk"
    if status == "LONG_FAMILY_AVOID_ONLY":
        return "selected bad zones have worse hit/quality, higher stop and MAE"
    if status == "LONG_FAMILY_FAILED":
        return "net quality or hit lift negative, or stop/MAE worsened materially"
    if status == "INSUFFICIENT_DATA":
        return "not enough samples after selection"
    return "mixed or weak lift profile"


def status_priority(status: str) -> int:
    order = {
        "LONG_FAMILY_PROMISING": 0,
        "LONG_FAMILY_AVOID_ONLY": 1,
        "LONG_FAMILY_MIXED": 2,
        "LONG_FAMILY_WEAK": 3,
        "LONG_FAMILY_FAILED": 4,
        "INSUFFICIENT_DATA": 5,
    }
    return order.get(status, 9)


def select_best_by_symbol(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for symbol in sorted({r["symbol"] for r in rows}):
        candidates = [r for r in rows if r["symbol"] == symbol]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (
            status_priority(str(r.get("family_status"))),
            -float(r.get("net_quality_lift_after_costs") or r.get("avoid_usefulness_score") or -999),
            -float(r.get("hit_lift") or 0.0),
            float(r.get("p90_mae_delta") or 999),
            int(r.get("horizon_candles") or 999),
        ))
        best.append(candidates[0])
    return best


def normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def db_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    return s.replace("USDT", "/USDT")


def load_candles(db_path: Path, symbol: str, lookback_days: int) -> pd.DataFrame:
    rows = max(5000, int(lookback_days * 288) + 300)
    con = sqlite3.connect(db_path)
    query = """
        SELECT timestamp, open, high, low, close, volume, buy_volume
        FROM ohlcv_data
        WHERE symbol = ? AND timeframe = '5m'
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, con, params=(db_symbol(symbol), rows))
    con.close()
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume", "buy_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def add_features(df: pd.DataFrame, btc: pd.DataFrame | None = None, eth: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].replace(0, np.nan)
    for n in (1, 3, 6, 12, 24):
        out[f"return_{n}"] = close.pct_change(n)
    for n in (12, 24):
        rolling_high = high.rolling(n, min_periods=max(3, n // 2)).max()
        rolling_low = low.rolling(n, min_periods=max(3, n // 2)).min()
        denom = (rolling_high - rolling_low).replace(0, np.nan)
        out[f"close_location_{n}"] = ((close - rolling_low) / denom).clip(0, 1)
        out[f"range_high_{n}"] = rolling_high
        out[f"range_low_{n}"] = rolling_low
    candle_range = (high - low).replace(0, np.nan)
    out["upper_wick_ratio"] = ((high - np.maximum(out["open"], close)) / candle_range).clip(0, 1)
    out["lower_wick_ratio"] = ((np.minimum(out["open"], close) - low) / candle_range).clip(0, 1)
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    out["range_expansion_12"] = out["range_pct"] / out["range_pct"].rolling(12, min_periods=6).mean().replace(0, np.nan)
    out["volume_ratio_12"] = volume / volume.rolling(12, min_periods=6).mean()
    out["volume_ratio_24"] = volume / volume.rolling(24, min_periods=12).mean()
    for span in (9, 21, 25, 99, 200):
        ema = close.ewm(span=span, adjust=False, min_periods=max(3, span // 3)).mean()
        out[f"ema{span}"] = ema
        out[f"distance_ema{span}"] = close / ema.replace(0, np.nan) - 1.0
        out[f"ema{span}_slope"] = ema.pct_change(6)
    out["ema_stack_bull"] = ((out["ema9"] > out["ema21"]) & (out["ema21"] > out["ema99"])).astype(float)
    for n in (12, 24):
        out[f"trend_efficiency_{n}"] = (close - close.shift(n)).abs() / close.diff().abs().rolling(n, min_periods=max(3, n // 2)).sum().replace(0, np.nan)
        out[f"realized_vol_{n}"] = close.pct_change().rolling(n, min_periods=max(3, n // 2)).std()
    out["breakout_12"] = close / out["range_high_12"].shift(1).replace(0, np.nan) - 1.0
    out["breakdown_sweep_12"] = out["range_low_12"].shift(1) / low.replace(0, np.nan) - 1.0
    out["reclaim_strength"] = ((low < out["range_low_12"].shift(1)) & (close > out["range_low_12"].shift(1))).astype(float)
    out["pullback_depth"] = (out["range_high_24"].shift(1) - close) / close.replace(0, np.nan)
    out["overextension"] = np.maximum(out["distance_ema21"], out["distance_ema99"]).clip(lower=0)
    out["late_entry_risk"] = (percentile_rank(out["return_24"].to_numpy()) * 0.45 + percentile_rank(out["overextension"].to_numpy()) * 0.35 + percentile_rank(out["volume_ratio_24"].to_numpy()) * 0.20)
    out["upper_wick_rejection"] = percentile_rank(out["upper_wick_ratio"].to_numpy())
    out["volume_climax"] = percentile_rank(out["volume_ratio_24"].to_numpy())
    out["fake_breakout"] = ((out["breakout_12"] > 0) & (out["upper_wick_ratio"] > 0.35) & (out["close_location_12"] < 0.65)).astype(float)
    out["exhaustion"] = np.clip(out["late_entry_risk"] * 0.45 + out["upper_wick_rejection"] * 0.25 + out["volume_climax"] * 0.20 + out["fake_breakout"] * 0.10, 0, 1)
    for name, ctx in (("btc", btc), ("eth", eth)):
        if ctx is not None and not ctx.empty:
            ctx_small = ctx[["timestamp", "close"]].copy()
            ctx_small[f"{name}_return_6"] = ctx_small["close"].pct_change(6)
            ctx_small[f"{name}_return_12"] = ctx_small["close"].pct_change(12)
            ctx_small = ctx_small[["timestamp", f"{name}_return_6", f"{name}_return_12"]]
            out = out.merge(ctx_small, on="timestamp", how="left")
        else:
            out[f"{name}_return_6"] = np.nan
            out[f"{name}_return_12"] = np.nan
    out["btc_eth_agreement"] = (((out["btc_return_6"] > 0) & (out["eth_return_6"] > 0)).astype(float)).fillna(0.0)
    out["btc_eth_contradiction"] = (((out["btc_return_6"] < 0) | (out["eth_return_6"] < 0)).astype(float)).fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def family_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    def r(col: str) -> np.ndarray:
        return percentile_rank(frame[col].to_numpy() if col in frame else np.full(len(frame), np.nan))
    ret3, ret6, ret12, ret24 = r("return_3"), r("return_6"), r("return_12"), r("return_24")
    cl12, cl24 = r("close_location_12"), r("close_location_24")
    vol12, rng12 = r("volume_ratio_12"), r("range_expansion_12")
    ema_stack = np.nan_to_num(frame.get("ema_stack_bull", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float), nan=0.0)
    trend12, trend24 = r("trend_efficiency_12"), r("trend_efficiency_24")
    dist21, dist99 = r("distance_ema21"), r("distance_ema99")
    low_wick, up_wick = r("lower_wick_ratio"), r("upper_wick_ratio")
    breakout = r("breakout_12")
    reclaim = np.nan_to_num(frame.get("reclaim_strength", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float), nan=0.0)
    pullback_ok = 1.0 - np.abs(np.nan_to_num(frame.get("pullback_depth", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float), nan=0.0)).clip(0, 0.08) / 0.08
    exhaustion = np.nan_to_num(frame.get("exhaustion", pd.Series(0.5, index=frame.index)).to_numpy(dtype=float), nan=0.5)
    agreement = np.nan_to_num(frame.get("btc_eth_agreement", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float), nan=0.0)
    contradiction = np.nan_to_num(frame.get("btc_eth_contradiction", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float), nan=0.0)
    avoid = np.clip(exhaustion * 0.35 + up_wick * 0.20 + r("overextension") * 0.20 + r("volume_ratio_24") * 0.15 + contradiction * 0.10, 0, 1)
    scores = {
        "momentum_burst_long": np.clip(ret3 * 0.15 + ret6 * 0.25 + ret12 * 0.20 + cl12 * 0.15 + vol12 * 0.10 + rng12 * 0.10 + agreement * 0.10 - exhaustion * 0.20 - contradiction * 0.10, 0, 1),
        "momentum_ride_long": np.clip(ret12 * 0.20 + ret24 * 0.20 + trend24 * 0.20 + ema_stack * 0.15 + dist21 * 0.10 + agreement * 0.10 - exhaustion * 0.15, 0, 1),
        "breakout_momentum_long": np.clip(breakout * 0.25 + cl12 * 0.20 + vol12 * 0.15 + rng12 * 0.15 + ret6 * 0.10 + agreement * 0.10 - up_wick * 0.20 - exhaustion * 0.15, 0, 1),
        "pullback_continuation_long": np.clip(ema_stack * 0.20 + ret24 * 0.15 + trend24 * 0.15 + pullback_ok * 0.20 + cl12 * 0.10 + ret3 * 0.10 + agreement * 0.10 - contradiction * 0.10 - exhaustion * 0.10, 0, 1),
        "reclaim_momentum_long": np.clip(reclaim * 0.30 + low_wick * 0.20 + cl12 * 0.15 + ret3 * 0.15 + ret6 * 0.10 + agreement * 0.10 - contradiction * 0.15 - up_wick * 0.10, 0, 1),
        "capitulation_bounce_long": np.clip((1 - ret24) * 0.20 + low_wick * 0.25 + rng12 * 0.20 + vol12 * 0.15 + ret3 * 0.10 + (1 - contradiction) * 0.10 - up_wick * 0.10, 0, 1),
        "slow_trend_long": np.clip(ret12 * 0.15 + ret24 * 0.25 + trend24 * 0.20 + ema_stack * 0.15 + (1 - r("realized_vol_24")) * 0.10 + agreement * 0.10 - exhaustion * 0.15, 0, 1),
        "micro_roe_momentum_long": np.clip(ret3 * 0.25 + ret6 * 0.20 + cl12 * 0.15 + rng12 * 0.15 + vol12 * 0.10 + agreement * 0.10 - up_wick * 0.15 - exhaustion * 0.10, 0, 1),
        "avoid_only_bad_long_filter": avoid,
    }
    return scores


def valid_mask(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    numeric_ok = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy()
    mask = numeric_ok.copy()
    if horizon > 0:
        mask[-horizon:] = False
    mask[:220] = False
    return mask


def evaluate_selection_metrics(
    symbol: str,
    family: str,
    target_name: str,
    horizon: int,
    target: TargetResult,
    selected: np.ndarray,
    valid: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    base = valid & np.isfinite(target.quality)
    sel = selected & base
    sample_count = int(base.sum())
    selected_count = int(sel.sum())
    baseline_hit = mean(target.hit[base]) or 0.0
    selected_hit = mean(target.hit[sel]) if selected_count else None
    baseline_quality = mean(target.quality[base]) or 0.0
    selected_quality = mean(target.quality[sel]) if selected_count else None
    baseline_stop = mean(target.stop[base]) or 0.0
    selected_stop = mean(target.stop[sel]) if selected_count else None
    baseline_p90_mae = quantile(target.mae[base], 0.90) or 0.0
    selected_p90_mae = quantile(target.mae[sel], 0.90) if selected_count else None
    ttt = target.time_to_target[sel]
    ttt = ttt[ttt > 0]
    tts = target.time_to_stop[sel]
    tts = tts[tts > 0]
    avg_mfe = mean(target.mfe[sel]) if selected_count else None
    avg_mae = mean(target.mae[sel]) if selected_count else None
    row = {
        "symbol": symbol,
        "side": "LONG",
        "alpha_family": family,
        "target_name": target_name,
        "horizon_candles": horizon,
        "sample_count": sample_count,
        "selected_fraction": float(selected_count / sample_count) if sample_count else 0.0,
        "selected_count": selected_count,
        "baseline_hit_rate": baseline_hit,
        "selected_hit_rate": selected_hit,
        "hit_lift": (selected_hit - baseline_hit) if selected_hit is not None else None,
        "baseline_trade_quality": baseline_quality,
        "selected_trade_quality": selected_quality,
        "quality_lift": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "net_quality_lift_after_costs": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "stop_rate": selected_stop,
        "stop_delta": (selected_stop - baseline_stop) if selected_stop is not None else None,
        "p90_mae": selected_p90_mae,
        "p90_mae_delta": ((selected_p90_mae - baseline_p90_mae) / max(baseline_p90_mae, 1e-12)) if selected_p90_mae is not None else None,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "mfe_mae_ratio": (avg_mfe / max(avg_mae or 0.0, 1e-12)) if avg_mfe is not None else None,
        "time_to_target_avg": mean(ttt),
        "time_to_target_p50": quantile(ttt, 0.50),
        "time_to_stop_avg": mean(tts),
        "ambiguous_rate": mean(target.ambiguous[sel]) if selected_count else None,
        "btc_eth_agreement_rate": mean(frame.loc[sel, "btc_eth_agreement"].to_numpy()) if selected_count and "btc_eth_agreement" in frame else None,
        "exhaustion_rate": mean(frame.loc[sel, "exhaustion"].to_numpy()) if selected_count and "exhaustion" in frame else None,
        "late_entry_risk": mean(frame.loc[sel, "late_entry_risk"].to_numpy()) if selected_count and "late_entry_risk" in frame else None,
        "fake_breakout_rate": mean(frame.loc[sel, "fake_breakout"].to_numpy()) if selected_count and "fake_breakout" in frame else None,
        "upper_wick_rejection_rate": mean(frame.loc[sel, "upper_wick_rejection"].to_numpy()) if selected_count and "upper_wick_rejection" in frame else None,
        "overextension_rate": mean(frame.loc[sel, "overextension"].to_numpy()) if selected_count and "overextension" in frame else None,
        "avoid_selected_fraction": None,
        "avoid_hit_rate_delta": None,
        "avoid_quality_delta": None,
        "avoid_stop_rate_delta": None,
        "avoid_p90_mae_delta": None,
        "avoid_usefulness_score": None,
    }
    if family == "avoid_only_bad_long_filter":
        avoid_quality_delta = row["quality_lift"]
        avoid_stop_delta = row["stop_delta"]
        avoid_hit_delta = row["hit_lift"]
        avoid_p90_delta = row["p90_mae_delta"]
        usefulness = (-(avoid_quality_delta or 0.0)) + max(avoid_stop_delta or 0.0, 0.0) + max(avoid_p90_delta or 0.0, 0.0) * 0.01 + max(-(avoid_hit_delta or 0.0), 0.0)
        row.update({
            "avoid_selected_fraction": row["selected_fraction"],
            "avoid_hit_rate_delta": avoid_hit_delta,
            "avoid_quality_delta": avoid_quality_delta,
            "avoid_stop_rate_delta": avoid_stop_delta,
            "avoid_p90_mae_delta": avoid_p90_delta,
            "avoid_usefulness_score": usefulness,
        })
    row["family_status"] = classify_long_family_candidate(row)
    row["family_reason"] = _status_reason(row)
    return row


def evaluate_symbol(symbol: str, db_path: Path, lookback_days: int, families: tuple[str, ...]) -> list[dict[str, Any]]:
    df = load_candles(db_path, symbol, lookback_days)
    if df.empty or len(df) < 1000:
        return [{"symbol": symbol, "side": "LONG", "alpha_family": "ALL", "family_status": "INSUFFICIENT_DATA", "family_reason": "no local candles", "sample_count": len(df)}]
    btc = load_candles(db_path, "BTCUSDT", lookback_days) if symbol != "BTCUSDT" else df
    eth = load_candles(db_path, "ETHUSDT", lookback_days) if symbol != "ETHUSDT" else df
    frame = add_features(df, btc, eth).reset_index(drop=True)
    market = {col: frame[col].to_numpy(dtype=np.float64) for col in ["close", "high", "low"]}
    scores = family_scores(frame)
    target_cache: dict[tuple[str, int], TargetResult] = {}
    rows: list[dict[str, Any]] = []
    for family in families:
        target_defs = MICRO_TARGETS if family == "micro_roe_momentum_long" else CLASSIC_TARGETS
        fraction = 0.15 if family == "avoid_only_bad_long_filter" else (0.10 if family in MOMENTUM_SET else 0.12)
        for target_name, target_move, stop_move in target_defs:
            for horizon in HORIZONS:
                if family in MOMENTUM_SET and horizon == 24:
                    continue
                if family in {"slow_trend_long", "pullback_continuation_long"} and horizon == 6:
                    continue
                cache_key = (target_name, horizon)
                if cache_key not in target_cache:
                    target_cache[cache_key] = compute_long_target_arrays(market["close"], market["high"], market["low"], target_move, stop_move, horizon)
                valid = valid_mask(frame, horizon)
                score = scores[family]
                selected = top_fraction_mask(score, valid, fraction)
                rows.append(evaluate_selection_metrics(symbol, family, target_name, horizon, target_cache[cache_key], selected, valid, frame))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] = CSV_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in columns})


def write_reports(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    best = select_best_by_symbol(rows)
    promising = [r for r in rows if r.get("family_status") == "LONG_FAMILY_PROMISING"]
    avoid = [r for r in rows if r.get("family_status") == "LONG_FAMILY_AVOID_ONLY"]
    failed = [r for r in rows if r.get("family_status") == "LONG_FAMILY_FAILED"]
    summary_rows = []
    for family in sorted({r.get("alpha_family") for r in rows}):
        fam = [r for r in rows if r.get("alpha_family") == family]
        summary_rows.append({
            "alpha_family": family,
            "configs": len(fam),
            "promising": sum(r.get("family_status") == "LONG_FAMILY_PROMISING" for r in fam),
            "mixed": sum(r.get("family_status") == "LONG_FAMILY_MIXED" for r in fam),
            "avoid_only": sum(r.get("family_status") == "LONG_FAMILY_AVOID_ONLY" for r in fam),
            "failed": sum(r.get("family_status") == "LONG_FAMILY_FAILED" for r in fam),
            "best_net_quality_lift": max(float(r.get("net_quality_lift_after_costs") or -999) for r in fam),
            "best_hit_lift": max(float(r.get("hit_lift") or -999) for r in fam),
        })
    paths = {
        "md": str(out / f"aegis_long_alpha_family_profile_{stamp}.md"),
        "json": str(out / f"aegis_long_alpha_family_profile_{stamp}.json"),
        "summary": str(out / f"aegis_long_alpha_family_summary_{stamp}.csv"),
        "all_configs": str(out / f"aegis_long_alpha_family_all_configs_{stamp}.csv"),
        "best_by_symbol": str(out / f"aegis_long_alpha_family_best_by_symbol_{stamp}.csv"),
        "promising": str(out / f"aegis_long_alpha_family_promising_{stamp}.csv"),
        "avoid_only": str(out / f"aegis_long_alpha_family_avoid_only_{stamp}.csv"),
        "failed": str(out / f"aegis_long_alpha_family_failed_{stamp}.csv"),
    }
    write_csv(Path(paths["all_configs"]), rows)
    write_csv(Path(paths["best_by_symbol"]), best)
    write_csv(Path(paths["promising"]), promising)
    write_csv(Path(paths["avoid_only"]), avoid)
    write_csv(Path(paths["failed"]), failed)
    write_csv(Path(paths["summary"]), summary_rows, tuple(summary_rows[0].keys()) if summary_rows else ("alpha_family",))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "symbols": args.symbols,
        "lookback_days": args.lookback_days,
        "family_group": args.family_group,
        "rows": rows,
        "best_by_symbol": best,
        "summary": summary_rows,
        "safety": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2": True,
            "no_orders": True,
        },
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2) + "\n")
    momentum_best = [r for r in best if r.get("alpha_family") in MOMENTUM_SET]
    lines = [
        "# Aegis LONG Alpha Family Profile A",
        "",
        "## Safety",
        "- RESEARCH_ONLY",
        "- no live inference changes",
        "- no active_manifest changes",
        "- no YAML live changes",
        "- no PM2",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Symbols evaluated: `{len(set(r.get('symbol') for r in rows))}`",
        f"- Config rows: `{len(rows)}`",
        f"- Promising configs: `{len(promising)}`",
        f"- Avoid-only configs: `{len(avoid)}`",
        "",
        "## Best By Symbol",
        "| symbol | family | target | h | status | hit_lift | quality_lift | p90_delta | reason |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in best:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {row.get('target_name')} | {row.get('horizon_candles')} | {row.get('family_status')} | {float(row.get('hit_lift') or 0):.4f} | {float(row.get('quality_lift') or 0):.5f} | {float(row.get('p90_mae_delta') or 0):.4f} | {row.get('family_reason')} |")
    lines += [
        "",
        "## Momentum LONG Candidates",
        "| symbol | family | target | h | status | time_to_target_avg | hit_lift | net_lift |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted([r for r in rows if r.get("alpha_family") in MOMENTUM_SET and r.get("family_status") in {"LONG_FAMILY_PROMISING", "LONG_FAMILY_MIXED"}], key=lambda r: (status_priority(str(r.get("family_status"))), -float(r.get("net_quality_lift_after_costs") or -999)))[:40]:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {row.get('target_name')} | {row.get('horizon_candles')} | {row.get('family_status')} | {float(row.get('time_to_target_avg') or 0):.2f} | {float(row.get('hit_lift') or 0):.4f} | {float(row.get('net_quality_lift_after_costs') or 0):.5f} |")
    lines += [
        "",
        "## Avoid-only LONG",
        "| symbol | target | h | status | usefulness | hit_delta | quality_delta | stop_delta |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(avoid, key=lambda r: -float(r.get("avoid_usefulness_score") or 0))[:40]:
        lines.append(f"| {row.get('symbol')} | {row.get('target_name')} | {row.get('horizon_candles')} | {row.get('family_status')} | {float(row.get('avoid_usefulness_score') or 0):.5f} | {float(row.get('avoid_hit_rate_delta') or 0):.4f} | {float(row.get('avoid_quality_delta') or 0):.5f} | {float(row.get('avoid_stop_rate_delta') or 0):.4f} |")
    lines += [
        "",
        "## LONG-B/C Recommendation",
        "- Prioritize LONG_FAMILY_PROMISING and LONG_FAMILY_MIXED momentum rows for first model training.",
        "- Treat avoid-only rows as future reducer/filter candidates, not entry models.",
        "- Keep Phase O SHORT live untouched until a separate LONG confirmation phase freezes candidates.",
    ]
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--db-path", default="/home/jasan/Develop/trading_system/data/binance_candles.db")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--family-group", choices=("all", "momentum"), default="all")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    args.symbols = symbols
    if args.fast:
        args.lookback_days = min(args.lookback_days, 60)
    families = tuple(f for f in ALL_FAMILIES if args.family_group == "all" or f in MOMENTUM_SET)
    rows: list[dict[str, Any]] = []
    db_path = Path(args.db_path)
    for symbol in symbols:
        if symbol not in SYMBOLS:
            raise SystemExit(f"Unsupported symbol for LONG-A research: {symbol}")
        rows.extend(evaluate_symbol(symbol, db_path, args.lookback_days, families))
    paths = write_reports(rows, args)
    best = select_best_by_symbol(rows)
    print(json.dumps(json_safe({"reports": paths, "row_count": len(rows), "best_by_symbol": best}), indent=2))


if __name__ == "__main__":
    main()
