#!/usr/bin/env python3
"""Build an enriched causal feature dataset for future TRRM training.

Research-only. This script does not train models and does not write to active
or live paths. It reconstructs causal features from historical OHLCV using only
past/current candles at each candidate timestamp.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)


REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "binance_candles.db"
DEFAULT_OUTPUT_DIR = Path("/home/jasan/Develop")
DATASET_VERSION = "trrm_causal_features_d_v1"
LEAKAGE_PATTERNS = [
    "label",
    "target",
    "future",
    "mfe",
    "mae",
    "pnl",
    "net",
    "quality",
    "clean",
    "bad",
    "tail",
    "qmae",
    "management",
    "premium",
    "hit",
    "stop",
    "tp",
    "sl",
    "realized",
    "outcome",
    "profit",
    "loss",
    "trade_result",
    "close_reason",
]
CAUSAL_CLOSE_OVERRIDES = (
    "feature.close",
    "feature.close_to_open_return",
    "feature.close_position_in_range",
    "feature.open_position_in_range",
    "feature.close_vs_ema_",
    "feature.close_vs_atr_",
    "feature.close_below_rolling_low_",
    "feature.lower_wick_close_pct",
    "feature.upper_wick_close_pct",
)
ID_SOURCE_COLUMNS = ["symbol", "timestamp", "timeframe", "horizon"]
TARGET_SOURCE_COLUMNS = [
    "tail_risk_v4",
    "bad_entry_v4",
    "early_mae_v4",
    "qmae_target",
    "q95_mae_target",
]
LABEL_SOURCE_COLUMNS = [
    "clean_entry_v4",
    "premium_allowed_v4",
    "management_dependent_v4",
    "no_trade_v4",
]
FUTURE_EVAL_SOURCE_COLUMNS = [
    "future_mfe_roe_proxy",
    "future_mae_roe_proxy",
    "net_quality_after_costs",
    "mfe_mae_ratio",
    "time_to_mfe",
    "time_to_mae",
    "mfe_before_mae",
]
REQUESTED_FEATURES = {
    "price_returns": [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "log_ret_1", "log_ret_3", "log_ret_6",
        "close_to_open_return", "candle_body_pct", "candle_range_pct", "high_low_range_pct",
    ],
    "candle_structure": [
        "upper_wick_pct", "lower_wick_pct", "body_to_range", "close_position_in_range", "open_position_in_range",
        "is_red_candle", "is_green_candle", "consecutive_red_count", "consecutive_green_count",
        "lower_wick_close_pct", "upper_wick_close_pct",
    ],
    "volatility_atr": [
        "rolling_range_mean_6", "rolling_range_mean_12", "rolling_range_mean_24", "rolling_range_std_12",
        "rolling_range_std_24", "atr_proxy_6", "atr_proxy_12", "atr_proxy_24", "close_vs_atr_12",
        "volatility_compression_12_24", "volatility_expansion_6_24",
    ],
    "volume": [
        "volume_ret_1", "volume_zscore_12", "volume_zscore_24", "volume_ratio_6_24", "volume_spike_12",
        "volume_trend_12",
    ],
    "ema_trend": [
        "ema_6", "ema_12", "ema_24", "ema_48", "close_vs_ema_6", "close_vs_ema_12", "close_vs_ema_24",
        "ema_6_vs_12", "ema_12_vs_24", "ema_slope_6", "ema_slope_12", "ema_slope_24",
        "trend_stack_short", "trend_stack_long", "trend_compression",
    ],
    "momentum": [
        "momentum_3", "momentum_6", "momentum_12", "momentum_24", "momentum_accel_3_12",
        "return_zscore_12", "return_zscore_24", "downside_momentum_6", "upside_momentum_6",
    ],
    "short_setup_proxies": [
        "breakdown_proxy_12", "breakdown_proxy_24", "close_below_rolling_low_12",
        "close_below_rolling_low_24", "distance_to_rolling_low_12", "distance_to_rolling_low_24",
        "distance_to_rolling_high_12", "distance_to_rolling_high_24", "failed_breakdown_proxy",
        "fake_breakdown_risk_proxy", "room_to_fall_proxy_24", "extension_down_proxy",
        "exhaustion_down_proxy", "rebound_risk_proxy", "squeeze_risk_proxy_causal",
    ],
    "market_context": [
        "btc_ret_3", "btc_ret_6", "btc_ret_12", "eth_ret_3", "eth_ret_6", "eth_ret_12",
        "symbol_vs_btc_ret_6", "symbol_vs_eth_ret_6", "btc_volatility_12", "eth_volatility_12",
        "btc_trend_proxy", "eth_trend_proxy",
    ],
    "regime_proxies": [
        "chop_proxy_12", "trend_strength_proxy_12", "trend_strength_proxy_24", "range_contraction_proxy",
        "range_expansion_proxy", "high_vol_regime_proxy", "low_vol_regime_proxy",
    ],
    "risk_state_proxies": [
        "immediate_reversal_risk_proxy", "overextended_down_risk_proxy", "low_room_to_fall_risk_proxy",
        "high_wick_reclaim_risk_proxy", "squeeze_plus_reclaim_risk_proxy",
    ],
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_input_csv(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path | None:
    files = sorted(output_dir.glob("aegis_risk_v4_qmae_base_dataset_a_samples_*.csv"))
    return files[-1] if files else None


def latest_report_input() -> Path | None:
    for pattern in ("aegis_phase_c_trrm_classifier_*.json", "aegis_phase_b_qmae_dataset_review_*.json"):
        files = sorted(DEFAULT_OUTPUT_DIR.glob(pattern))
        if not files:
            continue
        try:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("dataset", "samples_csv"):
            value = data.get(key)
            if value and Path(value).exists():
                return Path(value)
    return latest_input_csv()


def normalize_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "")


def db_symbol(symbol: str) -> str:
    clean = normalize_symbol(symbol)
    return clean[:-4] + "/USDT" if clean.endswith("USDT") else clean


def safe_div(a: Any, b: Any) -> Any:
    return a / pd.Series(b).replace(0, np.nan) if not isinstance(b, (int, float)) else a / (b or np.nan)


def is_leakage_column(name: str) -> tuple[bool, str]:
    low = name.lower()
    if any(low.startswith(prefix) for prefix in CAUSAL_CLOSE_OVERRIDES):
        return False, "causal_close_override"
    for pattern in LEAKAGE_PATTERNS:
        if pattern in low:
            return True, f"pattern:{pattern}"
    return False, ""


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def load_ohlcv(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    start_buffer = (start - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    end_text = (end + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    query = """
        select timestamp, open, high, low, close, volume, buy_volume
        from ohlcv_data
        where symbol = ? and timeframe = ? and timestamp between ? and ?
        order by timestamp asc
    """
    with sqlite3.connect(db_path, timeout=5) as con:
        df = pd.read_sql_query(query, con, params=(db_symbol(symbol), timeframe, start_buffer, end_text))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def consecutive_count(condition: pd.Series) -> pd.Series:
    groups = (condition != condition.shift()).cumsum()
    return condition.astype(int).groupby(groups).cumsum().where(condition, 0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 3)).mean()
    std = series.rolling(window, min_periods=max(3, window // 3)).std()
    return (series - mean) / std.replace(0, np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def compute_causal_features(candles: pd.DataFrame) -> pd.DataFrame:
    out = candles[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    volume = out["volume"].astype(float) if "volume" in out else pd.Series(np.nan, index=out.index)
    eps = 1e-12

    for bars in (1, 2, 3, 6, 12, 24):
        out[f"ret_{bars}"] = close / close.shift(bars) - 1.0
        out[f"log_ret_{bars}"] = np.log(close / close.shift(bars))
        out[f"momentum_{bars}"] = out[f"ret_{bars}"]
    out["close_to_open_return"] = close / open_.replace(0, np.nan) - 1.0
    candle_range_abs = (high - low).replace(0, np.nan)
    out["candle_body_pct"] = (close - open_) / open_.replace(0, np.nan)
    out["candle_range_pct"] = candle_range_abs / close.replace(0, np.nan)
    out["high_low_range_pct"] = out["candle_range_pct"]
    upper_body = pd.concat([open_, close], axis=1).max(axis=1)
    lower_body = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick_pct"] = (high - upper_body) / candle_range_abs
    out["lower_wick_pct"] = (lower_body - low) / candle_range_abs
    out["upper_wick_close_pct"] = (high - close) / close.replace(0, np.nan)
    out["lower_wick_close_pct"] = (close - low) / close.replace(0, np.nan)
    out["body_to_range"] = (close - open_).abs() / candle_range_abs
    out["close_position_in_range"] = (close - low) / candle_range_abs
    out["open_position_in_range"] = (open_ - low) / candle_range_abs
    out["is_red_candle"] = (close < open_).astype(int)
    out["is_green_candle"] = (close > open_).astype(int)
    out["consecutive_red_count"] = consecutive_count(close < open_)
    out["consecutive_green_count"] = consecutive_count(close > open_)

    range_pct = (high - low) / close.replace(0, np.nan)
    tr_pct = true_range(out) / close.replace(0, np.nan)
    for window in (6, 12, 24):
        out[f"rolling_range_mean_{window}"] = range_pct.rolling(window, min_periods=max(3, window // 2)).mean()
        out[f"atr_proxy_{window}"] = tr_pct.rolling(window, min_periods=max(3, window // 2)).mean()
    for window in (12, 24):
        out[f"rolling_range_std_{window}"] = range_pct.rolling(window, min_periods=max(4, window // 2)).std()
    out["close_vs_atr_12"] = close.pct_change().abs() / out["atr_proxy_12"].replace(0, np.nan)
    out["volatility_compression_12_24"] = out["rolling_range_mean_12"] / out["rolling_range_mean_24"].replace(0, np.nan)
    out["volatility_expansion_6_24"] = out["rolling_range_mean_6"] / out["rolling_range_mean_24"].replace(0, np.nan)

    out["volume_ret_1"] = volume / volume.shift(1).replace(0, np.nan) - 1.0
    out["volume_zscore_12"] = rolling_zscore(volume, 12)
    out["volume_zscore_24"] = rolling_zscore(volume, 24)
    out["volume_ratio_6_24"] = volume.rolling(6, min_periods=3).mean() / volume.rolling(24, min_periods=8).mean().replace(0, np.nan)
    out["volume_spike_12"] = (volume > volume.rolling(12, min_periods=4).mean() * 1.8).astype(int)
    out["volume_trend_12"] = volume / volume.shift(12).replace(0, np.nan) - 1.0

    for span in (6, 12, 24, 48):
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
        out[f"close_vs_ema_{span}"] = close / out[f"ema_{span}"].replace(0, np.nan) - 1.0
        out[f"ema_slope_{span}"] = out[f"ema_{span}"] / out[f"ema_{span}"].shift(3).replace(0, np.nan) - 1.0
    out["ema_6_vs_12"] = out["ema_6"] / out["ema_12"].replace(0, np.nan) - 1.0
    out["ema_12_vs_24"] = out["ema_12"] / out["ema_24"].replace(0, np.nan) - 1.0
    out["trend_stack_short"] = ((out["ema_6"] < out["ema_12"]) & (out["ema_12"] < out["ema_24"])).astype(int)
    out["trend_stack_long"] = ((out["ema_6"] > out["ema_12"]) & (out["ema_12"] > out["ema_24"])).astype(int)
    out["trend_compression"] = (out["ema_6"] - out["ema_24"]).abs() / close.replace(0, np.nan)
    out["momentum_accel_3_12"] = out["momentum_3"] - out["momentum_12"]
    out["return_zscore_12"] = rolling_zscore(close.pct_change(), 12)
    out["return_zscore_24"] = rolling_zscore(close.pct_change(), 24)
    out["downside_momentum_6"] = np.minimum(out["momentum_6"], 0.0)
    out["upside_momentum_6"] = np.maximum(out["momentum_6"], 0.0)

    prior_low_12 = low.shift(1).rolling(12, min_periods=4).min()
    prior_low_24 = low.shift(1).rolling(24, min_periods=8).min()
    prior_high_12 = high.shift(1).rolling(12, min_periods=4).max()
    prior_high_24 = high.shift(1).rolling(24, min_periods=8).max()
    out["breakdown_proxy_12"] = np.maximum(0.0, (prior_low_12 - close) / close.replace(0, np.nan))
    out["breakdown_proxy_24"] = np.maximum(0.0, (prior_low_24 - close) / close.replace(0, np.nan))
    out["close_below_rolling_low_12"] = (close < prior_low_12).astype(int)
    out["close_below_rolling_low_24"] = (close < prior_low_24).astype(int)
    out["distance_to_rolling_low_12"] = (close - prior_low_12) / close.replace(0, np.nan)
    out["distance_to_rolling_low_24"] = (close - prior_low_24) / close.replace(0, np.nan)
    out["distance_to_rolling_high_12"] = (prior_high_12 - close) / close.replace(0, np.nan)
    out["distance_to_rolling_high_24"] = (prior_high_24 - close) / close.replace(0, np.nan)
    swept_12 = (low < prior_low_12) & (close >= prior_low_12)
    swept_24 = (low < prior_low_24) & (close >= prior_low_24)
    out["failed_breakdown_proxy"] = (swept_12 | swept_24).astype(int)
    out["fake_breakdown_risk_proxy"] = ((swept_12 | swept_24) & (out["lower_wick_pct"] > 0.35)).astype(int)
    out["room_to_fall_proxy_24"] = np.maximum(0.0, (close - prior_low_24) / close.replace(0, np.nan))
    out["extension_down_proxy"] = np.maximum(0.0, -out["close_vs_ema_24"])
    out["exhaustion_down_proxy"] = np.maximum(0.0, -out["momentum_12"]) * (1.0 + out["lower_wick_pct"].fillna(0.0))
    out["rebound_risk_proxy"] = ((out["lower_wick_pct"] > 0.45) & (out["momentum_3"] > 0)).astype(int)
    out["squeeze_risk_proxy_causal"] = ((out["momentum_12"] < -out["atr_proxy_12"].fillna(0.0) * 2.0) & (out["lower_wick_pct"] > 0.30)).astype(int)

    path_abs = (close - close.shift(12)).abs()
    path_sum = close.diff().abs().rolling(12, min_periods=4).sum()
    out["chop_proxy_12"] = 1.0 - (path_abs / path_sum.replace(0, np.nan)).clip(0, 1)
    out["trend_strength_proxy_12"] = (out["momentum_12"].abs() / out["atr_proxy_12"].replace(0, np.nan)).clip(0, 10)
    out["trend_strength_proxy_24"] = (out["momentum_24"].abs() / out["atr_proxy_24"].replace(0, np.nan)).clip(0, 10)
    out["range_contraction_proxy"] = (out["rolling_range_mean_6"] < out["rolling_range_mean_24"] * 0.75).astype(int)
    out["range_expansion_proxy"] = (out["rolling_range_mean_6"] > out["rolling_range_mean_24"] * 1.25).astype(int)
    out["high_vol_regime_proxy"] = (out["rolling_range_mean_24"] > out["rolling_range_mean_24"].rolling(96, min_periods=24).quantile(0.75)).astype(int)
    out["low_vol_regime_proxy"] = (out["rolling_range_mean_24"] < out["rolling_range_mean_24"].rolling(96, min_periods=24).quantile(0.25)).astype(int)
    out["immediate_reversal_risk_proxy"] = ((out["lower_wick_pct"] > 0.40) & (out["close_position_in_range"] > 0.60)).astype(int)
    out["overextended_down_risk_proxy"] = ((out["momentum_12"] < -out["atr_proxy_12"].fillna(0.0) * 3.0) & (out["close_vs_ema_24"] < 0)).astype(int)
    out["low_room_to_fall_risk_proxy"] = (out["room_to_fall_proxy_24"] < out["atr_proxy_12"].fillna(0.0)).astype(int)
    out["high_wick_reclaim_risk_proxy"] = ((out["lower_wick_pct"] > 0.45) & (out["failed_breakdown_proxy"] == 1)).astype(int)
    out["squeeze_plus_reclaim_risk_proxy"] = ((out["squeeze_risk_proxy_causal"] == 1) & (out["failed_breakdown_proxy"] == 1)).astype(int)
    return out


def add_market_context(features: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = features.copy()
    for prefix, ctx in (("btc", context.get("BTCUSDT")), ("eth", context.get("ETHUSDT"))):
        if ctx is None or ctx.empty:
            continue
        cols = ["timestamp", "ret_3", "ret_6", "ret_12", "rolling_range_mean_12", "trend_stack_short"]
        available = [c for c in cols if c in ctx.columns]
        renamed = ctx[available].rename(columns={
            "ret_3": f"{prefix}_ret_3",
            "ret_6": f"{prefix}_ret_6",
            "ret_12": f"{prefix}_ret_12",
            "rolling_range_mean_12": f"{prefix}_volatility_12",
            "trend_stack_short": f"{prefix}_trend_proxy",
        })
        out = out.merge(renamed, on="timestamp", how="left")
    if "btc_ret_6" in out:
        out["symbol_vs_btc_ret_6"] = out["ret_6"] - out["btc_ret_6"]
    if "eth_ret_6" in out:
        out["symbol_vs_eth_ret_6"] = out["ret_6"] - out["eth_ret_6"]
    return out


def namespace_input(df: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    out = pd.DataFrame(index=df.index)
    id_cols: list[str] = []
    target_cols: list[str] = []
    for col in ID_SOURCE_COLUMNS:
        if col in df:
            new = f"id.{col}" if col != "timestamp" else "id.timestamp"
            out[new] = df[col]
            id_cols.append(new)
    out["id.row_id"] = np.arange(len(df), dtype=int)
    out["id.source_path"] = str(source_path)
    out["id.dataset_version"] = DATASET_VERSION
    id_cols.extend(["id.row_id", "id.source_path", "id.dataset_version"])
    for col in TARGET_SOURCE_COLUMNS:
        if col in df:
            new = f"target.{col}"
            out[new] = df[col]
            target_cols.append(new)
    for col in LABEL_SOURCE_COLUMNS:
        if col in df:
            new = f"label.{col}"
            out[new] = df[col]
            target_cols.append(new)
    for col in FUTURE_EVAL_SOURCE_COLUMNS:
        if col in df:
            new = f"future_eval.{col}"
            out[new] = df[col]
            target_cols.append(new)
    return out, id_cols, target_cols, [c for c in df.columns if c not in set(ID_SOURCE_COLUMNS + TARGET_SOURCE_COLUMNS + LABEL_SOURCE_COLUMNS + FUTURE_EVAL_SOURCE_COLUMNS)]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_csv) if args.input_csv else latest_report_input()
    if not input_path or not input_path.exists():
        raise FileNotFoundError("Input Risk V4 / QMAE CSV not found")
    base = pd.read_csv(input_path)
    if args.max_rows:
        base = base.head(int(args.max_rows)).copy()
    if args.symbols:
        allowed = {normalize_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()}
        base = base[base["symbol"].map(normalize_symbol).isin(allowed)].copy()
    if args.horizons:
        allowed_h = {str(x.strip()) for x in args.horizons.split(",") if x.strip()}
        base = base[base["horizon"].astype(str).isin(allowed_h)].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce").dt.tz_localize(None)
    base["symbol"] = base["symbol"].map(normalize_symbol)
    base["timeframe"] = base["timeframe"].astype(str)
    namespaced, id_cols, target_cols, manual_source_cols = namespace_input(base, input_path)
    unavailable: list[dict[str, str]] = []
    feature_frames: list[pd.DataFrame] = []
    context_cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}

    for (symbol, timeframe), rows in base.groupby(["symbol", "timeframe"], dropna=False):
        start = rows["timestamp"].min()
        end = rows["timestamp"].max()
        candles = load_ohlcv(symbol, timeframe, start, end)
        if candles.empty:
            unavailable.append({"feature_group": "ohlcv", "reason": f"no_ohlcv:{symbol}:{timeframe}"})
            continue
        feats = compute_causal_features(candles)
        context: dict[str, pd.DataFrame] = {}
        for ctx_symbol in ("BTCUSDT", "ETHUSDT"):
            key = (ctx_symbol, timeframe, start, end)
            if key not in context_cache:
                ctx_candles = load_ohlcv(ctx_symbol, timeframe, start, end)
                context_cache[key] = compute_causal_features(ctx_candles) if not ctx_candles.empty else pd.DataFrame()
            if not context_cache[key].empty:
                context[ctx_symbol] = context_cache[key]
            else:
                unavailable.append({"feature_group": "market_context", "reason": f"no_context:{ctx_symbol}:{timeframe}"})
        feats = add_market_context(feats, context)
        subset = rows[["timestamp"]].copy()
        subset["_orig_index"] = rows.index
        aligned = subset.merge(feats, on="timestamp", how="left").set_index("_orig_index").sort_index()
        feature_frames.append(aligned.drop(columns=["timestamp", "open", "high", "low", "volume"], errors="ignore"))
    if feature_frames:
        raw_features = pd.concat(feature_frames, axis=0).sort_index()
    else:
        raw_features = pd.DataFrame(index=base.index)

    feature_cols: list[str] = []
    excluded: list[dict[str, str]] = []
    manual_review: list[dict[str, str]] = []
    feature_out = pd.DataFrame(index=base.index)
    for col in raw_features.columns:
        new_col = f"feature.{col}"
        leak, reason = is_leakage_column(new_col)
        if leak:
            excluded.append({"column": new_col, "reason": reason})
            continue
        if any(token in new_col.lower() for token in ("score", "decision", "guard", "vote", "reason", "action")):
            manual_review.append({"column": new_col, "reason": "ambiguous_semantic_name"})
            continue
        feature_out[new_col] = pd.to_numeric(raw_features[col], errors="coerce")
        feature_cols.append(new_col)
    for col in manual_source_cols:
        leak, reason = is_leakage_column(col)
        if leak:
            excluded.append({"column": col, "reason": reason})
        else:
            manual_review.append({"column": col, "reason": "source_column_not_auto_included"})

    final = pd.concat([namespaced, feature_out], axis=1)
    for col in feature_cols:
        final[col] = pd.to_numeric(final[col], errors="coerce")
    unavailable.extend(unavailable_requested_features(feature_cols))
    diagnostics = dataset_diagnostics(final, feature_cols, target_cols)
    decision, reason, next_phase = decide_dataset(feature_cols, diagnostics, excluded, target_cols)
    stamp = utc_stamp()
    csv_path = output_dir / f"aegis_trrm_causal_feature_dataset_d_{stamp}.csv"
    json_path = output_dir / f"aegis_phase_d_trrm_causal_features_{stamp}.json"
    md_path = output_dir / f"aegis_phase_d_trrm_causal_features_{stamp}.md"
    final.to_csv(csv_path, index=False)
    result = {
        "schema_version": DATASET_VERSION,
        "generated_at": stamp,
        "decision": decision,
        "reason": reason,
        "recommended_next_phase": next_phase,
        "input_dataset": str(input_path),
        "output_dataset": str(csv_path),
        "rows": int(len(final)),
        "symbols": sorted(final["id.symbol"].dropna().astype(str).unique().tolist()) if "id.symbol" in final else [],
        "horizons": sorted(final["id.horizon"].dropna().astype(str).unique().tolist()) if "id.horizon" in final else [],
        "timeframes": sorted(final["id.timeframe"].dropna().astype(str).unique().tolist()) if "id.timeframe" in final else [],
        "timestamp_min": str(final["id.timestamp"].min()) if "id.timestamp" in final else None,
        "timestamp_max": str(final["id.timestamp"].max()) if "id.timestamp" in final else None,
        "id_columns": id_cols,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "excluded_leakage_columns": excluded,
        "manual_review_columns": manual_review,
        "unavailable_features": unavailable,
        "included_feature_count": len(feature_cols),
        "excluded_leakage_count": len(excluded),
        "manual_review_count": len(manual_review),
        "unavailable_feature_count": len(unavailable),
        "causal_feature_groups": feature_groups_present(feature_cols),
        "diagnostics": diagnostics,
        "strict_causal": parse_bool(args.strict_causal),
        "artifacts": {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)},
        "safety_confirmations": {
            "no_live_touched": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_ts_touched": True,
            "no_push": True,
            "no_model_training": True,
            "no_model_promotion": True,
            "research_only_artifacts": True,
            "no_future_or_label_features": True,
        },
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"decision": decision, "reason": reason, "csv": str(csv_path), "md": str(md_path), "json": str(json_path)}, indent=2))
    return result


def unavailable_requested_features(feature_cols: list[str]) -> list[dict[str, str]]:
    present = {c.replace("feature.", "") for c in feature_cols}
    out: list[dict[str, str]] = []
    for group, names in REQUESTED_FEATURES.items():
        for name in names:
            if name not in present:
                out.append({"feature_group": group, "feature": name, "reason": "not_generated_or_not_available"})
    return out


def feature_groups_present(feature_cols: list[str]) -> dict[str, list[str]]:
    present = {c.replace("feature.", ""): c for c in feature_cols}
    groups: dict[str, list[str]] = {}
    for group, names in REQUESTED_FEATURES.items():
        groups[group] = [present[name] for name in names if name in present]
    return groups


def bool_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return series.astype(str).str.lower().isin({"1", "true", "yes"}).mean()


def standardized_mean_diff(feature: pd.Series, target: pd.Series) -> float:
    y = target.astype(str).str.lower().isin({"1", "true", "yes"})
    a = pd.to_numeric(feature[y], errors="coerce").dropna()
    b = pd.to_numeric(feature[~y], errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return 0.0
    pooled = math.sqrt((float(a.var()) + float(b.var())) / 2.0)
    if not math.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def dataset_diagnostics(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> dict[str, Any]:
    target_rates: dict[str, float] = {}
    for col in target_cols:
        if col.startswith(("target.", "label.")):
            target_rates[f"{col}_rate"] = float(bool_rate(df[col]))
    if {"target.tail_risk_v4", "target.bad_entry_v4"} <= set(df.columns):
        union = df["target.tail_risk_v4"].astype(str).str.lower().isin({"1", "true", "yes"}) | df["target.bad_entry_v4"].astype(str).str.lower().isin({"1", "true", "yes"})
        target_rates["target.union_tail_or_bad_rate"] = float(union.mean())
    missing = {c: float(df[c].isna().mean()) for c in feature_cols}
    constants = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    high_missing = [c for c, rate in missing.items() if rate > 0.40]
    duplicate_rows = int(df.duplicated(subset=["id.symbol", "id.timestamp", "id.timeframe", "id.horizon"] if {"id.symbol", "id.timestamp", "id.timeframe", "id.horizon"} <= set(df.columns) else None).sum())
    corr_warnings: list[dict[str, Any]] = []
    top_sep: list[dict[str, Any]] = []
    target_candidates = [c for c in ("target.tail_risk_v4", "target.bad_entry_v4") if c in df]
    if {"target.tail_risk_v4", "target.bad_entry_v4"} <= set(df.columns):
        df["_target_union_tmp"] = (
            df["target.tail_risk_v4"].astype(str).str.lower().isin({"1", "true", "yes"})
            | df["target.bad_entry_v4"].astype(str).str.lower().isin({"1", "true", "yes"})
        ).astype(int)
        target_candidates.append("_target_union_tmp")
    for f in feature_cols:
        x = pd.to_numeric(df[f], errors="coerce")
        if x.nunique(dropna=True) <= 1:
            continue
        for target in target_candidates:
            y = df[target].astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)
            corr = x.corr(y)
            if pd.notna(corr) and abs(float(corr)) > 0.95:
                corr_warnings.append({"feature": f, "target": target, "correlation": float(corr)})
            smd = standardized_mean_diff(x, df[target])
            top_sep.append({"feature": f, "target": target.replace("_target_union_tmp", "target.union_tail_or_bad"), "standardized_mean_diff": smd, "abs_smd": abs(smd)})
    top_sep = sorted(top_sep, key=lambda r: r["abs_smd"], reverse=True)[:30]
    coverage_symbol = df.groupby("id.symbol").size().to_dict() if "id.symbol" in df else {}
    coverage_horizon = df.groupby("id.horizon").size().to_dict() if "id.horizon" in df else {}
    feature_availability: list[dict[str, Any]] = []
    if {"id.symbol", "id.horizon"} <= set(df.columns):
        for (symbol, horizon), g in df.groupby(["id.symbol", "id.horizon"]):
            feature_availability.append({
                "symbol": symbol,
                "horizon": horizon,
                "rows": int(len(g)),
                "available_feature_count": int(sum(g[c].notna().mean() >= 0.60 for c in feature_cols)),
            })
    return {
        "rows": int(len(df)),
        "feature_count": len(feature_cols),
        "target_rates": target_rates,
        "missing_rate_by_feature": missing,
        "average_missing_rate": float(np.mean(list(missing.values()))) if missing else 1.0,
        "constant_feature_count": len(constants),
        "constant_features": constants,
        "highly_missing_features": high_missing,
        "suspicious_correlations": corr_warnings,
        "duplicate_rows": duplicate_rows,
        "per_symbol_row_count": {str(k): int(v) for k, v in coverage_symbol.items()},
        "per_horizon_row_count": {str(k): int(v) for k, v in coverage_horizon.items()},
        "feature_availability_by_symbol_horizon": feature_availability,
        "top_separation_features": top_sep,
    }


def decide_dataset(feature_cols: list[str], diagnostics: dict[str, Any], excluded: list[dict[str, str]], target_cols: list[str]) -> tuple[str, str, str]:
    if any(is_leakage_column(c)[0] for c in feature_cols):
        return "LEAKAGE_RISK_TOO_HIGH", "leakage pattern included in feature columns", "STOP"
    if diagnostics["suspicious_correlations"]:
        return "LEAKAGE_RISK_TOO_HIGH", "feature-target correlation above 0.95 detected", "STOP"
    if not {"target.tail_risk_v4", "target.bad_entry_v4"} & set(target_cols):
        return "DATASET_NOT_USABLE", "no usable target columns", "STOP"
    usable = len(feature_cols) - diagnostics["constant_feature_count"] - len(diagnostics["highly_missing_features"])
    if usable < 10:
        return "DATASET_NOT_USABLE", "fewer than 10 usable causal features", "STOP"
    if usable < 20:
        return "CAUSAL_FEATURE_DATASET_PARTIAL", "10-19 usable causal features after diagnostics", "FASE-E: retrain TRRM with enriched causal features"
    if diagnostics["average_missing_rate"] > 0.40:
        return "CAUSAL_FEATURE_DATASET_PARTIAL", "average missing rate is high but leakage controls passed", "FASE-D2: improve OHLCV alignment / BTC-ETH context"
    return "CAUSAL_FEATURE_DATASET_READY", "enriched causal dataset passed leakage and availability checks", "FASE-E: retrain TRRM with enriched causal features"


def render_markdown(result: dict[str, Any]) -> str:
    diag = result["diagnostics"]
    groups = result["causal_feature_groups"]
    lines = [
        "# FASE-D TRRM Causal Feature Dataset",
        "",
        "## 1. Estado",
        f"- decision: {result['decision']}",
        "- mode: research-only",
        f"- reason: {result['reason']}",
        "",
        "## 2. Input dataset",
        f"- path: {result['input_dataset']}",
        f"- rows: {result['rows']}",
        f"- symbols: {', '.join(result['symbols'])}",
        f"- horizons: {', '.join(result['horizons'])}",
        f"- timeframe: {', '.join(result['timeframes'])}",
        f"- timestamp range: {result['timestamp_min']} to {result['timestamp_max']}",
        "",
        "## 3. Output dataset",
        f"- path: {result['output_dataset']}",
        f"- rows: {result['rows']}",
        f"- feature count: {result['included_feature_count']}",
        f"- target columns: {len(result['target_columns'])}",
        f"- id columns: {len(result['id_columns'])}",
        "",
        "## 4. Feature groups generated",
        *[f"- {group}: {len(cols)}" for group, cols in groups.items()],
        "",
        "## 5. Feature columns",
        *[f"- {c}" for c in result["feature_columns"]],
        "",
        "## 6. Target/evaluation columns",
        *[f"- {c}" for c in result["target_columns"]],
        "",
        "## 7. Leakage exclusions",
        *[f"- {r['column']}: {r['reason']}" for r in result["excluded_leakage_columns"][:80]],
        "",
        "## 8. Manual review columns",
        *[f"- {r['column']}: {r['reason']}" for r in result["manual_review_columns"][:80]],
        "",
        "## 9. Unavailable features",
        *[f"- {r.get('feature_group')}: {r.get('feature', '')} {r.get('reason')}" for r in result["unavailable_features"][:120]],
        "",
        "## 10. Missing/constant feature diagnostics",
        f"- average_missing_rate: {diag['average_missing_rate']:.4f}",
        f"- constant_feature_count: {diag['constant_feature_count']}",
        f"- highly_missing_features: {len(diag['highly_missing_features'])}",
        f"- duplicate_rows: {diag['duplicate_rows']}",
        "",
        "## 11. Target rates",
        *[f"- {k}: {v:.4f}" for k, v in diag["target_rates"].items()],
        "",
        "## 12. Per-symbol/per-horizon coverage",
        *[f"- symbol {k}: {v}" for k, v in diag["per_symbol_row_count"].items()],
        *[f"- horizon {k}: {v}" for k, v in diag["per_horizon_row_count"].items()],
        "",
        "## 13. Basic target separation diagnostics",
        *[f"- {r['feature']} vs {r['target']}: smd={r['standardized_mean_diff']:.4f}" for r in diag["top_separation_features"][:25]],
        "",
        "## 14. Suspicious correlations",
        *([f"- {r['feature']} vs {r['target']}: corr={r['correlation']:.6f}" for r in diag["suspicious_correlations"]] or ["- none"]),
        "",
        "## 15. Limitations",
        "- Features are OHLCV-derived and still need FASE-E walk-forward model validation.",
        "- Market context is only included where BTC/ETH OHLCV aligns safely by timestamp/timeframe.",
        "- This dataset preserves labels/future columns for evaluation only; they are excluded from feature_columns.",
        "",
        "## 16. Decision",
        f"- {result['decision']}",
        "",
        "## 17. Recommended next phase",
        f"- {result['recommended_next_phase']}",
        "",
        "## 18. Safety confirmations",
        "- No toqué live.",
        "- No toqué active_manifest.",
        "- No toqué YAML.",
        "- No reinicié PM2.",
        "- No envié órdenes.",
        "- No toqué .env.",
        "- No toqué binance-futures-bot-ts.",
        "- No hice push.",
        "- No entrené modelos.",
        "- No promoví modelos.",
        "- Artifacts generados son research-only.",
        "- No usé labels/futuro como features.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build research-only TRRM causal feature dataset.")
    p.add_argument("--input-csv", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--symbols", default="")
    p.add_argument("--horizons", default="")
    p.add_argument("--strict-causal", default="true")
    p.add_argument("--write-report", default="true")
    return p


def main() -> int:
    build_dataset(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
