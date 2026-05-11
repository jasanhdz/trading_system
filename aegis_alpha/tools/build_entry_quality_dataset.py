#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FALLBACK))

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.edge.common import build_edge_feature_matrix
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG


DEFAULT_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)
LOOKBACK_DAYS = (7, 14, 30)
MAX_HORIZON_BARS = 96
HORIZONS = {
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
    "8h": 96,
}
FEATURE_WARMUP_BARS = 64
DEFAULT_OUTPUT = REPO_ROOT / "aegis_alpha/data/processed/entry_quality/entry_quality_dataset_v020.parquet"
INPUT_VALIDATION_DIR = REPO_ROOT / "aegis_alpha/logs/entry_quality"
REPORT_JSON = INPUT_VALIDATION_DIR / "entry_quality_dataset_report_v020.json"
REPORT_MD = INPUT_VALIDATION_DIR / "entry_quality_dataset_report_v020.md"
DEFAULT_TS_YAML = REPO_ROOT / "binance-futures-bot-ts/regime_config.live.yaml"
DATABASE_PATH = REPO_ROOT / "data/binance_candles.db"


@dataclass(frozen=True)
class SymbolDataset:
    rows: list[dict[str, Any]]
    validation: dict[str, Any]
    warnings: list[str]
    candidate_method: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def db_symbol(symbol: str) -> str:
    clean = normalize_symbol(symbol)
    return clean.replace("USDT", "/USDT") if "/" not in clean else clean


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "now":
        return pd.Timestamp.now(tz="UTC").tz_convert(None)
    parsed = pd.Timestamp(text)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def safe_float(value: Any) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def finite_or_none(value: Any) -> float | None:
    result = safe_float(value)
    return result if np.isfinite(result) else None


def timeframe_minutes(timeframe: str) -> int:
    text = timeframe.strip().lower()
    if text.endswith("m"):
        return int(text[:-1])
    if text.endswith("h"):
        return int(text[:-1]) * 60
    raise ValueError(f"unsupported_timeframe: {timeframe}")


def load_leverage_by_symbol(path: Path = DEFAULT_TS_YAML) -> tuple[dict[str, float], float, list[str]]:
    warnings: list[str] = []
    default_leverage = 20.0
    leverage_by_symbol: dict[str, float] = {}
    if yaml is None:
        return leverage_by_symbol, default_leverage, ["PyYAML unavailable; using default leverage=20"]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        default_leverage = float(raw.get("REGIMES", {}).get("AEGIS_TURBO", {}).get("leverage", default_leverage))
        overrides = raw.get("SYMBOL_OVERRIDES", {}) or {}
        for symbol, section in overrides.items():
            value = ((section or {}).get("AEGIS_TURBO") or {}).get("leverage")
            if value is not None:
                leverage_by_symbol[normalize_symbol(symbol)] = float(value)
    except Exception as exc:
        warnings.append(f"leverage_yaml_parse_failed: {exc!r}; using default leverage=20")
    return leverage_by_symbol, default_leverage, warnings


def load_candles(symbol: str, timeframe: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    formatted_symbol = db_symbol(symbol)
    query = """
        SELECT timestamp, open, high, low, close, volume, buy_volume
        FROM ohlcv_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp ASC
    """
    with sqlite3.connect(DATABASE_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(formatted_symbol, timeframe))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    df = df.set_index("timestamp", drop=True)
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_candles(symbol: str, timeframe: str, df: pd.DataFrame) -> dict[str, Any]:
    expected = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    if df.empty:
        return {
            "symbol": normalize_symbol(symbol),
            "rows": 0,
            "first_ts": None,
            "last_ts": None,
            "gaps": None,
            "duplicates": None,
            "valid": False,
            "reason": "empty",
        }
    timestamps = pd.Index(df.index)
    duplicate_count = int(timestamps.duplicated().sum())
    diffs = timestamps.to_series().diff().dropna()
    gap_count = int((diffs > expected).sum())
    max_gap_seconds = int(diffs.max().total_seconds()) if len(diffs) else 0
    return {
        "symbol": normalize_symbol(symbol),
        "rows": int(len(df)),
        "first_ts": str(df.index.min()),
        "last_ts": str(df.index.max()),
        "gaps": gap_count,
        "duplicates": duplicate_count,
        "expected_seconds": int(expected.total_seconds()),
        "max_gap_seconds": max_gap_seconds,
        "valid": bool(len(df) >= FEATURE_WARMUP_BARS + MAX_HORIZON_BARS + 50),
    }


def rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    arr = values.astype(float).to_numpy()
    out = np.full(len(arr), np.nan, dtype=np.float32)
    sorted_window: list[float] = []
    queue: list[float] = []
    for idx, value in enumerate(arr):
        if np.isfinite(value):
            bisect.insort(sorted_window, float(value))
            queue.append(float(value))
        else:
            queue.append(float("nan"))
        if len(queue) > window:
            old = queue.pop(0)
            if np.isfinite(old):
                pos = bisect.bisect_left(sorted_window, old)
                if pos < len(sorted_window):
                    sorted_window.pop(pos)
        if np.isfinite(value) and sorted_window:
            out[idx] = bisect.bisect_right(sorted_window, float(value)) / len(sorted_window)
    return pd.Series(out, index=values.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(high - low)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    quote_volume = close * volume

    for bars in (1, 2, 3, 6, 12):
        out[f"ret_{bars}"] = close / close.shift(bars) - 1.0
    out["momentum_3"] = out["ret_3"]
    out["momentum_6"] = out["ret_6"]
    out["momentum_12"] = out["ret_12"]
    out["candle_body_pct"] = (close - open_) / open_.replace(0, np.nan)
    candle_range = (high - low).replace(0, np.nan)
    out["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    out["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range
    out["green_candle_count_3"] = (close > open_).astype(float).rolling(3, min_periods=1).sum()
    out["red_candle_count_3"] = (close < open_).astype(float).rolling(3, min_periods=1).sum()

    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["price_to_ema_9"] = close / out["ema_9"] - 1.0
    out["price_to_ema_21"] = close / out["ema_21"] - 1.0
    out["ema_9_slope"] = out["ema_9"] / out["ema_9"].shift(3) - 1.0
    out["ema_21_slope"] = out["ema_21"] / out["ema_21"].shift(3) - 1.0

    tr = true_range(out)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)
    returns = close.pct_change()
    out["realized_vol_12"] = returns.rolling(12, min_periods=4).std()
    out["realized_vol_36"] = returns.rolling(36, min_periods=12).std()
    out["high_low_range_pct"] = (high - low) / close.replace(0, np.nan)
    out["atr_percentile_7d"] = rolling_percentile(out["atr_pct"], 7 * 24 * 12)

    vol_mean = volume.rolling(36, min_periods=12).mean()
    vol_std = volume.rolling(36, min_periods=12).std()
    out["volume_zscore_36"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    out["volume_ratio_12"] = volume / volume.rolling(12, min_periods=4).mean().replace(0, np.nan)
    out["quote_volume"] = quote_volume
    return out


def mtf_frame(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    resampled = pd.DataFrame(
        {
            "open": df["open"].resample(rule, label="right", closed="right").first(),
            "high": df["high"].resample(rule, label="right", closed="right").max(),
            "low": df["low"].resample(rule, label="right", closed="right").min(),
            "close": df["close"].resample(rule, label="right", closed="right").last(),
            "volume": df["volume"].resample(rule, label="right", closed="right").sum(),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    if resampled.empty:
        return pd.DataFrame(index=df.index)
    close = resampled["close"].astype(float)
    resampled[f"{prefix}_ret_1"] = close / close.shift(1) - 1.0
    resampled[f"{prefix}_ret_2"] = close / close.shift(2) - 1.0
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    resampled[f"{prefix}_ema_9_slope"] = ema_9 / ema_9.shift(2) - 1.0
    resampled[f"{prefix}_price_to_ema_21"] = close / ema_21 - 1.0
    atr = true_range(resampled).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    resampled[f"{prefix}_atr_pct"] = atr / close.replace(0, np.nan)
    trend = np.where(
        (close > ema_21) & (ema_9 > ema_21) & (resampled[f"{prefix}_ema_9_slope"] > 0),
        1,
        np.where((close < ema_21) & (ema_9 < ema_21) & (resampled[f"{prefix}_ema_9_slope"] < 0), -1, 0),
    )
    resampled[f"{prefix}_trend_direction"] = trend.astype(np.int8)
    cols = [
        f"{prefix}_ret_1",
        f"{prefix}_ret_2",
        f"{prefix}_ema_9_slope",
        f"{prefix}_price_to_ema_21",
        f"{prefix}_atr_pct",
        f"{prefix}_trend_direction",
    ]
    return resampled[cols].reindex(df.index, method="ffill")


def add_mtf_features(features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out = out.join(mtf_frame(candles, "15min", "mtf_15m"))
    out = out.join(mtf_frame(candles, "1h", "mtf_1h"))
    out["long_mtf_agreement"] = (
        (out["momentum_3"] > 0)
        & (out["mtf_15m_trend_direction"] > 0)
        & (out["mtf_1h_trend_direction"] >= 0)
    ).astype(np.int8)
    out["short_mtf_agreement"] = (
        (out["momentum_3"] < 0)
        & (out["mtf_15m_trend_direction"] < 0)
        & (out["mtf_1h_trend_direction"] <= 0)
    ).astype(np.int8)
    five_min_side = np.sign(out["momentum_6"].fillna(0.0))
    mtf_side = np.sign(out["mtf_15m_trend_direction"].fillna(0.0) + out["mtf_1h_trend_direction"].fillna(0.0))
    out["mtf_conflict"] = ((five_min_side != 0) & (mtf_side != 0) & (five_min_side != mtf_side)).astype(np.int8)
    return out


def model_path_from_manifest(symbol: str, side: str, lookback_days: int) -> Path:
    symbol_dir = REPO_ROOT / "aegis_alpha/models/turbo" / normalize_symbol(symbol)
    manifest_path = symbol_dir / "active_manifest.json"
    key = f"{side}_{lookback_days}d"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_path = (manifest.get("model_paths") or {}).get(key)
            if raw_path:
                path = Path(raw_path)
                if not path.is_absolute():
                    path = REPO_ROOT / path
                if path.exists():
                    return path
        except Exception:
            pass
    active = symbol_dir / "active" / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"
    if active.exists():
        return active
    return symbol_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"


def load_turbo_models(symbol: str) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    models: dict[str, Any] = {}
    paths: dict[str, str] = {}
    warnings: list[str] = []
    for lookback in LOOKBACK_DAYS:
        for side in ("long", "short"):
            key = f"{side}_{lookback}d"
            path = model_path_from_manifest(symbol, side, lookback)
            paths[key] = str(path)
            if not path.exists():
                warnings.append(f"{key}_model_missing: {path}")
                continue
            try:
                bundle = joblib.load(path)
                estimator = bundle.get("estimator") if isinstance(bundle, dict) else bundle
                feature_names = bundle.get("feature_names") if isinstance(bundle, dict) else None
                if feature_names is not None and len(feature_names) != 168:
                    warnings.append(f"{key}_feature_count_unexpected: {len(feature_names)}")
                models[key] = estimator
            except Exception as exc:
                warnings.append(f"{key}_model_load_failed: {exc!r}")
    return models, paths, warnings


def raw_turbo_score(direction: str, scores: dict[str, np.ndarray], votes: dict[str, np.ndarray]) -> np.ndarray:
    agreement = votes[direction] / max(len(LOOKBACK_DAYS), 1)
    side_stack = np.vstack([scores[f"{direction}_{lookback}d"] for lookback in LOOKBACK_DAYS])
    magnitude = np.nanmax(side_stack, axis=0)
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude_score = np.clip(magnitude / 0.003, 0.0, 1.0)
    return np.clip(0.85 * agreement + 0.15 * magnitude_score, 0.0, 1.0)


def score_turbo_models(symbol: str, candles: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    warnings: list[str] = []
    score_frame = pd.DataFrame(index=features.index)
    for lookback in LOOKBACK_DAYS:
        score_frame[f"long_score_{lookback}d"] = np.nan
        score_frame[f"short_score_{lookback}d"] = np.nan
    score_frame["votes_long"] = 0
    score_frame["votes_short"] = 0
    score_frame["votes_neutral"] = len(LOOKBACK_DAYS)
    score_frame["turbo_action"] = "HOLD"
    score_frame["turbo_score"] = np.nan
    score_frame["score_gap"] = np.nan

    models, model_paths, model_warnings = load_turbo_models(symbol)
    warnings.extend(model_warnings)
    if not models:
        score_frame["candidate_generation_method"] = "rules_only_no_turbo_models"
        return score_frame, "rules_only_no_turbo_models", warnings

    try:
        turbo_feature_frame = build_feature_frame(candles)
        base_features = turbo_feature_frame[FEATURE_COLUMNS].values.astype(np.float32)
        edge_x = build_edge_feature_matrix(base_features, DEFAULT_TURBO_CONFIG.model_dir and FEATURE_WARMUP_BARS)
        model_index = turbo_feature_frame.index[FEATURE_WARMUP_BARS:]
    except Exception as exc:
        warnings.append(f"turbo_feature_build_failed: {exc!r}")
        score_frame["candidate_generation_method"] = "rules_only_turbo_feature_error"
        return score_frame, "rules_only_turbo_feature_error", warnings

    prediction_columns: dict[str, np.ndarray] = {}
    for lookback in LOOKBACK_DAYS:
        for side in ("long", "short"):
            key = f"{side}_{lookback}d"
            arr = np.full(len(edge_x), np.nan, dtype=np.float32)
            estimator = models.get(key)
            if estimator is not None:
                try:
                    arr = np.asarray(estimator.predict(edge_x), dtype=np.float32)
                except Exception as exc:
                    warnings.append(f"{key}_predict_failed: {exc!r}")
            prediction_columns[key] = arr
            public_col = f"{side}_score_{lookback}d"
            score_frame.loc[model_index, public_col] = arr

    long_votes = np.zeros(len(edge_x), dtype=np.int16)
    short_votes = np.zeros(len(edge_x), dtype=np.int16)
    neutral_votes = np.zeros(len(edge_x), dtype=np.int16)
    for lookback in LOOKBACK_DAYS:
        long_values = prediction_columns[f"long_{lookback}d"]
        short_values = prediction_columns[f"short_{lookback}d"]
        valid = np.isfinite(long_values) & np.isfinite(short_values) & (np.maximum(long_values, short_values) > 0.0)
        long_votes += (valid & (long_values >= short_values)).astype(np.int16)
        short_votes += (valid & (short_values > long_values)).astype(np.int16)
        neutral_votes += (~valid).astype(np.int16)

    vote_arrays = {"long": long_votes, "short": short_votes, "neutral": neutral_votes}
    long_score = raw_turbo_score("long", prediction_columns, vote_arrays)
    short_score = raw_turbo_score("short", prediction_columns, vote_arrays)
    turbo_score = np.maximum(long_score, short_score)
    score_gap = long_score - short_score
    action = np.where(
        (long_votes >= 2) & (long_score >= short_score),
        "LONG",
        np.where((short_votes >= 2) & (short_score > long_score), "SHORT", "HOLD"),
    )
    score_frame.loc[model_index, "votes_long"] = long_votes
    score_frame.loc[model_index, "votes_short"] = short_votes
    score_frame.loc[model_index, "votes_neutral"] = neutral_votes
    score_frame.loc[model_index, "turbo_action"] = action
    score_frame.loc[model_index, "turbo_score"] = turbo_score
    score_frame.loc[model_index, "score_gap"] = score_gap
    score_frame["candidate_generation_method"] = "turbo_models_active_manifest"
    score_frame.attrs["model_paths"] = model_paths
    return score_frame, "turbo_models_active_manifest", warnings


def side_candidate_masks(frame: pd.DataFrame, candidate_threshold: float) -> tuple[pd.Series, pd.Series]:
    turbo_long = frame["turbo_action"].eq("LONG")
    turbo_short = frame["turbo_action"].eq("SHORT")
    long_model_score = frame["turbo_score"].ge(candidate_threshold) & frame["score_gap"].ge(0)
    short_model_score = frame["turbo_score"].ge(candidate_threshold) & frame["score_gap"].lt(0)
    long_votes = frame["votes_long"].fillna(0).ge(2)
    short_votes = frame["votes_short"].fillna(0).ge(2)
    long_rules = (frame["momentum_3"] > 0) & (frame["momentum_6"] > 0) & (frame["long_mtf_agreement"] > 0)
    short_rules = (frame["momentum_3"] < 0) & (frame["momentum_6"] < 0) & (frame["short_mtf_agreement"] > 0)
    return turbo_long | long_votes | long_model_score | long_rules, turbo_short | short_votes | short_model_score | short_rules


def first_hit(values: np.ndarray, threshold: float, greater: bool) -> int:
    if greater:
        hits = np.flatnonzero(values >= threshold)
    else:
        hits = np.flatnonzero(values <= threshold)
    return int(hits[0] + 1) if len(hits) else -1


def profit_before_loss(favorable: np.ndarray, adverse: np.ndarray, profit: float, loss: float) -> int:
    profit_bar = first_hit(favorable, profit, True)
    loss_bar = first_hit(adverse, -abs(loss), False)
    if profit_bar < 0:
        return 0
    if loss_bar < 0:
        return 1
    return int(profit_bar < loss_bar)


def loss_before_profit(favorable: np.ndarray, adverse: np.ndarray, loss: float, profit: float) -> int:
    loss_bar = first_hit(adverse, -abs(loss), False)
    profit_bar = first_hit(favorable, profit, True)
    if loss_bar < 0:
        return 0
    if profit_bar < 0:
        return 1
    return int(loss_bar <= profit_bar)


def classify_quality(
    good: int,
    bad: int,
    tail: int,
    future_mfe_roe: float,
    future_mae_roe: float,
    final_roe_8h: float,
    time_to_green_minutes: int,
) -> str:
    if tail:
        return "TAIL_RISK"
    if good and future_mfe_roe >= 0.15 and future_mae_roe > -0.06 and 0 <= time_to_green_minutes <= 15:
        return "EXCELLENT"
    if good:
        return "GOOD"
    if final_roe_8h > 0 and time_to_green_minutes > 30:
        return "SLOW_WIN"
    if bad and final_roe_8h > 0:
        return "BAD_ENTRY_WIN"
    if bad or (final_roe_8h <= 0 and future_mae_roe <= -0.12):
        return "BAD_ENTRY_LOSS"
    return "UNKNOWN"


def add_future_labels(row: dict[str, Any], candles: pd.DataFrame, idx: int, side: str, leverage: float) -> None:
    entry = float(candles["close"].iloc[idx])
    future = candles.iloc[idx + 1 : idx + MAX_HORIZON_BARS + 1]
    direction = 1.0 if side == "LONG" else -1.0
    if future.empty or entry <= 0:
        row.update(
            {
                "future_mfe_roe": np.nan,
                "future_mae_roe": np.nan,
                "time_to_green_minutes": -1,
                "quality_class": "UNKNOWN",
                "label_good_entry_v1": 0,
                "label_bad_entry_v1": 0,
                "label_tail_risk_v1": 0,
            }
        )
        return

    high = future["high"].to_numpy(dtype=float)
    low = future["low"].to_numpy(dtype=float)
    close = future["close"].to_numpy(dtype=float)
    if side == "LONG":
        favorable_raw = high / entry - 1.0
        adverse_raw = low / entry - 1.0
    else:
        favorable_raw = entry / np.maximum(low, 1e-12) - 1.0
        adverse_raw = entry / np.maximum(high, 1e-12) - 1.0
    close_raw = (close / entry - 1.0) * direction
    favorable = favorable_raw * leverage
    adverse = adverse_raw * leverage
    close_roe = close_raw * leverage
    future_mfe_roe = float(np.nanmax(favorable))
    future_mae_roe = float(np.nanmin(adverse))
    time_to_green_bar = first_hit(close_roe, 0.0, True)
    time_to_green_minutes = time_to_green_bar * 5 if time_to_green_bar > 0 else -1

    row["future_mfe_roe"] = future_mfe_roe
    row["future_mae_roe"] = future_mae_roe
    row["time_to_green_minutes"] = time_to_green_minutes
    row["time_to_5pct_roe"] = first_hit(favorable, 0.05, True) * 5
    row["time_to_8pct_roe"] = first_hit(favorable, 0.08, True) * 5
    row["time_to_10pct_roe"] = first_hit(favorable, 0.10, True) * 5
    row["hit_profit_8_before_loss_8"] = profit_before_loss(favorable, adverse, 0.08, 0.08)
    row["hit_profit_10_before_loss_10"] = profit_before_loss(favorable, adverse, 0.10, 0.10)
    row["hit_loss_15_before_profit_5"] = loss_before_profit(favorable, adverse, 0.15, 0.05)
    row["hit_stop_40_before_profit"] = loss_before_profit(favorable, adverse, 0.40, 0.08)
    row["final_roe_8h"] = float(close_roe[min(len(close_roe), MAX_HORIZON_BARS) - 1])
    row["final_raw_return_8h"] = row["final_roe_8h"] / leverage if leverage else np.nan
    for name, bars in HORIZONS.items():
        end = min(bars, len(favorable))
        row[f"future_mfe_roe_{name}"] = float(np.nanmax(favorable[:end]))
        row[f"future_mae_roe_{name}"] = float(np.nanmin(adverse[:end]))
        row[f"final_roe_{name}"] = float(close_roe[end - 1])

    good_time = time_to_green_minutes >= 0 and time_to_green_minutes <= 30
    label_good = int(row["hit_profit_8_before_loss_8"] == 1 and future_mae_roe > -0.12 and good_time)
    slow_or_weak = (time_to_green_minutes > 120 or time_to_green_minutes < 0) and row["final_roe_8h"] < 0.08
    label_bad = int(row["hit_loss_15_before_profit_5"] == 1 or future_mae_roe <= -0.20 or slow_or_weak)
    label_tail = int(loss_before_profit(favorable, adverse, 0.25, 0.08) == 1 or row["hit_stop_40_before_profit"] == 1)
    row["label_good_entry_v1"] = label_good
    row["label_bad_entry_v1"] = label_bad
    row["label_tail_risk_v1"] = label_tail
    row["quality_class"] = classify_quality(
        label_good,
        label_bad,
        label_tail,
        future_mfe_roe,
        future_mae_roe,
        row["final_roe_8h"],
        time_to_green_minutes,
    )


def feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_6",
        "ret_12",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "green_candle_count_3",
        "red_candle_count_3",
        "ema_9",
        "ema_21",
        "ema_50",
        "price_to_ema_9",
        "price_to_ema_21",
        "ema_9_slope",
        "ema_21_slope",
        "atr_14",
        "atr_pct",
        "realized_vol_12",
        "realized_vol_36",
        "high_low_range_pct",
        "atr_percentile_7d",
        "volume_zscore_36",
        "volume_ratio_12",
        "quote_volume",
        "mtf_15m_ret_1",
        "mtf_15m_ret_2",
        "mtf_15m_ema_9_slope",
        "mtf_15m_price_to_ema_21",
        "mtf_15m_atr_pct",
        "mtf_15m_trend_direction",
        "mtf_1h_ret_1",
        "mtf_1h_ret_2",
        "mtf_1h_ema_9_slope",
        "mtf_1h_price_to_ema_21",
        "mtf_1h_atr_pct",
        "mtf_1h_trend_direction",
        "long_mtf_agreement",
        "short_mtf_agreement",
        "mtf_conflict",
    ]


def label_columns() -> list[str]:
    cols = [
        "future_mfe_roe",
        "future_mae_roe",
        "time_to_green_minutes",
        "time_to_5pct_roe",
        "time_to_8pct_roe",
        "time_to_10pct_roe",
        "hit_profit_8_before_loss_8",
        "hit_profit_10_before_loss_10",
        "hit_loss_15_before_profit_5",
        "hit_stop_40_before_profit",
        "final_roe_8h",
        "final_raw_return_8h",
        "label_good_entry_v1",
        "label_bad_entry_v1",
        "label_tail_risk_v1",
        "quality_class",
    ]
    for name in HORIZONS:
        cols.extend([f"future_mfe_roe_{name}", f"future_mae_roe_{name}", f"final_roe_{name}"])
    return cols


def build_symbol_dataset(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    leverage: float,
    candidate_threshold: float,
    max_rows_per_symbol: int | None,
) -> SymbolDataset:
    warnings: list[str] = []
    candles = load_candles(symbol, timeframe, start, end)
    validation = validate_candles(symbol, timeframe, candles)
    if not validation["valid"]:
        return SymbolDataset([], validation, [f"skipped_symbol_invalid_input: {validation}"], "skipped")

    if int(validation.get("duplicates") or 0) > 0:
        warnings.append(f"deduplicated_in_memory duplicate_count={validation['duplicates']}")
    candles = candles[~candles.index.duplicated(keep="last")].sort_index()
    features = add_mtf_features(add_base_features(candles), candles)
    scores, method, score_warnings = score_turbo_models(symbol, candles, features)
    warnings.extend(score_warnings)
    frame = features.join(scores)
    frame["symbol"] = normalize_symbol(symbol)
    frame["entry_price"] = candles["close"]
    frame["leverage"] = leverage

    usable = np.arange(len(frame)) >= FEATURE_WARMUP_BARS
    usable &= np.arange(len(frame)) < len(frame) - MAX_HORIZON_BARS
    long_mask, short_mask = side_candidate_masks(frame, candidate_threshold)
    long_mask = long_mask & usable
    short_mask = short_mask & usable
    candidate_refs: list[tuple[int, str]] = [(int(idx), "LONG") for idx in np.flatnonzero(long_mask.to_numpy())]
    candidate_refs.extend((int(idx), "SHORT") for idx in np.flatnonzero(short_mask.to_numpy()))
    candidate_refs.sort(key=lambda item: (item[0], item[1]))
    if max_rows_per_symbol and max_rows_per_symbol > 0 and len(candidate_refs) > max_rows_per_symbol:
        positions = np.linspace(0, len(candidate_refs) - 1, max_rows_per_symbol).round().astype(int)
        candidate_refs = [candidate_refs[int(pos)] for pos in positions]
        warnings.append(f"max_rows_per_symbol_applied: {max_rows_per_symbol}")

    rows: list[dict[str, Any]] = []
    feature_cols = feature_columns()
    score_cols = [
        "long_score_7d",
        "long_score_14d",
        "long_score_30d",
        "short_score_7d",
        "short_score_14d",
        "short_score_30d",
        "votes_long",
        "votes_short",
        "votes_neutral",
        "turbo_action",
        "turbo_score",
        "score_gap",
        "candidate_generation_method",
    ]
    for idx, side in candidate_refs:
        series = frame.iloc[idx]
        row = {
            "symbol": normalize_symbol(symbol),
            "timestamp": str(frame.index[idx]),
            "side": side,
            "timeframe": timeframe,
            "entry_price": finite_or_none(series.get("entry_price")),
            "leverage": leverage,
        }
        for col in feature_cols + score_cols:
            value = series.get(col)
            row[col] = str(value) if col in {"turbo_action", "candidate_generation_method"} else finite_or_none(value)
        row["candidate_reason"] = candidate_reason(row, side, candidate_threshold)
        add_future_labels(row, candles, idx, side, leverage)
        rows.append(row)

    return SymbolDataset(rows, validation, warnings, method)


def candidate_reason(row: dict[str, Any], side: str, candidate_threshold: float) -> str:
    if row.get("turbo_action") == side:
        return "turbo_action"
    if side == "LONG" and (row.get("votes_long") or 0) >= 2:
        return "turbo_votes_long"
    if side == "SHORT" and (row.get("votes_short") or 0) >= 2:
        return "turbo_votes_short"
    if (row.get("turbo_score") or -999) >= candidate_threshold:
        return "turbo_score_threshold"
    if side == "LONG" and (row.get("long_mtf_agreement") or 0) > 0:
        return "rules_long_mtf_agreement"
    if side == "SHORT" and (row.get("short_mtf_agreement") or 0) > 0:
        return "rules_short_mtf_agreement"
    return "rules_momentum"


def save_npz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    string_cols = [col for col in df.columns if col not in numeric_cols]
    numeric = df[numeric_cols].to_numpy(dtype=np.float32) if numeric_cols else np.empty((len(df), 0), dtype=np.float32)
    payload: dict[str, Any] = {
        "numeric": numeric,
        "numeric_columns": np.asarray(numeric_cols),
        "string_columns": np.asarray(string_cols),
    }
    for col in string_cols:
        payload[f"str_{col}"] = df[col].fillna("").astype(str).to_numpy()
    np.savez_compressed(path, **payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Aegis Entry Quality Dataset v0.2.0",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Rows total: `{report['rows_total']}`",
        f"- Candidate generation: `{report['candidate_generation_method']}`",
        f"- Parquet written: `{report['outputs'].get('parquet_written')}`",
        f"- NPZ written: `{report['outputs'].get('npz_written')}`",
        "",
        "## Rows By Symbol",
        "",
    ]
    for symbol, count in report["rows_per_symbol"].items():
        side_counts = report["candidates_by_symbol_side"].get(symbol, {})
        labels = report["labels_by_symbol"].get(symbol, {})
        lines.append(
            f"- `{symbol}`: rows={count}, LONG={side_counts.get('LONG', 0)}, SHORT={side_counts.get('SHORT', 0)}, "
            f"good={labels.get('good', 0)}, bad={labels.get('bad', 0)}, tail={labels.get('tail', 0)}"
        )
    lines.extend(
        [
            "",
            "## Class Balance",
            "",
        ]
    )
    for klass, count in report["quality_class_balance"].items():
        lines.append(f"- `{klass}`: {count}")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendation", "", report["recommendation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    df: pd.DataFrame,
    validations: dict[str, Any],
    warnings: list[str],
    outputs: dict[str, Any],
    candidate_methods: Counter,
) -> dict[str, Any]:
    rows_per_symbol = {str(k): int(v) for k, v in df["symbol"].value_counts().sort_index().items()} if not df.empty else {}
    candidates_by_symbol_side: dict[str, dict[str, int]] = {}
    labels_by_symbol: dict[str, dict[str, int]] = {}
    if not df.empty:
        for symbol, group in df.groupby("symbol"):
            candidates_by_symbol_side[str(symbol)] = {str(k): int(v) for k, v in group["side"].value_counts().items()}
            labels_by_symbol[str(symbol)] = {
                "good": int(group["label_good_entry_v1"].sum()),
                "bad": int(group["label_bad_entry_v1"].sum()),
                "tail": int(group["label_tail_risk_v1"].sum()),
            }
    feature_missing = {}
    if not df.empty:
        for col in feature_columns():
            if col in df.columns:
                missing = int(df[col].isna().sum())
                if missing:
                    feature_missing[col] = missing
    rows_total = int(len(df))
    enough_rows = rows_total >= 5000 and len(rows_per_symbol) >= 5
    balanced = False if df.empty else df["label_bad_entry_v1"].mean() > 0.02 and df["label_tail_risk_v1"].sum() >= 25
    recommendation = (
        "Dataset is sufficient for Phase 2 prototype training; still validate per-symbol holdout before ENFORCE."
        if enough_rows and balanced
        else "Dataset is usable for exploration, but collect more candidates or tune sampling before production training."
    )
    return {
        "created_at": utc_now_iso(),
        "rows_total": rows_total,
        "rows_per_symbol": rows_per_symbol,
        "candidates_by_symbol_side": candidates_by_symbol_side,
        "labels_total": {
            "good": int(df["label_good_entry_v1"].sum()) if not df.empty else 0,
            "bad": int(df["label_bad_entry_v1"].sum()) if not df.empty else 0,
            "tail": int(df["label_tail_risk_v1"].sum()) if not df.empty else 0,
        },
        "labels_by_symbol": labels_by_symbol,
        "quality_class_balance": {str(k): int(v) for k, v in df["quality_class"].value_counts().items()} if not df.empty else {},
        "missing_features": feature_missing,
        "input_validation": validations,
        "leakage_checks": {
            "features_use_only_current_or_past_candles": True,
            "labels_use_future_candles_only": True,
            "max_horizon_bars_excluded_from_candidates": MAX_HORIZON_BARS,
            "mtf_resample_label_right_closed_right": True,
            "turbo_model_features_aligned_to_feature_frame_steps": True,
        },
        "warnings": warnings,
        "symbols_with_few_rows": [symbol for symbol, count in rows_per_symbol.items() if count < 100],
        "candidate_generation_method": dict(candidate_methods),
        "outputs": outputs,
        "recommendation": recommendation,
    }


def build_metadata(
    df: pd.DataFrame,
    symbols: list[str],
    timeframe: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    validations: dict[str, Any],
    candidate_methods: Counter,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    date_range_per_symbol = {}
    if not df.empty:
        for symbol, group in df.groupby("symbol"):
            date_range_per_symbol[str(symbol)] = {
                "first_candidate": str(group["timestamp"].min()),
                "last_candidate": str(group["timestamp"].max()),
            }
    return {
        "created_at": utc_now_iso(),
        "version": "entry_quality_dataset_v020",
        "symbols": symbols,
        "timeframe": timeframe,
        "requested_start": str(start) if start is not None else None,
        "requested_end": str(end) if end is not None else "now",
        "rows_total": int(len(df)),
        "rows_per_symbol": {str(k): int(v) for k, v in df["symbol"].value_counts().sort_index().items()} if not df.empty else {},
        "feature_columns": feature_columns(),
        "label_columns": label_columns(),
        "date_range_per_symbol": date_range_per_symbol,
        "input_validation": validations,
        "candidate_generation_method": dict(candidate_methods),
        "leakage_notes": [
            "Base 5m features are computed from the current and previous closed candles only.",
            "15m/1h features are resampled with right labels and forward-filled, using closed higher-timeframe candles only.",
            "Future labels are calculated after candidate selection and are not referenced by features.",
            "Last 96 5m bars per symbol are excluded from candidates to preserve the 8h horizon.",
        ],
        "horizons": {"max_horizon_bars": MAX_HORIZON_BARS, "metrics": HORIZONS},
        "label_definitions": {
            "label_good_entry_v1": "hit +8% ROE before -8% ROE, future_mae_roe > -12%, time_to_green <= 30m",
            "label_bad_entry_v1": "hit -15% ROE before +5% ROE, or future_mae_roe <= -20%, or time_to_green > 120m/no green with weak final outcome",
            "label_tail_risk_v1": "hit -25% ROE or -40% ROE before +8% ROE",
            "quality_class": "EXCELLENT, GOOD, SLOW_WIN, BAD_ENTRY_WIN, BAD_ENTRY_LOSS, TAIL_RISK, UNKNOWN",
        },
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Aegis Turbo v0.2 entry quality historical replay dataset.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", default="2025-01-01", help="UTC start date; defaults to 2025-01-01 for v0.2 Phase 1.")
    parser.add_argument("--end", default="now")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--also-npz", default="false")
    parser.add_argument("--max-rows-per-symbol", type=int)
    parser.add_argument("--candidate-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    symbols = [normalize_symbol(item) for item in str(args.symbols).split(",") if item.strip()]
    timeframe = str(args.timeframe)
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    output = Path(args.output)
    also_npz = parse_bool(args.also_npz)

    leverage_by_symbol, default_leverage, leverage_warnings = load_leverage_by_symbol()
    warnings: list[str] = list(leverage_warnings)
    validations: dict[str, Any] = {}
    candidate_methods: Counter = Counter()
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        leverage = leverage_by_symbol.get(symbol, default_leverage)
        result = build_symbol_dataset(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            leverage=leverage,
            candidate_threshold=float(args.candidate_threshold),
            max_rows_per_symbol=args.max_rows_per_symbol,
        )
        validations[symbol] = result.validation
        candidate_methods[result.candidate_method] += 1
        warnings.extend(f"{symbol}: {warning}" for warning in result.warnings)
        rows.extend(result.rows)
        print(
            f"{symbol}: candles={result.validation.get('rows', 0)} candidates={len(result.rows)} "
            f"method={result.candidate_method}"
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        ordered = [
            "symbol",
            "timestamp",
            "side",
            "timeframe",
            "entry_price",
            "leverage",
            "candidate_reason",
        ]
        ordered.extend(feature_columns())
        ordered.extend(
            [
                "long_score_7d",
                "long_score_14d",
                "long_score_30d",
                "short_score_7d",
                "short_score_14d",
                "short_score_30d",
                "votes_long",
                "votes_short",
                "votes_neutral",
                "turbo_action",
                "turbo_score",
                "score_gap",
                "candidate_generation_method",
            ]
        )
        ordered.extend(label_columns())
        ordered = [col for col in ordered if col in df.columns]
        df = df[ordered].sort_values(["symbol", "timestamp", "side"]).reset_index(drop=True)

    outputs: dict[str, Any] = {
        "requested_output": str(output),
        "parquet_written": False,
        "npz_written": False,
        "runtime_seconds": round(time.time() - started, 3),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output, index=False)
            outputs["parquet_written"] = True
            outputs["parquet_path"] = str(output)
        except Exception as exc:
            warnings.append(f"parquet_write_failed_falling_back_to_npz: {exc!r}")
    else:
        warnings.append(f"parquet_not_requested_output_suffix={output.suffix}")

    npz_path = output.with_suffix(".npz")
    if also_npz or not outputs["parquet_written"]:
        save_npz(df, npz_path)
        outputs["npz_written"] = True
        outputs["npz_path"] = str(npz_path)

    validation_path = INPUT_VALIDATION_DIR / f"dataset_input_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(validation_path, {"created_at": utc_now_iso(), "validations": validations})
    outputs["input_validation_path"] = str(validation_path)
    outputs["runtime_seconds"] = round(time.time() - started, 3)

    meta_path = output.with_name(f"{output.stem}_meta.json")
    outputs["metadata_path"] = str(meta_path)
    meta = build_metadata(df, symbols, timeframe, start, end, validations, candidate_methods, outputs)
    write_json(meta_path, meta)

    report = build_report(df, validations, warnings, outputs, candidate_methods)
    write_json(REPORT_JSON, report)
    write_markdown_report(REPORT_MD, report)
    print(
        json.dumps(
            {
                "rows_total": int(len(df)),
                "labels_total": report["labels_total"],
                "quality_class_balance": report["quality_class_balance"],
                "outputs": outputs,
                "runtime_seconds": outputs["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
