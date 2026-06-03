#!/usr/bin/env python3
"""Research-only trainer for LONG alpha candidate families.

The script trains candidate LONG models into aegis_alpha/models/research only.
It never writes active manifests, active model directories, YAML, PM2 state, or
live inference code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    import joblib
except Exception:  # pragma: no cover - joblib is expected with sklearn, but keep tests robust.
    joblib = None

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - optional dependency.
    spearmanr = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.profile_long_alpha_families_a import (  # noqa: E402
    CLASSIC_TARGETS,
    MICRO_TARGETS,
    MOMENTUM_SET,
    SYMBOLS,
    TargetResult,
    add_features,
    compute_long_target_arrays,
    db_symbol,
    json_safe,
    load_candles,
    mean,
    quantile,
    top_fraction_mask,
)

SCHEMA_VERSION = "aegis_long_c_research_train_v1"
SIDE = "LONG"
DEFAULT_DB_PATH = Path("/home/jasan/Develop/trading_system/data/binance_candles.db")
DEFAULT_MODEL_DIR = Path("/home/jasan/Develop/trading_system/aegis_alpha/models/research/long_alpha")
REPO_ROOT = Path(__file__).resolve().parents[2]

CORE_FAMILIES = ("slow_trend_long", "pullback_continuation_long")
MOMENTUM_FAMILIES = (
    "momentum_burst_long",
    "momentum_ride_long",
    "breakout_momentum_long",
    "micro_roe_momentum_long",
)
ALL_TRAIN_FAMILIES = CORE_FAMILIES + MOMENTUM_FAMILIES

TARGET_MAP = {
    "long_hit3_before_minus2": (0.03, 0.02),
    "long_hit5_before_minus3": (0.05, 0.03),
    "long_hit6_before_minus4": (0.06, 0.04),
    "long_hit8_before_minus5": (0.08, 0.05),
    "long_hit10_before_minus8": (0.10, 0.08),
    "long_roe6_before_minus4": (0.06 / 20.0, 0.04 / 20.0),
    "long_roe8_before_minus5": (0.08 / 20.0, 0.05 / 20.0),
    "long_roe10_before_minus6": (0.10 / 20.0, 0.06 / 20.0),
    "long_roe12_before_minus8": (0.12 / 20.0, 0.08 / 20.0),
}

BASE_TARGET_BY_SYMBOL = {
    "BTCUSDT": ("long_hit3_before_minus2", 24),
    "ETHUSDT": ("long_hit3_before_minus2", 24),
    "SOLUSDT": ("long_hit3_before_minus2", 24),
    "BNBUSDT": ("long_hit3_before_minus2", 24),
    "XRPUSDT": ("long_hit3_before_minus2", 24),
    "DOGEUSDT": ("long_hit6_before_minus4", 24),
    "ADAUSDT": ("long_hit3_before_minus2", 24),
    "AVAXUSDT": ("long_hit3_before_minus2", 24),
    "LINKUSDT": ("long_hit3_before_minus2", 24),
    "SUIUSDT": ("long_hit5_before_minus3", 24),
    "LTCUSDT": ("long_hit8_before_minus5", 24),
}

MOMENTUM_CONFIGS = (
    ("SUIUSDT", "breakout_momentum_long", "long_hit3_before_minus2", 12),
    ("SUIUSDT", "momentum_ride_long", "long_hit3_before_minus2", 12),
    ("LINKUSDT", "breakout_momentum_long", "long_hit3_before_minus2", 12),
    ("AVAXUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 6),
    ("AVAXUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 12),
    ("AVAXUSDT", "breakout_momentum_long", "long_hit3_before_minus2", 12),
    ("ETHUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 6),
    ("ETHUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 12),
    ("ETHUSDT", "breakout_momentum_long", "long_hit3_before_minus2", 12),
    ("DOGEUSDT", "breakout_momentum_long", "long_hit3_before_minus2", 12),
    ("DOGEUSDT", "momentum_burst_long", "long_hit3_before_minus2", 12),
    ("BTCUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 12),
    ("BNBUSDT", "micro_roe_momentum_long", "long_roe12_before_minus8", 12),
)

FAMILY_FEATURES = {
    "slow_trend_long": (
        "return_30m", "return_60m", "return_120m", "trend_efficiency_24",
        "trend_efficiency_64", "distance_ema25", "distance_ema99", "ema25_slope",
        "ema99_slope", "close_location_24", "realized_vol_24", "realized_vol_64",
        "atr_ratio_14", "btc_eth_long_agreement", "btc_eth_short_contradiction",
    ),
    "pullback_continuation_long": (
        "pullback_depth_6", "pullback_depth_12", "pullback_recovery_speed",
        "distance_ema25", "distance_ema99", "lower_wick_ratio", "close_location_12",
        "close_location_24", "volume_ratio_12", "trend_efficiency_24",
        "local_trend_up_score", "local_chop_score", "btc_eth_long_agreement",
    ),
    "breakout_momentum_long": (
        "breakout_strength_12", "breakout_strength_24", "range_expansion_12",
        "close_location_12", "close_location_24", "volume_ratio_12", "realized_vol_12",
        "upper_wick_ratio", "fake_breakout_risk", "distance_to_recent_high",
        "btc_eth_long_agreement",
    ),
    "momentum_burst_long": (
        "return_15m", "return_30m", "acceleration_15_30", "range_expansion_12",
        "volume_ratio_12", "close_location_12", "realized_vol_12",
        "trend_efficiency_12", "upper_wick_ratio", "overextension_risk",
    ),
    "momentum_ride_long": (
        "return_30m", "return_60m", "trend_efficiency_24", "trend_efficiency_64",
        "pullback_depth_6", "close_location_24", "ema25_slope", "ema99_slope",
        "realized_vol_24", "btc_eth_long_agreement",
    ),
    "micro_roe_momentum_long": (
        "return_15m", "return_30m", "range_expansion_12", "atr_ratio_14",
        "close_location_12", "volume_ratio_12", "realized_vol_12",
        "overextension_risk", "upper_wick_ratio", "btc_eth_long_agreement",
    ),
}

PROXY_FEATURES = {
    "return_15m": "return_3",
    "return_30m": "return_6",
    "return_60m": "return_12",
    "return_120m": "return_24",
    "btc_eth_long_agreement": "btc_eth_agreement",
    "btc_eth_short_contradiction": "btc_eth_contradiction",
    "overextension_risk": "overextension",
    "fake_breakout_risk": "fake_breakout",
    "breakout_strength_12": "breakout_12",
    "breakout_strength_24": "breakout_12",
    "distance_to_recent_high": "distance_recent_high_24",
    "pullback_depth_6": "pullback_depth",
    "pullback_depth_12": "pullback_depth",
    "pullback_recovery_speed": "return_3",
    "local_trend_up_score": "ema_stack_bull",
    "local_chop_score": "chop_score",
    "atr_ratio_14": "range_expansion_12",
    "trend_efficiency_64": "trend_efficiency_24",
    "realized_vol_64": "realized_vol_24",
}

CSV_COLUMNS = (
    "symbol", "side", "alpha_family", "target_name", "horizon_candles", "feature_mode",
    "model_status", "model_reason", "train_samples", "validation_samples", "test_samples",
    "feature_count", "missing_features", "proxy_features_used", "rare_target_warning",
    "hit_positive_rate", "hit_auc", "hit_average_precision", "hit_top_decile_hit_rate",
    "hit_top_decile_hit_lift", "quality_mae", "quality_rmse", "quality_r2",
    "quality_corr", "quality_spearman_corr", "top_decile_actual_quality",
    "quality_lift", "danger_auc", "danger_average_precision", "danger_filter_usefulness",
    "exhaustion_auc", "exhaustion_average_precision", "exhaustion_filter_usefulness",
    "selected_count", "selected_fraction", "baseline_hit_rate", "selected_hit_rate",
    "hit_lift", "baseline_quality", "selected_quality", "net_quality_lift_after_costs",
    "p90_mae_delta", "stop_rate_delta", "time_to_target_avg", "time_to_target_p50",
    "ambiguous_rate", "fast_hit_rate", "late_entry_rate", "exhaustion_filtered_quality_lift",
    "saved_model_count", "model_bundle_path",
)


@dataclass(frozen=True)
class TrainConfig:
    symbol: str
    alpha_family: str
    target_name: str
    horizon_candles: int


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def assert_research_only_path(path: str | Path) -> None:
    text = str(path)
    bad_parts = (
        "/active/",
        "active_manifest.json",
        "phase_o_short_manifest.json",
    )
    if any(part in text for part in bad_parts) or ("/models/turbo/" in text and "/active" in text):
        raise ValueError(f"refusing non research-only path: {text}")


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def feature_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]




def load_candles_research(db_path: Path, symbol: str, lookback_days: int, retries: int = 3) -> pd.DataFrame:
    rows = max(5000, int(lookback_days * 288) + 300)
    query = """
        SELECT timestamp, open, high, low, close, volume, buy_volume
        FROM ohlcv_data
        WHERE symbol = ? AND timeframe = '5m'
        ORDER BY timestamp DESC
        LIMIT ?
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
            try:
                df = pd.read_sql_query(query, con, params=(db_symbol(symbol), rows))
            finally:
                con.close()
            if df.empty:
                return df
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            for col in ["open", "high", "low", "close", "volume", "buy_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
        except Exception as exc:  # sqlite may be briefly locked by candle refresh jobs.
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to load candles for {symbol}: {last_error}")

def add_long_c_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    for alias, source in (
        ("return_15m", "return_3"),
        ("return_30m", "return_6"),
        ("return_60m", "return_12"),
        ("return_120m", "return_24"),
    ):
        if alias not in out and source in out:
            out[alias] = out[source]
    out["acceleration_15_30"] = out.get("return_15m", 0.0) - out.get("return_30m", 0.0)
    for n in (6, 12):
        recent_high = high.rolling(n, min_periods=max(3, n // 2)).max()
        out[f"pullback_depth_{n}"] = (recent_high.shift(1) - close) / close.replace(0, np.nan)
    out["pullback_recovery_speed"] = out.get("return_3", 0.0)
    out["breakout_strength_12"] = out.get("breakout_12", 0.0)
    high24 = high.rolling(24, min_periods=12).max().shift(1)
    out["breakout_strength_24"] = close / high24.replace(0, np.nan) - 1.0
    out["distance_recent_high_24"] = close / high24.replace(0, np.nan) - 1.0
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()
    out["atr_ratio_14"] = atr14 / close.replace(0, np.nan)
    out["trend_efficiency_64"] = (close - close.shift(64)).abs() / close.diff().abs().rolling(64, min_periods=24).sum().replace(0, np.nan)
    out["realized_vol_64"] = close.pct_change().rolling(64, min_periods=24).std()
    out["btc_eth_long_agreement"] = out.get("btc_eth_agreement", 0.0)
    out["btc_eth_short_contradiction"] = out.get("btc_eth_contradiction", 0.0)
    out["overextension_risk"] = out.get("overextension", 0.0)
    out["fake_breakout_risk"] = out.get("fake_breakout", 0.0)
    out["local_trend_up_score"] = out.get("ema_stack_bull", 0.0)
    chop_base = 1.0 - out.get("trend_efficiency_24", pd.Series(0.5, index=out.index)).clip(0, 1)
    out["local_chop_score"] = chop_base
    return out.replace([np.inf, -np.inf], np.nan)


def select_long_family_features(dataset: pd.DataFrame, family: str, feature_mode: str = "selected_family") -> tuple[list[str], list[str], list[str]]:
    if feature_mode == "combined_v3_all":
        names = [c for c in dataset.columns if c not in {"timestamp"} and pd.api.types.is_numeric_dtype(dataset[c])]
        names = [c for c in names if c not in {"open", "high", "low", "close", "volume", "buy_volume"}]
        return names, [], []
    wanted = FAMILY_FEATURES.get(family, FAMILY_FEATURES["slow_trend_long"])
    selected: list[str] = []
    missing: list[str] = []
    proxies: list[str] = []
    for name in wanted:
        if name in dataset:
            selected.append(name)
            continue
        proxy = PROXY_FEATURES.get(name)
        if proxy and proxy in dataset:
            selected.append(proxy)
            proxies.append(f"{name}->{proxy}")
        else:
            missing.append(name)
    deduped = list(dict.fromkeys(selected))
    return deduped, missing, proxies


def temporal_split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def one_class(y: np.ndarray) -> bool:
    finite = y[np.isfinite(y)]
    return len(np.unique(finite)) < 2


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    try:
        if one_class(y):
            return None
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    try:
        if one_class(y):
            return None
        return float(average_precision_score(y, score))
    except Exception:
        return None


def safe_brier(y: np.ndarray, score: np.ndarray) -> float | None:
    try:
        return float(brier_score_loss(y, np.clip(score, 0, 1)))
    except Exception:
        return None


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if spearmanr is None:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    try:
        corr = spearmanr(a[mask], b[mask]).correlation
        return finite_float(corr)
    except Exception:
        return None


def build_probability_buckets(
    score: np.ndarray,
    hit: np.ndarray,
    quality: np.ndarray,
    danger: np.ndarray,
    exhaustion: np.ndarray,
    *,
    symbol: str,
    family: str,
    target_name: str,
    horizon: int,
    bucket_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = np.isfinite(score)
    if valid.sum() == 0:
        return rows
    quantiles = np.quantile(score[valid], np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == 9:
            mask = valid & (score >= lo) & (score <= hi)
        else:
            mask = valid & (score >= lo) & (score < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "symbol": symbol,
            "alpha_family": family,
            "target_name": target_name,
            "horizon_candles": horizon,
            "bucket_source": bucket_source,
            "bucket": i + 1,
            "count": count,
            "avg_pred": mean(score[mask]),
            "actual_hit_rate": mean(hit[mask]),
            "actual_trade_quality": mean(quality[mask]),
            "mae_danger_rate": mean(danger[mask]),
            "exhaustion_rate": mean(exhaustion[mask]),
        })
    return rows


def compute_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray, frame: pd.DataFrame, target_name: str, horizon: int) -> dict[str, np.ndarray]:
    target_move, stop_move = TARGET_MAP[target_name]
    target = compute_long_target_arrays(close, high, low, target_move, stop_move, horizon)
    valid_quality = np.isfinite(target.quality)
    mae_q75 = np.nanquantile(target.mae[valid_quality], 0.75) if valid_quality.any() else stop_move
    mae_danger = ((target.stop == 1) | (target.mae >= mae_q75)).astype(np.int8)
    future_upper = frame["upper_wick_ratio"].shift(-1).rolling(max(2, min(4, horizon)), min_periods=1).max().to_numpy(dtype=float)
    fake = frame.get("fake_breakout", pd.Series(0, index=frame.index)).to_numpy(dtype=float)
    exhaustion_feature = frame.get("exhaustion", pd.Series(0.5, index=frame.index)).to_numpy(dtype=float)
    low_mfe_high_mae = (target.mfe < target_move * 0.35) & (target.mae > stop_move * 0.55)
    late_entry_exhaustion = (
        (target.stop == 1)
        | (target.ambiguous == 1)
        | low_mfe_high_mae
        | (future_upper > 0.45)
        | (fake > 0)
        | (exhaustion_feature > 0.78)
    ).astype(np.int8)
    fast_cutoff = float(horizon) * 0.65
    fast_hit = ((target.hit == 1) & (target.time_to_target > 0) & (target.time_to_target <= fast_cutoff)).astype(np.int8)
    return {
        "hit": target.hit.astype(np.int8),
        "stop": target.stop.astype(np.int8),
        "quality": target.quality,
        "mfe": target.mfe,
        "mae": target.mae,
        "time_to_target": target.time_to_target,
        "time_to_stop": target.time_to_stop,
        "ambiguous": target.ambiguous.astype(np.int8),
        "mae_danger": mae_danger,
        "late_entry_exhaustion": late_entry_exhaustion,
        "fast_hit": fast_hit,
    }


def classifier(random_state: int, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=random_state,
    )


def regressor(random_state: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=random_state,
    )


def model_predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), np.nan)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x), dtype=float)


def train_or_skip_classifier(name: str, x_train: np.ndarray, y_train: np.ndarray, random_state: int, max_iter: int) -> tuple[Any | None, str | None]:
    if one_class(y_train):
        return None, f"{name}:INSUFFICIENT_CLASS_DIVERSITY"
    return classifier(random_state, max_iter).fit(x_train, y_train), None


def selection_mask(
    hit_pred: np.ndarray,
    quality_pred: np.ndarray,
    danger_pred: np.ndarray,
    exhaustion_pred: np.ndarray,
    valid: np.ndarray,
    family: str,
) -> np.ndarray:
    score = np.nan_to_num(quality_pred, nan=np.nanmedian(quality_pred[np.isfinite(quality_pred)]) if np.isfinite(quality_pred).any() else 0.0)
    if np.isfinite(hit_pred).any():
        score = score + np.nan_to_num(hit_pred, nan=0.0) * 0.5
    if np.isfinite(danger_pred).any():
        score = score - np.nan_to_num(danger_pred, nan=0.5) * 0.25
    if np.isfinite(exhaustion_pred).any():
        score = score - np.nan_to_num(exhaustion_pred, nan=0.5) * 0.25
    fraction = 0.10 if family in MOMENTUM_SET else 0.12
    return top_fraction_mask(score, valid, fraction)


def classify_long_model_candidate(row: dict[str, Any]) -> str:
    if row.get("model_status") == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    if int(row.get("test_samples") or 0) < 100 or int(row.get("feature_count") or 0) == 0:
        return "INSUFFICIENT_DATA"
    if row.get("hit_training_status") == "INSUFFICIENT_CLASS_DIVERSITY":
        return "INSUFFICIENT_DATA"
    selected_fraction = float(row.get("selected_fraction") or 0.0)
    net = float(row.get("net_quality_lift_after_costs") or 0.0)
    hit_lift = float(row.get("hit_lift") or 0.0)
    quality_lift = float(row.get("quality_lift") or 0.0)
    p90_delta = float(row.get("p90_mae_delta") or 0.0)
    stop_delta = float(row.get("stop_rate_delta") or 0.0)
    auc = row.get("hit_auc")
    top_lift = float(row.get("hit_top_decile_hit_lift") or 0.0)
    corr = row.get("quality_corr")
    time_to_target = row.get("time_to_target_avg")
    horizon = float(row.get("horizon_candles") or 1)
    family = str(row.get("alpha_family") or "")
    fast_ok = True
    if family in MOMENTUM_SET:
        fast_ok = time_to_target is not None and float(time_to_target) <= horizon * 0.65
    predictive = (auc is not None and float(auc) >= 0.54) or top_lift > 0.02
    hit_ok = hit_lift > 0 or top_lift > 0
    quality_ok = (corr is not None and float(corr) > 0) or quality_lift > 0
    if predictive and hit_ok and quality_ok and net > 0 and p90_delta <= 0.10 and stop_delta <= 0.05 and 0.05 <= selected_fraction <= 0.25 and fast_ok:
        return "LONG_MODEL_PROMISING"
    if net < 0 or hit_lift < 0 or p90_delta > 0.20 or stop_delta > 0.08:
        return "LONG_MODEL_FAILED"
    if hit_lift > 0 or quality_lift > 0 or top_lift > 0 or net > 0:
        return "LONG_MODEL_MIXED"
    return "LONG_MODEL_WEAK"


def status_reason(row: dict[str, Any]) -> str:
    status = row.get("model_status")
    if status == "LONG_MODEL_PROMISING":
        return "predictive hit/quality with positive net lift and controlled risk"
    if status == "LONG_MODEL_FAILED":
        return "net quality, hit lift, MAE, or stop risk failed out-of-sample"
    if status == "INSUFFICIENT_DATA":
        return "insufficient samples, features, or class diversity"
    if status == "LONG_MODEL_MIXED":
        return "some out-of-sample lift, but not enough confirmation"
    return "weak or unstable out-of-sample profile"


def _required_valid(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    valid = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy().copy()
    valid[:220] = False
    if horizon > 0:
        valid[-horizon:] = False
    return valid


def _metrics_for_classifier(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pred = (score >= 0.5).astype(int)
    return {
        "accuracy": finite_float(accuracy_score(y, pred)),
        "precision": finite_float(precision_score(y, pred, zero_division=0)),
        "recall": finite_float(recall_score(y, pred, zero_division=0)),
        "f1": finite_float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_auc(y, score),
        "average_precision": safe_ap(y, score),
        "brier_score": safe_brier(y, score),
    }


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "mae": finite_float(mean_absolute_error(y, pred)),
        "rmse": finite_float(math.sqrt(mean_squared_error(y, pred))),
        "r2": finite_float(r2_score(y, pred)),
        "corr": safe_corr(y, pred),
        "spearman_corr": safe_spearman(y, pred),
    }


def save_model_bundle(
    bundle_dir: Path,
    models: dict[str, Any],
    metadata: dict[str, Any],
    *,
    save_models: bool,
) -> tuple[list[str], int]:
    assert_research_only_path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_files: list[str] = []
    if save_models and joblib is not None:
        for name, model in models.items():
            if model is None:
                continue
            path = bundle_dir / f"{name}.joblib"
            assert_research_only_path(path)
            joblib.dump(model, path)
            model_files.append(str(path))
    metadata = dict(metadata)
    metadata.update({
        "research_only": True,
        "model_files": model_files,
        "saved_model_count": len(model_files),
    })
    meta_path = bundle_dir / "metadata.json"
    assert_research_only_path(meta_path)
    meta_path.write_text(json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n")
    return model_files, len(model_files)


def train_config(
    config: TrainConfig,
    db_path: Path,
    model_dir: Path,
    *,
    lookback_days: int,
    feature_mode: str,
    min_train_samples: int,
    min_test_samples: int,
    max_iter: int,
    save_models: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = load_candles_research(db_path, config.symbol, lookback_days)
    if df.empty:
        return ({
            "symbol": config.symbol,
            "side": SIDE,
            "alpha_family": config.alpha_family,
            "target_name": config.target_name,
            "horizon_candles": config.horizon_candles,
            "model_status": "INSUFFICIENT_DATA",
            "model_reason": "no local candles",
            "train_samples": 0,
            "validation_samples": 0,
            "test_samples": 0,
            "feature_count": 0,
        }, [])
    btc = load_candles_research(db_path, "BTCUSDT", lookback_days) if config.symbol != "BTCUSDT" else df
    eth = load_candles_research(db_path, "ETHUSDT", lookback_days) if config.symbol != "ETHUSDT" else df
    frame = add_long_c_features(add_features(df, btc, eth)).reset_index(drop=True)
    labels = compute_labels(
        frame["close"].to_numpy(dtype=float),
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        frame,
        config.target_name,
        config.horizon_candles,
    )
    feature_names, missing_features, proxies = select_long_family_features(frame, config.alpha_family, feature_mode)
    valid = _required_valid(frame, config.horizon_candles)
    for name in feature_names:
        valid &= np.isfinite(frame[name].to_numpy(dtype=float))
    valid &= np.isfinite(labels["quality"])
    idx = np.flatnonzero(valid)
    if len(idx) == 0 or len(feature_names) == 0:
        row = {
            "symbol": config.symbol,
            "side": SIDE,
            "alpha_family": config.alpha_family,
            "target_name": config.target_name,
            "horizon_candles": config.horizon_candles,
            "feature_mode": feature_mode,
            "model_status": "INSUFFICIENT_DATA",
            "model_reason": "feature_count=0 or no valid rows",
            "train_samples": 0,
            "validation_samples": 0,
            "test_samples": 0,
            "feature_count": len(feature_names),
            "missing_features": ";".join(missing_features),
            "proxy_features_used": ";".join(proxies),
        }
        return row, []
    train_i, val_i, test_i = temporal_split_indices(len(idx))
    train_idx, val_idx, test_idx = idx[train_i], idx[val_i], idx[test_i]
    if len(train_idx) < min_train_samples or len(test_idx) < min_test_samples:
        row = {
            "symbol": config.symbol,
            "side": SIDE,
            "alpha_family": config.alpha_family,
            "target_name": config.target_name,
            "horizon_candles": config.horizon_candles,
            "feature_mode": feature_mode,
            "model_status": "INSUFFICIENT_DATA",
            "model_reason": "not enough train/test samples",
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "test_samples": len(test_idx),
            "feature_count": len(feature_names),
            "missing_features": ";".join(missing_features),
            "proxy_features_used": ";".join(proxies),
        }
        return row, []
    x = frame[feature_names].to_numpy(dtype=float)
    x_train, x_test = x[train_idx], x[test_idx]
    y_hit_train = labels["hit"][train_idx]
    y_hit_test = labels["hit"][test_idx]
    y_quality_train = labels["quality"][train_idx]
    y_quality_test = labels["quality"][test_idx]
    y_danger_train = labels["mae_danger"][train_idx]
    y_danger_test = labels["mae_danger"][test_idx]
    y_exh_train = labels["late_entry_exhaustion"][train_idx]
    y_exh_test = labels["late_entry_exhaustion"][test_idx]
    seed_base = int(hashlib.sha256(f"{config.symbol}:{config.alpha_family}:{config.target_name}:{config.horizon_candles}".encode()).hexdigest()[:8], 16)
    training_notes: list[str] = []
    hit_model, note = train_or_skip_classifier("hit", x_train, y_hit_train, seed_base % 2_000_000_000, max_iter)
    if note:
        training_notes.append(note)
    danger_model, note = train_or_skip_classifier("danger", x_train, y_danger_train, (seed_base + 1) % 2_000_000_000, max_iter)
    if note:
        training_notes.append(note)
    exhaustion_model, note = train_or_skip_classifier("exhaustion", x_train, y_exh_train, (seed_base + 2) % 2_000_000_000, max_iter)
    if note:
        training_notes.append(note)
    quality_model = regressor((seed_base + 3) % 2_000_000_000, max_iter).fit(x_train, y_quality_train)
    hit_pred = model_predict_proba(hit_model, x_test)
    danger_pred = model_predict_proba(danger_model, x_test)
    exhaustion_pred = model_predict_proba(exhaustion_model, x_test)
    quality_pred = quality_model.predict(x_test)
    test_valid = np.ones(len(test_idx), dtype=bool)
    selected = selection_mask(hit_pred, quality_pred, danger_pred, exhaustion_pred, test_valid, config.alpha_family)
    baseline_hit = mean(y_hit_test) or 0.0
    selected_hit = mean(y_hit_test[selected]) if selected.any() else None
    baseline_quality = mean(y_quality_test) or 0.0
    selected_quality = mean(y_quality_test[selected]) if selected.any() else None
    baseline_stop = mean(labels["stop"][test_idx]) or 0.0
    selected_stop = mean(labels["stop"][test_idx][selected]) if selected.any() else None
    baseline_p90_mae = quantile(labels["mae"][test_idx], 0.90) or 0.0
    selected_p90_mae = quantile(labels["mae"][test_idx][selected], 0.90) if selected.any() else None
    ttt = labels["time_to_target"][test_idx][selected]
    ttt = ttt[ttt > 0]
    cls_metrics = _metrics_for_classifier(y_hit_test, np.nan_to_num(hit_pred, nan=baseline_hit)) if hit_model is not None else {}
    danger_metrics = _metrics_for_classifier(y_danger_test, np.nan_to_num(danger_pred, nan=mean(y_danger_test) or 0.0)) if danger_model is not None else {}
    exhaustion_metrics = _metrics_for_classifier(y_exh_test, np.nan_to_num(exhaustion_pred, nan=mean(y_exh_test) or 0.0)) if exhaustion_model is not None else {}
    reg_metrics = _regression_metrics(y_quality_test, quality_pred)
    top_decile = top_fraction_mask(np.nan_to_num(hit_pred, nan=-1), test_valid, 0.10)
    top_q_decile = top_fraction_mask(quality_pred, test_valid, 0.10)
    hit_top_rate = mean(y_hit_test[top_decile]) if top_decile.any() else None
    top_quality = mean(y_quality_test[top_q_decile]) if top_q_decile.any() else None
    high_exh = top_fraction_mask(np.nan_to_num(exhaustion_pred, nan=-1), test_valid, 0.20) if exhaustion_model is not None else np.zeros(len(test_idx), bool)
    low_exh = ~high_exh
    exhaustion_filter_lift = None
    if high_exh.any() and low_exh.any():
        exhaustion_filter_lift = (mean(y_quality_test[low_exh]) or 0.0) - (mean(y_quality_test[high_exh]) or 0.0)
    high_danger = top_fraction_mask(np.nan_to_num(danger_pred, nan=-1), test_valid, 0.20) if danger_model is not None else np.zeros(len(test_idx), bool)
    danger_filter = None
    if high_danger.any():
        danger_filter = (mean(labels["stop"][test_idx][high_danger]) or 0.0) - baseline_stop
    hit_positive_rate = mean(labels["hit"][idx]) or 0.0
    rare = hit_positive_rate < 0.01
    row = {
        "symbol": config.symbol,
        "side": SIDE,
        "alpha_family": config.alpha_family,
        "target_name": config.target_name,
        "horizon_candles": config.horizon_candles,
        "feature_mode": feature_mode,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "feature_count": len(feature_names),
        "missing_features": ";".join(missing_features),
        "proxy_features_used": ";".join(proxies),
        "feature_schema_hash": feature_hash(feature_names),
        "rare_target_warning": rare,
        "hit_positive_rate": hit_positive_rate,
        "hit_training_status": "INSUFFICIENT_CLASS_DIVERSITY" if hit_model is None else "TRAINED",
        "hit_auc": cls_metrics.get("roc_auc"),
        "hit_average_precision": cls_metrics.get("average_precision"),
        "hit_brier_score": cls_metrics.get("brier_score"),
        "hit_accuracy": cls_metrics.get("accuracy"),
        "hit_precision": cls_metrics.get("precision"),
        "hit_recall": cls_metrics.get("recall"),
        "hit_f1": cls_metrics.get("f1"),
        "hit_top_decile_hit_rate": hit_top_rate,
        "hit_top_decile_hit_lift": (hit_top_rate - baseline_hit) if hit_top_rate is not None else None,
        "quality_mae": reg_metrics.get("mae"),
        "quality_rmse": reg_metrics.get("rmse"),
        "quality_r2": reg_metrics.get("r2"),
        "quality_corr": reg_metrics.get("corr"),
        "quality_spearman_corr": reg_metrics.get("spearman_corr"),
        "top_decile_actual_quality": top_quality,
        "quality_lift": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "danger_auc": danger_metrics.get("roc_auc"),
        "danger_average_precision": danger_metrics.get("average_precision"),
        "danger_brier_score": danger_metrics.get("brier_score"),
        "danger_filter_usefulness": danger_filter,
        "exhaustion_auc": exhaustion_metrics.get("roc_auc"),
        "exhaustion_average_precision": exhaustion_metrics.get("average_precision"),
        "exhaustion_brier_score": exhaustion_metrics.get("brier_score"),
        "exhaustion_filter_usefulness": exhaustion_filter_lift,
        "selected_count": int(selected.sum()),
        "selected_fraction": float(selected.mean()) if len(selected) else 0.0,
        "baseline_hit_rate": baseline_hit,
        "selected_hit_rate": selected_hit,
        "hit_lift": (selected_hit - baseline_hit) if selected_hit is not None else None,
        "baseline_quality": baseline_quality,
        "selected_quality": selected_quality,
        "net_quality_lift_after_costs": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "p90_mae_delta": ((selected_p90_mae - baseline_p90_mae) / max(baseline_p90_mae, 1e-12)) if selected_p90_mae is not None else None,
        "stop_rate_delta": (selected_stop - baseline_stop) if selected_stop is not None else None,
        "time_to_target_avg": mean(ttt),
        "time_to_target_p50": quantile(ttt, 0.50),
        "ambiguous_rate": mean(labels["ambiguous"][test_idx][selected]) if selected.any() else None,
        "fast_hit_rate": mean(labels["fast_hit"][test_idx][selected]) if selected.any() else None,
        "late_entry_rate": mean(y_exh_test[selected]) if selected.any() else None,
        "exhaustion_filtered_quality_lift": exhaustion_filter_lift,
        "model_notes": ";".join(training_notes),
    }
    row["model_status"] = classify_long_model_candidate(row)
    row["model_reason"] = status_reason(row)
    model_bundle_path = model_dir / config.symbol / config.alpha_family / f"{config.target_name}_h{config.horizon_candles}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "symbol": config.symbol,
        "side": SIDE,
        "alpha_family": config.alpha_family,
        "target_name": config.target_name,
        "horizon_candles": config.horizon_candles,
        "feature_mode": feature_mode,
        "feature_names": feature_names,
        "feature_schema_hash": row["feature_schema_hash"],
        "missing_features": missing_features,
        "proxy_features_used": proxies,
        "status": row["model_status"],
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
    }
    models = {
        "long_hit_classifier": hit_model,
        "long_quality_regressor": quality_model,
        "long_mae_danger_classifier": danger_model,
        "long_exhaustion_classifier": exhaustion_model,
    }
    files, saved_count = save_model_bundle(model_bundle_path, models, metadata, save_models=save_models)
    row["model_files"] = ";".join(files)
    row["saved_model_count"] = saved_count
    row["model_bundle_path"] = str(model_bundle_path)
    bucket_rows: list[dict[str, Any]] = []
    bucket_rows.extend(build_probability_buckets(np.nan_to_num(hit_pred, nan=baseline_hit), y_hit_test, y_quality_test, y_danger_test, y_exh_test, symbol=config.symbol, family=config.alpha_family, target_name=config.target_name, horizon=config.horizon_candles, bucket_source="hit_probability"))
    bucket_rows.extend(build_probability_buckets(quality_pred, y_hit_test, y_quality_test, y_danger_test, y_exh_test, symbol=config.symbol, family=config.alpha_family, target_name=config.target_name, horizon=config.horizon_candles, bucket_source="quality_prediction"))
    return row, bucket_rows


def configs_for_args(args: argparse.Namespace) -> list[TrainConfig]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if any(s not in SYMBOLS for s in symbols):
        bad = [s for s in symbols if s not in SYMBOLS]
        raise SystemExit(f"Unsupported LONG-C symbols: {','.join(bad)}")
    families_filter = None
    if args.families:
        families_filter = {f.strip() for f in args.families.split(",") if f.strip()}
    configs: list[TrainConfig] = []
    if args.family_group in {"core", "all"}:
        for symbol in symbols:
            target, horizon = BASE_TARGET_BY_SYMBOL[symbol]
            for family in CORE_FAMILIES:
                if families_filter and family not in families_filter:
                    continue
                configs.append(TrainConfig(symbol, family, target, horizon))
    if args.family_group in {"momentum", "all"}:
        for symbol, family, target, horizon in MOMENTUM_CONFIGS:
            if symbol not in symbols:
                continue
            if families_filter and family not in families_filter:
                continue
            configs.append(TrainConfig(symbol, family, target, horizon))
    return configs


def status_priority(status: str) -> int:
    return {
        "LONG_MODEL_PROMISING": 0,
        "LONG_MODEL_MIXED": 1,
        "LONG_MODEL_WEAK": 2,
        "LONG_MODEL_FAILED": 3,
        "INSUFFICIENT_DATA": 4,
    }.get(status, 9)


def select_best_by_symbol(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = []
    for symbol in sorted({str(r.get("symbol")) for r in rows}):
        items = [r for r in rows if r.get("symbol") == symbol]
        items.sort(key=lambda r: (
            status_priority(str(r.get("model_status"))),
            -float(r.get("net_quality_lift_after_costs") or -999),
            -float(r.get("hit_top_decile_hit_lift") or -999),
            float(r.get("p90_mae_delta") or 999),
            int(r.get("horizon_candles") or 999),
        ))
        if items:
            best.append(items[0])
    return best


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = tuple(keys or ("empty",))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in columns})


def write_reports(rows: list[dict[str, Any]], buckets: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    best = select_best_by_symbol(rows)
    promising = [r for r in rows if r.get("model_status") == "LONG_MODEL_PROMISING"]
    mixed = [r for r in rows if r.get("model_status") == "LONG_MODEL_MIXED"]
    failed = [r for r in rows if r.get("model_status") == "LONG_MODEL_FAILED"]
    summary_rows: list[dict[str, Any]] = []
    for family in sorted({str(r.get("alpha_family")) for r in rows}):
        fam = [r for r in rows if r.get("alpha_family") == family]
        summary_rows.append({
            "alpha_family": family,
            "configs": len(fam),
            "promising": sum(r.get("model_status") == "LONG_MODEL_PROMISING" for r in fam),
            "mixed": sum(r.get("model_status") == "LONG_MODEL_MIXED" for r in fam),
            "weak": sum(r.get("model_status") == "LONG_MODEL_WEAK" for r in fam),
            "failed": sum(r.get("model_status") == "LONG_MODEL_FAILED" for r in fam),
            "insufficient": sum(r.get("model_status") == "INSUFFICIENT_DATA" for r in fam),
            "best_net_quality_lift": max(float(r.get("net_quality_lift_after_costs") or -999) for r in fam),
            "best_hit_auc": max(float(r.get("hit_auc") or -999) for r in fam),
        })
    paths = {
        "md": str(out / f"aegis_long_c_research_train_{stamp}.md"),
        "json": str(out / f"aegis_long_c_research_train_{stamp}.json"),
        "summary": str(out / f"aegis_long_c_research_summary_{stamp}.csv"),
        "all_configs": str(out / f"aegis_long_c_research_all_configs_{stamp}.csv"),
        "buckets": str(out / f"aegis_long_c_research_buckets_{stamp}.csv"),
        "best_by_symbol": str(out / f"aegis_long_c_research_best_by_symbol_{stamp}.csv"),
        "promising": str(out / f"aegis_long_c_research_promising_{stamp}.csv"),
        "mixed": str(out / f"aegis_long_c_research_mixed_{stamp}.csv"),
        "failed": str(out / f"aegis_long_c_research_failed_{stamp}.csv"),
    }
    write_csv(Path(paths["all_configs"]), rows, CSV_COLUMNS)
    write_csv(Path(paths["buckets"]), buckets)
    write_csv(Path(paths["best_by_symbol"]), best, CSV_COLUMNS)
    write_csv(Path(paths["promising"]), promising, CSV_COLUMNS)
    write_csv(Path(paths["mixed"]), mixed, CSV_COLUMNS)
    write_csv(Path(paths["failed"]), failed, CSV_COLUMNS)
    write_csv(Path(paths["summary"]), summary_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "safety": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2": True,
            "no_orders": True,
        },
        "args": vars(args),
        "configs_trained": len(rows),
        "models_saved": sum(int(r.get("saved_model_count") or 0) for r in rows),
        "summary": summary_rows,
        "best_by_symbol": best,
        "rows": rows,
        "bucket_rows": buckets,
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    lines = [
        "# Aegis LONG-C Research Training",
        "",
        "## Safety",
        "- RESEARCH_ONLY",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Configs trained: `{len(rows)}`",
        f"- Models saved: `{sum(int(r.get('saved_model_count') or 0) for r in rows)}`",
        f"- Promising: `{len(promising)}`",
        f"- Mixed: `{len(mixed)}`",
        f"- Failed: `{len(failed)}`",
        "",
        "## Best By Symbol",
        "| symbol | family | target | h | status | auc | top_hit_lift | net_lift | reason |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in best:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {row.get('target_name')} | {row.get('horizon_candles')} | {row.get('model_status')} | {float(row.get('hit_auc') or 0):.4f} | {float(row.get('hit_top_decile_hit_lift') or 0):.4f} | {float(row.get('net_quality_lift_after_costs') or 0):.5f} | {row.get('model_reason')} |")
    lines += [
        "",
        "## Core LONG Results",
        "| symbol | family | status | auc | quality_corr | net_lift | p90_delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted([r for r in rows if r.get("alpha_family") in CORE_FAMILIES], key=lambda r: (status_priority(str(r.get("model_status"))), -float(r.get("net_quality_lift_after_costs") or -999)))[:40]:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {row.get('model_status')} | {float(row.get('hit_auc') or 0):.4f} | {float(row.get('quality_corr') or 0):.4f} | {float(row.get('net_quality_lift_after_costs') or 0):.5f} | {float(row.get('p90_mae_delta') or 0):.4f} |")
    lines += [
        "",
        "## Momentum LONG Results",
        "| symbol | family | status | h | avg_ttt | fast_hit | net_lift |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted([r for r in rows if r.get("alpha_family") in MOMENTUM_SET], key=lambda r: (status_priority(str(r.get("model_status"))), -float(r.get("net_quality_lift_after_costs") or -999)))[:40]:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {row.get('model_status')} | {row.get('horizon_candles')} | {float(row.get('time_to_target_avg') or 0):.2f} | {float(row.get('fast_hit_rate') or 0):.4f} | {float(row.get('net_quality_lift_after_costs') or 0):.5f} |")
    lines += [
        "",
        "## Exhaustion / Late-entry",
        "| symbol | family | exhaustion_auc | usefulness | late_entry_rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: -float(r.get("exhaustion_filter_usefulness") or -999))[:25]:
        lines.append(f"| {row.get('symbol')} | {row.get('alpha_family')} | {float(row.get('exhaustion_auc') or 0):.4f} | {float(row.get('exhaustion_filter_usefulness') or 0):.5f} | {float(row.get('late_entry_rate') or 0):.4f} |")
    lines += [
        "",
        "## LONG-D Recommendation",
        "- Move LONG_MODEL_PROMISING and strongest LONG_MODEL_MIXED configs to walk-forward confirmation.",
        "- Keep model bundles under research only until a frozen confirmation phase passes.",
        "- Do not promote LONG models or modify active manifests from this research output.",
    ]
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = configs_for_args(args)
    db_path = Path(args.db_path)
    model_dir = Path(args.model_dir)
    assert_research_only_path(model_dir)
    max_iter = 30 if args.fast else 120
    save_models = not args.no_save_models
    if args.save_models:
        save_models = True
    rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for config in configs:
        row, buckets = train_config(
            config,
            db_path,
            model_dir,
            lookback_days=args.lookback_days,
            feature_mode=args.feature_mode,
            min_train_samples=args.min_train_samples,
            min_test_samples=args.min_test_samples,
            max_iter=max_iter,
            save_models=save_models,
        )
        rows.append(row)
        bucket_rows.extend(buckets)
    paths = write_reports(rows, bucket_rows, args)
    return {"reports": paths, "row_count": len(rows), "best_by_symbol": select_best_by_symbol(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--families", default="")
    parser.add_argument("--family-group", choices=("core", "momentum", "all"), default="core")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--feature-mode", choices=("selected_family", "combined_v3_all"), default="selected_family")
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.lookback_days = min(args.lookback_days, 60)
    result = run(args)
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
