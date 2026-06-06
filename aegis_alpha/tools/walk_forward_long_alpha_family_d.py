#!/usr/bin/env python3
"""Research-only walk-forward runner for LONG alpha families.

Phase LONG-D supports only explicitly approved LONG families. It reads
local candles, trains fold-local models in memory by default, and writes reports
only. It does not touch active manifests, YAML, PM2, orders, or live inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.profile_long_alpha_families_a import SYMBOLS, add_features, json_safe, mean, quantile, top_fraction_mask
from aegis_alpha.turbo.long_research_cache import LongResearchCache
from aegis_alpha.tools.train_long_alpha_candidates_c import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    TARGET_MAP,
    add_long_c_features,
    assert_research_only_path,
    build_probability_buckets,
    classifier,
    compute_labels,
    feature_hash,
    load_candles_research,
    model_predict_proba,
    one_class,
    regressor,
    safe_ap,
    safe_auc,
    safe_corr,
    safe_spearman,
    select_long_family_features,
    train_or_skip_classifier,
)

SCHEMA_VERSION = "aegis_long_walkforward_family_d_v2"
ALLOWED_FAMILY = "pullback_continuation_long"
ALLOWED_FAMILIES = ("pullback_continuation_long", "slow_trend_long", "micro_roe_momentum_long")
PRIMARY_SYMBOLS = ("AVAXUSDT", "ETHUSDT", "SUIUSDT")
SECONDARY_CONFIGS = {
    "pullback_continuation_long": {
        "XRPUSDT": ("long_hit3_before_minus2", 24),
        "DOGEUSDT": ("long_hit6_before_minus4", 24),
        "SOLUSDT": ("long_hit3_before_minus2", 24),
        "ADAUSDT": ("long_hit3_before_minus2", 24),
    },
    "slow_trend_long": {
        "ADAUSDT": ("long_hit3_before_minus2", 24),
        "AVAXUSDT": ("long_hit3_before_minus2", 24),
        "LINKUSDT": ("long_hit3_before_minus2", 24),
        "DOGEUSDT": ("long_hit6_before_minus4", 24),
        "SUIUSDT": ("long_hit5_before_minus3", 24),
        "LTCUSDT": ("long_hit8_before_minus5", 24),
    },
    "micro_roe_momentum_long": {
        "ETHUSDT": ("long_roe12_before_minus8", 12),
        "AVAXUSDT": ("long_roe12_before_minus8", 12),
        "SOLUSDT": ("long_roe10_before_minus6", 6),
        "XRPUSDT": ("long_roe10_before_minus6", 6),
    },
}
CSV_FOLD_COLUMNS = (
    "symbol", "family", "target", "horizon", "fold_index", "train_start", "train_end",
    "test_start", "test_end", "train_samples", "test_samples", "feature_count",
    "baseline_hit_rate", "baseline_quality", "baseline_stop_rate", "baseline_p90_mae",
    "baseline_time_to_target", "hit_auc", "hit_average_precision", "hit_top_decile_hit_rate",
    "hit_top_decile_hit_lift", "quality_corr", "quality_spearman_corr", "quality_lift",
    "net_quality_lift_after_costs", "danger_auc", "exhaustion_auc", "exhaustion_filter_usefulness",
    "selected_fraction", "selected_count", "selected_hit_rate", "selected_quality",
    "selected_stop_rate", "selected_p90_mae", "selected_time_to_target_avg",
    "selected_fast_hit_rate", "selected_late_entry_rate", "hit_lift", "stop_rate_delta",
    "p90_mae_delta", "time_to_target_delta", "trend_persistence_score", "baseline_fast_hit_rate",
    "fast_hit_lift", "exhaustion_rate_selected", "late_entry_rate_selected", "overextension_proxy_avg",
    "upper_wick_risk_avg", "range_expansion_avg", "fold_status", "fold_reason",
)
CSV_SUMMARY_COLUMNS = (
    "symbol", "family", "target", "horizon", "d_status", "d1_status", "d2_status", "d3_status", "score", "valid_folds",
    "negative_folds", "mean_hit_lift", "latest_fold_hit_lift", "mean_net_quality_lift_after_costs",
    "latest_fold_net_quality_lift_after_costs", "mean_p90_mae_delta", "mean_stop_rate_delta",
    "mean_selected_fraction", "mean_hit_auc", "mean_hit_top_decile_hit_lift", "trend_persistence_score",
    "slow_trend_quality_stability", "slow_trend_hit_stability", "mean_time_to_target",
    "latest_time_to_target", "fast_hit_rate", "fast_hit_lift", "momentum_quality_stability",
    "momentum_hit_stability", "exhaustion_rate_selected", "late_entry_rate_selected",
    "overextension_proxy_avg", "upper_wick_risk_avg", "range_expansion_avg", "recommendation",
)


@dataclass(frozen=True)
class WalkConfig:
    symbol: str
    family: str
    target: str
    horizon: int


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def build_expanding_folds(n: int, fold_count: int, min_train: int, min_test: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if fold_count <= 0:
        raise ValueError("fold_count must be positive")
    if n < min_train + min_test:
        return []
    remaining = n - min_train
    test_size = remaining // fold_count
    if test_size < min_test:
        return []
    folds = []
    idx = np.arange(n)
    for fold in range(fold_count):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = n if fold == fold_count - 1 else test_start + test_size
        if test_end - test_start < min_test:
            continue
        folds.append((idx[:train_end], idx[test_start:test_end]))
    return folds


def selection_mask(hit_pred: np.ndarray, quality_pred: np.ndarray, danger_pred: np.ndarray, exhaustion_pred: np.ndarray) -> np.ndarray:
    valid = np.ones(len(quality_pred), dtype=bool)
    q = np.nan_to_num(quality_pred, nan=np.nanmedian(quality_pred[np.isfinite(quality_pred)]) if np.isfinite(quality_pred).any() else 0.0)
    score = q + np.nan_to_num(hit_pred, nan=0.0) * 0.5 - np.nan_to_num(danger_pred, nan=0.5) * 0.25 - np.nan_to_num(exhaustion_pred, nan=0.5) * 0.25
    return top_fraction_mask(score, valid, 0.12)


def fold_status(row: dict[str, Any]) -> str:
    net = float(row.get("net_quality_lift_after_costs") or 0.0)
    hit = float(row.get("hit_lift") or 0.0)
    p90 = float(row.get("p90_mae_delta") or 0.0)
    stop = float(row.get("stop_rate_delta") or 0.0)
    if net > 0 and hit > 0 and p90 <= 0.10 and stop <= 0.05:
        return "POSITIVE"
    if net < 0 or hit < 0 or p90 > 0.20 or stop > 0.08:
        return "NEGATIVE"
    return "MIXED"


def train_fold(
    *,
    config: WalkConfig,
    frame,
    labels: dict[str, np.ndarray],
    feature_names: list[str],
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_iter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = frame[feature_names].to_numpy(dtype=float)
    x_train, x_test = x[train_idx], x[test_idx]
    y_hit_train, y_hit_test = labels["hit"][train_idx], labels["hit"][test_idx]
    y_quality_train, y_quality_test = labels["quality"][train_idx], labels["quality"][test_idx]
    y_danger_train, y_danger_test = labels["mae_danger"][train_idx], labels["mae_danger"][test_idx]
    y_exh_train, y_exh_test = labels["late_entry_exhaustion"][train_idx], labels["late_entry_exhaustion"][test_idx]
    seed = abs(hash((config.symbol, config.family, config.target, config.horizon, fold_index))) % 2_000_000_000
    hit_model, _ = train_or_skip_classifier("hit", x_train, y_hit_train, seed, max_iter)
    danger_model, _ = train_or_skip_classifier("danger", x_train, y_danger_train, seed + 1, max_iter)
    exhaustion_model, _ = train_or_skip_classifier("exhaustion", x_train, y_exh_train, seed + 2, max_iter)
    quality_model = regressor(seed + 3, max_iter).fit(x_train, y_quality_train)
    hit_pred = model_predict_proba(hit_model, x_test)
    danger_pred = model_predict_proba(danger_model, x_test)
    exhaustion_pred = model_predict_proba(exhaustion_model, x_test)
    quality_pred = quality_model.predict(x_test)
    selected = selection_mask(hit_pred, quality_pred, danger_pred, exhaustion_pred)
    baseline_hit = mean(y_hit_test) or 0.0
    selected_hit = mean(y_hit_test[selected]) if selected.any() else None
    baseline_quality = mean(y_quality_test) or 0.0
    selected_quality = mean(y_quality_test[selected]) if selected.any() else None
    baseline_stop = mean(labels["stop"][test_idx]) or 0.0
    selected_stop = mean(labels["stop"][test_idx][selected]) if selected.any() else None
    baseline_mae = quantile(labels["mae"][test_idx], 0.90) or 0.0
    selected_mae = quantile(labels["mae"][test_idx][selected], 0.90) if selected.any() else None
    baseline_ttt = labels["time_to_target"][test_idx]
    baseline_ttt = baseline_ttt[baseline_ttt > 0]
    selected_ttt = labels["time_to_target"][test_idx][selected]
    selected_ttt = selected_ttt[selected_ttt > 0]
    top_decile = top_fraction_mask(np.nan_to_num(hit_pred, nan=-1), np.ones(len(test_idx), dtype=bool), 0.10)
    hit_top = mean(y_hit_test[top_decile]) if top_decile.any() else None
    high_exh = top_fraction_mask(np.nan_to_num(exhaustion_pred, nan=-1), np.ones(len(test_idx), dtype=bool), 0.20) if exhaustion_model is not None else np.zeros(len(test_idx), bool)
    low_exh = ~high_exh
    exhaustion_lift = None
    if high_exh.any() and low_exh.any():
        exhaustion_lift = (mean(y_quality_test[low_exh]) or 0.0) - (mean(y_quality_test[high_exh]) or 0.0)
    baseline_fast_hit = mean(labels["fast_hit"][test_idx]) or 0.0
    selected_fast_hit = mean(labels["fast_hit"][test_idx][selected]) if selected.any() else None
    selected_frame = frame.iloc[test_idx[selected]] if selected.any() else frame.iloc[[]]
    row = {
        "symbol": config.symbol,
        "family": config.family,
        "target": config.target,
        "horizon": config.horizon,
        "fold_index": fold_index,
        "train_start": str(frame.iloc[int(train_idx[0])]["timestamp"]),
        "train_end": str(frame.iloc[int(train_idx[-1])]["timestamp"]),
        "test_start": str(frame.iloc[int(test_idx[0])]["timestamp"]),
        "test_end": str(frame.iloc[int(test_idx[-1])]["timestamp"]),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "feature_count": int(len(feature_names)),
        "baseline_hit_rate": baseline_hit,
        "baseline_quality": baseline_quality,
        "baseline_stop_rate": baseline_stop,
        "baseline_p90_mae": baseline_mae,
        "baseline_time_to_target": mean(baseline_ttt),
        "hit_auc": safe_auc(y_hit_test, np.nan_to_num(hit_pred, nan=baseline_hit)) if hit_model is not None else None,
        "hit_average_precision": safe_ap(y_hit_test, np.nan_to_num(hit_pred, nan=baseline_hit)) if hit_model is not None else None,
        "hit_top_decile_hit_rate": hit_top,
        "hit_top_decile_hit_lift": (hit_top - baseline_hit) if hit_top is not None else None,
        "quality_corr": safe_corr(y_quality_test, quality_pred),
        "quality_spearman_corr": safe_spearman(y_quality_test, quality_pred),
        "quality_lift": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "net_quality_lift_after_costs": (selected_quality - baseline_quality) if selected_quality is not None else None,
        "danger_auc": safe_auc(y_danger_test, np.nan_to_num(danger_pred, nan=mean(y_danger_test) or 0.0)) if danger_model is not None else None,
        "exhaustion_auc": safe_auc(y_exh_test, np.nan_to_num(exhaustion_pred, nan=mean(y_exh_test) or 0.0)) if exhaustion_model is not None else None,
        "exhaustion_filter_usefulness": exhaustion_lift,
        "selected_fraction": float(selected.mean()) if len(selected) else 0.0,
        "selected_count": int(selected.sum()),
        "selected_hit_rate": selected_hit,
        "selected_quality": selected_quality,
        "selected_stop_rate": selected_stop,
        "selected_p90_mae": selected_mae,
        "selected_time_to_target_avg": mean(selected_ttt),
        "selected_fast_hit_rate": selected_fast_hit,
        "selected_late_entry_rate": mean(y_exh_test[selected]) if selected.any() else None,
        "hit_lift": (selected_hit - baseline_hit) if selected_hit is not None else None,
        "stop_rate_delta": (selected_stop - baseline_stop) if selected_stop is not None else None,
        "p90_mae_delta": ((selected_mae - baseline_mae) / max(baseline_mae, 1e-12)) if selected_mae is not None else None,
        "time_to_target_delta": ((mean(selected_ttt) or 0.0) - (mean(baseline_ttt) or 0.0)) if len(selected_ttt) else None,
        "trend_persistence_score": mean(selected_frame["trend_efficiency_24"].to_numpy(dtype=float)) if selected.any() and "trend_efficiency_24" in frame else None,
        "baseline_fast_hit_rate": baseline_fast_hit,
        "fast_hit_lift": (selected_fast_hit - baseline_fast_hit) if selected_fast_hit is not None else None,
        "exhaustion_rate_selected": mean(y_exh_test[selected]) if selected.any() else None,
        "late_entry_rate_selected": mean(y_exh_test[selected]) if selected.any() else None,
        "overextension_proxy_avg": mean(selected_frame["overextension_risk"].to_numpy(dtype=float)) if selected.any() and "overextension_risk" in frame else None,
        "upper_wick_risk_avg": mean(selected_frame["upper_wick_ratio"].to_numpy(dtype=float)) if selected.any() and "upper_wick_ratio" in frame else None,
        "range_expansion_avg": mean(selected_frame["range_expansion_12"].to_numpy(dtype=float)) if selected.any() and "range_expansion_12" in frame else None,
    }
    row["fold_status"] = fold_status(row)
    row["fold_reason"] = "positive hit/net/risk" if row["fold_status"] == "POSITIVE" else ("negative hit/net/risk" if row["fold_status"] == "NEGATIVE" else "mixed fold")
    buckets = []
    buckets.extend(build_probability_buckets(np.nan_to_num(hit_pred, nan=baseline_hit), y_hit_test, y_quality_test, y_danger_test, y_exh_test, symbol=config.symbol, family=config.family, target_name=config.target, horizon=config.horizon, bucket_source=f"fold_{fold_index}_hit_probability"))
    return row, buckets


def classify_d_symbol(summary: dict[str, Any], family: str = ALLOWED_FAMILY) -> str:
    if int(summary.get("valid_folds") or 0) < 3:
        return "INSUFFICIENT_DATA"
    mean_hit = float(summary.get("mean_hit_lift") or 0.0)
    latest_hit = float(summary.get("latest_fold_hit_lift") or 0.0)
    mean_net = float(summary.get("mean_net_quality_lift_after_costs") or 0.0)
    latest_net = float(summary.get("latest_fold_net_quality_lift_after_costs") or 0.0)
    p90 = float(summary.get("mean_p90_mae_delta") or 0.0)
    stop = float(summary.get("mean_stop_rate_delta") or 0.0)
    frac = float(summary.get("mean_selected_fraction") or 0.0)
    auc = summary.get("mean_hit_auc")
    top = float(summary.get("mean_hit_top_decile_hit_lift") or 0.0)
    negative = int(summary.get("negative_folds") or 0)
    if family == "micro_roe_momentum_long":
        horizon = float(summary.get("horizon") or 1.0)
        mean_ttt = summary.get("mean_time_to_target")
        fast_lift = float(summary.get("fast_hit_lift") or 0.0)
        fast_rate = float(summary.get("fast_hit_rate") or 0.0)
        time_ok = mean_ttt is not None and float(mean_ttt) <= horizon * 0.65
        fast_ok = fast_lift > 0 or fast_rate >= 0.05
        if mean_hit > 0 and latest_hit >= 0 and mean_net > 0 and latest_net >= 0 and p90 <= 0.12 and stop <= 0.06 and 0.05 <= frac <= 0.25 and ((auc is not None and float(auc) >= 0.53) or top > 0.02) and negative <= 1 and time_ok and fast_ok:
            return "LONG_D3_CONFIRMED"
        if mean_net < 0 or mean_hit < 0 or p90 > 0.20 or stop > 0.08 or not time_ok or negative >= max(2, int(summary.get("valid_folds") or 0) - 1):
            return "LONG_D3_FAILED"
        if mean_hit > 0 or mean_net > 0 or latest_hit > 0 or latest_net > 0 or fast_lift > 0:
            return "LONG_D3_MIXED"
        return "LONG_D3_WEAK"
    if mean_hit > 0 and latest_hit >= 0 and mean_net > 0 and latest_net >= 0 and p90 <= 0.10 and stop <= 0.05 and 0.05 <= frac <= 0.25 and ((auc is not None and float(auc) >= 0.53) or top > 0.02) and negative <= 1:
        return f"{status_prefix(family)}_CONFIRMED"
    if mean_net < 0 or mean_hit < 0 or p90 > 0.20 or stop > 0.08 or negative >= max(2, int(summary.get("valid_folds") or 0) - 1):
        return f"{status_prefix(family)}_FAILED"
    if mean_hit > 0 or mean_net > 0 or latest_hit > 0 or latest_net > 0:
        return f"{status_prefix(family)}_MIXED"
    return f"{status_prefix(family)}_WEAK"

def status_prefix(family: str) -> str:
    if family == "slow_trend_long":
        return "LONG_D2"
    if family == "micro_roe_momentum_long":
        return "LONG_D3"
    return "LONG_D1"


def classify_d1_symbol(summary: dict[str, Any]) -> str:
    return classify_d_symbol(summary, "pullback_continuation_long")


def phase_slug(family: str) -> str:
    if family == "slow_trend_long":
        return "d2_slowtrend"
    if family == "micro_roe_momentum_long":
        return "d3_micro_roe"
    return "d1_pullback"


def phase_title(family: str) -> str:
    if family == "slow_trend_long":
        return "LONG-D2 Slow-trend Walk-forward"
    if family == "micro_roe_momentum_long":
        return "LONG-D3 Micro-ROE Momentum Walk-forward"
    return "LONG-D1 Pullback Walk-forward"


def score_summary(summary: dict[str, Any]) -> float:
    mean_net = float(summary.get("mean_net_quality_lift_after_costs") or 0.0)
    mean_hit = float(summary.get("mean_hit_lift") or 0.0)
    mean_top = float(summary.get("mean_hit_top_decile_hit_lift") or 0.0)
    p90 = max(float(summary.get("mean_p90_mae_delta") or 0.0), 0.0)
    stop = max(float(summary.get("mean_stop_rate_delta") or 0.0), 0.0)
    latest_net = float(summary.get("latest_fold_net_quality_lift_after_costs") or 0.0)
    latest_hit = float(summary.get("latest_fold_hit_lift") or 0.0)
    if summary.get("family") == "micro_roe_momentum_long":
        fast_lift = float(summary.get("fast_hit_lift") or 0.0)
        late = float(summary.get("late_entry_rate_selected") or 0.0)
        return 2.0 * mean_net + 1.5 * mean_hit + 1.0 * mean_top + 0.75 * fast_lift - p90 - stop - 0.5 * late + 0.5 * latest_net + 0.5 * latest_hit
    return 2.0 * mean_net + 1.5 * mean_hit + 1.0 * mean_top - p90 - stop + 0.5 * latest_net + 0.5 * latest_hit


def summarize_symbol(config: WalkConfig, folds: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [f for f in folds if f.get("fold_status")]
    latest = valid[-1] if valid else {}
    def avg(key: str) -> float | None:
        vals = [finite(f.get(key)) for f in valid]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    summary = {
        "symbol": config.symbol,
        "family": config.family,
        "target": config.target,
        "horizon": config.horizon,
        "valid_folds": len(valid),
        "negative_folds": sum(f.get("fold_status") == "NEGATIVE" for f in valid),
        "mean_hit_lift": avg("hit_lift"),
        "latest_fold_hit_lift": latest.get("hit_lift"),
        "mean_net_quality_lift_after_costs": avg("net_quality_lift_after_costs"),
        "latest_fold_net_quality_lift_after_costs": latest.get("net_quality_lift_after_costs"),
        "mean_p90_mae_delta": avg("p90_mae_delta"),
        "mean_stop_rate_delta": avg("stop_rate_delta"),
        "mean_selected_fraction": avg("selected_fraction"),
        "mean_hit_auc": avg("hit_auc"),
        "mean_hit_top_decile_hit_lift": avg("hit_top_decile_hit_lift"),
    }
    summary["trend_persistence_score"] = avg("trend_persistence_score")
    summary["slow_trend_quality_stability"] = (sum((finite(f.get("net_quality_lift_after_costs")) or 0.0) >= 0 for f in valid) / len(valid)) if valid else None
    summary["slow_trend_hit_stability"] = (sum((finite(f.get("hit_lift")) or 0.0) >= 0 for f in valid) / len(valid)) if valid else None
    summary["mean_time_to_target"] = avg("selected_time_to_target_avg")
    summary["latest_time_to_target"] = latest.get("selected_time_to_target_avg")
    summary["fast_hit_rate"] = avg("selected_fast_hit_rate")
    summary["fast_hit_lift"] = avg("fast_hit_lift")
    summary["momentum_quality_stability"] = (sum((finite(f.get("net_quality_lift_after_costs")) or 0.0) >= 0 for f in valid) / len(valid)) if valid else None
    summary["momentum_hit_stability"] = (sum((finite(f.get("hit_lift")) or 0.0) >= 0 for f in valid) / len(valid)) if valid else None
    summary["exhaustion_rate_selected"] = avg("exhaustion_rate_selected")
    summary["late_entry_rate_selected"] = avg("late_entry_rate_selected")
    summary["overextension_proxy_avg"] = avg("overextension_proxy_avg")
    summary["upper_wick_risk_avg"] = avg("upper_wick_risk_avg")
    summary["range_expansion_avg"] = avg("range_expansion_avg")
    summary["d_status"] = classify_d_symbol(summary, config.family)
    summary["d1_status"] = summary["d_status"] if config.family == "pullback_continuation_long" else None
    summary["d2_status"] = summary["d_status"] if config.family == "slow_trend_long" else None
    summary["d3_status"] = summary["d_status"] if config.family == "micro_roe_momentum_long" else None
    summary["score"] = score_summary(summary)
    prefix = status_prefix(config.family)
    if summary["d_status"] == f"{prefix}_CONFIRMED":
        summary["recommendation"] = "pass_to_frozen_confirmation"
    elif summary["d_status"] == f"{prefix}_MIXED":
        summary["recommendation"] = "repeat_with_more_data_or_lockbox"
    else:
        summary["recommendation"] = f"keep_research_or_discard_{config.family}"
    return summary


def _cache_get(cache: LongResearchCache | None, namespace: str, parts: tuple[Any, ...], factory):
    if cache is None:
        return factory()
    return cache.get_or_set(namespace, parts, factory)


def prepare_symbol(config: WalkConfig, db_path: Path, lookback_days: int, feature_mode: str, cache: LongResearchCache | None = None):
    df = _cache_get(cache, "ohlcv", cache.ohlcv_key(config.symbol, lookback_days, db_path) if cache else (), lambda: load_candles_research(db_path, config.symbol, lookback_days))
    if df.empty:
        return None, None, [], [], []
    btc = df if config.symbol == "BTCUSDT" else _cache_get(cache, "ohlcv", cache.ohlcv_key("BTCUSDT", lookback_days, db_path) if cache else (), lambda: load_candles_research(db_path, "BTCUSDT", lookback_days))
    eth = df if config.symbol == "ETHUSDT" else _cache_get(cache, "ohlcv", cache.ohlcv_key("ETHUSDT", lookback_days, db_path) if cache else (), lambda: load_candles_research(db_path, "ETHUSDT", lookback_days))
    base = _cache_get(cache, "feature_base", cache.feature_key(config.symbol, lookback_days, f"base:{feature_mode}", db_path) if cache else (), lambda: add_features(df, btc, eth))
    frame = _cache_get(cache, "feature_long_c", cache.feature_key(config.symbol, lookback_days, f"long_c:{config.family}:{feature_mode}", db_path) if cache else (), lambda: add_long_c_features(base).reset_index(drop=True))
    labels = _cache_get(cache, "labels", cache.labels_key(config.symbol, config.target, config.horizon, lookback_days, db_path) if cache else (), lambda: compute_labels(frame["close"].to_numpy(float), frame["high"].to_numpy(float), frame["low"].to_numpy(float), frame, config.target, config.horizon))
    feature_names, missing, proxies = select_long_family_features(frame, config.family, feature_mode)
    valid = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy().copy()
    valid[:220] = False
    valid[-config.horizon:] = False
    for name in feature_names:
        valid &= np.isfinite(frame[name].to_numpy(float))
    valid &= np.isfinite(labels["quality"])
    idx = np.flatnonzero(valid)
    return frame, labels, feature_names, idx, {"missing_features": missing, "proxy_features_used": proxies}


def run_config(config: WalkConfig, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cache = getattr(args, "_long_research_cache", None) if getattr(args, "use_cache", False) else None
    frame, labels, features, idx, feature_meta = prepare_symbol(config, Path(args.db_path), args.lookback_days, args.feature_mode, cache)
    if frame is None or labels is None or len(features) == 0:
        summary = {"symbol": config.symbol, "family": config.family, "target": config.target, "horizon": config.horizon, "valid_folds": 0, "negative_folds": 0, "d1_status": "INSUFFICIENT_DATA", "score": -999, "recommendation": "no_data"}
        return summary, [], []
    if cache is not None:
        folds_idx = cache.get_or_set("folds", cache.folds_key(len(idx), args.fold_count, args.min_train_samples, args.min_test_samples), lambda: build_expanding_folds(len(idx), args.fold_count, args.min_train_samples, args.min_test_samples))
    else:
        folds_idx = build_expanding_folds(len(idx), args.fold_count, args.min_train_samples, args.min_test_samples)
    fold_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    max_iter = 30 if args.fast else 120
    for fold_no, (train_local, test_local) in enumerate(folds_idx, start=1):
        train_idx = idx[train_local]
        test_idx = idx[test_local]
        row, buckets = train_fold(config=config, frame=frame, labels=labels, feature_names=features, fold_index=fold_no, train_idx=train_idx, test_idx=test_idx, max_iter=max_iter)
        row["missing_features"] = ";".join(feature_meta["missing_features"])
        row["proxy_features_used"] = ";".join(feature_meta["proxy_features_used"])
        row["feature_schema_hash"] = feature_hash(features)
        fold_rows.append(row)
        bucket_rows.extend(buckets)
    summary = summarize_symbol(config, fold_rows)
    summary["feature_count"] = len(features)
    summary["feature_schema_hash"] = feature_hash(features)
    summary["missing_features"] = ";".join(feature_meta["missing_features"])
    summary["proxy_features_used"] = ";".join(feature_meta["proxy_features_used"])
    return summary, fold_rows, bucket_rows


def configs_from_args(args: argparse.Namespace) -> list[WalkConfig]:
    if args.family not in ALLOWED_FAMILIES:
        raise SystemExit("LONG-D only allows --family pullback_continuation_long, slow_trend_long, or micro_roe_momentum_long")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bad = [s for s in symbols if s not in SYMBOLS]
    if bad:
        raise SystemExit(f"Unsupported symbols: {','.join(bad)}")
    configs = []
    for symbol in symbols:
        secondary = SECONDARY_CONFIGS.get(args.family, {})
        if args.include_secondary and symbol in secondary:
            target, horizon = secondary[symbol]
        else:
            target, horizon = args.target, args.horizon
        configs.append(WalkConfig(symbol, args.family, target, horizon))
    return configs


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys = []
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


def write_reports(summaries: list[dict[str, Any]], folds: list[dict[str, Any]], buckets: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    ranking = sorted(summaries, key=lambda r: -float(r.get("score") or -999))
    prefix = status_prefix(args.family)
    slug = phase_slug(args.family)
    confirmed = [r for r in summaries if r.get("d_status") == f"{prefix}_CONFIRMED"]
    mixed = [r for r in summaries if r.get("d_status") == f"{prefix}_MIXED"]
    failed = [r for r in summaries if r.get("d_status") in {f"{prefix}_FAILED", f"{prefix}_WEAK", "INSUFFICIENT_DATA"}]
    paths = {
        "md": str(out / f"aegis_long_{slug}_walkforward_{stamp}.md"),
        "json": str(out / f"aegis_long_{slug}_walkforward_{stamp}.json"),
        "summary": str(out / f"aegis_long_{slug}_summary_{stamp}.csv"),
        "folds": str(out / f"aegis_long_{slug}_folds_{stamp}.csv"),
        "ranking": str(out / f"aegis_long_{slug}_ranking_{stamp}.csv"),
        "confirmed": str(out / f"aegis_long_{slug}_confirmed_{stamp}.csv"),
        "mixed": str(out / f"aegis_long_{slug}_mixed_{stamp}.csv"),
        "failed": str(out / f"aegis_long_{slug}_failed_{stamp}.csv"),
    }
    write_csv(Path(paths["summary"]), summaries, CSV_SUMMARY_COLUMNS)
    write_csv(Path(paths["folds"]), folds, CSV_FOLD_COLUMNS)
    write_csv(Path(paths["ranking"]), ranking, CSV_SUMMARY_COLUMNS)
    write_csv(Path(paths["confirmed"]), confirmed, CSV_SUMMARY_COLUMNS)
    write_csv(Path(paths["mixed"]), mixed, CSV_SUMMARY_COLUMNS)
    write_csv(Path(paths["failed"]), failed, CSV_SUMMARY_COLUMNS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "safety": {"no_live": True, "no_active_manifest": True, "no_yaml": True, "no_pm2": True, "no_orders": True},
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "summaries": summaries,
        "folds": folds,
        "buckets": buckets,
        "ranking": ranking,
        "cache_stats": args._long_research_cache.summary() if getattr(args, "_long_research_cache", None) is not None else {"enabled": False},
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    best = ranking[0] if ranking else {}
    lines = [
        f"# Aegis {phase_title(args.family)}",
        "",
        "## Safety",
        "- research-only",
        "- no live",
        "- no active_manifest",
        "- no YAML",
        "- no PM2",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Confirmed symbols: `{','.join(r['symbol'] for r in confirmed) or 'none'}`",
        f"- Mixed symbols: `{','.join(r['symbol'] for r in mixed) or 'none'}`",
        f"- Failed/weak/insufficient symbols: `{','.join(r['symbol'] for r in failed) or 'none'}`",
        f"- Best symbol: `{best.get('symbol', 'none')}`",
        "",
        "## Symbol Ranking",
        "| symbol | status | score | mean_hit_lift | mean_net_lift | latest_hit_lift | latest_net_lift | p90_delta | stop_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(f"| {row.get('symbol')} | {row.get('d_status')} | {float(row.get('score') or 0):.5f} | {float(row.get('mean_hit_lift') or 0):.5f} | {float(row.get('mean_net_quality_lift_after_costs') or 0):.5f} | {float(row.get('latest_fold_hit_lift') or 0):.5f} | {float(row.get('latest_fold_net_quality_lift_after_costs') or 0):.5f} | {float(row.get('mean_p90_mae_delta') or 0):.5f} | {float(row.get('mean_stop_rate_delta') or 0):.5f} |")
    lines += [
        "",
        "## Fold Details",
        "| symbol | fold | status | hit_lift | net_lift | p90_delta | stop_delta | selected_fraction |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(f"| {row.get('symbol')} | {row.get('fold_index')} | {row.get('fold_status')} | {float(row.get('hit_lift') or 0):.5f} | {float(row.get('net_quality_lift_after_costs') or 0):.5f} | {float(row.get('p90_mae_delta') or 0):.5f} | {float(row.get('stop_rate_delta') or 0):.5f} | {float(row.get('selected_fraction') or 0):.4f} |")
    lines += [
        "",
        "## Recommendation",
        "- Confirmed symbols can move to frozen/lockbox confirmation.",
        "- Mixed symbols stay research and should be rerun with more data or tighter selection.",
        "- Failed/weak symbols should not advance for this family.",
    ]
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_research_only_path(args.model_dir)
    args._long_research_cache = LongResearchCache(args.cache_max_items) if getattr(args, "use_cache", False) else None
    configs = configs_from_args(args)
    summaries: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    buckets: list[dict[str, Any]] = []
    for config in configs:
        summary, fold_rows, bucket_rows = run_config(config, args)
        summaries.append(summary)
        folds.extend(fold_rows)
        buckets.extend(bucket_rows)
    paths = write_reports(summaries, folds, buckets, args)
    return {"reports": paths, "summaries": summaries, "fold_count": len(folds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(PRIMARY_SYMBOLS))
    parser.add_argument("--family", default=ALLOWED_FAMILY)
    parser.add_argument("--target", default="long_hit3_before_minus2")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--feature-mode", choices=("selected_family", "combined_v3_all"), default="selected_family")
    parser.add_argument("--fast", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", action="store_true")
    cache_group.add_argument("--no-cache", action="store_false", dest="use_cache")
    parser.set_defaults(use_cache=False)
    parser.add_argument("--cache-max-items", type=int, default=64)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-save-models", action="store_true", default=True)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=250)
    parser.add_argument("--include-secondary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
