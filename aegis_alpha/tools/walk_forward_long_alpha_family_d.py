#!/usr/bin/env python3
"""Research-only walk-forward runner for LONG alpha families.

Phase LONG-D1 is intentionally limited to pullback_continuation_long. It reads
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

SCHEMA_VERSION = "aegis_long_d1_pullback_walkforward_v1"
ALLOWED_FAMILY = "pullback_continuation_long"
PRIMARY_SYMBOLS = ("AVAXUSDT", "ETHUSDT", "SUIUSDT")
SECONDARY_CONFIGS = {
    "XRPUSDT": ("long_hit3_before_minus2", 24),
    "DOGEUSDT": ("long_hit6_before_minus4", 24),
    "SOLUSDT": ("long_hit3_before_minus2", 24),
    "ADAUSDT": ("long_hit3_before_minus2", 24),
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
    "p90_mae_delta", "time_to_target_delta", "fold_status", "fold_reason",
)
CSV_SUMMARY_COLUMNS = (
    "symbol", "family", "target", "horizon", "d1_status", "score", "valid_folds",
    "negative_folds", "mean_hit_lift", "latest_fold_hit_lift", "mean_net_quality_lift_after_costs",
    "latest_fold_net_quality_lift_after_costs", "mean_p90_mae_delta", "mean_stop_rate_delta",
    "mean_selected_fraction", "mean_hit_auc", "mean_hit_top_decile_hit_lift", "recommendation",
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
        "selected_fast_hit_rate": mean(labels["fast_hit"][test_idx][selected]) if selected.any() else None,
        "selected_late_entry_rate": mean(y_exh_test[selected]) if selected.any() else None,
        "hit_lift": (selected_hit - baseline_hit) if selected_hit is not None else None,
        "stop_rate_delta": (selected_stop - baseline_stop) if selected_stop is not None else None,
        "p90_mae_delta": ((selected_mae - baseline_mae) / max(baseline_mae, 1e-12)) if selected_mae is not None else None,
        "time_to_target_delta": ((mean(selected_ttt) or 0.0) - (mean(baseline_ttt) or 0.0)) if len(selected_ttt) else None,
    }
    row["fold_status"] = fold_status(row)
    row["fold_reason"] = "positive hit/net/risk" if row["fold_status"] == "POSITIVE" else ("negative hit/net/risk" if row["fold_status"] == "NEGATIVE" else "mixed fold")
    buckets = []
    buckets.extend(build_probability_buckets(np.nan_to_num(hit_pred, nan=baseline_hit), y_hit_test, y_quality_test, y_danger_test, y_exh_test, symbol=config.symbol, family=config.family, target_name=config.target, horizon=config.horizon, bucket_source=f"fold_{fold_index}_hit_probability"))
    return row, buckets


def classify_d1_symbol(summary: dict[str, Any]) -> str:
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
    if mean_hit > 0 and latest_hit >= 0 and mean_net > 0 and latest_net >= 0 and p90 <= 0.10 and stop <= 0.05 and 0.05 <= frac <= 0.25 and ((auc is not None and float(auc) >= 0.53) or top > 0.02) and negative <= 1:
        return "LONG_D1_CONFIRMED"
    if mean_net < 0 or mean_hit < 0 or p90 > 0.20 or stop > 0.08 or negative >= max(2, int(summary.get("valid_folds") or 0) - 1):
        return "LONG_D1_FAILED"
    if mean_hit > 0 or mean_net > 0 or latest_hit > 0 or latest_net > 0:
        return "LONG_D1_MIXED"
    return "LONG_D1_WEAK"


def score_summary(summary: dict[str, Any]) -> float:
    mean_net = float(summary.get("mean_net_quality_lift_after_costs") or 0.0)
    mean_hit = float(summary.get("mean_hit_lift") or 0.0)
    mean_top = float(summary.get("mean_hit_top_decile_hit_lift") or 0.0)
    p90 = max(float(summary.get("mean_p90_mae_delta") or 0.0), 0.0)
    stop = max(float(summary.get("mean_stop_rate_delta") or 0.0), 0.0)
    latest_net = float(summary.get("latest_fold_net_quality_lift_after_costs") or 0.0)
    latest_hit = float(summary.get("latest_fold_hit_lift") or 0.0)
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
    summary["d1_status"] = classify_d1_symbol(summary)
    summary["score"] = score_summary(summary)
    if summary["d1_status"] == "LONG_D1_CONFIRMED":
        summary["recommendation"] = "pass_to_frozen_confirmation"
    elif summary["d1_status"] == "LONG_D1_MIXED":
        summary["recommendation"] = "repeat_with_more_data_or_lockbox"
    else:
        summary["recommendation"] = "keep_research_or_discard_pullback"
    return summary


def prepare_symbol(config: WalkConfig, db_path: Path, lookback_days: int, feature_mode: str):
    df = load_candles_research(db_path, config.symbol, lookback_days)
    if df.empty:
        return None, None, [], [], []
    btc = load_candles_research(db_path, "BTCUSDT", lookback_days) if config.symbol != "BTCUSDT" else df
    eth = load_candles_research(db_path, "ETHUSDT", lookback_days) if config.symbol != "ETHUSDT" else df
    frame = add_long_c_features(add_features(df, btc, eth)).reset_index(drop=True)
    labels = compute_labels(frame["close"].to_numpy(float), frame["high"].to_numpy(float), frame["low"].to_numpy(float), frame, config.target, config.horizon)
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
    frame, labels, features, idx, feature_meta = prepare_symbol(config, Path(args.db_path), args.lookback_days, args.feature_mode)
    if frame is None or labels is None or len(features) == 0:
        summary = {"symbol": config.symbol, "family": config.family, "target": config.target, "horizon": config.horizon, "valid_folds": 0, "negative_folds": 0, "d1_status": "INSUFFICIENT_DATA", "score": -999, "recommendation": "no_data"}
        return summary, [], []
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
    if args.family != ALLOWED_FAMILY:
        raise SystemExit("LONG-D1 only allows --family pullback_continuation_long")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bad = [s for s in symbols if s not in SYMBOLS]
    if bad:
        raise SystemExit(f"Unsupported symbols: {','.join(bad)}")
    configs = []
    for symbol in symbols:
        if args.include_secondary and symbol in SECONDARY_CONFIGS:
            target, horizon = SECONDARY_CONFIGS[symbol]
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
    confirmed = [r for r in summaries if r.get("d1_status") == "LONG_D1_CONFIRMED"]
    mixed = [r for r in summaries if r.get("d1_status") == "LONG_D1_MIXED"]
    failed = [r for r in summaries if r.get("d1_status") in {"LONG_D1_FAILED", "LONG_D1_WEAK", "INSUFFICIENT_DATA"}]
    paths = {
        "md": str(out / f"aegis_long_d1_pullback_walkforward_{stamp}.md"),
        "json": str(out / f"aegis_long_d1_pullback_walkforward_{stamp}.json"),
        "summary": str(out / f"aegis_long_d1_pullback_summary_{stamp}.csv"),
        "folds": str(out / f"aegis_long_d1_pullback_folds_{stamp}.csv"),
        "ranking": str(out / f"aegis_long_d1_pullback_ranking_{stamp}.csv"),
        "confirmed": str(out / f"aegis_long_d1_pullback_confirmed_{stamp}.csv"),
        "mixed": str(out / f"aegis_long_d1_pullback_mixed_{stamp}.csv"),
        "failed": str(out / f"aegis_long_d1_pullback_failed_{stamp}.csv"),
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
        "args": vars(args),
        "summaries": summaries,
        "folds": folds,
        "buckets": buckets,
        "ranking": ranking,
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    best = ranking[0] if ranking else {}
    lines = [
        "# Aegis LONG-D1 Pullback Walk-forward",
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
        lines.append(f"| {row.get('symbol')} | {row.get('d1_status')} | {float(row.get('score') or 0):.5f} | {float(row.get('mean_hit_lift') or 0):.5f} | {float(row.get('mean_net_quality_lift_after_costs') or 0):.5f} | {float(row.get('latest_fold_hit_lift') or 0):.5f} | {float(row.get('latest_fold_net_quality_lift_after_costs') or 0):.5f} | {float(row.get('mean_p90_mae_delta') or 0):.5f} | {float(row.get('mean_stop_rate_delta') or 0):.5f} |")
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
        "- LONG_D1_CONFIRMED symbols can move to frozen/lockbox confirmation.",
        "- LONG_D1_MIXED symbols stay research and should be rerun with more data or tighter selection.",
        "- LONG_D1_FAILED/WEAK symbols should not advance for pullback_continuation_long.",
    ]
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_research_only_path(args.model_dir)
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
