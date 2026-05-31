#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import build_last_block_fold  # noqa: E402
from aegis_alpha.tools.final_repair_sol_link_short_m import _mean, _quantile  # noqa: E402
from aegis_alpha.tools.profile_failed_alpha_symbols import ALT_HIT_RULES  # noqa: E402,F401
from aegis_alpha.tools.research_link_short_alpha_n import (  # noqa: E402
    _feature_column,
    _feature_values,
    _first,
    _json_safe,
    _sparse_features,
    link_failed_retest_mask,
    link_pullback_rejection_mask,
)
from aegis_alpha.tools.train_short_alpha_family_l2_research import (  # noqa: E402
    _percentile_ranks,
    _top_fraction_mask,
    select_alpha_family_features,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_operable_edge_v2 import classification_metrics, safe_corr  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
SCHEMA_VERSION = "aegis_link_micro_roe_short_n1_v1"
LOCKBOX_MODES = ("last-block", "rolling-forward", "recent-only")
FEATURE_MODES = ("selected_family", "combined_v3_all")
ROE_CONFIGS = (
    {"target_roe": 0.06, "stop_roe": 0.04},
    {"target_roe": 0.08, "stop_roe": 0.05},
    {"target_roe": 0.10, "stop_roe": 0.06},
    {"target_roe": 0.12, "stop_roe": 0.08},
)
HORIZONS = (6, 12)
ENTRY_MODES = (
    "micro_quality_primary",
    "micro_quality_danger_filtered",
    "micro_hit_primary",
    "micro_consensus",
    "pullback_rejection_quality",
    "failed_retest_consensus",
    "top_bucket_only",
)
AVOID_MODES = (
    "avoid_by_micro_danger",
    "avoid_by_chop_reclaim",
    "avoid_by_fake_breakdown",
    "avoid_by_bad_micro_quality",
)
SLOW_TREND_PULLBACK_FEATURES = (
    "local_trend_down_score",
    "local_momentum_down_score",
    "local_chop_score",
    "btc_eth_long_contradiction",
    "btc_eth_short_agreement",
    "short_room_to_fall_12",
    "short_overhead_risk_12",
    "distance_ema25",
    "distance_ema99",
    "ema25_slope",
    "ema99_slope",
    "trend_efficiency_12",
    "trend_efficiency_24",
    "close_location_12",
    "upper_wick_ratio",
    "volume_ratio_12",
    "realized_vol_12",
    "realized_vol_24",
)
PULLBACK_REJECTION_FEATURES = (
    "pullback_strength_3",
    "pullback_strength_6",
    "pullback_volume_weakness",
    "upper_wick_ratio",
    "close_location_12",
    "local_trend_down_score",
    "btc_eth_long_contradiction",
    "short_room_to_fall_12",
    "short_overhead_risk_12",
)
FAILED_RETEST_FEATURES = (
    "short_breakdown_strength_12",
    "short_breakdown_retest_distance",
    "short_retest_failed",
    "short_retest_rejection_wick",
    "short_reclaim_range_risk",
    "short_failed_breakdown_risk_12",
    "short_lower_wick_sweep_risk",
    "local_chop_score",
    "btc_eth_long_contradiction",
)
MICRO_SCALP_FEATURES = (
    "realized_vol_12",
    "realized_vol_24",
    "range_expansion_12",
    "atr_ratio_14",
    "volume_ratio_12",
    "close_location_12",
    "upper_wick_ratio",
    "local_momentum_down_score",
    "btc_eth_long_contradiction",
    "short_room_to_fall_12",
)
AVOID_FEATURES = (
    "local_chop_score",
    "short_reclaim_range_risk",
    "short_failed_breakdown_risk_12",
    "short_lower_wick_sweep_risk",
    "btc_eth_long_contradiction",
    "short_adverse_rebound_risk",
    "upper_wick_ratio",
    "mae_danger_probability",
)
ALPHA_FEATURES = {
    "slow_trend_pullback_short": SLOW_TREND_PULLBACK_FEATURES,
    "pullback_rejection_short": PULLBACK_REJECTION_FEATURES,
    "failed_retest_short": FAILED_RETEST_FEATURES,
    "micro_scalp_short": MICRO_SCALP_FEATURES,
    "avoid_only_bad_short_filter": tuple(dict.fromkeys(AVOID_FEATURES + FAILED_RETEST_FEATURES + MICRO_SCALP_FEATURES)),
}
PROXY_FEATURES = {
    "pullback_strength_3": ("return_15m", "future_return_6", "close_location_12"),
    "pullback_strength_6": ("return_30m", "future_return_12", "close_location_12"),
    "pullback_volume_weakness": ("volume_ratio_12",),
    "realized_vol_12": ("realized_vol_24", "range_expansion_12"),
    "atr_ratio_14": ("realized_vol_24", "range_expansion_12"),
    "short_breakdown_retest_distance": ("short_breakdown_strength_12", "close_location_12"),
    "short_retest_failed": ("upper_wick_ratio", "short_reclaim_range_risk"),
    "short_retest_rejection_wick": ("upper_wick_ratio",),
    "short_adverse_rebound_risk": ("short_reclaim_range_risk", "upper_wick_ratio"),
}
CSV_COLUMNS = (
    "symbol", "side", "alpha_family", "feature_set", "feature_mode", "lookback_days",
    "leverage", "target_roe", "stop_roe", "horizon_candles", "decision_mode",
    "lockbox_mode", "fold", "train_samples", "validation_samples", "test_samples",
    "feature_count", "missing_features", "proxy_features_used", "sparse_features",
    "baseline_micro_hit_rate", "baseline_micro_quality", "baseline_micro_stop_rate",
    "baseline_p90_mae_roe", "baseline_avg_time_to_target", "baseline_ambiguous_rate",
    "selected_fraction", "selected_count", "selected_micro_hit_rate",
    "selected_micro_hit_lift", "selected_micro_quality",
    "selected_micro_quality_lift", "selected_net_micro_quality_lift",
    "selected_micro_stop_rate", "selected_micro_stop_delta", "selected_p90_mae_roe",
    "selected_p90_mae_delta", "selected_avg_time_to_target",
    "selected_avg_time_to_stop", "selected_ambiguous_rate", "selected_avg_mfe_roe",
    "selected_avg_mae_roe", "micro_hit_auc", "micro_hit_average_precision",
    "micro_quality_corr", "micro_danger_auc", "danger_filter_usefulness",
    "gross_edge_roe", "cost_proxy_roe", "cost_to_edge_ratio",
    "net_edge_after_costs", "avoid_selected_fraction", "avoid_selected_count",
    "avoid_quality_delta_vs_baseline", "avoid_stop_rate_delta",
    "avoid_p90_mae_delta", "avoid_hit_rate_delta", "avoid_usefulness_score",
    "n1_status", "n1_reason", "recommended_next_step",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _model_seed(symbol: str, family: str, target_roe: float, stop_roe: float, horizon: int, name: str) -> int:
    raw = f"{symbol}:{family}:{target_roe:.4f}:{stop_roe:.4f}:{horizon}:{name}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def _classifier(max_iter: int, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=random_state,
    )


def _regressor(max_iter: int, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=random_state,
    )


def compute_micro_roe_short_targets(
    market: Any,
    steps: np.ndarray,
    leverage: float,
    target_roe: float,
    stop_roe: float,
    horizon: int,
    *,
    fee_bps: float = 8.0,
    slippage_bps: float = 3.0,
) -> dict[str, np.ndarray]:
    target_move = float(target_roe) / max(float(leverage), 1e-12)
    stop_move = float(stop_roe) / max(float(leverage), 1e-12)
    cost_proxy_price = (float(fee_bps) + float(slippage_bps)) / 10000.0
    cost_proxy_roe = cost_proxy_price * float(leverage)
    hit: list[int] = []
    stop: list[int] = []
    mfe: list[float] = []
    mae: list[float] = []
    quality: list[float] = []
    time_to_target: list[float] = []
    time_to_stop: list[float] = []
    ambiguous: list[int] = []
    net_roe: list[float] = []
    for raw_step in np.asarray(steps, dtype=np.int64):
        step = int(raw_step)
        entry = float(market.close[step])
        highs = np.asarray(market.high[step + 1 : step + horizon + 1], dtype=np.float64)
        lows = np.asarray(market.low[step + 1 : step + horizon + 1], dtype=np.float64)
        if entry <= 0.0 or len(highs) != horizon or len(lows) != horizon:
            hit.append(0)
            stop.append(0)
            mfe.append(0.0)
            mae.append(0.0)
            quality.append(float(-cost_proxy_roe))
            time_to_target.append(-1.0)
            time_to_stop.append(-1.0)
            ambiguous.append(0)
            net_roe.append(float(-cost_proxy_roe))
            continue
        target_hits = lows <= entry * (1.0 - target_move)
        stop_hits = highs >= entry * (1.0 + stop_move)
        target_indices = np.flatnonzero(target_hits)
        stop_indices = np.flatnonzero(stop_hits)
        t_target = int(target_indices[0] + 1) if len(target_indices) else -1
        t_stop = int(stop_indices[0] + 1) if len(stop_indices) else -1
        same_candle = bool(np.any(target_hits & stop_hits))
        is_hit = bool(t_target >= 0 and (t_stop < 0 or t_target < t_stop))
        is_stop = bool(t_stop >= 0 and (t_target < 0 or t_stop <= t_target))
        price_mfe = max(0.0, float(np.max(1.0 - lows / entry)))
        price_mae = max(0.0, float(np.max(highs / entry - 1.0)))
        mfe_roe = price_mfe * float(leverage)
        mae_roe = price_mae * float(leverage)
        base = float(target_roe) if is_hit else (-float(stop_roe) if is_stop else 0.0)
        value = (
            base
            + min(mfe_roe, float(target_roe)) * 0.25
            - min(mae_roe, float(stop_roe) * 2.0) * 0.50
            - cost_proxy_roe
        )
        hit.append(int(is_hit))
        stop.append(int(is_stop))
        mfe.append(float(mfe_roe))
        mae.append(float(mae_roe))
        quality.append(float(np.clip(value, -1.0, 1.0)))
        time_to_target.append(float(t_target))
        time_to_stop.append(float(t_stop))
        ambiguous.append(int(same_candle))
        net_roe.append(float(base - cost_proxy_roe))
    return {
        "micro_hit_before_stop": np.asarray(hit, dtype=np.int8),
        "micro_stop_before_hit": np.asarray(stop, dtype=np.int8),
        "micro_mfe": np.asarray(mfe, dtype=np.float32),
        "micro_mae": np.asarray(mae, dtype=np.float32),
        "micro_trade_quality": np.asarray(quality, dtype=np.float32),
        "micro_time_to_target": np.asarray(time_to_target, dtype=np.float32),
        "micro_time_to_stop": np.asarray(time_to_stop, dtype=np.float32),
        "micro_ambiguous_hit_stop": np.asarray(ambiguous, dtype=np.int8),
        "micro_net_roe_after_costs": np.asarray(net_roe, dtype=np.float32),
        "target_price_move": np.asarray([target_move], dtype=np.float32),
        "stop_price_move": np.asarray([stop_move], dtype=np.float32),
        "cost_proxy_roe": np.asarray([cost_proxy_roe], dtype=np.float32),
    }


def select_link_micro_roe_features(
    dataset: dict[str, Any],
    alpha_family: str,
    feature_mode: str,
) -> dict[str, Any]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    if feature_mode == "combined_v3_all":
        selected = select_alpha_family_features(dataset, "slow_trend_short", "combined_v3_all")
        selected["alpha_family"] = alpha_family
        selected["proxy_features_used"] = []
        return selected
    names = np.asarray(dataset.get("feature_names", [])).astype(str)
    x = np.asarray(dataset["X"], dtype=np.float32)
    expected = ALPHA_FEATURES.get(alpha_family, ())
    index_by_name = {name: index for index, name in enumerate(names)}
    indices: list[int] = []
    output_names: list[str] = []
    missing: list[str] = []
    proxies: list[str] = []
    for name in expected:
        if name in index_by_name:
            indices.append(index_by_name[name])
            output_names.append(name)
            continue
        used_proxy = None
        for proxy in PROXY_FEATURES.get(name, ()):
            if proxy in index_by_name:
                used_proxy = proxy
                break
        if used_proxy is None:
            missing.append(name)
            continue
        indices.append(index_by_name[used_proxy])
        output_names.append(used_proxy)
        proxies.append(f"{name}<-{used_proxy}")
    unique_indices: list[int] = []
    unique_names: list[str] = []
    for index, name in zip(indices, output_names):
        if index not in unique_indices:
            unique_indices.append(index)
            unique_names.append(name)
    selected = dict(dataset)
    selected["X"] = x[:, unique_indices] if unique_indices else np.zeros((len(x), 0), dtype=np.float32)
    selected["feature_names"] = np.asarray(unique_names, dtype=str)
    selected["missing_family_features"] = missing
    selected["proxy_features_used"] = proxies
    selected["feature_mode"] = feature_mode
    selected["alpha_family"] = alpha_family
    selected["family_feature_count"] = int(len(unique_indices))
    return selected


def default_configs(include_avoid_only: bool, feature_mode: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for family in ("slow_trend_pullback_short", "pullback_rejection_short", "failed_retest_short", "micro_scalp_short"):
        for roe in ROE_CONFIGS:
            for horizon in HORIZONS:
                configs.append({
                    "symbol": "LINKUSDT",
                    "side": SIDE,
                    "alpha_family": family,
                    "feature_set": "combined_v3",
                    "feature_mode": feature_mode,
                    "lookback_days": 30,
                    "target_roe": float(roe["target_roe"]),
                    "stop_roe": float(roe["stop_roe"]),
                    "horizon_candles": int(horizon),
                    "decision_modes": list(ENTRY_MODES),
                })
    if include_avoid_only:
        for roe in ROE_CONFIGS:
            for horizon in HORIZONS:
                configs.append({
                    "symbol": "LINKUSDT",
                    "side": SIDE,
                    "alpha_family": "avoid_only_bad_short_filter",
                    "feature_set": "combined_v3",
                    "feature_mode": feature_mode,
                    "lookback_days": 30,
                    "target_roe": float(roe["target_roe"]),
                    "stop_roe": float(roe["stop_roe"]),
                    "horizon_candles": int(horizon),
                    "decision_modes": list(AVOID_MODES),
                })
    return configs


def _required_selected_count(test_samples: int) -> int:
    return min(30, max(1, int(math.ceil(float(test_samples) * 0.05))))


def _folds_for_mode(sample_count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.lockbox_mode == "rolling-forward":
        return temporal_folds(
            sample_count,
            fold_count=4,
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=float(args.lockbox_test_ratio),
            expanding_window=True,
            min_train_samples=int(args.min_train_samples),
            min_test_samples=int(args.min_test_samples),
        )
    fold = build_last_block_fold(
        sample_count,
        test_ratio=float(args.lockbox_test_ratio),
        min_train_samples=int(args.min_train_samples),
        min_test_samples=int(args.min_test_samples),
        recent_only=args.lockbox_mode == "recent-only",
    )
    return [fold] if fold is not None else []


def fit_micro_models(
    dataset: dict[str, Any],
    targets: dict[str, np.ndarray],
    split: dict[str, Any],
    config: dict[str, Any],
    max_iter: int,
) -> dict[str, Any]:
    x = np.asarray(dataset["X"], dtype=np.float32)
    train = np.asarray(split["train"], dtype=np.int64)
    validation = np.asarray(split["validation"], dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    base = {
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "test_samples": int(len(test)),
    }
    if x.ndim != 2 or x.shape[1] == 0:
        return {**base, "model_status": "missing_family_features"}
    if len(np.unique(targets["micro_hit_before_stop"][train])) < 2 or len(np.unique(targets["micro_stop_before_hit"][train])) < 2:
        return {**base, "model_status": "insufficient_class_diversity"}
    if float(np.std(targets["micro_trade_quality"][train])) <= 1e-12:
        return {**base, "model_status": "insufficient_quality_variance"}
    symbol = str(config["symbol"])
    family = str(config["alpha_family"])
    target_roe = float(config["target_roe"])
    stop_roe = float(config["stop_roe"])
    horizon = int(config["horizon_candles"])
    hit_model = _classifier(max_iter, _model_seed(symbol, family, target_roe, stop_roe, horizon, "micro_hit"))
    quality_model = _regressor(max_iter, _model_seed(symbol, family, target_roe, stop_roe, horizon, "micro_quality"))
    danger_model = _classifier(max_iter, _model_seed(symbol, family, target_roe, stop_roe, horizon, "micro_danger"))
    hit_model.fit(x[train], targets["micro_hit_before_stop"][train])
    quality_model.fit(x[train], targets["micro_trade_quality"][train])
    danger_model.fit(x[train], targets["micro_stop_before_hit"][train])
    hit_prob = hit_model.predict_proba(x[test])[:, 1]
    quality_pred = quality_model.predict(x[test])
    danger_prob = danger_model.predict_proba(x[test])[:, 1]
    hit_metrics = classification_metrics(targets["micro_hit_before_stop"][test], hit_prob)
    danger_metrics = classification_metrics(targets["micro_stop_before_hit"][test], danger_prob)
    return {
        **base,
        "model_status": "trained",
        "test": test,
        "micro_hit_prob": hit_prob,
        "micro_quality_pred": quality_pred,
        "micro_danger_prob": danger_prob,
        "micro_hit_auc": hit_metrics.get("roc_auc"),
        "micro_hit_average_precision": hit_metrics.get("average_precision"),
        "micro_quality_corr": safe_corr(quality_pred, targets["micro_trade_quality"][test]),
        "micro_danger_auc": danger_metrics.get("roc_auc"),
        "danger_filter_usefulness": safe_corr(danger_prob, targets["micro_stop_before_hit"][test]),
    }


def micro_decision_mask(mode: str, trained: dict[str, Any], dataset: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    hit = np.asarray(trained["micro_hit_prob"], dtype=np.float64)
    quality = np.asarray(trained["micro_quality_pred"], dtype=np.float64)
    danger = np.asarray(trained["micro_danger_prob"], dtype=np.float64)
    if mode == "micro_quality_primary":
        return _top_fraction_mask(quality), [], []
    if mode == "micro_quality_danger_filtered":
        return _top_fraction_mask(quality, eligible=danger <= np.quantile(danger, 0.80)), [], []
    if mode == "micro_hit_primary":
        return _top_fraction_mask(hit), [], []
    if mode == "micro_consensus":
        score = _percentile_ranks(hit) + _percentile_ranks(quality) - _percentile_ranks(danger)
        return _top_fraction_mask(score), [], []
    if mode == "top_bucket_only":
        return _top_fraction_mask(quality), [], []
    if mode == "pullback_rejection_quality":
        return link_pullback_rejection_mask(dataset, {
            **trained,
            "quality_pred": quality,
            "hit_prob": hit,
            "danger_prob": danger,
        })
    if mode == "failed_retest_consensus":
        return link_failed_retest_mask(dataset, {
            **trained,
            "quality_pred": quality,
            "hit_prob": hit,
            "danger_prob": danger,
        })
    if mode in AVOID_MODES:
        test = np.asarray(trained["test"], dtype=np.int64)
        values, missing = _feature_values(dataset, test, AVOID_FEATURES)
        sparse = _sparse_features(values, AVOID_FEATURES)
        score = _percentile_ranks(danger) + _percentile_ranks(-quality)
        fake = _first(values, "short_failed_breakdown_risk_12", "short_reclaim_range_risk")
        chop = values.get("local_chop_score")
        contradiction = values.get("btc_eth_long_contradiction")
        if mode == "avoid_by_chop_reclaim":
            if chop is not None:
                score += _percentile_ranks(chop)
            if fake is not None:
                score += _percentile_ranks(fake)
        elif mode == "avoid_by_fake_breakdown":
            if fake is not None:
                score += _percentile_ranks(fake) * 1.4
        elif mode == "avoid_by_bad_micro_quality":
            score += _percentile_ranks(-quality) * 1.2
        elif mode == "avoid_by_micro_danger":
            score += _percentile_ranks(danger) * 1.2
        if contradiction is not None:
            score += _percentile_ranks(contradiction) * 0.5
        return _top_fraction_mask(score, fraction=0.20), missing, sparse
    raise ValueError(f"unsupported micro decision mode: {mode}")


def selection_stats(trained: dict[str, Any], targets: dict[str, np.ndarray], mask: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    test = np.asarray(trained["test"], dtype=np.int64)
    hit = targets["micro_hit_before_stop"][test]
    stop = targets["micro_stop_before_hit"][test]
    quality = targets["micro_trade_quality"][test]
    mae = targets["micro_mae"][test]
    mfe = targets["micro_mfe"][test]
    time_target = targets["micro_time_to_target"][test]
    time_stop = targets["micro_time_to_stop"][test]
    ambiguous = targets["micro_ambiguous_hit_stop"][test]
    selected_count = int(mask.sum())
    baseline_hit = _mean(hit)
    baseline_quality = _mean(quality)
    baseline_stop = _mean(stop)
    baseline_p90 = _quantile(mae, 0.90)
    baseline_time = _mean(time_target[time_target > 0])
    baseline_ambiguous = _mean(ambiguous)
    selected_hit = _mean(hit[mask]) if selected_count else None
    selected_quality = _mean(quality[mask]) if selected_count else None
    selected_stop = _mean(stop[mask]) if selected_count else None
    selected_p90 = _quantile(mae[mask], 0.90) if selected_count else None
    selected_time_target = _mean(time_target[mask][time_target[mask] > 0]) if selected_count else None
    selected_time_stop = _mean(time_stop[mask][time_stop[mask] > 0]) if selected_count else None
    quality_lift = None if selected_quality is None or baseline_quality is None else selected_quality - baseline_quality
    cost_proxy_roe = float(targets["cost_proxy_roe"][0])
    gross_edge = quality_lift
    net_edge = None if gross_edge is None else gross_edge - cost_proxy_roe
    cost_ratio = None if gross_edge is None or gross_edge <= 0.0 else cost_proxy_roe / max(gross_edge, 1e-12)
    return {
        "baseline_micro_hit_rate": baseline_hit,
        "baseline_micro_quality": baseline_quality,
        "baseline_micro_stop_rate": baseline_stop,
        "baseline_p90_mae_roe": baseline_p90,
        "baseline_avg_time_to_target": baseline_time,
        "baseline_ambiguous_rate": baseline_ambiguous,
        "selected_fraction": selected_count / max(1, len(test)),
        "selected_count": selected_count,
        "selected_micro_hit_rate": selected_hit,
        "selected_micro_hit_lift": None if selected_hit is None or baseline_hit is None else selected_hit - baseline_hit,
        "selected_micro_quality": selected_quality,
        "selected_micro_quality_lift": quality_lift,
        "selected_net_micro_quality_lift": net_edge,
        "selected_micro_stop_rate": selected_stop,
        "selected_micro_stop_delta": None if selected_stop is None or baseline_stop is None else selected_stop - baseline_stop,
        "selected_p90_mae_roe": selected_p90,
        "selected_p90_mae_delta": None if selected_p90 is None or baseline_p90 is None else selected_p90 - baseline_p90,
        "selected_avg_time_to_target": selected_time_target,
        "selected_avg_time_to_stop": selected_time_stop,
        "selected_ambiguous_rate": _mean(ambiguous[mask]) if selected_count else None,
        "selected_avg_mfe_roe": _mean(mfe[mask]) if selected_count else None,
        "selected_avg_mae_roe": _mean(mae[mask]) if selected_count else None,
        "gross_edge_roe": gross_edge,
        "cost_proxy_roe": cost_proxy_roe,
        "cost_to_edge_ratio": cost_ratio,
        "net_edge_after_costs": net_edge,
    }


def avoid_stats(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "avoid_selected_fraction": row.get("selected_fraction"),
        "avoid_selected_count": row.get("selected_count"),
        "avoid_quality_delta_vs_baseline": row.get("selected_micro_quality_lift"),
        "avoid_stop_rate_delta": row.get("selected_micro_stop_delta"),
        "avoid_p90_mae_delta": row.get("selected_p90_mae_delta"),
        "avoid_hit_rate_delta": row.get("selected_micro_hit_lift"),
        "avoid_usefulness_score": (
            -(finite(row.get("selected_micro_quality_lift"), 0.0) or 0.0)
            + (finite(row.get("selected_micro_stop_delta"), 0.0) or 0.0)
            + (finite(row.get("selected_p90_mae_delta"), 0.0) or 0.0)
            - min(finite(row.get("selected_micro_hit_lift"), 0.0) or 0.0, 0.0)
        ),
    }


def classify_link_micro_roe_candidate(row: dict[str, Any]) -> str:
    if (
        row.get("model_status") != "trained"
        or int(row.get("test_samples") or 0) <= 0
        or int(row.get("feature_count") or 0) <= 0
    ):
        return "LINK_MICRO_ROE_INSUFFICIENT_DATA"
    mode = str(row.get("decision_mode"))
    if mode in AVOID_MODES:
        fraction = finite(row.get("avoid_selected_fraction"), 0.0) or 0.0
        confirmed = (
            0.05 <= fraction <= 0.30
            and (finite(row.get("avoid_quality_delta_vs_baseline"), 0.0) or 0.0) < 0.0
            and (finite(row.get("avoid_stop_rate_delta"), 0.0) or 0.0) > 0.0
            and (finite(row.get("avoid_p90_mae_delta"), 0.0) or 0.0) > 0.0
            and (finite(row.get("avoid_hit_rate_delta"), 0.0) or 0.0) <= 0.0
            and (finite(row.get("avoid_usefulness_score"), 0.0) or 0.0) > 0.0
        )
        return "LINK_MICRO_ROE_AVOID_ONLY" if confirmed else "LINK_MICRO_ROE_WEAK"
    selected_count = int(row.get("selected_count") or 0)
    test_samples = int(row.get("test_samples") or 0)
    hit_lift = finite(row.get("selected_micro_hit_lift"), 0.0) or 0.0
    quality_lift = finite(row.get("selected_micro_quality_lift"), 0.0) or 0.0
    net_lift = finite(row.get("selected_net_micro_quality_lift"), 0.0) or 0.0
    stop_rate = finite(row.get("selected_micro_stop_rate"), float("inf")) or float("inf")
    baseline_stop = finite(row.get("baseline_micro_stop_rate"), 0.0) or 0.0
    selected_p90 = finite(row.get("selected_p90_mae_roe"), float("inf")) or float("inf")
    baseline_p90 = finite(row.get("baseline_p90_mae_roe"), 0.0) or 0.0
    time_target = finite(row.get("selected_avg_time_to_target"), float("inf")) or float("inf")
    horizon = finite(row.get("horizon_candles"), 0.0) or 0.0
    cost_ratio = finite(row.get("cost_to_edge_ratio"), float("inf")) or float("inf")
    ambiguity = finite(row.get("selected_ambiguous_rate"), float("inf")) or float("inf")
    baseline_ambiguity = finite(row.get("baseline_ambiguous_rate"), 0.0) or 0.0
    auc = finite(row.get("micro_hit_auc"), 0.0) or 0.0
    corr = finite(row.get("micro_quality_corr"), 0.0) or 0.0
    fraction = finite(row.get("selected_fraction"), 0.0) or 0.0
    if (
        net_lift <= 0.0
        or hit_lift <= 0.0
        or stop_rate > max(1e-12, baseline_stop) * 1.35
        or cost_ratio > 0.70
        or selected_p90 > max(1e-12, baseline_p90) * 1.30
    ):
        return "LINK_MICRO_ROE_FAILED"
    promising = (
        selected_count >= _required_selected_count(test_samples)
        and hit_lift > 0.0
        and quality_lift > 0.0
        and net_lift > 0.0
        and stop_rate <= baseline_stop * 1.05
        and selected_p90 <= baseline_p90
        and time_target <= horizon * 0.65
        and cost_ratio <= 0.40
        and ambiguity <= max(baseline_ambiguity * 1.10, baseline_ambiguity + 0.01)
        and (auc > 0.53 or corr > 0.03)
        and 0.05 <= fraction <= 0.20
    )
    if promising:
        return "LINK_MICRO_ROE_PROMISING"
    return "LINK_MICRO_ROE_WEAK"


def n1_reason(row: dict[str, Any]) -> str:
    status = str(row.get("n1_status"))
    if status == "LINK_MICRO_ROE_PROMISING":
        return "discovery_micro_roe_edge_requires_frozen_confirmation"
    if status == "LINK_MICRO_ROE_AVOID_ONLY":
        return "micro_roe_avoid_filter_detects_bad_zone_not_entry"
    if status == "LINK_MICRO_ROE_WEAK":
        return "partial_micro_roe_edge_but_gates_not_all_met"
    if status == "LINK_MICRO_ROE_FAILED":
        return "micro_roe_edge_or_risk_gate_failed"
    return "insufficient_samples_features_or_target_diversity"


def recommended_next_step(status: str) -> str:
    if status == "LINK_MICRO_ROE_PROMISING":
        return "prepare_phase_n2_frozen_confirmation_no_shadow_yet"
    if status == "LINK_MICRO_ROE_AVOID_ONLY":
        return "consider_future_avoid_or_reducer_metadata_shadow_not_entry"
    if status == "LINK_MICRO_ROE_WEAK":
        return "continue_research_no_shadow_entry_artifact"
    if status == "LINK_MICRO_ROE_FAILED":
        return "turn_off_link_entry_model_for_now"
    return "collect_more_data_before_decision"


def evaluate_mode(
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    dataset: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    *,
    fold_index: int,
    lockbox_mode: str,
) -> dict[str, Any]:
    base = {
        **config,
        "decision_mode": mode,
        "fold": fold_index,
        "lockbox_mode": lockbox_mode,
        "model_status": trained.get("model_status"),
        "train_samples": trained.get("train_samples"),
        "validation_samples": trained.get("validation_samples"),
        "test_samples": trained.get("test_samples"),
        "feature_count": int(np.asarray(dataset.get("X", np.zeros((0, 0)))).shape[1]),
        "missing_features": ",".join(dataset.get("missing_family_features", [])),
        "proxy_features_used": ",".join(dataset.get("proxy_features_used", [])),
        "sparse_features": "",
        "micro_hit_auc": trained.get("micro_hit_auc"),
        "micro_hit_average_precision": trained.get("micro_hit_average_precision"),
        "micro_quality_corr": trained.get("micro_quality_corr"),
        "micro_danger_auc": trained.get("micro_danger_auc"),
        "danger_filter_usefulness": trained.get("danger_filter_usefulness"),
    }
    if trained.get("model_status") != "trained":
        row = {**base, "n1_status": "LINK_MICRO_ROE_INSUFFICIENT_DATA"}
        row["n1_reason"] = n1_reason(row)
        row["recommended_next_step"] = recommended_next_step(row["n1_status"])
        return row
    mask, missing, sparse = micro_decision_mask(mode, trained, dataset)
    row = {**base, **selection_stats(trained, targets, mask, config)}
    combined_missing = list(dict.fromkeys(list(dataset.get("missing_family_features", [])) + missing))
    row["missing_features"] = ",".join(combined_missing)
    row["sparse_features"] = ",".join(sparse)
    if mode in AVOID_MODES:
        row.update(avoid_stats(row))
    else:
        row.update({
            "avoid_selected_fraction": None,
            "avoid_selected_count": None,
            "avoid_quality_delta_vs_baseline": None,
            "avoid_stop_rate_delta": None,
            "avoid_p90_mae_delta": None,
            "avoid_hit_rate_delta": None,
            "avoid_usefulness_score": None,
        })
    row["n1_status"] = classify_link_micro_roe_candidate(row)
    row["n1_reason"] = n1_reason(row)
    row["recommended_next_step"] = recommended_next_step(row["n1_status"])
    return row


def evaluate_config(config: dict[str, Any], dataset: dict[str, Any], targets: dict[str, np.ndarray], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = _folds_for_mode(len(np.asarray(dataset["X"])), args)
    if not splits:
        for mode in config["decision_modes"]:
            row = {
                **config,
                "decision_mode": mode,
                "lockbox_mode": args.lockbox_mode,
                "model_status": "insufficient_split_samples",
                "test_samples": 0,
                "feature_count": int(np.asarray(dataset["X"]).shape[1]),
                "n1_status": "LINK_MICRO_ROE_INSUFFICIENT_DATA",
            }
            row["n1_reason"] = n1_reason(row)
            row["recommended_next_step"] = recommended_next_step(row["n1_status"])
            rows.append(row)
        return rows
    for fold_index, split in enumerate(splits, start=1):
        trained = fit_micro_models(
            dataset,
            targets,
            split,
            config,
            max_iter=60 if args.fast else 120,
        )
        for mode in config["decision_modes"]:
            rows.append(evaluate_mode(
                trained,
                targets,
                dataset,
                config,
                mode,
                fold_index=fold_index,
                lockbox_mode=args.lockbox_mode,
            ))
    return rows


def select_best_link_micro_roe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {
        "LINK_MICRO_ROE_PROMISING": 5,
        "LINK_MICRO_ROE_AVOID_ONLY": 4,
        "LINK_MICRO_ROE_WEAK": 3,
        "LINK_MICRO_ROE_FAILED": 2,
        "LINK_MICRO_ROE_INSUFFICIENT_DATA": 1,
    }

    def score(row: dict[str, Any]) -> tuple[float, ...]:
        status = str(row.get("n1_status"))
        if status == "LINK_MICRO_ROE_AVOID_ONLY":
            primary = finite(row.get("avoid_usefulness_score"), -999.0) or -999.0
            cost = 0.0
        else:
            primary = finite(row.get("selected_net_micro_quality_lift"), -999.0) or -999.0
            cost = -(finite(row.get("cost_to_edge_ratio"), 999.0) or 999.0)
        return (
            priority.get(status, 0),
            primary,
            cost,
            -(finite(row.get("selected_p90_mae_delta"), 999.0) or 999.0),
            -(finite(row.get("selected_avg_time_to_target"), 999.0) or 999.0),
            finite(row.get("micro_hit_auc"), -999.0) or -999.0,
            -(finite(row.get("target_roe"), 999.0) or 999.0),
            -(finite(row.get("horizon_candles"), 999.0) or 999.0),
        )

    return dict(max(rows, key=score)) if rows else {
        "symbol": "LINKUSDT",
        "side": SIDE,
        "n1_status": "LINK_MICRO_ROE_INSUFFICIENT_DATA",
        "n1_reason": "no_rows_evaluated",
        "recommended_next_step": "collect_more_data_before_decision",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _num(value: Any) -> str:
    number = finite(value)
    return "null" if number is None else f"{number:.4f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["best_link_micro_roe_result"]
    lines = [
        f"# Aegis LINK Micro-ROE SHORT N.1 {report['created_at']}",
        "",
        "## Safety",
        "",
        "- `RESEARCH_ONLY`.",
        "- No shadow artifacts are generated.",
        "- No active models, active manifests or live inference are changed.",
        "",
        "## Micro-ROE Assumption",
        "",
        f"- Leverage: `{report['leverage']}x`.",
        "- ROE targets: `6%`, `8%`, `10%`, `12%`.",
        "- Stop ROE: `4%`, `5%`, `6%`, `8%`.",
        f"- Cost proxy in ROE: `{_num(best.get('cost_proxy_roe'))}`.",
        "",
        "## Best LINK Micro-ROE Result",
        "",
        "| Family | Target ROE | Stop ROE | Horizon | Mode | Status | Hit Lift | Quality Lift | Net Lift | Stop Delta | P90 MAE Delta | Time To Target | Cost/Edge | Reason |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {best.get('alpha_family')} | {_num(best.get('target_roe'))} | {_num(best.get('stop_roe'))} | "
            f"{best.get('horizon_candles')} | {best.get('decision_mode')} | {best.get('n1_status')} | "
            f"{_num(best.get('selected_micro_hit_lift'))} | {_num(best.get('selected_micro_quality_lift'))} | "
            f"{_num(best.get('selected_net_micro_quality_lift'))} | {_num(best.get('selected_micro_stop_delta'))} | "
            f"{_num(best.get('selected_p90_mae_delta'))} | {_num(best.get('selected_avg_time_to_target'))} | "
            f"{_num(best.get('cost_to_edge_ratio'))} | {best.get('n1_reason')} |"
        ),
    ]
    for key, title in (
        ("promising", "Promising"),
        ("weak", "Weak"),
        ("failed", "Failed"),
        ("avoid_only", "Avoid-Only"),
    ):
        lines.extend(["", f"## {title}", ""])
        values = report[key]
        if not values:
            lines.append("- None.")
            continue
        lines.append("| Family | Target | Stop | Horizon | Mode | Status | Net Lift | Cost/Edge |")
        lines.append("|---|---:|---:|---:|---|---|---:|---:|")
        for row in values[:25]:
            lines.append(
                f"| {row.get('alpha_family')} | {_num(row.get('target_roe'))} | {_num(row.get('stop_roe'))} | "
                f"{row.get('horizon_candles')} | {row.get('decision_mode')} | {row.get('n1_status')} | "
                f"{_num(row.get('selected_net_micro_quality_lift'))} | {_num(row.get('cost_to_edge_ratio'))} |"
            )
    recommendation = "LINK apagado por ahora."
    if best.get("n1_status") == "LINK_MICRO_ROE_PROMISING":
        recommendation = "Preparar Fase N.2 frozen confirmation; no shadow todavia."
    elif best.get("n1_status") == "LINK_MICRO_ROE_AVOID_ONLY":
        recommendation = "Mantener fuera de entry y considerar avoid/reducer futuro."
    elif best.get("n1_status") == "LINK_MICRO_ROE_WEAK":
        recommendation = "Mantener LINK en research-only."
    lines.extend(["", "## Recommendation", "", f"- {recommendation}"])
    if report["errors"]:
        lines.extend(["", "## Errors", "", f"- `{report['errors']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbol = normalize_turbo_symbol(str(args.symbol))
    if symbol != "LINKUSDT":
        raise ValueError(f"Fase N.1 only supports LINKUSDT: {symbol}")
    configs = default_configs(bool(args.include_avoid_only), str(args.feature_mode))
    context_markets: dict[str, Any] = {}
    context_warning: str | None = None
    if not bool(args.disable_cross_context):
        try:
            context_markets = {
                context_symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=context_symbol)
                for context_symbol in ("BTCUSDT", "ETHUSDT")
            }
        except Exception as exc:
            context_warning = f"cross_context_unavailable:{exc!r}"
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, int], tuple[Any, dict[str, Any]]] = {}
    for config in configs:
        try:
            config["leverage"] = float(args.leverage)
            cache_key = (config["symbol"], int(config["lookback_days"]))
            if cache_key not in data_cache:
                market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
                base = build_recent_dataset(config["symbol"], int(config["lookback_days"]), save=False, market=market)["dataset"]
                combined = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
                data_cache[cache_key] = (market, combined)
            market, combined = data_cache[cache_key]
            dataset = select_link_micro_roe_features(combined, config["alpha_family"], config["feature_mode"])
            targets = compute_micro_roe_short_targets(
                market,
                np.asarray(combined["step"], dtype=np.int64),
                float(args.leverage),
                float(config["target_roe"]),
                float(config["stop_roe"]),
                int(config["horizon_candles"]),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            all_rows.extend(evaluate_config(config, dataset, targets, args))
        except Exception as exc:
            errors.append({**config, "error": repr(exc)})
            for mode in config["decision_modes"]:
                all_rows.append({
                    **config,
                    "decision_mode": mode,
                    "lockbox_mode": args.lockbox_mode,
                    "model_status": "evaluation_error",
                    "test_samples": 0,
                    "feature_count": 0,
                    "n1_status": "LINK_MICRO_ROE_INSUFFICIENT_DATA",
                    "n1_reason": "configuration_evaluation_error",
                    "recommended_next_step": "collect_more_data_before_decision",
                    "config_error": repr(exc),
                })
    best = select_best_link_micro_roe(all_rows)
    promising = [row for row in all_rows if row.get("n1_status") == "LINK_MICRO_ROE_PROMISING"]
    weak = [row for row in all_rows if row.get("n1_status") == "LINK_MICRO_ROE_WEAK"]
    failed = [row for row in all_rows if row.get("n1_status") in {"LINK_MICRO_ROE_FAILED", "LINK_MICRO_ROE_INSUFFICIENT_DATA"}]
    avoid_only = [row for row in all_rows if row.get("n1_status") == "LINK_MICRO_ROE_AVOID_ONLY"]
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_link_micro_roe_n1_{stamp}.md",
        "json": out_dir / f"aegis_link_micro_roe_n1_{stamp}.json",
        "all_configs_csv": out_dir / f"aegis_link_micro_roe_n1_all_configs_{stamp}.csv",
        "best_csv": out_dir / f"aegis_link_micro_roe_n1_best_{stamp}.csv",
        "promising_csv": out_dir / f"aegis_link_micro_roe_n1_promising_{stamp}.csv",
        "weak_csv": out_dir / f"aegis_link_micro_roe_n1_weak_{stamp}.csv",
        "failed_csv": out_dir / f"aegis_link_micro_roe_n1_failed_{stamp}.csv",
        "avoid_only_csv": out_dir / f"aegis_link_micro_roe_n1_avoid_only_{stamp}.csv",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "symbol": "LINKUSDT",
        "side": SIDE,
        "leverage": float(args.leverage),
        "lockbox_mode": args.lockbox_mode,
        "feature_mode": args.feature_mode,
        "context_warning": context_warning,
        "configs": configs,
        "all_config_rows": all_rows,
        "best_link_micro_roe_result": best,
        "promising": promising,
        "weak": weak,
        "failed": failed,
        "avoid_only": avoid_only,
        "status_counts": dict(Counter(str(row.get("n1_status")) for row in all_rows)),
        "errors": errors,
        "models_trained_in_memory_only": True,
        "model_artifacts_written": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "yaml_changed": False,
        "thresholds_changed": False,
        "pm2_touched": False,
        "orders_sent": False,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    safe_report = _json_safe(report)
    paths["json"].write_text(json.dumps(safe_report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(paths["md"], safe_report)
    _write_csv(paths["all_configs_csv"], all_rows)
    _write_csv(paths["best_csv"], [best])
    _write_csv(paths["promising_csv"], promising)
    _write_csv(paths["weak_csv"], weak)
    _write_csv(paths["failed_csv"], failed)
    _write_csv(paths["avoid_only_csv"], avoid_only)
    return safe_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only LINKUSDT SHORT micro-ROE scalp N.1.")
    parser.add_argument("--symbol", default="LINKUSDT")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--lockbox-mode", choices=LOCKBOX_MODES, default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="selected_family")
    parser.add_argument("--leverage", type=float, default=20.0)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--disable-cross-context", action="store_true")
    parser.add_argument("--include-avoid-only", action="store_true")
    args = parser.parse_args()
    report = run(args)
    best = report["best_link_micro_roe_result"]
    print(json.dumps({
        "paths": report["paths"],
        "best_link_micro_roe_result": {
            "symbol": best.get("symbol"),
            "family": best.get("alpha_family"),
            "target_roe": best.get("target_roe"),
            "stop_roe": best.get("stop_roe"),
            "horizon": best.get("horizon_candles"),
            "mode": best.get("decision_mode"),
            "status": best.get("n1_status"),
        },
        "status_counts": report["status_counts"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
