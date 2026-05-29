#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import build_last_block_fold  # noqa: E402
from aegis_alpha.tools.final_repair_sol_link_short_m import _selection_stats  # noqa: E402
from aegis_alpha.tools.profile_failed_alpha_symbols import ALT_HIT_RULES  # noqa: E402,F401
from aegis_alpha.tools.train_short_alpha_family_l2_research import (  # noqa: E402
    _fit_fold_predictions,
    _percentile_ranks,
    _top_fraction_mask,
    compute_alpha_target_arrays,
    select_alpha_family_features,
    selection_mask as l2_selection_mask,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_operable_edge_v2 import safe_corr  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
SCHEMA_VERSION = "aegis_link_short_alpha_n_v1"
LOCKBOX_MODES = ("last-block", "rolling-forward", "recent-only")
FEATURE_MODES = ("selected_family", "combined_v3_all")
ENTRY_MODES = (
    "quality_primary",
    "quality_primary_danger_filtered",
    "trend_confirmed_quality",
    "pullback_rejection_quality",
    "failed_retest_quality",
    "failed_retest_consensus",
    "top_bucket_only",
    "hit_primary",
)
AVOID_MODES = (
    "avoid_by_danger_quality",
    "avoid_by_fake_breakdown",
    "avoid_by_chop_reclaim",
    "avoid_by_retest_failure_noise",
)
SLOW_TREND_PULLBACK_FEATURES = (
    "local_trend_down_score",
    "local_momentum_down_score",
    "local_chop_score",
    "btc_eth_long_contradiction",
    "btc_eth_short_agreement",
    "symbol_vs_btc_relative_strength_30m",
    "symbol_vs_eth_relative_strength_30m",
    "short_room_to_fall_12",
    "short_room_to_fall_24",
    "short_overhead_risk_12",
    "short_overhead_risk_24",
    "distance_ema25",
    "distance_ema99",
    "ema25_slope",
    "ema99_slope",
    "trend_efficiency_24",
    "trend_efficiency_64",
    "close_location_12",
    "close_location_24",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volume_ratio_12",
    "realized_vol_24",
)
FAILED_RETEST_FEATURES = (
    "short_breakdown_strength_12",
    "short_breakdown_strength_24",
    "short_breakdown_retest_distance",
    "short_retest_failed",
    "short_retest_success",
    "short_retest_volume_dryup",
    "short_retest_rejection_wick",
    "short_close_back_inside_range",
    "short_reclaim_range_risk",
    "short_failed_breakdown_risk_12",
    "short_failed_breakdown_risk_24",
    "short_lower_wick_sweep_risk",
    "upper_wick_ratio",
    "close_location_12",
    "close_location_24",
    "local_chop_score",
    "btc_eth_long_contradiction",
    "short_room_to_fall_12",
    "short_overhead_risk_12",
)
AVOID_FEATURES = (
    "mae_danger_probability",
    "short_failed_breakdown_risk_12",
    "short_failed_breakdown_risk_24",
    "short_reclaim_range_risk",
    "short_lower_wick_sweep_risk",
    "local_chop_score",
    "btc_eth_long_contradiction",
    "short_extension_below_ema21",
    "short_extension_below_ema200",
    "short_volume_climax_risk",
    "short_absorption_risk",
    "short_adverse_rebound_risk",
    "short_retest_success",
    "short_retest_failed",
)
ALPHA_FEATURES = {
    "slow_trend_pullback_short": SLOW_TREND_PULLBACK_FEATURES,
    "failed_retest_short": FAILED_RETEST_FEATURES,
    "avoid_only_bad_short_filter": tuple(dict.fromkeys(AVOID_FEATURES + FAILED_RETEST_FEATURES + SLOW_TREND_PULLBACK_FEATURES)),
}
CSV_COLUMNS = (
    "symbol", "side", "alpha_family", "feature_set", "feature_mode", "lookback_days",
    "target_name", "horizon_candles", "decision_mode", "lockbox_mode", "fold",
    "train_samples", "validation_samples", "test_samples", "valid_fold_count",
    "feature_count", "missing_features", "sparse_features",
    "baseline_target_hit_rate", "baseline_trade_quality", "baseline_mae_danger",
    "baseline_p90_mae", "selected_fraction", "selected_count",
    "selected_target_hit_rate", "selected_target_lift", "selected_quality_mean",
    "selected_quality_lift", "selected_net_quality_lift", "selected_mae_danger_rate",
    "selected_mae_danger_delta", "selected_p90_mae", "selected_p90_mae_delta",
    "selected_avg_mfe", "selected_avg_mae", "target_auc",
    "target_average_precision", "quality_corr", "danger_auc",
    "danger_filter_usefulness", "avoid_selected_fraction", "avoid_selected_count",
    "avoid_quality_mean", "avoid_quality_delta_vs_baseline",
    "avoid_mae_danger_rate", "avoid_mae_danger_delta", "avoid_p90_mae",
    "avoid_p90_mae_delta", "avoid_hit_rate", "avoid_hit_delta",
    "avoid_usefulness_score", "n_status", "n_reason", "recommended_next_step",
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


def _mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else None


def _quantile(values: np.ndarray, q: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _feature_column(dataset: dict[str, Any], name: str, test: np.ndarray) -> np.ndarray | None:
    names = np.asarray(dataset.get("feature_names", [])).astype(str)
    matches = np.flatnonzero(names == name)
    if not len(matches):
        return None
    return np.asarray(dataset["X"], dtype=np.float32)[test, int(matches[0])]


def _feature_values(dataset: dict[str, Any], test: np.ndarray, names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str]]:
    values: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in names:
        column = _feature_column(dataset, name, test)
        if column is None:
            missing.append(name)
        else:
            values[name] = np.asarray(column, dtype=np.float64)
    return values, missing


def _safe_rank(values: np.ndarray | None, default: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(0, default, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return array
    finite_array = np.where(np.isfinite(array), array, np.nanmedian(array[np.isfinite(array)]) if np.any(np.isfinite(array)) else default)
    return _percentile_ranks(finite_array)


def _first(values: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def _sparse_features(values: dict[str, np.ndarray], names: tuple[str, ...]) -> list[str]:
    sparse: list[str] = []
    for name in names:
        column = values.get(name)
        if column is None or not len(column):
            continue
        finite_column = column[np.isfinite(column)]
        if len(finite_column) == 0:
            sparse.append(name)
        elif np.unique(np.round(finite_column, 8)).size <= 1:
            sparse.append(name)
        elif float(np.mean(np.abs(finite_column) > 1e-12)) < 0.02:
            sparse.append(name)
    return sparse


def _required_selected_count(test_samples: int) -> int:
    return min(30, max(1, int(math.ceil(float(test_samples) * 0.05))))


def default_entry_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    modes_by_family = {
        "slow_trend_pullback_short": (
            "quality_primary",
            "quality_primary_danger_filtered",
            "trend_confirmed_quality",
            "pullback_rejection_quality",
            "top_bucket_only",
            "hit_primary",
        ),
        "failed_retest_short": (
            "failed_retest_quality",
            "failed_retest_consensus",
            "quality_primary",
            "quality_primary_danger_filtered",
            "top_bucket_only",
            "hit_primary",
        ),
    }
    for family, modes in modes_by_family.items():
        for target_name in ("hit3_before_minus2", "hit5_before_minus3"):
            for horizon in (12, 24):
                configs.append({
                    "symbol": "LINKUSDT",
                    "side": SIDE,
                    "alpha_family": family,
                    "feature_set": "combined_v3",
                    "feature_mode": "selected_family",
                    "lookback_days": 30,
                    "target_name": target_name,
                    "horizon_candles": horizon,
                    "decision_modes": list(modes),
                })
    return configs


def default_avoid_configs() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "LINKUSDT",
            "side": SIDE,
            "alpha_family": "avoid_only_bad_short_filter",
            "feature_set": "combined_v3",
            "feature_mode": "selected_family",
            "lookback_days": 30,
            "target_name": "hit3_before_minus2",
            "horizon_candles": 12,
            "decision_modes": [mode],
        }
        for mode in AVOID_MODES
    ]


def configs_for_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    symbol = normalize_turbo_symbol(str(args.symbol))
    if symbol != "LINKUSDT":
        raise ValueError(f"Fase N is LINKUSDT-only: {symbol}")
    if args.feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature_mode: {args.feature_mode}")
    configs = default_entry_configs()
    if bool(args.include_avoid_only):
        configs.extend(default_avoid_configs())
    for config in configs:
        config["feature_mode"] = args.feature_mode
    return configs


def link_slow_trend_pullback_mask(dataset: dict[str, Any], trained: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    test = np.asarray(trained["test"], dtype=np.int64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    values, missing = _feature_values(dataset, test, SLOW_TREND_PULLBACK_FEATURES)
    required = ("local_trend_down_score", "btc_eth_long_contradiction", "short_room_to_fall_12")
    missing_required = [name for name in required if name not in values]
    sparse = _sparse_features(values, SLOW_TREND_PULLBACK_FEATURES)
    if missing_required:
        return _top_fraction_mask(quality), missing, sparse
    trend = values["local_trend_down_score"]
    contradiction = values["btc_eth_long_contradiction"]
    room = np.maximum(values["short_room_to_fall_12"], values.get("short_room_to_fall_24", values["short_room_to_fall_12"]))
    chop = values.get("local_chop_score")
    momentum = values.get("local_momentum_down_score", trend)
    eligible = (
        trend >= np.nanquantile(trend, 0.50)
    ) & (
        momentum >= np.nanquantile(momentum, 0.45)
    ) & (
        contradiction <= np.nanquantile(contradiction, 0.65)
    ) & (
        room >= np.nanquantile(room, 0.50)
    )
    if chop is not None:
        eligible &= chop <= np.nanquantile(chop, 0.80)
    mask = _top_fraction_mask(quality, fraction=0.10, eligible=eligible)
    if not np.any(mask):
        return _top_fraction_mask(quality), missing + ["slow_trend_pullback_no_eligible_rows"], sparse
    return mask, missing, sparse


def link_pullback_rejection_mask(dataset: dict[str, Any], trained: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    test = np.asarray(trained["test"], dtype=np.int64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    values, missing = _feature_values(dataset, test, SLOW_TREND_PULLBACK_FEATURES)
    sparse = _sparse_features(values, SLOW_TREND_PULLBACK_FEATURES)
    trend = values.get("local_trend_down_score")
    upper = values.get("upper_wick_ratio")
    close_location = _first(values, "close_location_12", "close_location_24")
    room = _first(values, "short_room_to_fall_12", "short_room_to_fall_24")
    if trend is None or upper is None or close_location is None:
        return _top_fraction_mask(quality), missing, sparse
    eligible = (
        trend >= np.nanquantile(trend, 0.45)
    ) & (
        upper >= np.nanquantile(upper, 0.55)
    ) & (
        close_location <= np.nanquantile(close_location, 0.55)
    )
    if room is not None:
        eligible &= room >= np.nanquantile(room, 0.45)
    contradiction = values.get("btc_eth_long_contradiction")
    if contradiction is not None:
        eligible &= contradiction <= np.nanquantile(contradiction, 0.70)
    mask = _top_fraction_mask(quality, fraction=0.10, eligible=eligible)
    if not np.any(mask):
        return _top_fraction_mask(quality), missing + ["pullback_rejection_no_eligible_rows"], sparse
    return mask, missing, sparse


def link_failed_retest_mask(dataset: dict[str, Any], trained: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    test = np.asarray(trained["test"], dtype=np.int64)
    hit = np.asarray(trained["hit_prob"], dtype=np.float64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    danger = np.asarray(trained["danger_prob"], dtype=np.float64)
    values, missing = _feature_values(dataset, test, FAILED_RETEST_FEATURES)
    sparse = _sparse_features(values, FAILED_RETEST_FEATURES)
    breakdown = _first(values, "short_breakdown_strength_12", "short_breakdown_strength_24")
    reclaim = values.get("short_reclaim_range_risk")
    if breakdown is None:
        return _top_fraction_mask((hit + quality - danger), fraction=0.10), missing + ["retest_features_missing_or_sparse"], sparse
    failed = values.get("short_retest_failed")
    rejection = _first(values, "short_retest_rejection_wick", "upper_wick_ratio")
    dryup = values.get("short_retest_volume_dryup")
    eligible = breakdown >= np.nanquantile(breakdown, 0.50)
    if reclaim is not None:
        eligible &= reclaim <= np.nanquantile(reclaim, 0.70)
    if failed is not None and "short_retest_failed" not in sparse:
        eligible &= failed >= np.nanquantile(failed, 0.50)
    if rejection is not None:
        eligible &= rejection >= np.nanquantile(rejection, 0.45)
    score = _percentile_ranks(hit) + _percentile_ranks(quality) - _percentile_ranks(danger)
    if dryup is not None and "short_retest_volume_dryup" not in sparse:
        score += _percentile_ranks(dryup) * 0.25
    mask = _top_fraction_mask(score, fraction=0.10, eligible=eligible)
    local_missing = list(missing)
    if sparse:
        local_missing.append("retest_features_missing_or_sparse")
    if not np.any(mask):
        return _top_fraction_mask(score, fraction=0.10), local_missing + ["failed_retest_no_eligible_rows"], sparse
    return mask, local_missing, sparse


def link_avoid_mask(
    dataset: dict[str, Any],
    trained: dict[str, Any],
    avoid_mode: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    test = np.asarray(trained["test"], dtype=np.int64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    danger = np.asarray(trained["danger_prob"], dtype=np.float64)
    values, missing = _feature_values(dataset, test, AVOID_FEATURES)
    sparse = _sparse_features(values, AVOID_FEATURES)
    length = len(quality)
    score = _percentile_ranks(danger) + _percentile_ranks(-quality)
    fake = _first(values, "short_failed_breakdown_risk_12", "short_failed_breakdown_risk_24", "short_reclaim_range_risk")
    chop = values.get("local_chop_score")
    contradiction = values.get("btc_eth_long_contradiction")
    sweep = values.get("short_lower_wick_sweep_risk")
    absorption = _first(values, "short_volume_climax_risk", "short_absorption_risk", "short_adverse_rebound_risk")
    retest_noise = _first(values, "short_retest_success", "short_retest_failed")
    if avoid_mode == "avoid_by_fake_breakdown":
        if fake is not None:
            score += _percentile_ranks(fake) * 1.25
        if sweep is not None:
            score += _percentile_ranks(sweep) * 0.75
    elif avoid_mode == "avoid_by_chop_reclaim":
        if chop is not None:
            score += _percentile_ranks(chop) * 1.20
        if fake is not None:
            score += _percentile_ranks(fake) * 0.80
        if contradiction is not None:
            score += _percentile_ranks(contradiction) * 0.70
    elif avoid_mode == "avoid_by_retest_failure_noise":
        if retest_noise is not None:
            score += _percentile_ranks(retest_noise) * 0.80
        if absorption is not None:
            score += _percentile_ranks(absorption) * 0.80
        if fake is not None:
            score += _percentile_ranks(fake) * 0.50
    elif avoid_mode != "avoid_by_danger_quality":
        raise ValueError(f"unsupported avoid mode: {avoid_mode}")
    fraction = 0.20 if length >= 100 else 0.10
    return _top_fraction_mask(score, fraction=fraction), missing, sparse


def decision_mask(
    mode: str,
    trained: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[str]]:
    hit = np.asarray(trained["hit_prob"], dtype=np.float64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    danger = np.asarray(trained["danger_prob"], dtype=np.float64)
    if mode in {"hit_primary", "quality_primary", "quality_primary_danger_filtered"}:
        return l2_selection_mask(mode, hit, quality, danger), [], []
    if mode == "top_bucket_only":
        return _top_fraction_mask(quality, fraction=0.10), [], []
    if mode == "trend_confirmed_quality":
        return link_slow_trend_pullback_mask(dataset, trained)
    if mode == "pullback_rejection_quality":
        return link_pullback_rejection_mask(dataset, trained)
    if mode == "failed_retest_quality":
        return link_failed_retest_mask(dataset, trained)
    if mode == "failed_retest_consensus":
        mask, missing, sparse = link_failed_retest_mask(dataset, trained)
        if not np.any(mask):
            score = _percentile_ranks(hit) + _percentile_ranks(quality) - _percentile_ranks(danger)
            return _top_fraction_mask(score, fraction=0.10), missing, sparse
        return mask, missing, sparse
    if mode in AVOID_MODES:
        return link_avoid_mask(dataset, trained, mode)
    raise ValueError(f"unsupported LINK N decision mode: {mode}")


def avoid_stats(
    row: dict[str, Any],
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    test = np.asarray(trained["test"], dtype=np.int64)
    selected_count = int(mask.sum())
    hit = targets["hit"][test]
    quality = targets["quality"][test]
    danger = targets["danger"][test]
    mae = targets["mae"][test]
    baseline_hit = row.get("baseline_target_hit_rate")
    baseline_quality = row.get("baseline_trade_quality")
    baseline_danger = row.get("baseline_mae_danger")
    baseline_p90 = row.get("baseline_p90_mae")
    avoid_hit = _mean(hit[mask]) if selected_count else None
    avoid_quality = _mean(quality[mask]) if selected_count else None
    avoid_danger = _mean(danger[mask]) if selected_count else None
    avoid_p90 = _quantile(mae[mask], 0.90) if selected_count else None
    quality_delta = None if avoid_quality is None or baseline_quality is None else avoid_quality - float(baseline_quality)
    danger_delta = None if avoid_danger is None or baseline_danger is None else avoid_danger - float(baseline_danger)
    p90_delta = None if avoid_p90 is None or baseline_p90 is None else avoid_p90 - float(baseline_p90)
    hit_delta = None if avoid_hit is None or baseline_hit is None else avoid_hit - float(baseline_hit)
    usefulness = (
        -(quality_delta or 0.0)
        + (danger_delta or 0.0)
        + (p90_delta or 0.0)
        - max(hit_delta or 0.0, 0.0)
    )
    return {
        "avoid_selected_fraction": selected_count / max(1, len(test)),
        "avoid_selected_count": selected_count,
        "avoid_quality_mean": avoid_quality,
        "avoid_quality_delta_vs_baseline": quality_delta,
        "avoid_mae_danger_rate": avoid_danger,
        "avoid_mae_danger_delta": danger_delta,
        "avoid_p90_mae": avoid_p90,
        "avoid_p90_mae_delta": p90_delta,
        "avoid_hit_rate": avoid_hit,
        "avoid_hit_delta": hit_delta,
        "avoid_usefulness_score": usefulness,
    }


def _risk_strongly_worse(row: dict[str, Any]) -> bool:
    p90_delta = finite(row.get("selected_p90_mae_delta"), 0.0) or 0.0
    danger_delta = finite(row.get("selected_mae_danger_delta"), 0.0) or 0.0
    baseline_p90 = max(finite(row.get("baseline_p90_mae"), 0.0) or 0.0, 1e-12)
    baseline_danger = max(finite(row.get("baseline_mae_danger"), 0.0) or 0.0, 1e-12)
    return p90_delta > baseline_p90 * 0.30 or danger_delta > baseline_danger * 0.40


def classify_link_n_candidate(row: dict[str, Any]) -> str:
    if (
        row.get("model_status") != "trained"
        or int(row.get("test_samples") or 0) <= 0
        or int(row.get("feature_count") or 0) <= 0
    ):
        return "LINK_INSUFFICIENT_DATA"
    mode = str(row.get("decision_mode", ""))
    if mode in AVOID_MODES:
        fraction = finite(row.get("avoid_selected_fraction"), 0.0) or 0.0
        useful = (
            0.05 <= fraction <= 0.30
            and (finite(row.get("avoid_quality_delta_vs_baseline"), 0.0) or 0.0) < 0.0
            and (finite(row.get("avoid_mae_danger_delta"), 0.0) or 0.0) > 0.0
            and (finite(row.get("avoid_p90_mae_delta"), 0.0) or 0.0) > 0.0
            and (finite(row.get("avoid_usefulness_score"), 0.0) or 0.0) > 0.0
        )
        if useful:
            return "LINK_AVOID_ONLY_CONFIRMED"
        partial = (
            0.03 <= fraction <= 0.35
            and sum([
                (finite(row.get("avoid_quality_delta_vs_baseline"), 0.0) or 0.0) < 0.0,
                (finite(row.get("avoid_mae_danger_delta"), 0.0) or 0.0) > 0.0,
                (finite(row.get("avoid_p90_mae_delta"), 0.0) or 0.0) > 0.0,
                (finite(row.get("avoid_hit_delta"), 0.0) or 0.0) <= 0.0,
            ]) >= 2
        )
        return "LINK_AVOID_ONLY_WEAK" if partial else "LINK_RESEARCH_ONLY"
    selected_count = int(row.get("selected_count") or 0)
    test_samples = int(row.get("test_samples") or 0)
    if selected_count <= 0:
        return "LINK_ENTRY_FAILED"
    target_lift = finite(row.get("selected_target_lift"), 0.0) or 0.0
    quality_lift = finite(row.get("selected_quality_lift"), 0.0) or 0.0
    net_lift = finite(row.get("selected_net_quality_lift"), 0.0) or 0.0
    p90_delta = finite(row.get("selected_p90_mae_delta"), 0.0) or 0.0
    selected_p90 = finite(row.get("selected_p90_mae"), float("inf")) or float("inf")
    baseline_p90 = finite(row.get("baseline_p90_mae"), 0.0) or 0.0
    selected_danger = finite(row.get("selected_mae_danger_rate"), float("inf")) or float("inf")
    baseline_danger = finite(row.get("baseline_mae_danger"), 0.0) or 0.0
    auc = finite(row.get("target_auc"), 0.0) or 0.0
    corr = finite(row.get("quality_corr"), 0.0) or 0.0
    fraction = finite(row.get("selected_fraction"), 0.0) or 0.0
    if quality_lift <= 0.0 or net_lift <= 0.0 or _risk_strongly_worse(row):
        return "LINK_ENTRY_FAILED"
    confirmed = (
        selected_count >= _required_selected_count(test_samples)
        and target_lift > 0.0
        and quality_lift > 0.05
        and net_lift > 0.04
        and selected_p90 <= baseline_p90
        and selected_danger <= baseline_danger * 1.10
        and (auc > 0.53 or corr > 0.03)
        and 0.05 <= fraction <= 0.20
    )
    if confirmed:
        return "LINK_ENTRY_CONFIRMED"
    if target_lift > 0.0 or quality_lift > 0.0:
        if p90_delta <= max(baseline_p90 * 0.20, 0.002) and selected_danger <= max(1.0, baseline_danger * 1.30):
            return "LINK_ENTRY_WEAK"
    return "LINK_RESEARCH_ONLY"


def n_reason(row: dict[str, Any]) -> str:
    status = str(row.get("n_status"))
    if status == "LINK_ENTRY_CONFIRMED":
        return "link_specific_entry_edge_confirmed_with_controlled_risk"
    if status == "LINK_ENTRY_WEAK":
        return "partial_link_entry_edge_but_confirmation_gates_not_all_met"
    if status == "LINK_ENTRY_FAILED":
        return "entry_edge_or_risk_gate_failed"
    if status == "LINK_AVOID_ONLY_CONFIRMED":
        return "avoid_only_bad_zone_filter_detects_lower_quality_higher_risk_bucket"
    if status == "LINK_AVOID_ONLY_WEAK":
        return "avoid_only_signal_is_partial_not_confirmed"
    if status == "LINK_INSUFFICIENT_DATA":
        return "insufficient_samples_features_or_model_diversity"
    return "mixed_research_signal_no_actionable_link_alpha"


def recommended_next_step(status: str) -> str:
    if status == "LINK_ENTRY_CONFIRMED":
        return "eligible_for_metadata_only_shadow_candidate_after_review"
    if status == "LINK_AVOID_ONLY_CONFIRMED":
        return "prepare_future_avoid_or_reducer_metadata_shadow_not_entry_artifact"
    if status in {"LINK_ENTRY_WEAK", "LINK_AVOID_ONLY_WEAK", "LINK_RESEARCH_ONLY"}:
        return "continue_research_no_shadow_entry_artifact"
    if status == "LINK_ENTRY_FAILED":
        return "disable_link_entry_hypothesis_for_now"
    return "collect_more_data_before_decision"


def evaluate_mode(
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    dataset: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    cost_proxy: float,
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
        "valid_fold_count": 1 if trained.get("model_status") == "trained" else 0,
        "target_auc": trained.get("target_auc"),
        "target_average_precision": trained.get("target_average_precision"),
        "quality_corr": trained.get("quality_corr"),
        "danger_auc": trained.get("danger_auc"),
        "danger_filter_usefulness": trained.get("danger_filter_usefulness"),
        "feature_count": int(np.asarray(dataset.get("X", np.zeros((0, 0)))).shape[1]),
        "missing_features": ",".join(dataset.get("missing_family_features", [])),
        "sparse_features": "",
    }
    if trained.get("model_status") != "trained":
        row = {**base, "n_status": "LINK_INSUFFICIENT_DATA"}
        row["n_reason"] = n_reason(row)
        row["recommended_next_step"] = recommended_next_step(row["n_status"])
        return row
    mask, missing, sparse = decision_mask(mode, trained, dataset)
    row = {**base, **_selection_stats(trained, targets, mask, cost_proxy)}
    combined_missing = list(dict.fromkeys(list(dataset.get("missing_family_features", [])) + missing))
    row["missing_features"] = ",".join(combined_missing)
    row["sparse_features"] = ",".join(sparse)
    if mode in AVOID_MODES:
        row.update(avoid_stats(row, trained, targets, mask))
    else:
        row.update({
            "avoid_selected_fraction": None,
            "avoid_selected_count": None,
            "avoid_quality_mean": None,
            "avoid_quality_delta_vs_baseline": None,
            "avoid_mae_danger_rate": None,
            "avoid_mae_danger_delta": None,
            "avoid_p90_mae": None,
            "avoid_p90_mae_delta": None,
            "avoid_hit_rate": None,
            "avoid_hit_delta": None,
            "avoid_usefulness_score": None,
        })
    row["n_status"] = classify_link_n_candidate(row)
    row["n_reason"] = n_reason(row)
    row["recommended_next_step"] = recommended_next_step(row["n_status"])
    return row


def _folds_for_mode(sample_count: int, lockbox_mode: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if lockbox_mode == "rolling-forward":
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
        recent_only=lockbox_mode == "recent-only",
    )
    return [fold] if fold is not None else []


def evaluate_config(
    config: dict[str, Any],
    dataset: dict[str, Any],
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    lockbox_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = _folds_for_mode(len(np.asarray(dataset["X"])), lockbox_mode, args)
    if not splits:
        for mode in config["decision_modes"]:
            row = {
                **config,
                "decision_mode": mode,
                "lockbox_mode": lockbox_mode,
                "model_status": "insufficient_split_samples",
                "test_samples": 0,
                "feature_count": int(np.asarray(dataset["X"]).shape[1]),
                "n_status": "LINK_INSUFFICIENT_DATA",
            }
            row["n_reason"] = n_reason(row)
            row["recommended_next_step"] = recommended_next_step(row["n_status"])
            rows.append(row)
        return rows
    for fold_index, split in enumerate(splits, start=1):
        trained = _fit_fold_predictions(
            dataset,
            targets,
            split,
            symbol=config["symbol"],
            lookback_days=int(config["lookback_days"]),
            target_name=config["target_name"],
            max_iter=60 if args.fast else 120,
        )
        for mode in config["decision_modes"]:
            rows.append(evaluate_mode(
                trained,
                targets,
                dataset,
                config,
                mode,
                (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0,
                fold_index=fold_index,
                lockbox_mode=lockbox_mode,
            ))
    return rows


def select_best_link_n(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {
        "LINK_ENTRY_CONFIRMED": 7,
        "LINK_AVOID_ONLY_CONFIRMED": 6,
        "LINK_ENTRY_WEAK": 5,
        "LINK_AVOID_ONLY_WEAK": 4,
        "LINK_RESEARCH_ONLY": 3,
        "LINK_ENTRY_FAILED": 2,
        "LINK_INSUFFICIENT_DATA": 1,
    }
    target_preference = {"hit3_before_minus2": 2, "hit5_before_minus3": 1}

    def score(row: dict[str, Any]) -> tuple[float, ...]:
        status = str(row.get("n_status"))
        if status.startswith("LINK_AVOID"):
            primary = finite(row.get("avoid_usefulness_score"), -999.0) or -999.0
            secondary = -abs((finite(row.get("avoid_selected_fraction"), 0.0) or 0.0) - 0.15)
        else:
            primary = finite(row.get("selected_net_quality_lift"), -999.0) or -999.0
            secondary = -(finite(row.get("selected_p90_mae_delta"), 999.0) or 999.0)
        return (
            priority.get(status, 0),
            primary,
            secondary,
            finite(row.get("selected_target_lift"), -999.0) or -999.0,
            finite(row.get("target_auc"), -999.0) or -999.0,
            target_preference.get(str(row.get("target_name")), 0),
        )

    return dict(max(rows, key=score)) if rows else {
        "symbol": "LINKUSDT",
        "side": SIDE,
        "n_status": "LINK_INSUFFICIENT_DATA",
        "n_reason": "no_rows_evaluated",
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
    best = report["best_link_result"]
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for row in report["all_config_rows"]:
        family_rows.setdefault(str(row.get("alpha_family")), []).append(row)
    lines = [
        f"# Aegis LINK SHORT Alpha N {report['created_at']}",
        "",
        "## Safety",
        "",
        "- `RESEARCH_ONLY`.",
        "- No shadow artifacts are generated.",
        "- No active models, active manifests or live inference are changed.",
        "- No YAML, thresholds, PM2, orders, push or commit are performed.",
        "",
        "## Best LINK Result",
        "",
        "| Family | Target | Horizon | Mode | Status | Target Lift | Quality Lift | Net Quality Lift | P90 MAE Delta | Avoid Score | Reason |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        (
            f"| {best.get('alpha_family')} | {best.get('target_name')} | {best.get('horizon_candles')} | "
            f"{best.get('decision_mode')} | {best.get('n_status')} | {_num(best.get('selected_target_lift'))} | "
            f"{_num(best.get('selected_quality_lift'))} | {_num(best.get('selected_net_quality_lift'))} | "
            f"{_num(best.get('selected_p90_mae_delta'))} | {_num(best.get('avoid_usefulness_score'))} | "
            f"{best.get('n_reason')} |"
        ),
        "",
        "## Family Comparison",
        "",
        "| Family | Best status | Best mode | Target | Horizon | Net lift | Avoid score |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for family in ("slow_trend_pullback_short", "failed_retest_short", "avoid_only_bad_short_filter"):
        local = family_rows.get(family, [])
        if not local:
            lines.append(f"| {family} | not_evaluated | - | - | - | null | null |")
            continue
        local_best = select_best_link_n(local)
        lines.append(
            f"| {family} | {local_best.get('n_status')} | {local_best.get('decision_mode')} | "
            f"{local_best.get('target_name')} | {local_best.get('horizon_candles')} | "
            f"{_num(local_best.get('selected_net_quality_lift'))} | {_num(local_best.get('avoid_usefulness_score'))} |"
        )
    decision = "LINK sigue research"
    if best.get("n_status") == "LINK_ENTRY_CONFIRMED":
        decision = "LINK puede entrar como entry candidate"
    elif best.get("n_status") == "LINK_AVOID_ONLY_CONFIRMED":
        decision = "LINK solo sirve como avoid-only"
    elif best.get("n_status") == "LINK_ENTRY_FAILED":
        decision = "LINK apagado por ahora"
    lines.extend([
        "",
        "## Decision",
        "",
        f"- {decision}.",
    ])
    if report["errors"]:
        lines.extend(["", "## Errors", "", f"- `{report['errors']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = configs_for_args(args)
    context_markets: dict[str, Any] = {}
    context_warning: str | None = None
    if not bool(args.disable_cross_context):
        try:
            context_markets = {
                symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
                for symbol in ("BTCUSDT", "ETHUSDT")
            }
        except Exception as exc:
            context_warning = f"cross_context_unavailable:{exc!r}"
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, int], tuple[Any, dict[str, Any]]] = {}
    lockbox_modes = [args.lockbox_mode]
    if bool(args.include_rolling_forward) and args.lockbox_mode != "rolling-forward":
        lockbox_modes.append("rolling-forward")
    for config in configs:
        try:
            cache_key = (config["symbol"], int(config["lookback_days"]))
            if cache_key not in data_cache:
                market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
                base = build_recent_dataset(config["symbol"], int(config["lookback_days"]), save=False, market=market)["dataset"]
                combined = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
                data_cache[cache_key] = (market, combined)
            market, combined = data_cache[cache_key]
            family_features = ALPHA_FEATURES.get(config["alpha_family"], ())
            original = getattr(select_alpha_family_features, "__wrapped__", None)
            if config["feature_mode"] == "combined_v3_all":
                dataset = select_alpha_family_features(combined, "slow_trend_short", "combined_v3_all")
            else:
                # Reuse L2 selector, then override the family list locally to avoid changing L2 globals.
                names = np.asarray(combined["feature_names"]).astype(str)
                x = np.asarray(combined["X"], dtype=np.float32)
                index_by_name = {name: index for index, name in enumerate(names)}
                indices = np.asarray([index_by_name[name] for name in family_features if name in index_by_name], dtype=np.int64)
                dataset = dict(combined)
                dataset["X"] = x[:, indices] if len(indices) else np.zeros((len(x), 0), dtype=np.float32)
                dataset["feature_names"] = names[indices] if len(indices) else np.asarray([], dtype=str)
                dataset["missing_family_features"] = [name for name in family_features if name not in index_by_name]
                dataset["feature_mode"] = config["feature_mode"]
                dataset["alpha_family"] = config["alpha_family"]
                dataset["family_feature_count"] = int(len(indices))
            _ = original  # keeps the imported symbol intentionally referenced for research lineage.
            targets = compute_alpha_target_arrays(
                market,
                np.asarray(combined["step"], dtype=np.int64),
                side=SIDE,
                target_name=config["target_name"],
                horizon=int(config["horizon_candles"]),
                cost_proxy=(float(args.fee_bps) + float(args.slippage_bps)) / 10000.0,
            )
            for mode in lockbox_modes:
                all_rows.extend(evaluate_config(config, dataset, targets, args, lockbox_mode=mode))
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
                    "n_status": "LINK_INSUFFICIENT_DATA",
                    "n_reason": "configuration_evaluation_error",
                    "recommended_next_step": "collect_more_data_before_decision",
                    "config_error": repr(exc),
                })
    best = select_best_link_n(all_rows)
    entry_confirmed = [row for row in all_rows if row.get("n_status") == "LINK_ENTRY_CONFIRMED"]
    entry_weak = [row for row in all_rows if row.get("n_status") == "LINK_ENTRY_WEAK"]
    avoid_confirmed = [row for row in all_rows if row.get("n_status") == "LINK_AVOID_ONLY_CONFIRMED"]
    failed = [row for row in all_rows if row.get("n_status") in {"LINK_ENTRY_FAILED", "LINK_INSUFFICIENT_DATA"}]
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_link_short_alpha_n_{stamp}.md",
        "json": out_dir / f"aegis_link_short_alpha_n_{stamp}.json",
        "all_configs_csv": out_dir / f"aegis_link_short_alpha_n_all_configs_{stamp}.csv",
        "best_csv": out_dir / f"aegis_link_short_alpha_n_best_{stamp}.csv",
        "entry_confirmed_csv": out_dir / f"aegis_link_short_alpha_n_entry_confirmed_{stamp}.csv",
        "entry_weak_csv": out_dir / f"aegis_link_short_alpha_n_entry_weak_{stamp}.csv",
        "avoid_confirmed_csv": out_dir / f"aegis_link_short_alpha_n_avoid_confirmed_{stamp}.csv",
        "failed_csv": out_dir / f"aegis_link_short_alpha_n_failed_{stamp}.csv",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "symbol": "LINKUSDT",
        "side": SIDE,
        "lockbox_mode": args.lockbox_mode,
        "feature_mode": args.feature_mode,
        "context_warning": context_warning,
        "configs": configs,
        "all_config_rows": all_rows,
        "best_link_result": best,
        "entry_confirmed": entry_confirmed,
        "entry_weak": entry_weak,
        "avoid_confirmed": avoid_confirmed,
        "failed": failed,
        "status_counts": dict(Counter(str(row.get("n_status")) for row in all_rows)),
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
    _write_csv(paths["entry_confirmed_csv"], entry_confirmed)
    _write_csv(paths["entry_weak_csv"], entry_weak)
    _write_csv(paths["avoid_confirmed_csv"], avoid_confirmed)
    _write_csv(paths["failed_csv"], failed)
    return safe_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only LINKUSDT SHORT alpha redesign Fase N.")
    parser.add_argument("--symbol", default="LINKUSDT")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--lockbox-mode", choices=LOCKBOX_MODES, default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="selected_family")
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--disable-cross-context", action="store_true")
    parser.add_argument("--include-avoid-only", action="store_true", default=True)
    parser.add_argument("--include-rolling-forward", action="store_true")
    args = parser.parse_args()
    report = run(args)
    best = report["best_link_result"]
    print(json.dumps({
        "paths": report["paths"],
        "best_link_result": {
            "symbol": best.get("symbol"),
            "family": best.get("alpha_family"),
            "target": best.get("target_name"),
            "horizon": best.get("horizon_candles"),
            "mode": best.get("decision_mode"),
            "status": best.get("n_status"),
        },
        "status_counts": report["status_counts"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
