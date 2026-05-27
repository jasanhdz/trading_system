#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import build_last_block_fold  # noqa: E402
from aegis_alpha.tools.profile_failed_alpha_symbols import ALT_HIT_RULES  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset, compute_path_outcome  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_operable_edge_v2 import (  # noqa: E402
    _classifier,
    _model_seed,
    _regressor,
    classification_metrics,
    safe_corr,
)
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SCHEMA_VERSION = "aegis_short_alpha_family_l2_research_v1"
DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "LINKUSDT": {
        "alpha_family": "slow_trend_short",
        "feature_set": "combined_v3",
        "lookback_days": 30,
        "target_candidates": ("hit3_before_minus2", "hit5_before_minus3"),
        "horizons": (12, 24),
    },
    "SOLUSDT": {
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "lookback_days": 30,
        "target_candidates": ("hit5_before_minus3", "hit6_before_minus4"),
        "horizons": (12, 24),
    },
    "XRPUSDT": {
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "lookback_days": 30,
        "target_candidates": ("hit5_before_minus3", "hit6_before_minus4"),
        "horizons": (12, 24),
    },
    "BNBUSDT": {
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "lookback_days": 30,
        "target_candidates": ("hit5_before_minus3", "hit6_before_minus4"),
        "horizons": (12, 24),
    },
}
DECISION_MODES = (
    "hit_primary",
    "quality_primary",
    "quality_primary_danger_filtered",
    "top_bucket_consensus",
    "top_bucket_consensus_danger_filtered",
)
FEATURE_MODES = ("selected_family", "combined_v3_all")
FAMILY_FEATURES: dict[str, tuple[str, ...]] = {
    "slow_trend_short": (
        "local_trend_down_score",
        "ema_stack_bearish",
        "ema21_slope_12",
        "ema200_slope_64",
        "trend_efficiency_64",
        "atr_percentile_64",
        "realized_vol_24",
        "realized_vol_64",
        "btc_eth_short_agreement",
        "btc_eth_long_contradiction",
        "symbol_vs_btc_relative_strength_30m",
        "symbol_underperforming_btc_60m",
        "symbol_vs_eth_relative_strength_30m",
        "symbol_underperforming_eth_60m",
        "short_room_to_fall_12",
        "short_room_to_fall_24",
        "short_overhead_risk_12",
        "short_overhead_risk_24",
    ),
    "momentum_burst_lower_target": (
        "local_momentum_down_score",
        "local_breakdown_score",
        "short_breakdown_strength_12",
        "short_breakdown_strength_24",
        "short_breakdown_followthrough_3",
        "short_breakdown_followthrough_6",
        "short_breakdown_volume_confirmed",
        "volume_ratio_12",
        "range_expansion_12",
        "range_expansion_24",
        "trend_efficiency_12",
        "trend_efficiency_24",
        "btc_return_15m",
        "btc_return_30m",
        "btc_return_60m",
        "eth_return_15m",
        "eth_return_30m",
        "eth_return_60m",
        "btc_eth_short_agreement",
        "btc_eth_long_contradiction",
        "short_failed_breakdown_risk_12",
        "short_failed_breakdown_risk_24",
        "short_lower_wick_sweep_risk",
        "short_reclaim_range_risk",
    ),
}
ALL_CONFIG_COLUMNS = (
    "symbol", "side", "alpha_family", "feature_set", "feature_mode", "lookback_days",
    "horizon_candles", "target_name", "decision_mode", "l2_status", "model_status",
    "train_samples", "validation_samples", "test_samples", "valid_fold_count",
    "feature_count", "missing_family_features", "baseline_target_hit_rate",
    "baseline_trade_quality", "baseline_mae_danger", "baseline_p90_mae",
    "target_auc", "target_average_precision", "quality_corr", "danger_auc",
    "danger_filter_usefulness", "selected_fraction", "selected_count",
    "selected_target_hit_rate", "selected_target_lift", "selected_quality_lift",
    "selected_net_quality_lift", "selected_p90_mae", "selected_p90_mae_delta",
    "selected_mae_danger_rate", "selected_mae_danger_delta",
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


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else None


def _parse_csv(raw: str | None, caster: Any = str) -> list[Any] | None:
    if raw is None:
        return None
    return list(dict.fromkeys(caster(value.strip()) for value in raw.split(",") if value.strip()))


def parse_symbols(raw: str | None) -> list[str]:
    requested = _parse_csv(raw) if raw else list(DEFAULT_CONFIGS)
    symbols = [normalize_turbo_symbol(value) for value in requested or []]
    unsupported = [symbol for symbol in symbols if symbol not in DEFAULT_CONFIGS]
    if unsupported:
        raise ValueError(f"symbols do not have frozen L2 family configs: {unsupported}")
    return list(dict.fromkeys(symbols))


def select_alpha_family_features(
    feature_dataset: dict[str, Any],
    alpha_family: str,
    feature_mode: str = "selected_family",
) -> dict[str, Any]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    names = np.asarray(feature_dataset["feature_names"]).astype(str)
    x = np.asarray(feature_dataset["X"], dtype=np.float32)
    selected = dict(feature_dataset)
    expected = FAMILY_FEATURES.get(alpha_family, ())
    missing = [name for name in expected if name not in set(names)]
    if feature_mode == "combined_v3_all":
        indices = np.arange(len(names), dtype=np.int64)
    else:
        index_by_name = {name: index for index, name in enumerate(names)}
        indices = np.asarray([index_by_name[name] for name in expected if name in index_by_name], dtype=np.int64)
    selected["X"] = x[:, indices] if len(indices) else np.zeros((len(x), 0), dtype=np.float32)
    selected["feature_names"] = names[indices] if len(indices) else np.asarray([], dtype=str)
    selected["feature_mode"] = feature_mode
    selected["alpha_family"] = alpha_family
    selected["missing_family_features"] = missing
    selected["family_feature_count"] = int(len(indices))
    return selected


def compute_alpha_target_arrays(
    market: Any,
    steps: np.ndarray,
    *,
    side: str,
    target_name: str,
    horizon: int,
    cost_proxy: float,
) -> dict[str, np.ndarray]:
    if target_name not in ALT_HIT_RULES:
        raise ValueError(f"unsupported alternative target: {target_name}")
    target_return, stop_return = ALT_HIT_RULES[target_name]
    hit: list[float] = []
    quality: list[float] = []
    danger: list[float] = []
    mfe_values: list[float] = []
    mae_values: list[float] = []
    ratios: list[float] = []
    time_target: list[float] = []
    time_stop: list[float] = []
    for raw_step in np.asarray(steps, dtype=np.int64):
        step = int(raw_step)
        entry = float(market.close[step])
        high = np.asarray(market.high[step + 1 : step + horizon + 1], dtype=np.float32)
        low = np.asarray(market.low[step + 1 : step + horizon + 1], dtype=np.float32)
        outcome = compute_path_outcome(entry, high, low, side.lower(), target_return, stop_return)
        is_hit = float(bool(outcome["hit_before_stop"]))
        mfe = float(outcome["mfe"])
        mae = float(outcome["mae"])
        reward = is_hit + min(mfe / max(target_return, 1e-6), 1.0) * 0.30
        penalty = min(mae / max(stop_return, 1e-6), 1.0) * 0.70 + float(cost_proxy)
        hit.append(is_hit)
        quality.append(float(np.clip(reward - penalty, -1.0, 1.0)))
        danger.append(float(mae >= stop_return))
        mfe_values.append(mfe)
        mae_values.append(mae)
        ratios.append(mfe / max(mae, 1e-6))
        time_target.append(float(outcome["time_to_target"]))
        time_stop.append(float(outcome["time_to_stop"]))
    return {
        "hit": np.asarray(hit, dtype=np.int8),
        "quality": np.asarray(quality, dtype=np.float32),
        "danger": np.asarray(danger, dtype=np.int8),
        "mfe": np.asarray(mfe_values, dtype=np.float32),
        "mae": np.asarray(mae_values, dtype=np.float32),
        "mfe_mae_ratio": np.asarray(ratios, dtype=np.float32),
        "time_to_target": np.asarray(time_target, dtype=np.float32),
        "time_to_stop": np.asarray(time_stop, dtype=np.float32),
    }


def _top_fraction_mask(scores: np.ndarray, fraction: float = 0.10, eligible: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    allowed = np.isfinite(values) if eligible is None else np.asarray(eligible, dtype=bool) & np.isfinite(values)
    positions = np.flatnonzero(allowed)
    mask = np.zeros(len(values), dtype=bool)
    if not len(positions):
        return mask
    take = max(1, int(math.ceil(len(values) * fraction)))
    take = min(take, len(positions))
    ranked = positions[np.argsort(values[positions], kind="stable")[-take:]]
    mask[ranked] = True
    return mask


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    result = np.zeros(len(array), dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(array), endpoint=True) if len(array) else array
    return result


def selection_mask(
    repair_mode: str,
    hit_prob: np.ndarray,
    quality_pred: np.ndarray,
    danger_prob: np.ndarray,
) -> np.ndarray:
    if repair_mode not in DECISION_MODES:
        raise ValueError(f"unsupported decision mode: {repair_mode}")
    safe = np.asarray(danger_prob, dtype=np.float64) <= float(np.quantile(danger_prob, 0.80))
    if repair_mode == "hit_primary":
        return _top_fraction_mask(hit_prob)
    if repair_mode == "quality_primary":
        return _top_fraction_mask(quality_pred)
    if repair_mode == "quality_primary_danger_filtered":
        return _top_fraction_mask(quality_pred, eligible=safe)
    consensus = (_percentile_ranks(hit_prob) + _percentile_ranks(quality_pred)) / 2.0
    if repair_mode == "top_bucket_consensus":
        return _top_fraction_mask(consensus)
    return _top_fraction_mask(consensus, eligible=safe)


def _required_selection_count(test_samples: int) -> int:
    return min(30, max(1, int(math.ceil(float(test_samples) * 0.05))))


def classify_l2_alpha_candidate(row: dict[str, Any]) -> str:
    if row.get("model_status") != "trained" or int(row.get("test_samples") or 0) <= 0:
        return "L2_ALPHA_INSUFFICIENT"
    test_samples = int(row["test_samples"])
    selected_count = int(row.get("selected_count") or 0)
    if selected_count < _required_selection_count(test_samples):
        return "L2_ALPHA_MIXED" if selected_count else "L2_ALPHA_BAD"
    hit_lift = finite(row.get("selected_target_lift"), -1.0) or -1.0
    quality_lift = finite(row.get("selected_quality_lift"), -1.0) or -1.0
    net_lift = finite(row.get("selected_net_quality_lift"), -1.0) or -1.0
    baseline_p90 = finite(row.get("baseline_p90_mae"), 0.0) or 0.0
    selected_p90 = finite(row.get("selected_p90_mae"), float("inf")) or float("inf")
    danger_delta = finite(row.get("selected_mae_danger_delta"), 0.0) or 0.0
    auc = finite(row.get("target_auc"), 0.0) or 0.0
    quality_corr = finite(row.get("quality_corr"), 0.0) or 0.0
    if hit_lift < 0.0 or quality_lift < 0.0 or net_lift <= 0.0:
        return "L2_ALPHA_BAD"
    if baseline_p90 > 0.0 and selected_p90 > baseline_p90 * 1.30:
        return "L2_ALPHA_BAD"
    promising = (
        hit_lift > 0.0
        and quality_lift > 0.0
        and net_lift > 0.0
        and (baseline_p90 <= 0.0 or selected_p90 <= baseline_p90 * 1.15)
        and (auc > 0.53 or quality_corr > 0.03)
        and danger_delta <= 0.20
    )
    return "L2_ALPHA_PROMISING" if promising else "L2_ALPHA_MIXED"


def _fit_fold_predictions(
    dataset: dict[str, Any],
    targets: dict[str, np.ndarray],
    split: dict[str, Any],
    *,
    symbol: str,
    lookback_days: int,
    target_name: str,
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
    if len(np.unique(targets["hit"][train])) < 2 or len(np.unique(targets["danger"][train])) < 2:
        return {**base, "model_status": "insufficient_class_diversity"}
    if float(np.std(targets["quality"][train])) <= 1e-12:
        return {**base, "model_status": "insufficient_quality_variance"}
    hit_model = _classifier(max_iter, _model_seed(symbol, "short", lookback_days, f"{target_name}:hit"))
    quality_model = _regressor(max_iter, _model_seed(symbol, "short", lookback_days, f"{target_name}:quality"))
    danger_model = _classifier(max_iter, _model_seed(symbol, "short", lookback_days, f"{target_name}:danger"))
    hit_model.fit(x[train], targets["hit"][train])
    quality_model.fit(x[train], targets["quality"][train])
    danger_model.fit(x[train], targets["danger"][train])
    hit_prob = hit_model.predict_proba(x[test])[:, 1]
    quality_pred = quality_model.predict(x[test])
    danger_prob = danger_model.predict_proba(x[test])[:, 1]
    hit_metrics = classification_metrics(targets["hit"][test], hit_prob)
    danger_metrics = classification_metrics(targets["danger"][test], danger_prob)
    return {
        **base,
        "model_status": "trained",
        "test": test,
        "hit_prob": hit_prob,
        "quality_pred": quality_pred,
        "danger_prob": danger_prob,
        "target_auc": hit_metrics.get("roc_auc"),
        "target_average_precision": hit_metrics.get("average_precision"),
        "quality_corr": safe_corr(quality_pred, targets["quality"][test]),
        "danger_auc": danger_metrics.get("roc_auc"),
        "danger_filter_usefulness": safe_corr(danger_prob, targets["danger"][test]),
    }


def _mode_row(
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    *,
    config: dict[str, Any],
    decision_mode: str,
    cost_proxy: float,
) -> dict[str, Any]:
    row = {**config, "decision_mode": decision_mode, **{key: trained.get(key) for key in (
        "model_status", "train_samples", "validation_samples", "test_samples", "target_auc",
        "target_average_precision", "quality_corr", "danger_auc", "danger_filter_usefulness",
    )}}
    if trained.get("model_status") != "trained":
        row["l2_status"] = "L2_ALPHA_INSUFFICIENT"
        return row
    test = np.asarray(trained["test"], dtype=np.int64)
    actual_hit = targets["hit"][test]
    actual_quality = targets["quality"][test]
    actual_danger = targets["danger"][test]
    actual_mae = targets["mae"][test]
    mask = selection_mask(decision_mode, trained["hit_prob"], trained["quality_pred"], trained["danger_prob"])
    selected_count = int(mask.sum())
    baseline_hit = _mean(actual_hit)
    baseline_quality = _mean(actual_quality)
    baseline_danger = _mean(actual_danger)
    baseline_p90 = _quantile(actual_mae, 0.90)
    selected_hit = _mean(actual_hit[mask]) if selected_count else None
    selected_quality = _mean(actual_quality[mask]) if selected_count else None
    selected_danger = _mean(actual_danger[mask]) if selected_count else None
    selected_p90 = _quantile(actual_mae[mask], 0.90) if selected_count else None
    row.update({
        "baseline_target_hit_rate": baseline_hit,
        "baseline_trade_quality": baseline_quality,
        "baseline_mae_danger": baseline_danger,
        "baseline_p90_mae": baseline_p90,
        "selected_fraction": selected_count / max(1, len(test)),
        "selected_count": selected_count,
        "selected_target_hit_rate": selected_hit,
        "selected_target_lift": None if selected_hit is None or baseline_hit is None else selected_hit - baseline_hit,
        "selected_quality_mean": selected_quality,
        "selected_quality_lift": None if selected_quality is None or baseline_quality is None else selected_quality - baseline_quality,
        "selected_net_quality_lift": (
            None if selected_quality is None or baseline_quality is None
            else selected_quality - baseline_quality - float(cost_proxy)
        ),
        "selected_mae_danger_rate": selected_danger,
        "selected_mae_danger_delta": (
            None if selected_danger is None or baseline_danger is None else selected_danger - baseline_danger
        ),
        "selected_p90_mae": selected_p90,
        "selected_p90_mae_delta": (
            None if selected_p90 is None or baseline_p90 is None else selected_p90 - baseline_p90
        ),
        "selected_avg_mfe": _mean(targets["mfe"][test][mask]) if selected_count else None,
        "selected_avg_mae": _mean(targets["mae"][test][mask]) if selected_count else None,
    })
    row["l2_status"] = classify_l2_alpha_candidate(row)
    return row


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = dict(rows[-1])
    numeric = {
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in {"lookback_days", "horizon_candles"}
    }
    for key in numeric:
        values = [finite(row.get(key)) for row in rows]
        numbers = [value for value in values if value is not None]
        latest[key] = float(np.mean(numbers)) if numbers else None
    latest["valid_fold_count"] = sum(1 for row in rows if row.get("model_status") == "trained")
    latest["test_samples"] = int(sum(int(row.get("test_samples") or 0) for row in rows))
    latest["selected_count"] = int(sum(int(row.get("selected_count") or 0) for row in rows))
    latest["l2_status"] = classify_l2_alpha_candidate(latest)
    return latest


def select_best_l2_alpha_by_symbol(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "L2_ALPHA_PROMISING": 3,
        "L2_ALPHA_MIXED": 2,
        "L2_ALPHA_BAD": 1,
        "L2_ALPHA_INSUFFICIENT": 0,
    }
    target_preference = {"hit3_before_minus2": 3, "hit5_before_minus3": 2, "hit6_before_minus4": 1}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["symbol"]), []).append(row)
    best: list[dict[str, Any]] = []
    for symbol, local_rows in grouped.items():
        selected = max(
            local_rows,
            key=lambda row: (
                order.get(str(row.get("l2_status")), -1),
                finite(row.get("selected_net_quality_lift"), -999.0),
                finite(row.get("selected_target_lift"), -999.0),
                -finite(row.get("selected_p90_mae_delta"), 999.0),
                finite(row.get("target_auc"), -999.0),
                target_preference.get(str(row.get("target_name")), 0),
            ),
        )
        selected = dict(selected)
        selected["best_reason"] = "discovery_best_requires_l3_frozen_confirmation"
        best.append(selected)
    return sorted(best, key=lambda row: (-order.get(str(row.get("l2_status")), -1), str(row["symbol"])))


def _splits(sample_count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.lockbox_mode == "last-block":
        fold = build_last_block_fold(
            sample_count,
            test_ratio=float(args.lockbox_test_ratio),
            min_train_samples=int(args.min_train_samples),
            min_test_samples=int(args.min_test_samples),
        )
        return [fold] if fold else []
    return temporal_folds(
        sample_count,
        fold_count=int(args.fold_count),
        min_train_samples=int(args.min_train_samples),
        min_test_samples=int(args.min_test_samples),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Aegis SHORT Alpha-Family L2 Research",
        "",
        "## Safety",
        "",
        "- RESEARCH_ONLY.",
        "- Models are trained in memory only; no shadow or active model artifacts are written.",
        "- Results compare candidate targets and decision modes and therefore require L3 frozen confirmation.",
        "- No live inference, active manifests, YAML, thresholds, PM2, or orders are changed.",
        "",
        "## Best By Symbol",
        "",
        "| Symbol | Alpha family | Target | Horizon | Decision mode | Status | Target lift | Quality lift | Net quality lift | p90 MAE delta |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["best_by_symbol"]:
        lines.append(
            f"| {row['symbol']} | {row['alpha_family']} | {row['target_name']} | "
            f"{row['horizon_candles']} | {row['decision_mode']} | {row['l2_status']} | "
            f"{finite(row.get('selected_target_lift'), 0.0):.4f} | "
            f"{finite(row.get('selected_quality_lift'), 0.0):.4f} | "
            f"{finite(row.get('selected_net_quality_lift'), 0.0):.4f} | "
            f"{finite(row.get('selected_p90_mae_delta'), 0.0):.4f} |"
        )
    for title, key in (
        ("Promising Candidates For L3", "promising"),
        ("Mixed Candidates", "mixed"),
        ("Bad Or Insufficient Candidates", "bad"),
    ):
        lines.extend(["", f"## {title}", ""])
        values = report[key]
        if not values:
            lines.append("- None.")
        for row in values:
            lines.append(
                f"- {row['symbol']} SHORT: `{row['target_name']}` h{row['horizon_candles']} "
                f"via `{row['decision_mode']}` -> `{row['l2_status']}`."
            )
    lines.extend([
        "",
        "## Method",
        "",
        "- Alternative targets are high/low path-aware; if target and stop occur in one candle, stop wins conservatively.",
        "- Quality rewards target hit and MFE, penalizes MAE and the configured fee/slippage proxy, bounded to [-1, 1].",
        "- `L2_ALPHA_PROMISING` is discovery evidence only because candidate target/mode selection occurs in this run.",
    ])
    if report["errors"]:
        lines.extend(["", "## Errors", "", f"- `{report['errors']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    feature_modes = _parse_csv(args.feature_mode) or ["selected_family"]
    if any(mode not in FEATURE_MODES for mode in feature_modes):
        raise ValueError(f"unsupported feature mode list: {feature_modes}")
    override_lookbacks = _parse_csv(args.lookback_days, int)
    override_horizons = _parse_csv(args.horizons, int)
    context_markets: dict[str, Any] = {}
    context_warning: str | None = None
    try:
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    except Exception as exc:
        context_warning = f"cross_symbol_context_unavailable:{exc!r}"
    cost_proxy = (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0
    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for symbol in symbols:
        config = DEFAULT_CONFIGS[symbol]
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for lookback_days in override_lookbacks or [int(config["lookback_days"])]:
                built = build_recent_dataset(symbol, lookback_days, save=False, market=market)
                base = built["dataset"]
                combined = apply_feature_set(base, market, "combined_v3", context_markets=context_markets)
                for feature_mode in feature_modes:
                    dataset = select_alpha_family_features(combined, config["alpha_family"], feature_mode)
                    for horizon in override_horizons or list(config["horizons"]):
                        if horizon <= 0 or horizon > 24:
                            raise ValueError(f"unsupported horizon {horizon}")
                        targets_for_config = config["target_candidates"]
                        for target_name in targets_for_config:
                            targets = compute_alpha_target_arrays(
                                market,
                                np.asarray(base["step"], dtype=np.int64),
                                side="SHORT",
                                target_name=target_name,
                                horizon=horizon,
                                cost_proxy=cost_proxy,
                            )
                            config_row = {
                                "symbol": symbol,
                                "side": "SHORT",
                                "alpha_family": config["alpha_family"],
                                "feature_set": config["feature_set"],
                                "feature_mode": feature_mode,
                                "lookback_days": lookback_days,
                                "horizon_candles": horizon,
                                "target_name": target_name,
                                "feature_count": int(np.asarray(dataset["X"]).shape[1]),
                                "missing_family_features": ",".join(dataset.get("missing_family_features", [])),
                                "lockbox_mode": args.lockbox_mode,
                            }
                            local: dict[str, list[dict[str, Any]]] = {mode: [] for mode in DECISION_MODES}
                            splits = _splits(len(base["step"]), args)
                            if not splits:
                                all_rows.append({
                                    **config_row,
                                    "decision_mode": "none",
                                    "model_status": "insufficient_split_samples",
                                    "test_samples": 0,
                                    "valid_fold_count": 0,
                                    "l2_status": "L2_ALPHA_INSUFFICIENT",
                                })
                                continue
                            for split_index, split in enumerate(splits, start=1):
                                trained = _fit_fold_predictions(
                                    dataset,
                                    targets,
                                    split,
                                    symbol=symbol,
                                    lookback_days=lookback_days,
                                    target_name=target_name,
                                    max_iter=60 if args.fast else 120,
                                )
                                for decision_mode in DECISION_MODES:
                                    result = _mode_row(
                                        trained,
                                        targets,
                                        config=config_row,
                                        decision_mode=decision_mode,
                                        cost_proxy=cost_proxy,
                                    )
                                    result["fold"] = split_index
                                    fold_rows.append(result)
                                    local[decision_mode].append(result)
                            for decision_mode, rows in local.items():
                                all_rows.append(_aggregate_rows(rows) if len(rows) > 1 else rows[0])
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    best_by_symbol = select_best_l2_alpha_by_symbol(all_rows)
    promising = [row for row in best_by_symbol if row.get("l2_status") == "L2_ALPHA_PROMISING"]
    mixed = [row for row in best_by_symbol if row.get("l2_status") == "L2_ALPHA_MIXED"]
    bad = [row for row in best_by_symbol if row.get("l2_status") in {"L2_ALPHA_BAD", "L2_ALPHA_INSUFFICIENT"}]
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_alpha_l2_research_{stamp}.md",
        "json": out_dir / f"aegis_short_alpha_l2_research_{stamp}.json",
        "all_configs_csv": out_dir / f"aegis_short_alpha_l2_all_configs_{stamp}.csv",
        "best_by_symbol_csv": out_dir / f"aegis_short_alpha_l2_best_by_symbol_{stamp}.csv",
        "promising_csv": out_dir / f"aegis_short_alpha_l2_promising_{stamp}.csv",
        "mixed_csv": out_dir / f"aegis_short_alpha_l2_mixed_{stamp}.csv",
        "bad_csv": out_dir / f"aegis_short_alpha_l2_bad_{stamp}.csv",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "symbols": symbols,
        "side": "SHORT",
        "feature_modes": feature_modes,
        "lockbox_mode": args.lockbox_mode,
        "discovery_only_requires_l3_frozen_confirmation": True,
        "candidate_configs": {symbol: DEFAULT_CONFIGS[symbol] for symbol in symbols},
        "cost_proxy": {"fee_bps": float(args.fee_bps), "slippage_bps": float(args.slippage_bps)},
        "quality_formula": "clip(hit + 0.30*min(mfe/target,1) - 0.70*min(mae/stop,1) - cost_proxy,-1,1)",
        "context_warning": context_warning,
        "all_config_rows": all_rows,
        "fold_rows": fold_rows,
        "best_by_symbol": best_by_symbol,
        "promising": promising,
        "mixed": mixed,
        "bad": bad,
        "errors": errors,
        "models_trained_in_memory_only": True,
        "model_artifacts_written": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(paths["md"], report)
    _write_csv(paths["all_configs_csv"], all_rows, ALL_CONFIG_COLUMNS)
    _write_csv(paths["best_by_symbol_csv"], best_by_symbol, ALL_CONFIG_COLUMNS + ("best_reason",))
    _write_csv(paths["promising_csv"], promising, ALL_CONFIG_COLUMNS + ("best_reason",))
    _write_csv(paths["mixed_csv"], mixed, ALL_CONFIG_COLUMNS + ("best_reason",))
    _write_csv(paths["bad_csv"], bad, ALL_CONFIG_COLUMNS + ("best_reason",))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only L2 alpha-family discovery for failed Turbo SHORT symbols.")
    parser.add_argument("--symbols")
    parser.add_argument("--lookback-days", help="Optional comma-separated override; default frozen 30.")
    parser.add_argument("--horizons", help="Optional comma-separated override; default frozen target horizons 12,24.")
    parser.add_argument("--feature-mode", default="selected_family", help="selected_family or combined_v3_all, comma-separated.")
    parser.add_argument("--lockbox-mode", choices=("last-block", "rolling-forward"), default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "best_by_symbol": [
            {
                "symbol": row["symbol"],
                "family": row["alpha_family"],
                "target": row["target_name"],
                "horizon": row["horizon_candles"],
                "mode": row["decision_mode"],
                "status": row["l2_status"],
            }
            for row in report["best_by_symbol"]
        ],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
