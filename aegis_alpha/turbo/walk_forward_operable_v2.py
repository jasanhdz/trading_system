#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aegis_alpha.turbo.train_operable_edge_v2 import (
    MODEL_FAMILIES,
    _classifier,
    _metadata,
    _model_seed,
    _regressor,
    _side_arrays,
    classification_metrics,
    finite_float,
    prediction_bucket_metrics,
    probability_bucket_metrics,
    regression_metrics,
    safe_corr,
    top_decile_metrics,
)


WALK_FORWARD_SCHEMA_VERSION = "aegis_turbo_operable_v2_walk_forward_v1"
DEFAULT_MIN_TRAIN_SAMPLES = 1000
DEFAULT_MIN_TEST_SAMPLES = 300


def temporal_folds(
    sample_count: int,
    *,
    fold_count: int = 4,
    train_ratio: float = 0.50,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    expanding_window: bool = True,
    min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
    min_test_samples: int = DEFAULT_MIN_TEST_SAMPLES,
) -> list[dict[str, Any]]:
    if sample_count <= 0 or fold_count <= 0:
        return []
    if min(train_ratio, validation_ratio, test_ratio) <= 0.0:
        raise ValueError("train, validation and test ratios must be positive")
    if train_ratio + validation_ratio + test_ratio > 1.0 + 1e-12:
        raise ValueError("train, validation and test ratios must sum to no more than 1")
    initial_train = max(int(sample_count * train_ratio), min_train_samples)
    validation_size = max(1, int(sample_count * validation_ratio))
    test_size = max(int(sample_count * test_ratio), min_test_samples)
    latest_train_end = sample_count - validation_size - test_size
    if latest_train_end < initial_train:
        return []
    train_ends = np.linspace(initial_train, latest_train_end, fold_count, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for position, train_end_value in enumerate(train_ends, start=1):
        train_end = int(train_end_value)
        validation_end = train_end + validation_size
        test_end = min(sample_count, validation_end + test_size)
        train_start = 0 if expanding_window else max(0, train_end - initial_train)
        key = (train_end, validation_end, test_end)
        if key in seen:
            continue
        seen.add(key)
        split = {
            "train": np.arange(train_start, train_end, dtype=np.int64),
            "validation": np.arange(train_end, validation_end, dtype=np.int64),
            "test": np.arange(validation_end, test_end, dtype=np.int64),
        }
        if len(split["train"]) < min_train_samples or len(split["test"]) < min_test_samples:
            continue
        folds.append({
            "fold": position,
            "expanding_window": bool(expanding_window),
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
            "ranges": {
                name: {"start": int(indices[0]), "end": int(indices[-1]), "count": int(len(indices))}
                for name, indices in split.items()
            },
        })
    return folds


def walkforward_model_path(
    run_dir: Path,
    fold_id: int,
    side: str,
    family: str,
    horizon: int,
    lookback_days: int,
) -> Path:
    if "active" in {part.lower() for part in run_dir.parts}:
        raise ValueError(f"walk-forward output directory must not contain active: {run_dir}")
    suffix = {
        "hit8_classifier": "hit8",
        "trade_quality_regressor": "quality",
        "mae_danger_classifier": "mae_danger",
    }[family]
    return run_dir / f"fold_{fold_id}" / f"v2_{side}_{suffix}_{horizon}_{lookback_days}d.joblib"


def _mean(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return finite_float(np.mean(finite)) if finite else None


def _minimum(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _low_decile_metrics(
    predictions: np.ndarray,
    hit8: np.ndarray,
    quality: np.ndarray,
    danger: np.ndarray,
    mae: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.float64)
    if not len(pred):
        return {"count": 0}
    threshold = float(np.quantile(pred, 0.10))
    mask = pred <= threshold
    return {
        "count": int(mask.sum()),
        "threshold": threshold,
        "avg_prediction": finite_float(np.mean(pred[mask])),
        "hit8_rate": finite_float(np.mean(hit8[mask])),
        "avg_trade_quality": finite_float(np.mean(quality[mask])),
        "mae_danger_rate": finite_float(np.mean(danger[mask])),
        "p90_mae": finite_float(np.quantile(mae[mask], 0.90)),
    }


def _add_lifts(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metrics["hit8_lift_vs_baseline"] = finite_float(
        float(metrics.get("hit8_rate") or 0.0) - float(baseline.get("hit8_rate") or 0.0)
    )
    metrics["quality_lift_vs_baseline"] = finite_float(
        float(metrics.get("avg_trade_quality") or 0.0) - float(baseline.get("avg_trade_quality") or 0.0)
    )
    metrics["danger_delta_vs_baseline"] = finite_float(
        float(metrics.get("mae_danger_rate") or 0.0) - float(baseline.get("mae_danger_rate") or 0.0)
    )
    metrics["p90_mae_delta_vs_baseline"] = finite_float(
        float(metrics.get("p90_mae") or 0.0) - float(baseline.get("p90_mae") or 0.0)
    )
    return metrics


def train_fold_models(
    dataset: dict[str, Any],
    *,
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    fold: dict[str, Any],
    run_dir: Path,
    save_models: bool = False,
    fast: bool = False,
) -> dict[str, Any]:
    normalized_side = side.lower()
    split = {name: np.asarray(fold[name], dtype=np.int64) for name in ("train", "validation", "test")}
    x = np.asarray(dataset["X"], dtype=np.float32)
    arrays = _side_arrays(dataset, normalized_side, horizon)
    feature_names = [str(item) for item in np.asarray(dataset["feature_names"]).tolist()]
    test = split["test"]
    baseline = {
        "hit8_rate": finite_float(np.mean(arrays["hit8"][test])),
        "avg_trade_quality": finite_float(np.mean(arrays["quality"][test])),
        "mae_danger_rate": finite_float(np.mean(arrays["danger"][test])),
        "p90_mae": finite_float(np.quantile(arrays["mae"][test], 0.90)),
    }
    v1_reference = _add_lifts(
        top_decile_metrics(
            arrays["v1_return"][test],
            arrays["hit8"][test],
            arrays["quality"][test],
            arrays["danger"][test],
            arrays["mae"][test],
        ),
        baseline,
    )
    result: dict[str, Any] = {
        "symbol": symbol,
        "side": normalized_side.upper(),
        "lookback_days": int(lookback_days),
        "horizon_candles": int(horizon),
        "fold": int(fold["fold"]),
        "ranges": fold.get("ranges", {}),
        "model_status": "trained",
        "sample_count": int(len(x)),
        "split_samples": {name: int(len(indices)) for name, indices in split.items()},
        "baseline_test": baseline,
        "v1_target_reference": {
            "warning": "outcome_target_reference_not_v1_model_prediction",
            "corr_hit8": safe_corr(arrays["v1_return"][test], arrays["hit8"][test]),
            "corr_trade_quality": safe_corr(arrays["v1_return"][test], arrays["quality"][test]),
            "top_decile": v1_reference,
        },
        "families": {},
    }
    max_iter = 60 if fast else 140
    for family, spec in MODEL_FAMILIES.items():
        target_name = spec["target_suffix"]
        target_key = f"{normalized_side}_{target_name}_{horizon}"
        target_alias = {"hit8_before_minus5": "hit8", "trade_quality": "quality", "mae_danger": "danger"}[target_name]
        y = arrays[target_alias]
        seed = _model_seed(symbol, normalized_side, lookback_days, f"{family}:fold:{fold['fold']}")
        metadata = _metadata(
            symbol=symbol,
            side=normalized_side,
            lookback_days=lookback_days,
            horizon=horizon,
            family=family,
            target_key=target_key,
            feature_names=feature_names,
            split=split,
            random_state=seed,
        )
        metadata.update({
            "walk_forward_only": True,
            "fold": int(fold["fold"]),
            "ranges": fold.get("ranges", {}),
        })
        path = walkforward_model_path(run_dir, int(fold["fold"]), normalized_side, family, horizon, lookback_days)
        family_result: dict[str, Any] = {
            "model_family": family,
            "target_key": target_key,
            "kind": spec["kind"],
            "model_path": str(path) if save_models else None,
            "metadata": metadata,
        }
        if spec["kind"] == "classifier" and len(np.unique(y[split["train"]])) < 2:
            family_result.update({"model_status": "insufficient_class_diversity", "reason": "single_class_in_train"})
            result["families"][family] = family_result
            result["model_status"] = "insufficient_class_diversity"
            continue
        if spec["kind"] == "regressor" and float(np.std(y[split["train"]])) <= 1e-12:
            family_result.update({"model_status": "insufficient_target_diversity", "reason": "constant_target_in_train"})
            result["families"][family] = family_result
            result["model_status"] = "insufficient_target_diversity"
            continue
        estimator = _classifier(max_iter, seed) if spec["kind"] == "classifier" else _regressor(max_iter, seed)
        estimator.fit(x[split["train"]], y[split["train"]])
        if spec["kind"] == "classifier":
            validation_pred = estimator.predict_proba(x[split["validation"]])[:, 1]
            test_pred = estimator.predict_proba(x[test])[:, 1]
            family_result["validation_metrics"] = classification_metrics(y[split["validation"]], validation_pred)
            family_result["test_metrics"] = classification_metrics(y[test], test_pred)
            family_result["buckets"] = probability_bucket_metrics(
                test_pred, y[test], arrays["quality"][test], arrays["mae"][test], arrays["danger"][test], arrays["hit8"][test]
            )
        else:
            validation_pred = estimator.predict(x[split["validation"]])
            test_pred = estimator.predict(x[test])
            family_result["validation_metrics"] = regression_metrics(y[split["validation"]], validation_pred)
            family_result["test_metrics"] = regression_metrics(y[test], test_pred)
            family_result["buckets"] = prediction_bucket_metrics(
                test_pred, arrays["quality"][test], arrays["hit8"][test], arrays["danger"][test], arrays["mae"][test]
            )
        family_result["test_comparisons"] = {
            "prediction_vs_hit8": safe_corr(test_pred, arrays["hit8"][test]),
            "prediction_vs_trade_quality": safe_corr(test_pred, arrays["quality"][test]),
            "prediction_vs_mae_danger": safe_corr(test_pred, arrays["danger"][test]),
        }
        top = _add_lifts(
            top_decile_metrics(test_pred, arrays["hit8"][test], arrays["quality"][test], arrays["danger"][test], arrays["mae"][test]),
            baseline,
        )
        family_result["top_decile"] = top
        if family == "mae_danger_classifier":
            safe_bucket = _add_lifts(
                _low_decile_metrics(test_pred, arrays["hit8"][test], arrays["quality"][test], arrays["danger"][test], arrays["mae"][test]),
                baseline,
            )
            family_result["safe_low_danger_decile"] = safe_bucket
            family_result["usefulness_as_filter"] = finite_float(
                float(top.get("mae_danger_rate") or 0.0) - float(safe_bucket.get("mae_danger_rate") or 0.0)
            )
        family_result["model_status"] = "trained"
        if save_models:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"metadata": metadata, "feature_names": feature_names, "estimator": estimator}, path)
        result["families"][family] = family_result
    return result


def evaluate_fold(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return train_fold_models(*args, **kwargs)


def _family_value(fold: dict[str, Any], family: str, *keys: str) -> Any:
    current: Any = (fold.get("families") or {}).get(family, {})
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def classify_walk_forward_status(summary: dict[str, Any], valid_folds: list[dict[str, Any]]) -> str:
    if int(summary.get("valid_fold_count") or 0) < 3:
        return "INSUFFICIENT_DATA"
    hit_positive = sum(
        float(_family_value(fold, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline") or 0.0) > 0.0
        for fold in valid_folds
    )
    quality_positive = sum(
        float(_family_value(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline") or 0.0) > 0.0
        for fold in valid_folds
    )
    quality_above_zero = sum(
        float(_family_value(fold, "trade_quality_regressor", "top_decile", "avg_trade_quality") or -1.0) > 0.0
        for fold in valid_folds
    )
    required_majority = math.ceil(len(valid_folds) / 2)
    latest = valid_folds[-1]
    latest_hit = float(_family_value(latest, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline") or 0.0)
    latest_quality = float(_family_value(latest, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline") or 0.0)
    latest_p90 = float(_family_value(latest, "trade_quality_regressor", "top_decile", "p90_mae_delta_vs_baseline") or 0.0)
    stable = (
        hit_positive >= required_majority
        and quality_positive >= required_majority
        and quality_above_zero >= required_majority
        and latest_hit >= 0.0
        and latest_quality >= 0.0
        and latest_p90 <= 0.001
    )
    if stable:
        return "WALK_FORWARD_PROMISING"
    if hit_positive == 0 and quality_positive == 0:
        return "WALK_FORWARD_BAD"
    if latest_hit < 0.0 and latest_quality < 0.0 and quality_positive < required_majority:
        return "WALK_FORWARD_BAD"
    return "WALK_FORWARD_MIXED"


def aggregate_walk_forward_results(
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    fold_results: list[dict[str, Any]],
    requested_fold_count: int,
) -> dict[str, Any]:
    valid = [fold for fold in fold_results if fold.get("model_status") == "trained"]
    hit_lifts = [_family_value(fold, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline") for fold in valid]
    quality_lifts = [_family_value(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline") for fold in valid]
    quality_top_values = [_family_value(fold, "trade_quality_regressor", "top_decile", "avg_trade_quality") for fold in valid]
    auc_values = [_family_value(fold, "hit8_classifier", "test_metrics", "roc_auc") for fold in valid]
    quality_corr_values = [_family_value(fold, "trade_quality_regressor", "test_metrics", "spearman") for fold in valid]
    danger_auc_values = [_family_value(fold, "mae_danger_classifier", "test_metrics", "roc_auc") for fold in valid]
    usefulness_values = [_family_value(fold, "mae_danger_classifier", "usefulness_as_filter") for fold in valid]
    positive_hit_fraction = _mean([1.0 if float(value or 0.0) > 0 else 0.0 for value in hit_lifts])
    positive_quality_fraction = _mean([1.0 if float(value or 0.0) > 0 else 0.0 for value in quality_lifts])
    positive_actual_quality_fraction = _mean([1.0 if float(value or -1.0) > 0 else 0.0 for value in quality_top_values])
    stability_components = [
        positive_hit_fraction,
        positive_quality_fraction,
        positive_actual_quality_fraction,
        _mean([max(0.0, min(1.0, (float(value) - 0.5) * 2.0)) for value in auc_values if value is not None]),
    ]
    stability = _mean(stability_components)
    summary: dict[str, Any] = {
        "symbol": symbol,
        "side": side.upper(),
        "lookback_days": int(lookback_days),
        "horizon_candles": int(horizon),
        "fold_count": int(requested_fold_count),
        "generated_fold_count": int(len(fold_results)),
        "valid_fold_count": int(len(valid)),
        "baseline_hit8_mean": _mean([fold.get("baseline_test", {}).get("hit8_rate") for fold in valid]),
        "baseline_quality_mean": _mean([fold.get("baseline_test", {}).get("avg_trade_quality") for fold in valid]),
        "baseline_danger_mean": _mean([fold.get("baseline_test", {}).get("mae_danger_rate") for fold in valid]),
        "v1_corr_quality_mean": _mean([fold.get("v1_target_reference", {}).get("corr_trade_quality") for fold in valid]),
        "v2_hit8_auc_mean": _mean(auc_values),
        "v2_hit8_auc_min": _minimum(auc_values),
        "v2_quality_corr_mean": _mean(quality_corr_values),
        "v2_quality_corr_min": _minimum(quality_corr_values),
        "v2_danger_auc_mean": _mean(danger_auc_values),
        "hit8_top_decile_lift_mean": _mean(hit_lifts),
        "hit8_top_decile_lift_min": _minimum(hit_lifts),
        "quality_top_decile_lift_mean": _mean(quality_lifts),
        "quality_top_decile_lift_min": _minimum(quality_lifts),
        "danger_filter_usefulness_mean": _mean(usefulness_values),
        "stability_score": stability,
        "decay_score": (
            finite_float(float(quality_lifts[-1]) - float(quality_lifts[0]))
            if len(quality_lifts) >= 2 and quality_lifts[-1] is not None and quality_lifts[0] is not None
            else None
        ),
        "v1_reference_warning": "outcome_target_reference_not_v1_model_prediction",
    }
    summary["recommendation"] = classify_walk_forward_status(summary, valid)
    return summary


def run_walk_forward(
    dataset: dict[str, Any],
    *,
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    fold_count: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    expanding_window: bool,
    min_train_samples: int,
    min_test_samples: int,
    run_dir: Path,
    save_models: bool = False,
    fast: bool = False,
) -> dict[str, Any]:
    folds = temporal_folds(
        len(np.asarray(dataset["X"])),
        fold_count=fold_count,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        expanding_window=expanding_window,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
    )
    fold_results = [
        evaluate_fold(
            dataset,
            symbol=symbol,
            side=side,
            lookback_days=lookback_days,
            horizon=horizon,
            fold=fold,
            run_dir=run_dir,
            save_models=save_models,
            fast=fast,
        )
        for fold in folds
    ]
    return {
        "summary": aggregate_walk_forward_results(symbol, side, lookback_days, horizon, fold_results, fold_count),
        "folds": fold_results,
    }
