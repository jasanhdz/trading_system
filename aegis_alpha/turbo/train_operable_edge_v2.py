#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
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


MODEL_SCHEMA_VERSION = "aegis_turbo_operable_v2_model_v1"
MIN_TRAIN_SAMPLES = 500
PROBABILITY_BUCKET_EDGES = tuple(i / 10.0 for i in range(11))
MODEL_FAMILIES = {
    "hit8_classifier": {"kind": "classifier", "target_suffix": "hit8_before_minus5"},
    "trade_quality_regressor": {"kind": "regressor", "target_suffix": "trade_quality"},
    "mae_danger_classifier": {"kind": "classifier", "target_suffix": "mae_danger"},
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def temporal_split_indices(sample_count: int) -> dict[str, np.ndarray]:
    if sample_count < 3:
        raise ValueError("at least three samples required for temporal split")
    train_end = max(1, int(sample_count * 0.60))
    validation_end = max(train_end + 1, int(sample_count * 0.80))
    validation_end = min(validation_end, sample_count - 1)
    return {
        "train": np.arange(0, train_end, dtype=np.int64),
        "validation": np.arange(train_end, validation_end, dtype=np.int64),
        "test": np.arange(validation_end, sample_count, dtype=np.int64),
    }


def feature_schema_hash(feature_names: list[str] | np.ndarray) -> str:
    names = [str(item) for item in list(feature_names)]
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def safe_corr(left: np.ndarray, right: np.ndarray, *, method: str = "pearson") -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or float(np.std(x[valid])) <= 1e-12 or float(np.std(y[valid])) <= 1e-12:
        return None
    if method == "spearman":
        result = spearmanr(x[valid], y[valid]).statistic
    else:
        result = np.corrcoef(x[valid], y[valid])[0, 1]
    return finite_float(result)


def distribution_summary(values: np.ndarray, *, classifier: bool = False) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    result: dict[str, float | int | None] = {
        "count": int(len(array)),
        "mean": finite_float(np.mean(array)),
        "median": finite_float(np.median(array)),
        "p25": finite_float(np.quantile(array, 0.25)),
        "p75": finite_float(np.quantile(array, 0.75)),
    }
    if classifier:
        result["positive_rate"] = finite_float(np.mean(array > 0.5))
    return result


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=np.int8)
    prob = np.asarray(probabilities, dtype=np.float64)
    predicted = (prob >= 0.5).astype(np.int8)
    metrics: dict[str, float | int | None] = {
        "count": int(len(y)),
        "positive_rate": finite_float(np.mean(y)) if len(y) else None,
        "accuracy": finite_float(accuracy_score(y, predicted)) if len(y) else None,
        "precision": finite_float(precision_score(y, predicted, zero_division=0)) if len(y) else None,
        "recall": finite_float(recall_score(y, predicted, zero_division=0)) if len(y) else None,
        "f1": finite_float(f1_score(y, predicted, zero_division=0)) if len(y) else None,
        "brier_score": finite_float(brier_score_loss(y, prob)) if len(y) else None,
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(y)) >= 2:
        metrics["roc_auc"] = finite_float(roc_auc_score(y, prob))
        metrics["average_precision"] = finite_float(average_precision_score(y, prob))
    return metrics


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if not len(y):
        return {"count": 0, "mae": None, "rmse": None, "r2": None, "pearson": None, "spearman": None}
    return {
        "count": int(len(y)),
        "mae": finite_float(mean_absolute_error(y, pred)),
        "rmse": finite_float(np.sqrt(mean_squared_error(y, pred))),
        "r2": finite_float(r2_score(y, pred)) if len(y) >= 2 else None,
        "pearson": safe_corr(pred, y),
        "spearman": safe_corr(pred, y, method="spearman"),
    }


def probability_bucket_metrics(
    probabilities: np.ndarray,
    actual: np.ndarray,
    trade_quality: np.ndarray,
    mae: np.ndarray,
    danger: np.ndarray,
    hit8: np.ndarray,
) -> list[dict[str, Any]]:
    prob = np.asarray(probabilities, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, lower in enumerate(PROBABILITY_BUCKET_EDGES[:-1]):
        upper = PROBABILITY_BUCKET_EDGES[index + 1]
        mask = (prob >= lower) & (prob < upper) if upper < 1.0 else (prob >= lower) & (prob <= upper)
        rows.append({
            "bucket": f"{lower:.1f}-{upper:.1f}",
            "count": int(mask.sum()),
            "avg_pred": finite_float(np.mean(prob[mask])) if mask.any() else None,
            "actual_rate": finite_float(np.mean(actual[mask])) if mask.any() else None,
            "avg_trade_quality": finite_float(np.mean(trade_quality[mask])) if mask.any() else None,
            "avg_mae": finite_float(np.mean(mae[mask])) if mask.any() else None,
            "mae_danger_rate": finite_float(np.mean(danger[mask])) if mask.any() else None,
            "hit8_rate": finite_float(np.mean(hit8[mask])) if mask.any() else None,
        })
    return rows


def prediction_bucket_metrics(
    predictions: np.ndarray,
    actual_quality: np.ndarray,
    hit8: np.ndarray,
    danger: np.ndarray,
    mae: np.ndarray,
    bucket_count: int = 10,
) -> list[dict[str, Any]]:
    pred = np.asarray(predictions, dtype=np.float64)
    if not len(pred):
        return []
    order = np.argsort(pred)
    ranks = np.empty(len(pred), dtype=np.int64)
    ranks[order] = np.arange(len(pred), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for bucket in range(bucket_count):
        lower = bucket / bucket_count
        upper = (bucket + 1) / bucket_count
        mask = (ranks >= math.floor(len(pred) * lower)) & (ranks < math.floor(len(pred) * upper))
        if bucket == bucket_count - 1:
            mask = ranks >= math.floor(len(pred) * lower)
        rows.append({
            "bucket": f"q{bucket + 1}",
            "count": int(mask.sum()),
            "avg_pred_quality": finite_float(np.mean(pred[mask])) if mask.any() else None,
            "actual_avg_quality": finite_float(np.mean(actual_quality[mask])) if mask.any() else None,
            "actual_hit8_rate": finite_float(np.mean(hit8[mask])) if mask.any() else None,
            "actual_mae_danger_rate": finite_float(np.mean(danger[mask])) if mask.any() else None,
            "actual_p90_mae": finite_float(np.quantile(mae[mask], 0.90)) if mask.any() else None,
        })
    return rows


def top_decile_metrics(
    predictions: np.ndarray,
    hit8: np.ndarray,
    quality: np.ndarray,
    danger: np.ndarray,
    mae: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.float64)
    if not len(pred):
        return {"count": 0}
    threshold = float(np.quantile(pred, 0.90))
    mask = pred >= threshold
    return {
        "count": int(mask.sum()),
        "threshold": threshold,
        "avg_prediction": finite_float(np.mean(pred[mask])),
        "hit8_rate": finite_float(np.mean(hit8[mask])),
        "avg_trade_quality": finite_float(np.mean(quality[mask])),
        "mae_danger_rate": finite_float(np.mean(danger[mask])),
        "p90_mae": finite_float(np.quantile(mae[mask], 0.90)),
    }


def research_model_path(run_dir: Path, side: str, family: str, horizon: int, lookback_days: int) -> Path:
    if "active" in {part.lower() for part in run_dir.parts}:
        raise ValueError(f"research output directory must not contain active: {run_dir}")
    suffix = {
        "hit8_classifier": "hit8",
        "trade_quality_regressor": "quality",
        "mae_danger_classifier": "mae_danger",
    }[family]
    return run_dir / f"v2_{side}_{suffix}_{horizon}_{lookback_days}d.joblib"


def _classifier(max_iter: int, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        l2_regularization=0.08,
        max_leaf_nodes=15,
        early_stopping=True,
        random_state=random_state,
    )


def _regressor(max_iter: int, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=0.05,
        l2_regularization=0.08,
        max_leaf_nodes=15,
        early_stopping=True,
        random_state=random_state,
    )


def _model_seed(symbol: str, side: str, lookback_days: int, family: str) -> int:
    encoded = f"{symbol}:{side}:{lookback_days}:{family}".encode("utf-8")
    return 1000 + int(hashlib.sha256(encoded).hexdigest()[:6], 16) % 100000


def _metadata(
    *,
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    family: str,
    target_key: str,
    feature_names: list[str],
    split: dict[str, np.ndarray],
    random_state: int,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": utc_iso(),
        "symbol": symbol,
        "side": side,
        "lookback_days": int(lookback_days),
        "horizon_candles": int(horizon),
        "model_family": family,
        "target_key": target_key,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_schema_hash": feature_schema_hash(feature_names),
        "sklearn_version": sklearn_version,
        "train_samples": int(len(split["train"])),
        "validation_samples": int(len(split["validation"])),
        "test_samples": int(len(split["test"])),
        "random_state": int(random_state),
        "research_only": True,
        "not_live_promoted": True,
    }


def _side_arrays(dataset: dict[str, Any], side: str, horizon: int) -> dict[str, np.ndarray]:
    prefix = side.lower()
    return {
        "hit8": np.asarray(dataset[f"{prefix}_hit8_before_minus5_{horizon}"], dtype=np.int8),
        "quality": np.asarray(dataset[f"{prefix}_trade_quality_{horizon}"], dtype=np.float32),
        "danger": np.asarray(dataset[f"{prefix}_mae_danger_{horizon}"], dtype=np.int8),
        "mae": np.asarray(dataset[f"{prefix}_mae_{horizon}"], dtype=np.float32),
        "v1_return": np.asarray(dataset[f"{prefix}_net_return_12"], dtype=np.float32),
    }


def classify_research_status(result: dict[str, Any]) -> str:
    if result.get("model_status") != "trained":
        return "INSUFFICIENT_DATA"
    baseline = float(result["baseline_test"]["hit8_rate"] or 0.0)
    hit_top = result["families"]["hit8_classifier"].get("top_decile", {})
    quality_top = result["families"]["trade_quality_regressor"].get("top_decile", {})
    danger_metrics = result["families"]["mae_danger_classifier"].get("test_metrics", {})
    hit_promising = (
        int(hit_top.get("count") or 0) >= 10
        and float(hit_top.get("hit8_rate") or 0.0) >= baseline * 1.30
        and float(hit_top.get("avg_trade_quality") or -1.0) > 0.0
    )
    quality_promising = (
        int(quality_top.get("count") or 0) >= 10
        and float(quality_top.get("hit8_rate") or 0.0) >= baseline * 1.30
        and float(quality_top.get("avg_trade_quality") or -1.0) > 0.0
        and float(result["families"]["trade_quality_regressor"]["test_metrics"].get("spearman") or 0.0) > 0.0
    )
    danger_helpful = float(danger_metrics.get("roc_auc") or 0.0) >= 0.55
    if hit_promising and quality_promising and danger_helpful:
        return "PROMISING_RESEARCH"
    if hit_promising or quality_promising or danger_helpful:
        return "MIXED_RESEARCH"
    return "BAD_RESEARCH"


def train_side_models(
    dataset: dict[str, Any],
    *,
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    run_dir: Path,
    save_models: bool = True,
    fast: bool = False,
) -> dict[str, Any]:
    normalized_side = side.lower()
    x = np.asarray(dataset["X"], dtype=np.float32)
    arrays = _side_arrays(dataset, normalized_side, horizon)
    feature_names = [str(item) for item in dataset["feature_names"].tolist()]
    feature_set = str(dataset.get("feature_set", "base"))
    base_feature_count = int(dataset.get("base_feature_count", len(feature_names)))
    new_feature_count = int(dataset.get("new_feature_count", 0))
    operable_v2_feature_count = int(dataset.get("operable_v2_feature_count", 0))
    operable_v3_feature_count = int(dataset.get("operable_v3_feature_count", 0))
    if len(x) < MIN_TRAIN_SAMPLES:
        return {
            "symbol": symbol,
            "side": normalized_side.upper(),
            "lookback_days": int(lookback_days),
            "horizon_candles": int(horizon),
            "model_status": "insufficient_data",
            "reason": "minimum_sample_count_not_met",
            "sample_count": int(len(x)),
            "research_status": "INSUFFICIENT_DATA",
        }
    split = temporal_split_indices(len(x))
    distributions = {
        name: {
            section: distribution_summary(values[indices], classifier=name in {"hit8", "danger"})
            for section, indices in split.items()
        }
        for name, values in arrays.items()
        if name != "mae"
    }
    result: dict[str, Any] = {
        "symbol": symbol,
        "side": normalized_side.upper(),
        "lookback_days": int(lookback_days),
        "horizon_candles": int(horizon),
        "model_status": "trained",
        "sample_count": int(len(x)),
        "feature_set": feature_set,
        "feature_count": int(len(feature_names)),
        "base_feature_count": base_feature_count,
        "new_feature_count": new_feature_count,
        "operable_v2_feature_count": operable_v2_feature_count,
        "operable_v3_feature_count": operable_v3_feature_count,
        "feature_schema_hash": feature_schema_hash(feature_names),
        "feature_diagnostics": dataset.get("feature_diagnostics"),
        "split_samples": {name: int(len(indices)) for name, indices in split.items()},
        "target_distributions": distributions,
        "baseline_test": {
            "hit8_rate": finite_float(np.mean(arrays["hit8"][split["test"]])),
            "avg_trade_quality": finite_float(np.mean(arrays["quality"][split["test"]])),
            "mae_danger_rate": finite_float(np.mean(arrays["danger"][split["test"]])),
            "p90_mae": finite_float(np.quantile(arrays["mae"][split["test"]], 0.90)),
            "v1_corr_hit8": safe_corr(arrays["v1_return"][split["test"]], arrays["hit8"][split["test"]]),
            "v1_corr_trade_quality": safe_corr(arrays["v1_return"][split["test"]], arrays["quality"][split["test"]]),
        },
        "families": {},
    }
    max_iter = 60 if fast else 140
    for family, spec in MODEL_FAMILIES.items():
        target_name = spec["target_suffix"]
        target_key = f"{normalized_side}_{target_name}_{horizon}"
        y = arrays[{"hit8_before_minus5": "hit8", "trade_quality": "quality", "mae_danger": "danger"}[target_name]]
        seed = _model_seed(symbol, normalized_side, lookback_days, family)
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
            "feature_set": feature_set,
            "base_feature_count": base_feature_count,
            "new_feature_count": new_feature_count,
            "operable_v2_feature_count": operable_v2_feature_count,
            "operable_v3_feature_count": operable_v3_feature_count,
            "feature_diagnostics": dataset.get("feature_diagnostics"),
        })
        path = research_model_path(run_dir, normalized_side, family, horizon, lookback_days)
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
            test_pred = estimator.predict_proba(x[split["test"]])[:, 1]
            family_result["validation_metrics"] = classification_metrics(y[split["validation"]], validation_pred)
            family_result["test_metrics"] = classification_metrics(y[split["test"]], test_pred)
            family_result["buckets"] = probability_bucket_metrics(
                test_pred,
                y[split["test"]],
                arrays["quality"][split["test"]],
                arrays["mae"][split["test"]],
                arrays["danger"][split["test"]],
                arrays["hit8"][split["test"]],
            )
            family_result["top_decile"] = top_decile_metrics(
                test_pred,
                arrays["hit8"][split["test"]],
                arrays["quality"][split["test"]],
                arrays["danger"][split["test"]],
                arrays["mae"][split["test"]],
            )
            family_result["test_comparisons"] = {
                "prediction_vs_hit8": safe_corr(test_pred, arrays["hit8"][split["test"]]),
                "prediction_vs_trade_quality": safe_corr(test_pred, arrays["quality"][split["test"]]),
                "prediction_vs_mae_danger": safe_corr(test_pred, arrays["danger"][split["test"]]),
            }
        else:
            validation_pred = estimator.predict(x[split["validation"]])
            test_pred = estimator.predict(x[split["test"]])
            family_result["validation_metrics"] = regression_metrics(y[split["validation"]], validation_pred)
            family_result["test_metrics"] = regression_metrics(y[split["test"]], test_pred)
            family_result["buckets"] = prediction_bucket_metrics(
                test_pred,
                arrays["quality"][split["test"]],
                arrays["hit8"][split["test"]],
                arrays["danger"][split["test"]],
                arrays["mae"][split["test"]],
            )
            family_result["top_decile"] = top_decile_metrics(
                test_pred,
                arrays["hit8"][split["test"]],
                arrays["quality"][split["test"]],
                arrays["danger"][split["test"]],
                arrays["mae"][split["test"]],
            )
            family_result["test_comparisons"] = {
                "prediction_vs_hit8": safe_corr(test_pred, arrays["hit8"][split["test"]]),
                "prediction_vs_trade_quality": safe_corr(test_pred, arrays["quality"][split["test"]]),
                "prediction_vs_mae_danger": safe_corr(test_pred, arrays["danger"][split["test"]]),
            }
        family_result["model_status"] = "trained"
        if save_models:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"metadata": metadata, "feature_names": feature_names, "estimator": estimator}, path)
        result["families"][family] = family_result
    result["research_status"] = classify_research_status(result)
    return result
