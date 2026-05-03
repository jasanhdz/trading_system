#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, roc_auc_score

from aegis_alpha.edge.common import save_model_bundle, safe_float, write_json


def _class_weights(y: np.ndarray) -> np.ndarray:
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    return weights.astype(np.float32)


def _classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "positive_rate": safe_float(np.mean(y)),
        "prob_mean": safe_float(np.mean(proba)),
        "log_loss": safe_float(log_loss(y, np.clip(proba, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = safe_float(roc_auc_score(y, proba))
        out["average_precision"] = safe_float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = 0.0
        out["average_precision"] = 0.0
    return out


def train_edge_model(
    dataset_path: Path,
    output_path: Path,
    report_path: Path,
    test_pct: float,
    max_iter: int,
    learning_rate: float,
    l2_regularization: float,
    max_leaf_nodes: int,
) -> None:
    data = np.load(dataset_path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    long_good = data["long_good"].astype(np.int8)
    short_good = data["short_good"].astype(np.int8)
    long_return = data["long_return"].astype(np.float32)
    short_return = data["short_return"].astype(np.float32)
    feature_names = data["feature_names"].astype(str).tolist()

    split = int(len(x) * (1.0 - test_pct))
    if split <= 0 or split >= len(x):
        raise ValueError(f"Invalid chronological split for n={len(x)}, test_pct={test_pct}")

    x_train, x_test = x[:split], x[split:]
    y_long_train, y_long_test = long_good[:split], long_good[split:]
    y_short_train, y_short_test = short_good[:split], short_good[split:]
    long_ret_train, long_ret_test = long_return[:split], long_return[split:]
    short_ret_train, short_ret_test = short_return[:split], short_return[split:]

    clf_kwargs: dict[str, Any] = {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "max_leaf_nodes": max_leaf_nodes,
        "early_stopping": True,
        "random_state": 4667,
    }
    reg_kwargs = dict(clf_kwargs)
    long_clf = HistGradientBoostingClassifier(**clf_kwargs)
    short_clf = HistGradientBoostingClassifier(**clf_kwargs)
    long_reg = HistGradientBoostingRegressor(loss="absolute_error", **reg_kwargs)
    short_reg = HistGradientBoostingRegressor(loss="absolute_error", **reg_kwargs)

    print("Training LONG classifier")
    long_clf.fit(x_train, y_long_train, sample_weight=_class_weights(y_long_train))
    print("Training SHORT classifier")
    short_clf.fit(x_train, y_short_train, sample_weight=_class_weights(y_short_train))
    print("Training LONG return regressor")
    long_reg.fit(x_train, long_ret_train)
    print("Training SHORT return regressor")
    short_reg.fit(x_train, short_ret_train)

    long_proba = long_clf.predict_proba(x_test)[:, 1]
    short_proba = short_clf.predict_proba(x_test)[:, 1]
    pred_long_return = long_reg.predict(x_test)
    pred_short_return = short_reg.predict(x_test)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "schema_version": "aegis_edge_model_v1",
        "created_at": created_at,
        "dataset_path": str(dataset_path),
        "test_pct": test_pct,
        "split_index": split,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "feature_count": int(x.shape[1]),
        "model_type": "sklearn.HistGradientBoosting",
        "params": {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "max_leaf_nodes": max_leaf_nodes,
        },
    }
    report = {
        **metadata,
        "metrics": {
            "long_classifier": _classification_metrics(y_long_test, long_proba),
            "short_classifier": _classification_metrics(y_short_test, short_proba),
            "long_return_mae": safe_float(mean_absolute_error(long_ret_test, pred_long_return)),
            "short_return_mae": safe_float(mean_absolute_error(short_ret_test, pred_short_return)),
            "actual_long_return_mean": safe_float(np.mean(long_ret_test)),
            "actual_short_return_mean": safe_float(np.mean(short_ret_test)),
            "pred_long_return_mean": safe_float(np.mean(pred_long_return)),
            "pred_short_return_mean": safe_float(np.mean(pred_short_return)),
        },
    }

    bundle = {
        "metadata": metadata,
        "feature_names": feature_names,
        "long_classifier": long_clf,
        "short_classifier": short_clf,
        "long_return_regressor": long_reg,
        "short_return_regressor": short_reg,
    }
    save_model_bundle(output_path, bundle)
    write_json(report_path, report)
    print(f"Model saved -> {output_path}")
    print(f"Train report saved -> {report_path}")
    print(
        "Holdout: "
        f"long_auc={report['metrics']['long_classifier']['roc_auc']:.3f} "
        f"short_auc={report['metrics']['short_classifier']['roc_auc']:.3f} "
        f"long_ap={report['metrics']['long_classifier']['average_precision']:.3f} "
        f"short_ap={report['metrics']['short_classifier']['average_precision']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/edge_dataset_v030.npz")
    parser.add_argument("--output", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--report", default="aegis_alpha/logs/edge/edge_train_report_v030.json")
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    args = parser.parse_args()

    train_edge_model(
        Path(args.dataset),
        Path(args.output),
        Path(args.report),
        args.test_pct,
        args.max_iter,
        args.learning_rate,
        args.l2_regularization,
        args.max_leaf_nodes,
    )


if __name__ == "__main__":
    main()
