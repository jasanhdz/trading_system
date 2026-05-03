#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from aegis_alpha.edge.common import save_model_bundle, safe_float, write_json


def _class_weights(y: np.ndarray) -> np.ndarray:
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    return weights.astype(np.float32)


def _classification_metrics(y: np.ndarray, proba: np.ndarray, returns: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "samples": safe_float(len(y)),
        "positive_rate": safe_float(np.mean(y)),
        "prob_mean": safe_float(np.mean(proba)),
        "avg_trade_return": safe_float(np.mean(returns)),
        "log_loss": safe_float(log_loss(y, np.clip(proba, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = safe_float(roc_auc_score(y, proba))
        out["average_precision"] = safe_float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = 0.0
        out["average_precision"] = 0.0

    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        mask = proba >= threshold
        prefix = f"threshold_{threshold:.2f}".replace(".", "p")
        out[f"{prefix}_kept"] = safe_float(mask.sum())
        out[f"{prefix}_kept_pct"] = safe_float(np.mean(mask))
        out[f"{prefix}_win_rate"] = safe_float(np.mean(y[mask])) if np.any(mask) else 0.0
        out[f"{prefix}_avg_return"] = safe_float(np.mean(returns[mask])) if np.any(mask) else 0.0
    return out


def train_meta_filter(
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
    y = data["y"].astype(np.int8)
    returns = data["simulated_trade_return"].astype(np.float32)
    steps = data["step"].astype(np.int64)
    feature_names = data["feature_names"].astype(str).tolist()

    order = np.argsort(steps, kind="stable")
    x = x[order]
    y = y[order]
    returns = returns[order]
    steps = steps[order]

    split = int(len(x) * (1.0 - test_pct))
    if split <= 0 or split >= len(x):
        raise ValueError(f"Invalid chronological split for n={len(x)}, test_pct={test_pct}")

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    ret_train, ret_test = returns[:split], returns[split:]

    params: dict[str, Any] = {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "max_leaf_nodes": max_leaf_nodes,
        "early_stopping": True,
        "random_state": 4667,
    }
    clf = HistGradientBoostingClassifier(**params)
    print("Training LONG edge meta-filter")
    clf.fit(x_train, y_train, sample_weight=_class_weights(y_train))

    train_proba = clf.predict_proba(x_train)[:, 1]
    test_proba = clf.predict_proba(x_test)[:, 1]
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "schema_version": "aegis_long_edge_meta_filter_v1",
        "created_at": created_at,
        "dataset_path": str(dataset_path),
        "test_pct": test_pct,
        "split_index": int(split),
        "split_step": int(steps[split]),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "feature_count": int(x.shape[1]),
        "target": "profitable_net_long_candidate",
        "model_type": "sklearn.HistGradientBoostingClassifier",
        "params": params,
    }
    report = {
        **metadata,
        "metrics": {
            "train": _classification_metrics(y_train, train_proba, ret_train),
            "test": _classification_metrics(y_test, test_proba, ret_test),
        },
    }
    bundle = {
        "metadata": metadata,
        "feature_names": feature_names,
        "classifier": clf,
    }
    save_model_bundle(output_path, bundle)
    write_json(report_path, report)
    print(f"Model saved -> {output_path}")
    print(f"Train report saved -> {report_path}")
    print(
        "Holdout: "
        f"auc={report['metrics']['test']['roc_auc']:.3f} "
        f"ap={report['metrics']['test']['average_precision']:.3f} "
        f"positive_rate={report['metrics']['test']['positive_rate']:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/long_edge_candidates_v040.npz")
    parser.add_argument("--output", default="aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib")
    parser.add_argument("--report", default="aegis_alpha/logs/edge/long_edge_meta_filter_train_v040.json")
    parser.add_argument("--test-pct", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2-regularization", type=float, default=0.10)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    args = parser.parse_args()
    train_meta_filter(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        report_path=Path(args.report),
        test_pct=args.test_pct,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        max_leaf_nodes=args.max_leaf_nodes,
    )


if __name__ == "__main__":
    main()
