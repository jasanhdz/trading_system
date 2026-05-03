#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import save_model_bundle, safe_float, write_json  # noqa: E402
from aegis_alpha.signals.signal_registry import SIGNAL_REGISTRY, SignalSpec  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_key: str
    kind: str


MODEL_SPECS: tuple[ModelSpec, ...] = tuple(
    ModelSpec(
        name=spec.name,
        target_key=f"h{spec.horizon}_{spec.target_type}",
        kind=spec.model_type,
    )
    for spec in SIGNAL_REGISTRY
)


def _class_weights(y: np.ndarray) -> np.ndarray:
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    return weights.astype(np.float32)


def _classification_report(y: np.ndarray, proba: np.ndarray, baseline_return: float) -> dict[str, float]:
    out: dict[str, float] = {
        "samples": safe_float(len(y)),
        "positive_rate": safe_float(np.mean(y)),
        "prob_mean": safe_float(np.mean(proba)),
        "baseline_return": safe_float(baseline_return),
        "log_loss": safe_float(log_loss(y, np.clip(proba, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = safe_float(roc_auc_score(y, proba))
        out["average_precision"] = safe_float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = 0.0
        out["average_precision"] = 0.0
    return out


def _regression_report(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    baseline = np.full_like(y, np.mean(y), dtype=np.float32)
    out: dict[str, float] = {
        "samples": safe_float(len(y)),
        "positive_rate": safe_float(np.mean(y > 0.0)),
        "baseline_return": safe_float(np.mean(y)),
        "mae": safe_float(mean_absolute_error(y, pred)),
        "rmse": safe_float(np.sqrt(mean_squared_error(y, pred))),
        "baseline_mae": safe_float(mean_absolute_error(y, baseline)),
        "baseline_rmse": safe_float(np.sqrt(mean_squared_error(y, baseline))),
    }
    return out


def _maybe_feature_importance(estimator: Any, feature_names: list[str]) -> dict[str, float] | None:
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(getattr(estimator, "feature_importances_"), dtype=np.float32)
        if len(raw) == len(feature_names):
            order = np.argsort(raw)[::-1]
            return {feature_names[i]: safe_float(raw[i]) for i in order[:20]}
    return None


def train_signal_models(
    dataset_path: Path,
    output_dir: Path,
    report_path: Path,
    test_pct: float,
    max_iter: int,
    learning_rate: float,
    l2_regularization: float,
    max_leaf_nodes: int,
) -> dict[str, Any]:
    data = np.load(dataset_path, allow_pickle=True)
    x = data["X"].astype(np.float32) if "X" in data else data["x"].astype(np.float32)
    steps = data["step"].astype(np.int64)
    timestamps = data["timestamp"].astype(str)
    feature_names = data["feature_names"].astype(str).tolist()

    order = np.argsort(steps, kind="stable")
    x = x[order]
    steps = steps[order]
    timestamps = timestamps[order]

    split = int(len(x) * (1.0 - test_pct))
    if split <= 0 or split >= len(x):
        raise ValueError(f"Invalid chronological split for n={len(x)}, test_pct={test_pct}")

    x_train, x_test = x[:split], x[split:]
    train_range = (str(timestamps[0]), str(timestamps[split - 1]))
    test_range = (str(timestamps[split]), str(timestamps[-1]))

    params: dict[str, Any] = {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "max_leaf_nodes": max_leaf_nodes,
        "early_stopping": True,
        "random_state": 4667,
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    report_models: dict[str, Any] = {}
    bundle_metadata = {
        "schema_version": "aegis_signal_models_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_path": str(dataset_path),
        "feature_count": int(x.shape[1]),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "holdout_range": {"train": train_range, "test": test_range},
        "params": params,
    }

    for spec in MODEL_SPECS:
        y = data[spec.target_key]
        y = y.astype(np.float32) if spec.kind == "regressor" else y.astype(np.int8)
        y = y[order]
        y_train, y_test = y[:split], y[split:]
        print(f"Training {spec.name} -> {spec.target_key}")
        if spec.kind == "classifier":
            estimator = HistGradientBoostingClassifier(**params)
            estimator.fit(x_train, y_train, sample_weight=_class_weights(y_train))
            train_proba = estimator.predict_proba(x_train)[:, 1]
            test_proba = estimator.predict_proba(x_test)[:, 1]
            train_report = _classification_report(y_train, train_proba, float(np.mean(y_train)))
            test_report = _classification_report(y_test, test_proba, float(np.mean(y_test)))
        else:
            estimator = HistGradientBoostingRegressor(loss="absolute_error", **params)
            estimator.fit(x_train, y_train)
            train_pred = estimator.predict(x_train)
            test_pred = estimator.predict(x_test)
            train_report = _regression_report(y_train, train_pred)
            test_report = _regression_report(y_test, test_pred)

        model_path = output_dir / f"aegis_{spec.name}_v050.joblib"
        model_bundle = {
            "metadata": {
                **bundle_metadata,
                "signal_name": spec.name,
                "target_key": spec.target_key,
                "model_kind": spec.kind,
            },
            "feature_names": feature_names,
            "estimator": estimator,
            "signal_name": spec.name,
            "target_key": spec.target_key,
            "model_kind": spec.kind,
        }
        save_model_bundle(model_path, model_bundle)
        report_models[spec.name] = {
            "target_key": spec.target_key,
            "model_kind": spec.kind,
            "artifact_path": str(model_path),
            "train": train_report,
            "test": test_report,
            "feature_importance": _maybe_feature_importance(estimator, feature_names),
        }
        print(
            f"  holdout: baseline={test_report['baseline_return']:.6g} "
            + (f"auc={test_report.get('roc_auc', 0.0):.3f} ap={test_report.get('average_precision', 0.0):.3f}" if spec.kind == "classifier" else f"mae={test_report['mae']:.6g} rmse={test_report['rmse']:.6g}")
        )

    report = {
        "schema_version": "aegis_signal_models_train_v1",
        "created_at": bundle_metadata["created_at"],
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "holdout_range": bundle_metadata["holdout_range"],
        "params": params,
        "models": report_models,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"Training report saved -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
    parser.add_argument("--output-dir", default="aegis_alpha/models/signals")
    parser.add_argument("--report", default="aegis_alpha/logs/signals/signal_models_train_v050.json")
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    args = parser.parse_args()
    train_signal_models(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        test_pct=args.test_pct,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        max_leaf_nodes=args.max_leaf_nodes,
    )


if __name__ == "__main__":
    main()
