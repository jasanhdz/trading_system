#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import save_model_bundle, safe_float, write_json  # noqa: E402
from aegis_alpha.signals.horizon_targets import HORIZONS, build_tail_risk_targets_from_dataset  # noqa: E402


@dataclass(frozen=True)
class TailRiskSpec:
    horizon: int
    name: str


TAIL_RISK_SPECS: tuple[TailRiskSpec, ...] = (
    TailRiskSpec(12, "long_tail_risk_h12"),
    TailRiskSpec(24, "long_tail_risk_h24"),
    TailRiskSpec(48, "long_tail_risk_h48"),
)


def _class_weights(y: np.ndarray) -> np.ndarray:
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    return weights.astype(np.float32)


def _classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "samples": safe_float(len(y)),
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


def train_tail_risk_models(
    dataset_path: Path,
    output_dir: Path,
    report_path: Path,
    model_version: str,
    test_pct: float,
    max_iter: int,
    learning_rate: float,
    l2_regularization: float,
    max_leaf_nodes: int,
) -> dict[str, Any]:
    raw = np.load(dataset_path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    x = data["X"].astype(np.float32) if "X" in data else data["x"].astype(np.float32)
    steps = data["step"].astype(np.int64)
    timestamps = data["timestamp"].astype(str)
    feature_names = data["feature_names"].astype(str).tolist()
    tail_targets = build_tail_risk_targets_from_dataset(data, horizons=HORIZONS)

    order = np.argsort(steps, kind="stable")
    x = x[order]
    steps = steps[order]
    timestamps = timestamps[order]
    tail_targets = {key: values[order] for key, values in tail_targets.items()}

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
        "schema_version": "aegis_tail_risk_models_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sklearn_version": sklearn.__version__,
        "dataset_path": str(dataset_path),
        "feature_count": int(x.shape[1]),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "train_date_range": train_range,
        "holdout_date_range": test_range,
        "holdout_range": {"train": train_range, "test": test_range},
        "params": params,
    }

    for spec in TAIL_RISK_SPECS:
        y = tail_targets[f"h{spec.horizon}_tail_loss_gt_0p20"].astype(np.int8)
        y_train, y_test = y[:split], y[split:]
        print(f"Training {spec.name} -> h{spec.horizon}_tail_loss_gt_0p20")
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(x_train, y_train, sample_weight=_class_weights(y_train))
        train_proba = clf.predict_proba(x_train)[:, 1]
        test_proba = clf.predict_proba(x_test)[:, 1]
        model_path = output_dir / f"aegis_{spec.name}_{model_version}.joblib"
        model_bundle = {
            "metadata": {
                **bundle_metadata,
                "signal_name": spec.name,
                "target_key": f"h{spec.horizon}_tail_loss_gt_0p20",
                "model_kind": "classifier",
            },
            "feature_names": feature_names,
            "estimator": clf,
            "signal_name": spec.name,
            "target_key": f"h{spec.horizon}_tail_loss_gt_0p20",
            "model_kind": "classifier",
        }
        save_model_bundle(model_path, model_bundle)
        holdout = _classification_metrics(y_test, test_proba)
        holdout.update(
            {
                "tail_loss_gt_0p20_rate": safe_float(np.mean(y_test)),
                "tail_loss_gt_0p35_rate": safe_float(np.mean(tail_targets[f"h{spec.horizon}_tail_loss_gt_0p35"][split:])),
                "mae_gt_mfe_rate": safe_float(np.mean(tail_targets[f"h{spec.horizon}_mae_gt_mfe"][split:])),
                "edge_deterioration_rate": safe_float(np.mean(tail_targets[f"h{spec.horizon}_edge_deterioration_loss"][split:])),
                "regime_shift_rate": safe_float(np.mean(tail_targets[f"h{spec.horizon}_regime_shift_loss"][split:])),
                "baseline_return": safe_float(np.mean(y_test)),
            }
        )
        report_models[spec.name] = {
            "target_key": f"h{spec.horizon}_tail_loss_gt_0p20",
            "artifact_path": str(model_path),
            "model_path": str(model_path),
            "train": _classification_metrics(y_train, train_proba),
            "test": holdout,
        }
        print(
            f"  holdout: auc={holdout['roc_auc']:.3f} ap={holdout['average_precision']:.3f} "
            f"positive_rate={holdout['positive_rate']:.2%}"
        )

    report = {
        "schema_version": "aegis_tail_risk_models_train_v1",
        "created_at": bundle_metadata["created_at"],
        "sklearn_version": sklearn.__version__,
        "dataset_path": str(dataset_path),
        "train_date_range": train_range,
        "holdout_date_range": test_range,
        "holdout_range": bundle_metadata["holdout_range"],
        "params": params,
        "models": report_models,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"Tail-risk training report saved -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
    parser.add_argument("--output-dir", default="aegis_alpha/models/signals")
    parser.add_argument("--report", default="aegis_alpha/logs/signals/tail_risk_train_v052.json")
    parser.add_argument("--model-version", default="v052")
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    args = parser.parse_args()
    train_tail_risk_models(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        model_version=args.model_version,
        test_pct=args.test_pct,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        max_leaf_nodes=args.max_leaf_nodes,
    )


if __name__ == "__main__":
    main()
