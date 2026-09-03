#!/usr/bin/env python3
"""Exploratory fixed-model target diagnosis; never a promotion result."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.directional_contract_v15 import contract_indices
from aegis.research.entry_label_audit import normalize_entry_label_row
from aegis.utils import sha256_file


TARGETS = (
    "v11_clean",
    "clean_fast_success",
    "target_before_stop",
    "positive_utility",
    "current_ts_positive",
)


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("target diagnosis requires timezone-aware timestamps")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/calibrated_horizon_v11/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/experiments/aegis_directional_contract_v15_research.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/v19_design/target_separability.json")
    )
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    rows = []
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            normalized = normalize_entry_label_row(source)
            if not normalized["independent"]:
                continue
            features = np.asarray(source["v9_features"], dtype=np.float64)
            if features.shape != (176,) or not np.isfinite(features).all():
                raise ValueError("invalid V9 feature contract")
            rows.append({**normalized, "time": timestamp(normalized["timestamp"]), "features": features})

    train_end = timestamp("2026-05-31T23:55:00+00:00")
    validation_start = timestamp("2026-06-01T00:00:00+00:00")
    result: dict[str, object] = {
        "schema_id": "aegis-v19-exploratory-target-separability-v1",
        "selection_effect": "NONE",
        "promotion_authority": False,
        "holdout_accesses": 0,
        "source_dataset": str(args.dataset),
        "source_dataset_sha256": sha256_file(args.dataset),
        "fixed_estimator": "STANDARD_SCALER_LOGISTIC_REGRESSION_BALANCED_C1_SEED_190019",
        "sides": {},
    }
    for side in ("LONG", "SHORT"):
        indices = contract_indices(contract, side)
        train = [row for row in rows if row["side"] == side and row["time"] <= train_end]
        validation = [
            row for row in rows if row["side"] == side and row["time"] >= validation_start
        ]
        x_train = np.asarray(
            [[row["features"][index] for index in indices] for row in train], dtype=np.float64
        )
        x_validation = np.asarray(
            [[row["features"][index] for index in indices] for row in validation],
            dtype=np.float64,
        )
        side_result: dict[str, object] = {
            "feature_count": len(indices),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "targets": {},
        }
        ordered_times = sorted({row["time"] for row in train})
        calibration_boundary = ordered_times[int(len(ordered_times) * 0.80) - 1]
        fit = [row for row in train if row["time"] <= calibration_boundary]
        calibration = [
            row for row in train if row["time"] > calibration_boundary + timedelta(minutes=120)
        ]
        x_fit = np.asarray(
            [[row["features"][index] for index in indices] for row in fit], dtype=np.float64
        )
        x_calibration = np.asarray(
            [[row["features"][index] for index in indices] for row in calibration],
            dtype=np.float64,
        )
        clean_fit = np.asarray([int(bool(row["v11_clean"])) for row in fit], dtype=np.int8)
        clean_calibration = np.asarray(
            [int(bool(row["v11_clean"])) for row in calibration], dtype=np.int8
        )
        v18_base = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=180018 + (0 if side == "LONG" else 100),
            ),
        ).fit(x_fit, clean_fit)
        calibrator = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=180022 + (0 if side == "LONG" else 100),
        ).fit(
            v18_base.decision_function(x_calibration).reshape(-1, 1), clean_calibration
        )
        raw_validation_score = v18_base.decision_function(x_validation)
        calibrated_validation_probability = calibrator.predict_proba(
            raw_validation_score.reshape(-1, 1)
        )[:, 1]
        clean_validation = np.asarray(
            [int(bool(row["v11_clean"])) for row in validation], dtype=np.int8
        )
        side_result["v18_calibration_reproduction"] = {
            "fit_rows": len(fit),
            "calibration_rows": len(calibration),
            "platt_slope": float(calibrator.coef_[0, 0]),
            "raw_average_precision": float(
                average_precision_score(clean_validation, raw_validation_score)
            ),
            "raw_roc_auc": float(roc_auc_score(clean_validation, raw_validation_score)),
            "calibrated_average_precision": float(
                average_precision_score(clean_validation, calibrated_validation_probability)
            ),
            "calibrated_roc_auc": float(
                roc_auc_score(clean_validation, calibrated_validation_probability)
            ),
            "ranking_inverted": bool(calibrator.coef_[0, 0] < 0.0),
        }
        for target_index, target in enumerate(TARGETS):
            y_train = np.asarray([int(bool(row[target])) for row in train], dtype=np.int8)
            y_validation = np.asarray(
                [int(bool(row[target])) for row in validation], dtype=np.int8
            )
            if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
                raise ValueError(f"target lacks both classes: {side}:{target}")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=190019 + target_index,
                ),
            ).fit(x_train, y_train)
            probability = model.predict_proba(x_validation)[:, 1]
            cutoff = float(np.quantile(probability, 0.90))
            selected = [
                row for row, score in zip(validation, probability, strict=True) if score >= cutoff
            ]
            utility = [float(row["utility"]) for row in selected]
            mae = [float(row["mae"]) for row in selected]
            prevalence = float(np.mean(y_validation))
            ap = float(average_precision_score(y_validation, probability))
            target_result = {
                "validation_prevalence": prevalence,
                "average_precision": ap,
                "average_precision_lift_ratio": ap / prevalence,
                "roc_auc": float(roc_auc_score(y_validation, probability)),
                "top_decile_rows": len(selected),
                "top_decile_mean_utility": float(np.mean(utility)),
                "top_decile_positive_utility_rate": float(
                    np.mean([row["positive_utility"] for row in selected])
                ),
                "top_decile_mean_mae": float(np.mean(mae)),
            }
            if not all(
                math.isfinite(float(value))
                for value in target_result.values()
                if isinstance(value, float)
            ):
                raise ValueError("non-finite target separability result")
            side_result["targets"][target] = target_result
        result["sides"][side] = side_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": "COMPLETE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
