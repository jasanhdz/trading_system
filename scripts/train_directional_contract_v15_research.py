#!/usr/bin/env python3
"""Evaluate preregistered V15 directional contracts without runtime authority."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.competing_barrier_v10_training import utility_metrics
from aegis.research.directional_contract_v15 import (
    contract_indices,
    entry_quality_score,
    select_at_most_one_per_timestamp,
)
from aegis.research.feature_information_v14 import (
    binary_probability_metrics,
    quantile_pinball_loss,
)
from aegis.research.joint_path_v12 import path_quality_metrics
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _mapping


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise BarrierResearchError("V15 timestamps must be timezone-aware")
    return parsed


def _verify_authority(root: Path, config: Mapping[str, Any]) -> None:
    authority = _mapping(config["authority"], "authority")
    for path_key, hash_key in (
        ("source_dataset", "source_dataset_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("source_v11_config", "source_v11_config_sha256"),
        ("source_v14_validation", "source_v14_validation_sha256"),
        ("source_v14_config", "source_v14_config_sha256"),
        ("source_v14_report", "source_v14_report_sha256"),
    ):
        path = root / str(authority[path_key])
        if sha256_file(path) != str(authority[hash_key]):
            raise BarrierResearchError(f"V15 authority mismatch: {path_key}")


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = _mapping(json.loads(line), f"dataset:{line_number}")
            contract = _mapping(source["v10_contract_outcomes"], "contracts")[
                "ROE_10_H12"
            ]
            features = tuple(float(value) for value in source["v9_features"])
            if len(features) != 176 or not all(
                math.isfinite(value) for value in features
            ):
                raise BarrierResearchError("invalid V15 feature vector")
            rows.append(
                {
                    "timestamp": str(source["timestamp"]),
                    "timestamp_value": _time(str(source["timestamp"])),
                    "symbol": str(source["symbol"]),
                    "side": str(source["side"]),
                    "independent": bool(source["independent"]),
                    "features": features,
                    "danger": str(contract["outcome"])
                    in {"ADVERSE_FIRST", "SAME_BAR_AMBIGUOUS"},
                    "clean": bool(source["v11_clean_entry_label"]),
                    "mae_fraction": float(source["mae_fraction"]),
                    "adverse_fraction": float(contract["adverse_fraction"]),
                    "actual_utility": float(contract["realized_utility"]),
                    "actual_outcome": str(contract["outcome"]),
                    "v11_clean_entry_label": bool(source["v11_clean_entry_label"]),
                    "v11_path_diagnostics": source["v11_path_diagnostics"],
                }
            )
    return sorted(
        rows, key=lambda row: (row["timestamp_value"], row["symbol"], row["side"])
    )


def _partition(
    rows: Sequence[Mapping[str, Any]], fold: Mapping[str, Any], embargo_minutes: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train_end = _time(str(fold["train_end"]))
    calibration_end = _time(str(fold["calibration_end"]))
    test_end = _time(str(fold["test_end"]))
    embargo = timedelta(minutes=embargo_minutes)
    train = [row for row in rows if row["timestamp_value"] <= train_end]
    calibration = [
        row
        for row in rows
        if train_end + embargo < row["timestamp_value"] <= calibration_end
        and row["independent"]
    ]
    test = [
        row
        for row in rows
        if calibration_end + embargo < row["timestamp_value"] <= test_end
        and row["independent"]
    ]
    return train, calibration, test


def _matrix(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [[row["features"][index] for index in indices] for row in rows],
        dtype=np.float64,
    )


def _classifier(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=250,
            solver="liblinear",
            random_state=seed,
        ),
    )


def _fit_bundle(
    train: Sequence[Mapping[str, Any]], indices: Sequence[int], *, seed: int
) -> Mapping[str, Any]:
    matrix = _matrix(train, indices)
    danger = np.asarray([bool(row["danger"]) for row in train], dtype=np.int8)
    clean = np.asarray([bool(row["clean"]) for row in train], dtype=np.int8)
    if len(np.unique(danger)) != 2 or len(np.unique(clean)) != 2:
        raise BarrierResearchError("V15 training partition lacks both classes")
    mae = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.9,
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=4.0,
        early_stopping=False,
        random_state=seed + 2,
    ).fit(matrix.astype(np.float32), [float(row["mae_fraction"]) for row in train])
    return {
        "danger": _classifier(seed).fit(matrix, danger),
        "clean": _classifier(seed + 1).fit(matrix, clean),
        "mae": mae,
        "indices": tuple(indices),
    }


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    matrix = _matrix(rows, bundle["indices"])
    danger = bundle["danger"].predict_proba(matrix)[:, 1]
    clean = bundle["clean"].predict_proba(matrix)[:, 1]
    mae = np.maximum(0.0, bundle["mae"].predict(matrix.astype(np.float32)))
    return [
        {
            **row,
            "danger_probability": float(danger[index]),
            "clean_probability": float(clean[index]),
            "mae_q90": float(mae[index]),
            "score": entry_quality_score(
                clean_probability=float(clean[index]),
                danger_probability=float(danger[index]),
                mae_q90=float(mae[index]),
                adverse=float(row["adverse_fraction"]),
            ),
        }
        for index, row in enumerate(rows)
    ]


def _predictive_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    danger = binary_probability_metrics(
        [int(bool(row["danger"])) for row in rows],
        [float(row["danger_probability"]) for row in rows],
    )
    clean = binary_probability_metrics(
        [int(bool(row["clean"])) for row in rows],
        [float(row["clean_probability"]) for row in rows],
    )
    mae = quantile_pinball_loss(
        [float(row["mae_fraction"]) for row in rows],
        [float(row["mae_q90"]) for row in rows],
        quantile=0.9,
    )
    return {"danger": danger, "clean": clean, "mae_q90_pinball_loss": mae}


def _selection_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "utility": utility_metrics(rows),
        "path_quality": path_quality_metrics(rows),
    }


def _selected(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[Mapping[str, Any]]:
    mask = select_at_most_one_per_timestamp(rows, minimum_score=threshold)
    return [row for row, selected in zip(rows, mask) if selected]


def _policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    choices = []
    scores = np.asarray([float(row["score"]) for row in calibration])
    minimum = int(config["validation"]["minimum_selected_per_fold"])
    for quantile in config["validation"]["score_quantiles"]:
        threshold = float(np.quantile(scores, float(quantile)))
        selected = _selected(calibration, threshold)
        metrics = _selection_metrics(selected)
        utility = metrics["utility"]
        valid = (
            int(utility["count"]) >= minimum
            and utility["mean_utility"] is not None
            and utility["cvar"] is not None
        )
        choices.append(
            {
                "quantile": float(quantile),
                "minimum_score": threshold,
                "metrics": metrics,
                "valid": valid,
            }
        )
    valid = [choice for choice in choices if choice["valid"]]
    if not valid:
        raise BarrierResearchError("V15 calibration cannot derive a valid policy")
    return max(
        valid,
        key=lambda choice: (
            float(choice["metrics"]["utility"]["mean_utility"]),
            float(choice["metrics"]["utility"]["cvar"]),
            -float(choice["metrics"]["path_quality"]["mean_mae_fraction"]),
        ),
    )


def _evaluate_contract(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    bundle = _fit_bundle(train, indices, seed=seed)
    calibration_predictions = _predict(calibration, bundle)
    policy = _policy(calibration_predictions, config)
    test_predictions = _predict(test, bundle)
    selected = _selected(test_predictions, float(policy["minimum_score"]))
    return {
        "feature_count": len(indices),
        "policy": policy,
        "predictive": _predictive_metrics(test_predictions),
        "selected": _selection_metrics(selected),
    }


def _better(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> Mapping[str, bool]:
    candidate_predictive = candidate["predictive"]
    baseline_predictive = baseline["predictive"]
    candidate_selected = candidate["selected"]
    baseline_selected = baseline["selected"]
    cu = candidate_selected["utility"]
    bu = baseline_selected["utility"]
    cp = candidate_selected["path_quality"]
    bp = baseline_selected["path_quality"]
    return {
        "danger": candidate_predictive["danger"]["log_loss"]
        < baseline_predictive["danger"]["log_loss"]
        and candidate_predictive["danger"]["average_precision"]
        > baseline_predictive["danger"]["average_precision"],
        "clean": candidate_predictive["clean"]["log_loss"]
        < baseline_predictive["clean"]["log_loss"]
        and candidate_predictive["clean"]["average_precision"]
        > baseline_predictive["clean"]["average_precision"],
        "mae": candidate_predictive["mae_q90_pinball_loss"]
        < baseline_predictive["mae_q90_pinball_loss"],
        "mean_utility": cu["mean_utility"] is not None
        and bu["mean_utility"] is not None
        and cu["mean_utility"] > bu["mean_utility"],
        "cvar": cu["cvar"] is not None
        and bu["cvar"] is not None
        and cu["cvar"] >= bu["cvar"],
        "adverse_first": cp["adverse_first_rate"] is not None
        and bp["adverse_first_rate"] is not None
        and cp["adverse_first_rate"] <= bp["adverse_first_rate"],
        "selected_mae": cp["mean_mae_fraction"] is not None
        and bp["mean_mae_fraction"] is not None
        and cp["mean_mae_fraction"] <= bp["mean_mae_fraction"],
    }


def train(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    _verify_authority(root, config)
    dataset = root / str(config["authority"]["source_dataset"])
    rows = _load(dataset)
    minimum = int(config["validation"]["minimum_rows_per_partition"])
    reports = {}
    for side_offset, side in enumerate(("LONG", "SHORT")):
        population = [row for row in rows if row["side"] == side]
        baseline_indices = contract_indices(config, "BASELINE")
        candidate_indices = contract_indices(config, side)
        folds = []
        for fold in config["validation"]["folds"]:
            train_rows, calibration, test = _partition(
                population, fold, int(config["validation"]["embargo_minutes"])
            )
            if min(len(train_rows), len(calibration), len(test)) < minimum:
                raise BarrierResearchError("V15 partition has insufficient rows")
            fold_id = int(fold["id"])
            baseline = _evaluate_contract(
                train_rows,
                calibration,
                test,
                baseline_indices,
                config,
                seed=2026081500 + side_offset * 100 + fold_id * 10,
            )
            candidate = _evaluate_contract(
                train_rows,
                calibration,
                test,
                candidate_indices,
                config,
                seed=2026081500 + side_offset * 100 + fold_id * 10,
            )
            folds.append(
                {
                    "fold": fold_id,
                    "role": str(fold["role"]),
                    "train_rows": len(train_rows),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "baseline": baseline,
                    "candidate": candidate,
                    "improvements": _better(candidate, baseline),
                }
            )
        counts = {
            metric: sum(bool(fold["improvements"][metric]) for fold in folds)
            for metric in folds[0]["improvements"]
        }
        required = int(config["validation"]["candidate_required_improvement_folds"])
        final = folds[-1]["improvements"]
        passed = all(count >= required for count in counts.values()) and all(
            final.values()
        )
        reports[side] = {
            "baseline_feature_count": len(baseline_indices),
            "candidate_feature_count": len(candidate_indices),
            "folds": folds,
            "improvement_folds": counts,
            "post_v14_holdout": final,
            "passed": passed,
        }
    passed = all(report["passed"] for report in reports.values())
    return {
        "schema_id": "aegis-directional-contract-v15-validation-v1",
        "experiment_id": str(config["experiment_id"]),
        "mode": "RESEARCH_ONLY",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "sides": reports,
        "candidate_passed": passed,
        "verdict": (
            "RESEARCH_CANDIDATE_ADMISSIBLE_NOT_PROMOTED"
            if passed
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "model_exported": False,
        "shadow_changed": False,
        "live_changed": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_directional_contract_v15_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/directional_contract_v15/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    result = train(root, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "candidate_passed": result["candidate_passed"],
                "output": str(output),
                "verdict": result["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
