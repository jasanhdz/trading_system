#!/usr/bin/env python3
"""Run the frozen V18 TRAIN/VALIDATION experiment without opening holdout."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, mean_pinball_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.decomposed_economic_selector_v18 import (
    economic_metrics,
    moving_block_intervals,
    offline_side_gate,
    select_candidates,
    temporal_thirds,
)
from aegis.research.directional_contract_v15 import contract_indices
from aegis.utils import Sha256HashProvider, sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BarrierResearchError(f"V18 {name} must be a mapping")
    return value


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise BarrierResearchError("V18 timestamps must be timezone-aware")
    return parsed


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = _mapping(json.loads(line), f"dataset:{line_number}")
            contract = _mapping(source["v10_contract_outcomes"], "contracts")["ROE_10_H12"]
            features = tuple(float(value) for value in source["v9_features"])
            values = (
                *features,
                float(source["mae_fraction"]),
                float(source["mfe_fraction"]),
                float(contract["realized_utility"]),
            )
            if len(features) != 176 or not all(math.isfinite(value) for value in values):
                raise BarrierResearchError("V18 dataset contains invalid numeric evidence")
            rows.append(
                {
                    "timestamp": str(source["timestamp"]),
                    "timestamp_value": _time(str(source["timestamp"])),
                    "symbol": str(source["symbol"]),
                    "side": str(source["side"]),
                    "independent": bool(source["independent"]),
                    "features": features,
                    "clean": bool(source["v11_clean_entry_label"]),
                    "danger": str(contract["outcome"])
                    in {"ADVERSE_FIRST", "SAME_BAR_AMBIGUOUS"},
                    "mae_fraction": float(source["mae_fraction"]),
                    "mfe_fraction": float(source["mfe_fraction"]),
                    "adverse_fraction": float(contract["adverse_fraction"]),
                    "actual_utility": float(contract["realized_utility"]),
                    "regime": str(source["v11_causal_regime"]),
                }
            )
    return sorted(rows, key=lambda row: (row["timestamp_value"], row["symbol"], row["side"]))


def _matrix(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(
        [[row["features"][index] for index in indices] for row in rows], dtype=np.float64
    )
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise BarrierResearchError("V18 feature matrix invalid")
    return matrix


def _split_train(
    rows: Sequence[Mapping[str, Any]], calibration_fraction: float, purge_minutes: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    timestamps = sorted({row["timestamp_value"] for row in rows})
    boundary = timestamps[max(0, int(len(timestamps) * (1.0 - calibration_fraction)) - 1)]
    purge = timedelta(minutes=purge_minutes)
    fit = [row for row in rows if row["timestamp_value"] <= boundary]
    calibration = [
        row for row in rows if row["timestamp_value"] > boundary + purge and row["independent"]
    ]
    if min(len(fit), len(calibration)) < 500:
        raise BarrierResearchError("V18 inner train split is insufficient")
    return fit, calibration


def _classifier(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
    )


def _fit_platt(scores: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    if len(np.unique(labels)) != 2:
        raise BarrierResearchError("V18 calibration partition lacks both classes")
    return LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        scores.reshape(-1, 1), labels
    )


def _fit_heads(
    fit: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    matrix = _matrix(fit, indices)
    clean_y = np.asarray([int(bool(row["clean"])) for row in fit], dtype=np.int8)
    danger_y = np.asarray([int(bool(row["danger"])) for row in fit], dtype=np.int8)
    if len(np.unique(clean_y)) != 2 or len(np.unique(danger_y)) != 2:
        raise BarrierResearchError("V18 fit partition lacks both safety classes")
    clean = _classifier(seed).fit(matrix, clean_y)
    danger = _classifier(seed + 1).fit(matrix, danger_y)
    models = _mapping(config["models"], "models")
    mae_config = _mapping(models["mae_q90"], "mae_q90")
    utility_config = _mapping(models["expected_utility"], "expected_utility")
    mae = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(mae_config["quantile"]),
        learning_rate=float(mae_config["learning_rate"]),
        max_iter=int(mae_config["max_iter"]),
        max_leaf_nodes=int(mae_config["max_leaf_nodes"]),
        l2_regularization=float(mae_config["l2_regularization"]),
        early_stopping=False,
        random_state=seed + 2,
    ).fit(matrix.astype(np.float32), [float(row["mae_fraction"]) for row in fit])
    utility = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(utility_config["learning_rate"]),
        max_iter=int(utility_config["max_iter"]),
        max_leaf_nodes=int(utility_config["max_leaf_nodes"]),
        l2_regularization=float(utility_config["l2_regularization"]),
        early_stopping=False,
        random_state=seed + 3,
    ).fit(matrix.astype(np.float32), [float(row["actual_utility"]) for row in fit])
    calibration_matrix = _matrix(calibration, indices)
    clean_platt = _fit_platt(
        clean.decision_function(calibration_matrix),
        np.asarray([int(bool(row["clean"])) for row in calibration]),
        seed + 4,
    )
    danger_platt = _fit_platt(
        danger.decision_function(calibration_matrix),
        np.asarray([int(bool(row["danger"])) for row in calibration]),
        seed + 5,
    )
    return {
        "indices": tuple(indices),
        "clean": clean,
        "danger": danger,
        "clean_platt": clean_platt,
        "danger_platt": danger_platt,
        "mae": mae,
        "utility": utility,
    }


def _predict(rows: Sequence[Mapping[str, Any]], heads: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = _matrix(rows, heads["indices"])
    clean = heads["clean_platt"].predict_proba(
        heads["clean"].decision_function(matrix).reshape(-1, 1)
    )[:, 1]
    danger = heads["danger_platt"].predict_proba(
        heads["danger"].decision_function(matrix).reshape(-1, 1)
    )[:, 1]
    mae = np.maximum(0.0, heads["mae"].predict(matrix.astype(np.float32)))
    utility = heads["utility"].predict(matrix.astype(np.float32))
    if not all(np.isfinite(values).all() for values in (clean, danger, mae, utility)):
        raise BarrierResearchError("V18 model produced non-finite output")
    return [
        {
            **row,
            "clean_probability": float(clean[index]),
            "danger_probability": float(danger[index]),
            "mae_q90": float(mae[index]),
            "expected_utility": float(utility[index]),
        }
        for index, row in enumerate(rows)
    ]


def _random_control(
    rows: Sequence[Mapping[str, Any]], count: int, *, seed: int
) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["timestamp"])].append(row)
    rng = np.random.default_rng(seed)
    timestamps = sorted(grouped)
    rng.shuffle(timestamps)
    return [
        grouped[timestamp][int(rng.integers(0, len(grouped[timestamp])))]
        for timestamp in timestamps[:count]
    ]


def _simple_control(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = []
    for row in rows:
        score = (
            float(row["clean_probability"])
            - float(row["danger_probability"])
            - float(row["mae_q90"]) / float(row["adverse_fraction"])
        )
        if score > 0.0:
            eligible.append({**row, "simple_score": score})
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row["timestamp"])].append(row)
    return [
        max(grouped[timestamp], key=lambda row: (float(row["simple_score"]), str(row["symbol"])))
        for timestamp in sorted(grouped)
    ]


def _prediction_diagnostics(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> Mapping[str, Any]:
    quantile_levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    distributions = {
        field: {
            str(level): float(value)
            for level, value in zip(
                quantile_levels,
                np.quantile([float(row[field]) for row in rows], quantile_levels),
            )
        }
        for field in ("clean_probability", "danger_probability", "mae_q90", "expected_utility")
    }
    clean_pass = [
        row
        for row in rows
        if float(row["clean_probability"]) >= float(policy["minimum_clean_probability"])
    ]
    danger_pass = [
        row
        for row in clean_pass
        if float(row["danger_probability"]) <= float(policy["maximum_danger_probability"])
    ]
    mae_pass = [
        row
        for row in danger_pass
        if float(row["mae_q90"]) <= float(policy["maximum_mae_fraction"])
    ]
    utility_pass = [
        row
        for row in mae_pass
        if float(row["expected_utility"]) > float(policy["minimum_expected_utility"])
    ]
    clean_labels = [int(bool(row["clean"])) for row in rows]
    danger_labels = [int(bool(row["danger"])) for row in rows]
    return {
        "distributions": distributions,
        "staged_survivors": {
            "population": len(rows),
            "after_clean": len(clean_pass),
            "after_danger": len(danger_pass),
            "after_mae": len(mae_pass),
            "after_expected_utility": len(utility_pass),
        },
        "predictive": {
            "clean_average_precision": float(
                average_precision_score(clean_labels, [float(row["clean_probability"]) for row in rows])
            ),
            "clean_log_loss": float(
                log_loss(clean_labels, [float(row["clean_probability"]) for row in rows])
            ),
            "danger_average_precision": float(
                average_precision_score(danger_labels, [float(row["danger_probability"]) for row in rows])
            ),
            "danger_log_loss": float(
                log_loss(danger_labels, [float(row["danger_probability"]) for row in rows])
            ),
            "mae_q90_pinball_loss": float(
                mean_pinball_loss(
                    [float(row["mae_fraction"]) for row in rows],
                    [float(row["mae_q90"]) for row in rows],
                    alpha=0.9,
                )
            ),
            "utility_rmse": float(
                math.sqrt(
                    mean_squared_error(
                        [float(row["actual_utility"]) for row in rows],
                        [float(row["expected_utility"]) for row in rows],
                    )
                )
            ),
        },
    }


def _side_report(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    partitions = _mapping(config["partitions"], "partitions")
    fit, calibration = _split_train(
        train,
        float(partitions["train_inner_calibration_fraction"]),
        int(partitions["purge_minutes"]),
    )
    heads = _fit_heads(fit, calibration, indices, config, seed=seed)
    train_predictions = _predict(train, heads)
    validation_predictions = _predict(validation, heads)
    policy = _mapping(config["selection_policy"], "selection_policy")
    selected_train = select_candidates(train_predictions, policy)
    selected_validation = select_candidates(validation_predictions, policy)
    validation_metrics = economic_metrics(selected_validation)
    uncertainty = _mapping(config["economics"], "economics")["uncertainty"]
    intervals = moving_block_intervals(
        selected_validation,
        resamples=int(uncertainty["resamples"]),
        seed=int(uncertainty["seed"]),
    )
    thirds = temporal_thirds(selected_validation)
    random = _random_control(validation, len(selected_validation), seed=seed)
    random_metrics = economic_metrics(random)
    simple = _simple_control(validation_predictions)
    gate = offline_side_gate(
        validation_metrics,
        intervals,
        thirds,
        _mapping(config["offline_gate"], "offline_gate"),
        random_expectancy=random_metrics["net_expectancy"],
    )
    return {
        "feature_count": len(indices),
        "fit_rows": len(fit),
        "inner_calibration_rows": len(calibration),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train": {
            "selected": economic_metrics(selected_train),
            "selected_count": len(selected_train),
        },
        "validation": {
            "selected": validation_metrics,
            "prediction_diagnostics": _prediction_diagnostics(validation_predictions, policy),
            "uncertainty": intervals,
            "temporal_thirds": thirds,
            "selected_by_symbol": {
                symbol: economic_metrics(
                    [row for row in selected_validation if row["symbol"] == symbol]
                )
                for symbol in sorted({str(row["symbol"]) for row in selected_validation})
            },
        },
        "controls": {
            "simple_causal": economic_metrics(simple),
            "random": random_metrics,
            "v15": "FROZEN_REFERENCE_REPORTED_SEPARATELY",
            "v17": "FROZEN_REFERENCE_REPORTED_SEPARATELY",
        },
        "gate": gate,
    }


def train(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = root / str(config["authority"]["source_dataset"])
    if sha256_file(dataset) != str(config["authority"]["source_dataset_sha256"]):
        raise BarrierResearchError("V18 source dataset authority mismatch")
    rows = _load(dataset)
    partitions = _mapping(config["partitions"], "partitions")
    train_end = _time(str(partitions["train"]["end"]))
    validation_start = _time(str(partitions["validation"]["start"]))
    validation_end = _time(str(partitions["validation"]["end"]))
    train_rows = [row for row in rows if row["timestamp_value"] <= train_end]
    validation_rows = [
        row
        for row in rows
        if validation_start <= row["timestamp_value"] <= validation_end and row["independent"]
    ]
    if len(train_rows) != int(partitions["train"]["expected_rows"]):
        raise BarrierResearchError("V18 TRAIN row count drift")
    if len([row for row in rows if validation_start <= row["timestamp_value"] <= validation_end]) != int(
        partitions["validation"]["expected_rows"]
    ):
        raise BarrierResearchError("V18 VALIDATION row count drift")
    v15_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_directional_contract_v15_research.yaml").read_text()
        ),
        "v15_config",
    )
    reports = {}
    for offset, side in enumerate(("LONG", "SHORT")):
        side_train = [row for row in train_rows if row["side"] == side]
        side_validation = [row for row in validation_rows if row["side"] == side]
        reports[side] = _side_report(
            side_train,
            side_validation,
            contract_indices(v15_config, side),
            config,
            seed=int(config["models"]["safety_clean"]["seed"]) + offset * 100,
        )
    validation_passed = all(bool(report["gate"]["passed"]) for report in reports.values())
    holdout = _mapping(partitions["final_holdout"], "final_holdout")
    return {
        "schema_id": "aegis-v18-train-validation-result-v1",
        "experiment_id": str(config["experiment_id"]),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "sides": reports,
        "validation_gate_passed": validation_passed,
        "final_holdout": {
            "status": str(holdout["status"]),
            "access_count": int(holdout["access_count"]),
            "opened": False,
        },
        "V18_READY_FOR_SHADOW": False,
        "V18_READY_FOR_LIVE": False,
        "blockers": (["VALIDATION_GATE_FAILED"] if not validation_passed else [])
        + ["FINAL_HOLDOUT_NOT_MATURE_OR_OPENED", "NEW_SHADOW_FORWARD_EVIDENCE_REQUIRED"],
        "model_exported": False,
        "shadow_changed": False,
        "live_changed": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/experiments/aegis_v18_preregistered.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/v18/train_validation_result.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    result = train(root, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "output": str(output),
                "validation_gate_passed": result["validation_gate_passed"],
                "V18_READY_FOR_SHADOW": result["V18_READY_FOR_SHADOW"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
