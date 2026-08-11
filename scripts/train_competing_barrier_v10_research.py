#!/usr/bin/env python3
"""Purged walk-forward validation for outcome-only competing-barrier V10."""

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
from sklearn.ensemble import HistGradientBoostingClassifier

from aegis.research.competing_barrier_v10 import (
    BarrierOutcome,
    conservative_utility,
    contracts_from_config,
)
from aegis.research.competing_barrier_v10_training import (
    fold_passes,
    select_cross_section,
    utility_metrics,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

DIRECTION_CLASSES = ("LONG", "SHORT", "ABSTAIN")
OUTCOME_CLASSES = tuple(value.value for value in BarrierOutcome)


def load_side(path: Path, side: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = dict(_mapping(json.loads(line), f"dataset:{line_number}"))
            if source.get("side") != side:
                continue
            features = tuple(float(value) for value in source["v9_features"])
            direction_features = tuple(
                float(value) for value in source["v9_direction_features"]
            )
            if not all(math.isfinite(value) for value in (*features, *direction_features)):
                raise ValueError("V10 dataset contains non-finite features")
            outcomes = _mapping(source["v10_contract_outcomes"], "outcomes")
            rows.append(
                {
                    **source,
                    "timestamp_value": datetime.fromisoformat(str(source["timestamp"])),
                    "features": features,
                    "direction_features": direction_features,
                    "direction_label": str(source["v10_direction_label"]),
                    "outcomes": outcomes,
                }
            )
    if not rows:
        raise ValueError(f"V10 dataset has no {side} episodes")
    return rows


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )


def _x(rows: Sequence[Mapping[str, Any]], key: str = "features") -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def _fit_multiclass(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    target: Any,
    classes: Sequence[str],
    feature_key: str,
    seed: int,
) -> Mapping[str, Any] | None:
    y_train = np.asarray([str(target(row)) for row in train])
    y_calibration = np.asarray([str(target(row)) for row in calibration])
    if len(set(y_train)) < 2 or not set(y_train).issubset(set(classes)):
        return None
    model = _classifier(seed)
    model.fit(_x(train, feature_key), y_train)
    raw = model.predict_proba(_x(calibration, feature_key))
    model_classes = tuple(str(value) for value in model.classes_)
    calibrators: dict[str, Any | None] = {}
    for name in classes:
        target_values = np.asarray(y_calibration == name, dtype=np.int8)
        if name in model_classes and len(np.unique(target_values)) == 2:
            calibrators[name] = fit_platt_calibrator(
                raw[:, model_classes.index(name)], target_values
            )
        else:
            calibrators[name] = None
    denominator = len(y_train) + len(classes)
    return {
        "model": model,
        "classes": tuple(classes),
        "model_classes": model_classes,
        "calibrators": calibrators,
        "priors": {
            name: (float(np.sum(y_train == name)) + 1.0) / denominator
            for name in classes
        },
    }


def _probabilities(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], feature_key: str
) -> list[Mapping[str, float]]:
    raw = bundle["model"].predict_proba(_x(rows, feature_key))
    model_classes = tuple(bundle["model_classes"])
    result: list[Mapping[str, float]] = []
    for index in range(len(rows)):
        values = {}
        for name in bundle["classes"]:
            if name not in model_classes:
                value = float(bundle["priors"][name])
            else:
                value = float(raw[index, model_classes.index(name)])
                calibrator = bundle["calibrators"][name]
                if calibrator is not None:
                    value = float(calibrator.apply(value))
            values[name] = max(1e-9, value)
        total = sum(values.values())
        result.append({name: value / total for name, value in values.items()})
    return result


def _fit_bundle(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    if len(train) < 500 or len(calibration) < 200:
        return None
    direction = _fit_multiclass(
        train,
        calibration,
        target=lambda row: row["direction_label"],
        classes=DIRECTION_CLASSES,
        feature_key="direction_features",
        seed=seed,
    )
    if direction is None:
        return None
    risks = {}
    for offset, contract in enumerate(contracts_from_config(config), start=10):
        fitted = _fit_multiclass(
            train,
            calibration,
            target=lambda row, name=contract.name: row["outcomes"][name]["outcome"],
            classes=OUTCOME_CLASSES,
            feature_key="features",
            seed=seed + offset,
        )
        if fitted is None:
            return None
        risks[contract.name] = fitted
    return {"direction": direction, "risks": risks}


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not rows:
        return []
    direction = _probabilities(rows, bundle["direction"], "direction_features")
    risk = {
        contract.name: _probabilities(rows, bundle["risks"][contract.name], "features")
        for contract in contracts_from_config(config)
    }
    penalty = float(config["utility"]["unknown_penalty_fraction_of_adverse"])
    result = []
    for index, row in enumerate(rows):
        choices = []
        for contract in contracts_from_config(config):
            probabilities = risk[contract.name][index]
            utility = conservative_utility(
                probabilities, contract, unknown_penalty_fraction=penalty
            )
            unknown = (
                probabilities[BarrierOutcome.SAME_BAR_AMBIGUOUS.value]
                + probabilities[BarrierOutcome.NEITHER_REACHED.value]
            )
            choices.append((utility, -unknown, contract.name, probabilities))
        utility, negative_unknown, contract_name, probabilities = max(choices)
        actual = row["outcomes"][contract_name]
        side = str(row["side"])
        result.append(
            {
                **row,
                "direction_probabilities": direction[index],
                "direction_probability": direction[index][side],
                "selected_contract": contract_name,
                "outcome_probabilities": probabilities,
                "unknown_probability": -negative_unknown,
                "predicted_utility": utility,
                "actual_utility": float(actual["realized_utility"]),
                "actual_outcome": str(actual["outcome"]),
            }
        )
    return result


def _multiclass_skill(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Mapping[str, float]],
    bundle: Mapping[str, Any],
    target: Any,
) -> Mapping[str, Any]:
    labels = [str(target(row)) for row in rows]
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [math.log(max(epsilon, values[label])) for values, label in zip(probabilities, labels)]
        )
    )
    prior_loss = -float(
        np.mean([math.log(max(epsilon, bundle["priors"][label])) for label in labels])
    )
    predicted = [max(values, key=values.get) for values in probabilities]
    accuracy = float(np.mean([left == right for left, right in zip(predicted, labels)]))
    majority = max(bundle["priors"], key=bundle["priors"].get)
    majority_accuracy = float(np.mean([label == majority for label in labels]))
    return {
        "count": len(rows),
        "log_loss": log_loss,
        "training_prior_log_loss": prior_loss,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "passed": log_loss < prior_loss and accuracy >= majority_accuracy,
    }


def _component_metrics(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    direction_probabilities = _probabilities(rows, bundle["direction"], "direction_features")
    direction = _multiclass_skill(
        rows,
        direction_probabilities,
        bundle["direction"],
        lambda row: row["direction_label"],
    )
    contracts = {}
    for contract in contracts_from_config(config):
        probabilities = _probabilities(rows, bundle["risks"][contract.name], "features")
        contracts[contract.name] = _multiclass_skill(
            rows,
            probabilities,
            bundle["risks"][contract.name],
            lambda row, name=contract.name: row["outcomes"][name]["outcome"],
        )
    skilled = sum(bool(value["passed"]) for value in contracts.values())
    return {
        "direction": direction,
        "competing_risk": {
            "contracts": contracts,
            "skilled_contracts": skilled,
            "required_contracts": int(
                config["validation"]["minimum_competing_risk_skilled_contracts_per_fold"]
            ),
            "passed": skilled
            >= int(config["validation"]["minimum_competing_risk_skilled_contracts_per_fold"]),
        },
    }


def _split(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    embargo_minutes: int,
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=embargo_minutes)
    train = [
        row
        for row in rows
        if row["timestamp_value"] <= train_end
        and (excluded_train_symbol is None or row["symbol"] != excluded_train_symbol)
    ]
    calibration = [
        row
        for row in rows
        if train_end + embargo < row["timestamp_value"] <= calibration_end
        and (excluded_train_symbol is None or row["symbol"] != excluded_train_symbol)
    ]
    test = [
        row
        for row in rows
        if calibration_end + embargo < row["timestamp_value"] <= test_end
        and (test_symbol is None or row["symbol"] == test_symbol)
    ]
    return train, calibration, test


def _candidate_policies(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = config["validation"]
    utilities = np.asarray([max(0.0, float(row["predicted_utility"])) for row in calibration])
    directions = np.asarray([float(row["direction_probability"]) for row in calibration])
    unknowns = np.asarray([float(row["unknown_probability"]) for row in calibration])
    return [
        {
            "minimum_utility": max(0.0, float(np.quantile(utilities, utility_q))),
            "minimum_direction_probability": float(np.quantile(directions, direction_q)),
            "maximum_unknown_probability": float(np.quantile(unknowns, unknown_q)),
            "maximum_selected_per_timestamp": int(validation["maximum_selected_per_timestamp"]),
            "source": "CALIBRATION_ONLY",
            "quantiles": {
                "utility": utility_q,
                "direction": direction_q,
                "unknown": unknown_q,
            },
        }
        for utility_q in validation["utility_quantiles"]
        for direction_q in validation["direction_probability_quantiles"]
        for unknown_q in validation["maximum_unknown_probability_quantiles"]
    ]


def _derive_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    minimum = int(config["validation"]["minimum_calibration_selections"])
    choices = []
    for policy in _candidate_policies(calibration, config):
        mask = select_cross_section(calibration, policy)
        selected = [row for row, keep in zip(calibration, mask) if keep]
        if len(selected) >= minimum:
            choices.append((policy, utility_metrics(selected)))
    if not choices:
        return None
    policy, metrics = max(
        choices,
        key=lambda item: (
            item[1]["mean_utility"],
            item[1]["cvar"],
            item[1]["count"],
        ),
    )
    return {**policy, "calibration_metrics": metrics, "policies_evaluated": len(choices)}


def _control(rows: Sequence[Mapping[str, Any]], primary: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "actual_utility": float(row["outcomes"][primary]["realized_utility"]),
        }
        for row in rows
    ]


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold: int,
    config: Mapping[str, Any],
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> Mapping[str, Any]:
    validation = config["validation"]
    train, calibration, test = _split(
        rows,
        boundaries,
        int(validation["embargo_minutes"]),
        excluded_train_symbol=excluded_train_symbol,
        test_symbol=test_symbol,
    )
    bundle = _fit_bundle(train, calibration, config, 20260823 + fold)
    if bundle is None or not test:
        return {
            "fold": fold,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted_calibration = _predict(calibration, bundle, config)
    policy = _derive_policy(predicted_calibration, config)
    if policy is None:
        return {
            "fold": fold,
            "status": "NO_CALIBRATION_POLICY",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted = _predict(test, bundle, config)
    mask = select_cross_section(predicted, policy)
    selected_rows = [row for row, keep in zip(predicted, mask) if keep]
    selected = utility_metrics(selected_rows)
    primary = str(config["models"]["direction"]["source_contract"])
    control = utility_metrics(_control(test, primary))
    components = _component_metrics(test, bundle, config)
    economic = fold_passes(
        selected,
        control,
        minimum_count=int(validation["minimum_test_selections_per_fold"]),
        minimum_payoff=float(validation["require_payoff_ratio_at_least"]),
        maximum_p95_gap_hours=float(validation["maximum_p95_opportunity_gap_hours"]),
    )
    return {
        "fold": fold,
        "status": "EVALUATED",
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "calibration": len(calibration),
        "test": len(test),
        "policy": policy,
        "selected": selected,
        "control_identity": "UNFILTERED_PRIMARY_CONTRACT",
        "control": control,
        "components": components,
        "economic_gate": economic,
        "passed": bool(
            components["direction"]["passed"]
            and components["competing_risk"]["passed"]
            and economic
        ),
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def evaluate_side(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    boundaries = _fold_boundaries(sorted({row["timestamp_value"] for row in rows}))
    folds = [
        _evaluate_fold(rows, boundary, index + 1, config)
        for index, boundary in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    direction_skilled = sum(bool(fold["components"]["direction"]["passed"]) for fold in evaluated)
    risk_skilled = sum(bool(fold["components"]["competing_risk"]["passed"]) for fold in evaluated)
    worst_non_negative = bool(
        evaluated
        and all(
            fold["selected"]["mean_utility"] is not None
            and float(fold["selected"]["mean_utility"]) >= 0.0
            for fold in evaluated
        )
    )
    validation = config["validation"]
    primary = bool(
        len(evaluated) == int(validation["folds"])
        and passing >= int(validation["minimum_positive_folds"])
        and direction_skilled >= int(validation["minimum_direction_skilled_folds"])
        and risk_skilled >= int(validation["minimum_competing_risk_skilled_folds"])
        and worst_non_negative
    )
    loso: Mapping[str, Any] = {
        "status": "NOT_RUN_PRIMARY_GATE_FAILED",
        "passing_symbols": 0,
        "required_symbols": int(validation["minimum_symbols_without_regression"]),
        "passed": False,
    }
    if primary:
        reports = {
            symbol: _evaluate_fold(
                rows,
                boundaries[-1],
                100 + index,
                config,
                excluded_train_symbol=symbol,
                test_symbol=symbol,
            )
            for index, symbol in enumerate(sorted({str(row["symbol"]) for row in rows}))
        }
        count = sum(bool(report.get("passed")) for report in reports.values())
        loso = {
            "status": "EVALUATED",
            "symbols": reports,
            "passing_symbols": count,
            "required_symbols": int(validation["minimum_symbols_without_regression"]),
            "passed": count >= int(validation["minimum_symbols_without_regression"]),
        }
    passed = primary and bool(loso["passed"])
    return {
        "rows": len(rows),
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "direction_skilled_folds": direction_skilled,
        "competing_risk_skilled_folds": risk_skilled,
        "worst_fold_non_negative": worst_non_negative,
        "primary_gate": primary,
        "leave_one_symbol_out": loso,
        "validation_pass": passed,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW"
            if passed
            else "HISTORICAL_VALIDATION_FAILED_RESEARCH_ONLY"
        ),
    }


def run(
    *, root: Path, config_path: Path, dataset: Path, manifest_path: Path, output: Path
) -> Mapping[str, Any]:
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    manifest = _mapping(json.loads(manifest_path.read_text()), "manifest")
    if sha256_file(dataset) != str(manifest["dataset_sha256"]):
        raise ValueError("V10 dataset hash mismatch")
    sides = {side: evaluate_side(load_side(dataset, side), config) for side in ("LONG", "SHORT")}
    passed = all(report["validation_pass"] for report in sides.values())
    result = {
        "schema_id": "aegis-competing-barrier-v10-validation-v1",
        "experiment_id": str(config["experiment_id"]),
        "config_sha256": sha256_file(config_path),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "manifest_sha256": sha256_file(manifest_path),
        "sides": sides,
        "validation_pass": passed,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW"
            if passed
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "model_exported": False,
        "shadow_activated": False,
        "live_activated": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_competing_barrier_v10_research.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/competing_barrier_v10/canonical_dataset.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/competing_barrier_v10/dataset_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/competing_barrier_v10/validation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = run(
        root=root,
        config_path=resolve(args.config),
        dataset=resolve(args.dataset),
        manifest_path=resolve(args.manifest),
        output=resolve(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
