#!/usr/bin/env python3
"""Train and validate regime-aware LONG/SHORT v6 specialists in research Shadow."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from aegis.models import CalibratorSpec
from aegis.research.regime_aware_directional_v6_training import (
    ABLATION_COMPONENTS,
    ablation_score,
    bootstrap_mean_interval,
    quality_score,
    reliability_table,
    select_ablation_cross_section,
    select_cross_section,
    selection_metrics,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

CLASSIFIER_TARGETS = {
    "protectable_probability": "protectable_advantage",
    "target_probability": "target_before_stop",
    "early_reversal_probability": "early_reversal",
}
REGRESSION_TARGETS = {
    "expected_protected_net": "full_lifecycle_worst_net_return",
    "mae_q90": "mae_fraction",
    "time_to_advantage": "time_to_protectable_fraction",
}


def load_side(path: Path, side: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), f"dataset:{line_number}")
            if row.get("side") != side:
                continue
            features = tuple(float(value) for value in row["features"])
            router_features = tuple(
                float(value) for value in row["regime_router_features"]
            )
            if (
                not features
                or not router_features
                or not all(math.isfinite(value) for value in features)
                or not all(math.isfinite(value) for value in router_features)
            ):
                raise ValueError("v6 dataset contains invalid features")
            rows.append(
                {
                    **row,
                    "timestamp_value": datetime.fromisoformat(str(row["timestamp"])),
                    "features": features,
                    "regime_router_features": router_features,
                }
            )
    if not rows:
        raise ValueError(f"v6 dataset contains no {side} rows")
    return rows


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def _regressor(
    seed: int, *, quantile: float | None = None
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile" if quantile is not None else "squared_error",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def _x(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([row["features"] for row in rows], dtype=np.float32)


def _unique_timestamps(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["timestamp"]), row)
    return list(unique.values())


def _fit_regime_router(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    train = _unique_timestamps(train)
    calibration = _unique_timestamps(calibration)
    classes = ("BULLISH", "NEUTRAL", "BEARISH")
    y_train = np.asarray([str(row["realized_global_regime"]) for row in train])
    y_calibration = np.asarray(
        [str(row["realized_global_regime"]) for row in calibration]
    )
    if set(y_train) != set(classes) or set(y_calibration) != set(classes):
        return None
    x_train = np.asarray(
        [row["regime_router_features"] for row in train], dtype=np.float32
    )
    x_calibration = np.asarray(
        [row["regime_router_features"] for row in calibration], dtype=np.float32
    )
    model = _classifier(seed)
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_calibration)
    model_classes = tuple(str(value) for value in model.classes_)
    calibrators = {}
    for class_name in classes:
        class_index = model_classes.index(class_name)
        binary = np.asarray(y_calibration == class_name, dtype=np.int8)
        calibrators[class_name] = fit_platt_calibrator(raw[:, class_index], binary)
    priors = {
        class_name: float(np.mean(y_train == class_name)) for class_name in classes
    }
    return {
        "model": model,
        "classes": classes,
        "model_classes": model_classes,
        "calibrators": calibrators,
        "training_priors": priors,
        "train_timestamps": len(train),
        "calibration_timestamps": len(calibration),
    }


def _regime_probabilities(
    rows: Sequence[Mapping[str, Any]], router: Mapping[str, Any]
) -> list[Mapping[str, float]]:
    features = np.asarray(
        [row["regime_router_features"] for row in rows], dtype=np.float32
    )
    raw = router["model"].predict_proba(features)
    model_classes = tuple(router["model_classes"])
    result = []
    for index in range(len(rows)):
        calibrated = {
            class_name: float(
                router["calibrators"][class_name].apply(
                    float(raw[index, model_classes.index(class_name)])
                )
            )
            for class_name in router["classes"]
        }
        denominator = sum(calibrated.values())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("v6 regime calibration produced invalid probabilities")
        result.append({name: value / denominator for name, value in calibrated.items()})
    return result


def _fit(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    if len(train) < 200 or len(calibration) < 80:
        return None
    x_train = _x(train)
    x_calibration = _x(calibration)
    classifiers: dict[str, tuple[Any, CalibratorSpec]] = {}
    for offset, (output, target) in enumerate(CLASSIFIER_TARGETS.items()):
        y_train = np.asarray([bool(row[target]) for row in train], dtype=np.int8)
        y_calibration = np.asarray(
            [bool(row[target]) for row in calibration], dtype=np.int8
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_calibration)) < 2:
            return None
        model = _classifier(seed + offset)
        model.fit(x_train, y_train)
        raw = model.predict_proba(x_calibration)[:, 1]
        calibrator = fit_platt_calibrator(raw, y_calibration)
        classifiers[output] = (model, calibrator)
    regressors: dict[str, Any] = {}
    for offset, (output, target) in enumerate(REGRESSION_TARGETS.items(), start=10):
        model = _regressor(
            seed + offset, quantile=0.90 if output == "mae_q90" else None
        )
        model.fit(
            x_train,
            np.asarray([float(row[target]) for row in train], dtype=np.float64),
        )
        regressors[output] = model
    router = _fit_regime_router(train, calibration, seed + 100)
    if router is None:
        return None
    return {
        "classifiers": classifiers,
        "regressors": regressors,
        "regime_router": router,
    }


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = _x(rows)
    outputs: dict[str, np.ndarray] = {}
    for output, (model, calibrator) in bundle["classifiers"].items():
        raw = model.predict_proba(features)[:, 1]
        outputs[output] = np.asarray(
            [calibrator.apply(float(value)) for value in raw], dtype=np.float64
        )
    for output, model in bundle["regressors"].items():
        values = np.asarray(model.predict(features), dtype=np.float64)
        if output in {"mae_q90", "time_to_advantage"}:
            values = np.maximum(values, 0.0)
        if output == "time_to_advantage":
            values = np.minimum(values, 1.0)
        outputs[output] = values
    regime = _regime_probabilities(rows, bundle["regime_router"])
    result = []
    for index, row in enumerate(rows):
        predicted = {name: float(values[index]) for name, values in outputs.items()}
        predicted["quality_score"] = quality_score(**predicted)
        router_probabilities = regime[index]
        predicted_regime = max(
            router_probabilities, key=lambda name: router_probabilities[name]
        )
        result.append(
            {
                **row,
                **predicted,
                "regime_probabilities": router_probabilities,
                "predicted_global_regime": predicted_regime,
            }
        )
    return result


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
        if bool(row["independent"])
        and calibration_end + embargo < row["timestamp_value"] <= test_end
        and (test_symbol is None or row["symbol"] == test_symbol)
    ]
    return train, calibration, test


def _candidate_policies(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    search = _mapping(config["policy_search"], "policy_search")
    scores = np.asarray([row["quality_score"] for row in calibration], dtype=np.float64)
    maes = np.asarray([row["mae_q90"] for row in calibration], dtype=np.float64)
    times = np.asarray(
        [row["time_to_advantage"] for row in calibration], dtype=np.float64
    )
    reversals = np.asarray(
        [row["early_reversal_probability"] for row in calibration], dtype=np.float64
    )
    return [
        {
            "minimum_score": float(np.quantile(scores, score_quantile)),
            "maximum_mae_q90": float(np.quantile(maes, mae_quantile)),
            "maximum_time_to_advantage": float(np.quantile(times, time_quantile)),
            "maximum_early_reversal_probability": float(
                np.quantile(reversals, reversal_quantile)
            ),
            "maximum_selected_per_timestamp": int(maximum_selected),
            "source": "CALIBRATION_ONLY",
            "quantiles": {
                "score": float(score_quantile),
                "mae": float(mae_quantile),
                "time": float(time_quantile),
                "reversal": float(reversal_quantile),
            },
        }
        for score_quantile in search["score_quantiles"]
        for mae_quantile in search["maximum_mae_quantiles"]
        for time_quantile in search["maximum_time_quantiles"]
        for reversal_quantile in search["maximum_reversal_quantiles"]
        for maximum_selected in search["maximum_selected_per_timestamp_grid"]
    ]


def _derive_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    minimum = int(config["validation"]["minimum_calibration_selections"])
    ranked = []
    for policy in _candidate_policies(calibration, config):
        mask = select_cross_section(calibration, policy)
        selected = [row for row, keep in zip(calibration, mask) if keep]
        if len(selected) < minimum:
            continue
        metrics = selection_metrics(selected)
        ranked.append((policy, metrics))
    if not ranked:
        return None
    policy, metrics = max(
        ranked,
        key=lambda item: (
            float(item[1]["mean_protected_net"]),
            float(item[1]["protectable_rate"]),
            -float(item[1]["mae_q90"]),
            int(item[1]["count"]),
        ),
    )
    return {**policy, "calibration_metrics": metrics, "policies_evaluated": len(ranked)}


def _probability_metrics(
    rows: Sequence[Mapping[str, Any]], probability: str, target: str
) -> Mapping[str, Any]:
    probabilities = np.asarray([float(row[probability]) for row in rows])
    labels = np.asarray([bool(row[target]) for row in rows], dtype=np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "count": len(rows),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "reliability": reliability_table(
            probabilities.tolist(), labels.astype(bool).tolist()
        ),
    }


def _regime_router_metrics(
    rows: Sequence[Mapping[str, Any]],
    router: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    unique = _unique_timestamps(rows)
    probabilities = _regime_probabilities(unique, router)
    classes = tuple(str(value) for value in router["classes"])
    labels = [str(row["realized_global_regime"]) for row in unique]
    if not unique or any(label not in classes for label in labels):
        raise ValueError("v6 realized regime labels are invalid")
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, probability[label]))
                for probability, label in zip(probabilities, labels)
            ]
        )
    )
    priors = {name: float(router["training_priors"][name]) for name in classes}
    prior_log_loss = -float(
        np.mean([math.log(max(epsilon, priors[label])) for label in labels])
    )
    predicted = [
        max(probability, key=lambda name: probability[name])
        for probability in probabilities
    ]
    accuracy = float(np.mean([left == right for left, right in zip(predicted, labels)]))
    counts = Counter(labels)
    majority_accuracy = max(counts.values()) / len(labels)
    brier = float(
        np.mean(
            [
                sum((probability[name] - float(label == name)) ** 2 for name in classes)
                for probability, label in zip(probabilities, labels)
            ]
        )
    )
    contract = _mapping(
        _mapping(config["regime"], "regime")["probabilistic_router"],
        "probabilistic_router",
    )
    passed = bool(
        (
            not bool(contract["require_log_loss_below_training_prior"])
            or log_loss < prior_log_loss
        )
        and (
            not bool(contract["require_accuracy_not_below_majority"])
            or accuracy >= majority_accuracy
        )
    )
    return {
        "timestamps": len(unique),
        "classes": list(classes),
        "label_counts": dict(sorted(counts.items())),
        "log_loss": log_loss,
        "training_prior_log_loss": prior_log_loss,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "multiclass_brier": brier,
        "passed": passed,
    }


def _group_audit(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"DIRECTION::{row['regime']['direction']}"].append(row)
        groups[f"ROLE::{row['directional_role']}"].append(row)
    result = {}
    for identity, values in sorted(groups.items()):
        result[identity] = {
            "metrics": selection_metrics(values),
            "protectable_calibration": _probability_metrics(
                values, "protectable_probability", "protectable_advantage"
            ),
        }
    return result


def _ablation_audit(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _mapping(config["ablations"], "ablations")
    quantile = float(raw["calibration_score_quantile"])
    maximum = int(raw["maximum_selected_per_timestamp"])
    configured = tuple(str(value) for value in raw["variants"])
    if not 0.0 < quantile < 1.0 or maximum <= 0:
        raise ValueError("v6 ablation configuration is invalid")
    unknown = sorted(set(configured).difference(ABLATION_COMPONENTS))
    if unknown:
        raise ValueError(f"unknown v6 ablation variants: {unknown}")
    reports: dict[str, Any] = {}
    for variant in configured:
        threshold = float(
            np.quantile(
                np.asarray(
                    [ablation_score(row, variant) for row in calibration],
                    dtype=np.float64,
                ),
                quantile,
            )
        )
        mask = select_ablation_cross_section(
            test,
            variant=variant,
            minimum_score=threshold,
            maximum_selected_per_timestamp=maximum,
        )
        selected = [row for row, keep in zip(test, mask) if keep]
        reports[variant] = {
            "source": "CALIBRATION_ONLY_FIXED_QUANTILE",
            "calibration_quantile": quantile,
            "minimum_score": threshold,
            "metrics": selection_metrics(selected),
            "promotion_eligible": False,
        }
    return reports


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_number: int,
    config: Mapping[str, Any],
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> Mapping[str, Any]:
    validation = _mapping(config["validation"], "validation")
    train, calibration, test = _split(
        rows,
        boundaries,
        int(validation["embargo_minutes"]),
        excluded_train_symbol=excluded_train_symbol,
        test_symbol=test_symbol,
    )
    bundle = _fit(train, calibration, 20260810 + fold_number)
    if bundle is None or not test:
        return {
            "fold": fold_number,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted_calibration = _predict(calibration, bundle)
    policy = _derive_policy(predicted_calibration, config)
    if policy is None:
        return {
            "fold": fold_number,
            "status": "NO_CALIBRATION_POLICY",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted_test = _predict(test, bundle)
    regime_router = _regime_router_metrics(
        predicted_test, bundle["regime_router"], config
    )
    mask = select_cross_section(predicted_test, policy)
    selected = [row for row, keep in zip(predicted_test, mask) if keep]
    baseline = selection_metrics(predicted_test)
    side = str(predicted_test[0]["side"])
    current_brain = selection_metrics(
        [row for row in predicted_test if row["entry_brain_action"] == side]
    )
    metrics = selection_metrics(selected)
    enough = len(selected) >= int(validation["minimum_test_selections_per_fold"])
    gap_ok = bool(
        metrics["p95_gap_hours"] is not None
        and float(metrics["p95_gap_hours"])
        <= float(validation["maximum_p95_gap_hours"])
    )
    passed = bool(
        enough
        and gap_ok
        and regime_router["passed"]
        and metrics["mean_protected_net"] is not None
        and float(metrics["mean_protected_net"]) > 0.0
        and float(metrics["mean_protected_net"]) > float(baseline["mean_protected_net"])
        and float(metrics["mae_q90"]) <= float(baseline["mae_q90"])
        and float(metrics["mean_underwater_bars"])
        <= float(baseline["mean_underwater_bars"])
        and float(metrics["protectable_rate"]) > float(baseline["protectable_rate"])
    )
    return {
        "fold": fold_number,
        "status": "EVALUATED",
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "calibration": len(calibration),
        "test": len(test),
        "policy": policy,
        "baseline": baseline,
        "current_brain_control": current_brain,
        "probabilistic_regime_router": regime_router,
        "selected": metrics,
        "opportunity_gap_gate_passed": gap_ok,
        "selected_net_bootstrap": bootstrap_mean_interval(
            [float(row["full_lifecycle_worst_net_return"]) for row in selected],
            samples=int(validation["bootstrap_samples"]),
            seed=20260810 + fold_number,
        ),
        "protectable_probability": _probability_metrics(
            predicted_test, "protectable_probability", "protectable_advantage"
        ),
        "diagnostic_ablations": _ablation_audit(
            predicted_calibration, predicted_test, config
        ),
        "group_audit": _group_audit(selected) if selected else {},
        "passed": passed,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def _leave_one_symbol_out(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    symbols = sorted({str(row["symbol"]) for row in rows})
    reports = {
        symbol: _evaluate_fold(
            rows,
            boundaries,
            100 + index,
            config,
            excluded_train_symbol=symbol,
            test_symbol=symbol,
        )
        for index, symbol in enumerate(symbols)
    }
    passing = sum(bool(report.get("passed")) for report in reports.values())
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "status": "EVALUATED",
        "symbols": reports,
        "passing_symbols": passing,
        "required_symbols": required,
        "passed": passing >= required,
    }


def evaluate_side(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    boundaries = _fold_boundaries(sorted({row["timestamp_value"] for row in rows}))
    folds = [
        _evaluate_fold(rows, boundary, index + 1, config)
        for index, boundary in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    minimum = int(config["validation"]["minimum_positive_folds"])
    router_skilled = sum(
        bool(fold["probabilistic_regime_router"]["passed"]) for fold in evaluated
    )
    minimum_router_skilled = int(
        config["regime"]["probabilistic_router"]["minimum_skilled_folds"]
    )
    worst_non_negative = bool(
        evaluated
        and all(
            fold["selected"]["mean_protected_net"] is not None
            and float(fold["selected"]["mean_protected_net"]) >= 0.0
            for fold in evaluated
        )
    )
    primary = bool(
        len(evaluated) == 4
        and passing >= minimum
        and worst_non_negative
        and router_skilled >= minimum_router_skilled
    )
    loso = (
        _leave_one_symbol_out(rows, boundaries[-1], config)
        if primary
        else {"status": "NOT_RUN_PRIMARY_GATE_FAILED", "passed": False}
    )
    passed = primary and bool(loso["passed"])
    return {
        "rows": len(rows),
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "regime_router_skilled_folds": router_skilled,
        "minimum_regime_router_skilled_folds": minimum_router_skilled,
        "worst_fold_non_negative": worst_non_negative,
        "primary_gate": primary,
        "leave_one_symbol_out": loso,
        "validation_pass": passed,
        "verdict": (
            "ELIGIBLE_FOR_PROSPECTIVE_SHADOW_DEPLOYMENT"
            if passed
            else "HISTORICAL_VALIDATION_FAILED_RESEARCH_ONLY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_regime_aware_directional_v6_shadow.yaml"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/regime_aware_directional_v6/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/regime_aware_directional_v6/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regime_aware_directional_v6/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    manifest_path = (
        args.dataset_manifest
        if args.dataset_manifest.is_absolute()
        else root / args.dataset_manifest
    )
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "v6_config")
    manifest = _mapping(json.loads(manifest_path.read_text()), "dataset_manifest")
    if manifest.get("dataset_sha256") != sha256_file(dataset):
        raise RuntimeError("AEGIS_V6_DATASET_HASH_MISMATCH")
    sides = {}
    for side in ("LONG", "SHORT"):
        print(json.dumps({"training_side": side}), flush=True)
        sides[side] = evaluate_side(load_side(dataset, side), config)
    validation_pass = all(bool(report["validation_pass"]) for report in sides.values())
    report = {
        "schema_id": "aegis-regime-aware-directional-v6-validation-v1",
        "experiment_id": config["experiment_id"],
        "mode": "RESEARCH_SHADOW",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_path": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": sha256_file(dataset),
        "source_evidence_start": manifest["evidence_start"],
        "source_evidence_end": manifest["evidence_end"],
        "sides": sides,
        "validation_pass": validation_pass,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_ACTIVATION"
            if validation_pass
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "shadow_runtime_enabled": False,
        "model_exported": False,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "content_hash": "",
    }
    report["content_hash"] = Sha256HashProvider().digest_value(
        {**report, "content_hash": ""}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "output": str(output),
                "content_hash": report["content_hash"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
