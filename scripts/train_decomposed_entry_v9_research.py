#!/usr/bin/env python3
"""Train and validate decomposed direction, timing, and trajectory V9."""

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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from aegis.models import CalibratorSpec
from aegis.research.decomposed_entry_v9 import DirectionClass, TimingFailure
from aegis.research.decomposed_entry_v9_training import (
    decomposed_quality_score,
    quantile_skill,
    regression_skill,
    select_decomposed_cross_section,
    v9_fold_passes,
    v9_selection_metrics,
)
from aegis.research.regime_aware_directional_v6_training import bootstrap_mean_interval
from aegis.research.tail_aware_entry_v8_training import binary_skill_metrics
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

TRAJECTORY_CLASSIFIERS = (
    "positive_current_ts_stress",
    "catastrophic_current_ts_stress",
)


def load_side(path: Path, side: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = dict(_mapping(json.loads(line), f"dataset:{line_number}"))
            if source.get("side") != side:
                continue
            timing = _mapping(source["v9_timing_labels"], "timing")
            trajectory = _mapping(source["v9_trajectory_targets"], "trajectory")
            features = tuple(float(value) for value in source["v9_features"])
            direction_features = tuple(
                float(value) for value in source["v9_direction_features"]
            )
            if (
                not features
                or not direction_features
                or not all(
                    math.isfinite(value) for value in (*features, *direction_features)
                )
            ):
                raise ValueError("V9 dataset contains invalid features")
            rows.append(
                {
                    **source,
                    "timestamp_value": datetime.fromisoformat(str(source["timestamp"])),
                    "features": features,
                    "direction_features": direction_features,
                    "direction_label": str(source["v9_direction_label"]["label"]),
                    "timing_targets": {
                        name.value: bool(timing[name.value]) for name in TimingFailure
                    },
                    "trajectory_targets": {
                        **{
                            name: bool(trajectory[name])
                            for name in TRAJECTORY_CLASSIFIERS
                        },
                        **{
                            name: float(trajectory[name])
                            for name in (
                                "current_ts_stress_net",
                                "mae_fraction",
                                "mfe_fraction",
                                "time_to_positive",
                            )
                        },
                    },
                }
            )
    if not rows:
        raise ValueError(f"V9 dataset has no {side} rows")
    return rows


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )


def _regressor(
    seed: int, quantile: float | None = None
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile" if quantile is not None else "squared_error",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )


def _x(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([row["features"] for row in rows], dtype=np.float32)


def _direction_x(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([row["direction_features"] for row in rows], dtype=np.float32)


def _fit_direction(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    classes = tuple(value.value for value in DirectionClass)
    y_train = np.asarray([row["direction_label"] for row in train])
    y_calibration = np.asarray([row["direction_label"] for row in calibration])
    if set(y_train) != set(classes) or set(y_calibration) != set(classes):
        return None
    model = _classifier(seed)
    model.fit(_direction_x(train), y_train)
    raw = model.predict_proba(_direction_x(calibration))
    model_classes = tuple(str(value) for value in model.classes_)
    calibrators = {
        name: fit_platt_calibrator(
            raw[:, model_classes.index(name)],
            np.asarray(y_calibration == name, dtype=np.int8),
        )
        for name in classes
    }
    return {
        "model": model,
        "classes": classes,
        "model_classes": model_classes,
        "calibrators": calibrators,
        "training_priors": {name: float(np.mean(y_train == name)) for name in classes},
    }


def _fit_binary(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    values: Any,
    seed: int,
) -> tuple[Any, CalibratorSpec] | None:
    y_train = np.asarray([bool(values(row)) for row in train], dtype=np.int8)
    y_calibration = np.asarray(
        [bool(values(row)) for row in calibration], dtype=np.int8
    )
    if len(np.unique(y_train)) < 2 or len(np.unique(y_calibration)) < 2:
        return None
    model = _classifier(seed)
    model.fit(_x(train), y_train)
    return (
        model,
        fit_platt_calibrator(model.predict_proba(_x(calibration))[:, 1], y_calibration),
    )


def _fit(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    if len(train) < 500 or len(calibration) < 200:
        return None
    direction = _fit_direction(train, calibration, seed)
    if direction is None:
        return None
    timing = {}
    for offset, name in enumerate(TimingFailure, start=10):
        fitted = _fit_binary(
            train,
            calibration,
            lambda row, key=name.value: row["timing_targets"][key],
            seed + offset,
        )
        if fitted is None:
            return None
        timing[name.value] = fitted
    trajectory_classifiers = {}
    for offset, name in enumerate(TRAJECTORY_CLASSIFIERS, start=30):
        fitted = _fit_binary(
            train,
            calibration,
            lambda row, key=name: row["trajectory_targets"][key],
            seed + offset,
        )
        if fitted is None:
            return None
        trajectory_classifiers[name] = fitted
    regressors = {}
    definitions = {
        "expected_stress_net": ("current_ts_stress_net", None),
        "mae_q90": ("mae_fraction", 0.90),
        "mfe_q50": ("mfe_fraction", 0.50),
        "time_to_positive": ("time_to_positive", None),
    }
    for offset, (name, (target, quantile)) in enumerate(definitions.items(), start=40):
        model = _regressor(seed + offset, quantile)
        model.fit(
            _x(train),
            np.asarray([row["trajectory_targets"][target] for row in train]),
        )
        regressors[name] = model
    return {
        "direction": direction,
        "timing": timing,
        "trajectory_classifiers": trajectory_classifiers,
        "regressors": regressors,
        "baselines": {
            "stress_mean": float(
                np.mean(
                    [
                        row["trajectory_targets"]["current_ts_stress_net"]
                        for row in train
                    ]
                )
            ),
            "mae_q90": float(
                np.quantile(
                    [row["trajectory_targets"]["mae_fraction"] for row in train], 0.90
                )
            ),
            "mfe_q50": float(
                np.quantile(
                    [row["trajectory_targets"]["mfe_fraction"] for row in train], 0.50
                )
            ),
        },
    }


def _calibrated_binary(
    rows: Sequence[Mapping[str, Any]], bundle: tuple[Any, CalibratorSpec]
) -> np.ndarray:
    model, calibrator = bundle
    return np.asarray(
        [
            calibrator.apply(float(value))
            for value in model.predict_proba(_x(rows))[:, 1]
        ],
        dtype=np.float64,
    )


def _direction_probabilities(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[Mapping[str, float]]:
    raw = bundle["model"].predict_proba(_direction_x(rows))
    model_classes = tuple(bundle["model_classes"])
    result = []
    for index in range(len(rows)):
        values = {
            name: float(
                bundle["calibrators"][name].apply(
                    float(raw[index, model_classes.index(name)])
                )
            )
            for name in bundle["classes"]
        }
        denominator = sum(values.values())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("V9 direction calibration is invalid")
        result.append({name: value / denominator for name, value in values.items()})
    return result


def _predict(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    stress_cost: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    direction = _direction_probabilities(rows, bundle["direction"])
    timing = {
        name: _calibrated_binary(rows, fitted)
        for name, fitted in bundle["timing"].items()
    }
    trajectory_probabilities = {
        name: _calibrated_binary(rows, fitted)
        for name, fitted in bundle["trajectory_classifiers"].items()
    }
    regressions = {
        name: np.asarray(model.predict(_x(rows)), dtype=np.float64)
        for name, model in bundle["regressors"].items()
    }
    result = []
    for index, row in enumerate(rows):
        side = str(row["side"])
        timing_probabilities = {
            name: float(values[index]) for name, values in timing.items()
        }
        maximum_timing = max(timing_probabilities.values())
        mae = max(0.0, float(regressions["mae_q90"][index]))
        mfe = max(0.0, float(regressions["mfe_q50"][index]))
        time_to_positive = min(
            1.0, max(0.0, float(regressions["time_to_positive"][index]))
        )
        positive = float(trajectory_probabilities["positive_current_ts_stress"][index])
        catastrophic = float(
            trajectory_probabilities["catastrophic_current_ts_stress"][index]
        )
        expected_stress = float(regressions["expected_stress_net"][index])
        score, reward_risk = decomposed_quality_score(
            direction_probability=direction[index][side],
            positive_probability=positive,
            maximum_timing_risk=maximum_timing,
            catastrophic_probability=catastrophic,
            expected_stress_net=expected_stress,
            mae_q90=mae,
            mfe_q50=mfe,
            time_to_positive=time_to_positive,
            stress_cost_fraction=stress_cost,
        )
        targets = row["trajectory_targets"]
        result.append(
            {
                **row,
                "direction_probabilities": direction[index],
                "direction_probability": direction[index][side],
                "timing_probabilities": timing_probabilities,
                "maximum_timing_risk": maximum_timing,
                "positive_probability": positive,
                "catastrophic_probability": catastrophic,
                "expected_stress_net": expected_stress,
                "mae_q90": mae,
                "mfe_q50": mfe,
                "predicted_time_to_positive": time_to_positive,
                "predicted_reward_risk": reward_risk,
                "v9_quality_score": score,
                "selected_profile": "CURRENT_TS",
                "selected_expected_net": targets["current_ts_stress_net"]
                + stress_cost
                - 0.001,
                "selected_stress_net": targets["current_ts_stress_net"],
                "selected_severe_net": targets["current_ts_stress_net"] - 0.0005,
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
        if row["independent"]
        and calibration_end + embargo < row["timestamp_value"] <= test_end
        and (test_symbol is None or row["symbol"] == test_symbol)
    ]
    return train, calibration, test


def _direction_metrics(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> Mapping[str, Any]:
    probabilities = _direction_probabilities(rows, bundle)
    labels = [str(row["direction_label"]) for row in rows]
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, probability[label]))
                for probability, label in zip(probabilities, labels)
            ]
        )
    )
    priors = bundle["training_priors"]
    prior_loss = -float(
        np.mean([math.log(max(epsilon, priors[label])) for label in labels])
    )
    predicted = [max(value, key=value.get) for value in probabilities]
    accuracy = float(np.mean([left == right for left, right in zip(predicted, labels)]))
    majority = max(bundle["classes"], key=lambda name: priors[name])
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
    rows: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    minimum_timing_heads: int,
) -> Mapping[str, Any]:
    timing = {
        name.value: binary_skill_metrics(
            [row["timing_probabilities"][name.value] for row in predicted],
            [row["timing_targets"][name.value] for row in rows],
        )
        for name in TimingFailure
    }
    skilled = sum(bool(value["passed"]) for value in timing.values())
    positive = binary_skill_metrics(
        [row["positive_probability"] for row in predicted],
        [row["trajectory_targets"]["positive_current_ts_stress"] for row in rows],
    )
    catastrophic = binary_skill_metrics(
        [row["catastrophic_probability"] for row in predicted],
        [row["trajectory_targets"]["catastrophic_current_ts_stress"] for row in rows],
    )
    stress = regression_skill(
        [row["expected_stress_net"] for row in predicted],
        [row["trajectory_targets"]["current_ts_stress_net"] for row in rows],
        bundle["baselines"]["stress_mean"],
    )
    mae = quantile_skill(
        [row["mae_q90"] for row in predicted],
        [row["trajectory_targets"]["mae_fraction"] for row in rows],
        bundle["baselines"]["mae_q90"],
        0.90,
    )
    mfe = quantile_skill(
        [row["mfe_q50"] for row in predicted],
        [row["trajectory_targets"]["mfe_fraction"] for row in rows],
        bundle["baselines"]["mfe_q50"],
        0.50,
    )
    trajectory_passed = all(
        value["passed"] for value in (positive, catastrophic, stress, mae, mfe)
    )
    return {
        "direction": _direction_metrics(rows, bundle["direction"]),
        "timing": {
            "heads": timing,
            "skilled_heads": skilled,
            "required_heads": minimum_timing_heads,
            "passed": skilled >= minimum_timing_heads,
        },
        "trajectory": {
            "positive_probability": positive,
            "catastrophic_probability": catastrophic,
            "stress_regression": stress,
            "mae_q90": mae,
            "mfe_q50": mfe,
            "passed": trajectory_passed,
        },
    }


def _candidate_policies(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = _mapping(config["validation"], "validation")
    arrays = {
        "score": np.asarray([row["v9_quality_score"] for row in calibration]),
        "direction": np.asarray([row["direction_probability"] for row in calibration]),
        "timing": np.asarray([row["maximum_timing_risk"] for row in calibration]),
        "mae": np.asarray([row["mae_q90"] for row in calibration]),
        "reward": np.asarray([row["predicted_reward_risk"] for row in calibration]),
    }
    return [
        {
            "minimum_score": float(np.quantile(arrays["score"], score_q)),
            "minimum_direction_probability": float(
                np.quantile(arrays["direction"], direction_q)
            ),
            "maximum_timing_risk": float(np.quantile(arrays["timing"], timing_q)),
            "maximum_mae_q90": float(np.quantile(arrays["mae"], mae_q)),
            "minimum_reward_risk": float(np.quantile(arrays["reward"], reward_q)),
            "maximum_selected_per_timestamp": int(
                validation["maximum_selected_per_timestamp"]
            ),
            "source": "CALIBRATION_ONLY",
            "quantiles": {
                "score": score_q,
                "direction": direction_q,
                "timing": timing_q,
                "mae": mae_q,
                "reward_risk": reward_q,
            },
        }
        for score_q in validation["score_quantiles"]
        for direction_q in validation["direction_probability_quantiles"]
        for timing_q in validation["maximum_timing_risk_quantiles"]
        for mae_q in validation["maximum_mae_quantiles"]
        for reward_q in validation["minimum_reward_risk_quantiles"]
    ]


def _derive_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    minimum = int(config["validation"]["minimum_calibration_selections"])
    tail = float(config["trajectory"]["tail_quantile"])
    choices = []
    for policy in _candidate_policies(calibration, config):
        mask = select_decomposed_cross_section(calibration, policy)
        selected = [row for row, keep in zip(calibration, mask) if keep]
        if len(selected) < minimum:
            continue
        choices.append((policy, v9_selection_metrics(selected, tail_quantile=tail)))
    if not choices:
        return None
    policy, metrics = max(
        choices,
        key=lambda item: (
            item[1]["mean_stress_net"],
            item[1]["stress_cvar"],
            -item[1]["mean_mae"],
            item[1]["count"],
        ),
    )
    return {
        **policy,
        "calibration_metrics": metrics,
        "policies_evaluated": len(choices),
    }


def _control_rows(
    rows: Sequence[Mapping[str, Any]], side: str, stress_cost: float
) -> tuple[list[dict[str, Any]], str]:
    matching = [row for row in rows if row["entry_brain_action"] == side]
    source = matching or list(rows)
    identity = "CURRENT_BRAIN" if matching else "UNFILTERED_SIDE_FALLBACK"
    return [
        {
            **row,
            "selected_profile": "CURRENT_TS",
            "selected_expected_net": row["trajectory_targets"]["current_ts_stress_net"]
            + stress_cost
            - 0.001,
            "selected_stress_net": row["trajectory_targets"]["current_ts_stress_net"],
            "selected_severe_net": row["trajectory_targets"]["current_ts_stress_net"]
            - 0.0005,
            "direction_probability": 0.0,
            "maximum_timing_risk": 0.0,
            "predicted_reward_risk": 0.0,
        }
        for row in source
    ], identity


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold: int,
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
    bundle = _fit(train, calibration, 20260822 + fold)
    if bundle is None or not test:
        return {
            "fold": fold,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    stress_cost = float(config["costs"]["stress_round_trip_fraction"])
    predicted_calibration = _predict(calibration, bundle, stress_cost)
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
    predicted_test = _predict(test, bundle, stress_cost)
    components = _component_metrics(
        test,
        predicted_test,
        bundle,
        int(config["timing"]["minimum_skilled_heads_per_fold"]),
    )
    mask = select_decomposed_cross_section(predicted_test, policy)
    selected = [row for row, keep in zip(predicted_test, mask) if keep]
    controls, control_identity = _control_rows(
        predicted_test, str(rows[0]["side"]), stress_cost
    )
    tail = float(config["trajectory"]["tail_quantile"])
    metrics = v9_selection_metrics(selected, tail_quantile=tail)
    control = v9_selection_metrics(controls, tail_quantile=tail)
    passed = bool(
        components["direction"]["passed"]
        and components["timing"]["passed"]
        and components["trajectory"]["passed"]
        and v9_fold_passes(
            metrics,
            control,
            minimum_count=int(validation["minimum_test_selections_per_fold"]),
            minimum_payoff=float(validation["require_payoff_ratio_at_least"]),
            maximum_p95_gap_hours=float(
                validation["maximum_p95_opportunity_gap_hours"]
            ),
        )
    )
    return {
        "fold": fold,
        "status": "EVALUATED",
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "calibration": len(calibration),
        "test": len(test),
        "policy": policy,
        "selected": metrics,
        "control_identity": control_identity,
        "control": control,
        "components": components,
        "stress_net_bootstrap": bootstrap_mean_interval(
            [row["selected_stress_net"] for row in selected],
            samples=int(validation["bootstrap_samples"]),
            seed=20260822 + fold,
        ),
        "passed": passed,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def evaluate_side(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    boundaries = _fold_boundaries(sorted({row["timestamp_value"] for row in rows}))
    folds = [
        _evaluate_fold(rows, value, index + 1, config)
        for index, value in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    direction_skilled = sum(
        bool(fold["components"]["direction"]["passed"]) for fold in evaluated
    )
    timing_skilled = sum(
        bool(fold["components"]["timing"]["passed"]) for fold in evaluated
    )
    trajectory_skilled = sum(
        bool(fold["components"]["trajectory"]["passed"]) for fold in evaluated
    )
    worst_non_negative = bool(
        evaluated
        and all(
            fold["selected"]["mean_stress_net"] is not None
            and fold["selected"]["mean_stress_net"] >= 0.0
            for fold in evaluated
        )
    )
    validation = _mapping(config["validation"], "validation")
    primary = bool(
        len(evaluated) == int(validation["folds"])
        and passing >= int(validation["minimum_positive_folds"])
        and direction_skilled >= int(config["direction"]["minimum_skilled_folds"])
        and timing_skilled >= int(config["timing"]["minimum_skilled_folds"])
        and trajectory_skilled >= int(config["trajectory"]["minimum_skilled_folds"])
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
        passing_symbols = sum(bool(value.get("passed")) for value in reports.values())
        loso = {
            "status": "EVALUATED",
            "symbols": reports,
            "passing_symbols": passing_symbols,
            "required_symbols": int(validation["minimum_symbols_without_regression"]),
            "passed": passing_symbols
            >= int(validation["minimum_symbols_without_regression"]),
        }
    passed = primary and bool(loso["passed"])
    return {
        "rows": len(rows),
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "direction_skilled_folds": direction_skilled,
        "timing_skilled_folds": timing_skilled,
        "trajectory_skilled_folds": trajectory_skilled,
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
    *,
    root: Path,
    config_path: Path,
    dataset: Path,
    manifest_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    manifest = _mapping(json.loads(manifest_path.read_text()), "manifest")
    if sha256_file(dataset) != str(manifest["dataset_sha256"]):
        raise ValueError("V9 dataset hash mismatch")
    sides = {
        side: evaluate_side(load_side(dataset, side), config)
        for side in ("LONG", "SHORT")
    }
    passed = all(value["validation_pass"] for value in sides.values())
    result = {
        "schema_id": "aegis-decomposed-entry-v9-validation-v1",
        "experiment_id": config["experiment_id"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "evidence_start": manifest["evidence_start"],
        "evidence_end": manifest["evidence_end"],
        "sides": sides,
        "validation_pass": passed,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW"
            if passed
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
    result["content_hash"] = Sha256HashProvider().digest_value(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_decomposed_entry_v9_research.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/decomposed_entry_v9/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/decomposed_entry_v9/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/decomposed_entry_v9/validation.json"),
    )
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
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "sides": {
                    side: {
                        "passing_folds": value["passing_folds"],
                        "direction_skilled_folds": value["direction_skilled_folds"],
                        "timing_skilled_folds": value["timing_skilled_folds"],
                        "trajectory_skilled_folds": value["trajectory_skilled_folds"],
                        "validation_pass": value["validation_pass"],
                    }
                    for side, value in result["sides"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
