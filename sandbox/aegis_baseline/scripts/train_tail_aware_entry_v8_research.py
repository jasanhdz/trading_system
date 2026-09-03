#!/usr/bin/env python3
"""Walk-forward validation for the preregistered V8 tail-aware experiment."""

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
from aegis.research.regime_aware_directional_v6_training import bootstrap_mean_interval
from aegis.research.tail_aware_entry_v8 import FORWARD_REGIMES
from aegis.research.tail_aware_entry_v8_training import (
    binary_skill_metrics,
    fold_passes,
    select_tail_aware_cross_section,
    tail_aware_quality_score,
    tail_selection_metrics,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

CLASSIFIER_TARGETS = (
    "clean_entry",
    "late_entry",
    "catastrophic_stress_loss",
    "positive_stress_net",
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
            labels = _mapping(source["v8_tail_labels"], "v8_tail_labels")
            costs = _mapping(source["v8_profile_cost_returns"], "profile_costs")
            profile_returns = {
                str(name): {
                    key: float(_mapping(value, str(name))[key])
                    for key in ("expected", "stress", "severe")
                }
                for name, value in costs.items()
            }
            first_positive = source.get("first_positive_after_cost_bar")
            features = tuple(float(value) for value in source["v8_features"])
            router_features = tuple(
                float(value) for value in source["regime_router_features"]
            )
            if (
                not features
                or not router_features
                or not all(
                    math.isfinite(value) for value in (*features, *router_features)
                )
            ):
                raise ValueError("V8 dataset contains invalid model features")
            rows.append(
                {
                    **source,
                    "timestamp_value": datetime.fromisoformat(str(source["timestamp"])),
                    "features": features,
                    "router_features": router_features,
                    "forward_regime": str(
                        source["forward_regime_multihorizon"]["label"]
                    ),
                    "profile_returns": profile_returns,
                    "time_to_positive": (
                        min(1.0, int(first_positive) / 24.0)
                        if first_positive is not None
                        else 1.0
                    ),
                    **{name: bool(labels[name]) for name in CLASSIFIER_TARGETS},
                }
            )
    if not rows:
        raise ValueError(f"V8 dataset has no {side} rows")
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
    seed: int, *, quantile: float | None = None
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


def _unique_timestamps(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["timestamp"]), row)
    return list(unique.values())


def _fit_router(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    train = _unique_timestamps(train)
    calibration = _unique_timestamps(calibration)
    y_train = np.asarray([row["forward_regime"] for row in train])
    y_cal = np.asarray([row["forward_regime"] for row in calibration])
    if set(y_train) != set(FORWARD_REGIMES) or set(y_cal) != set(FORWARD_REGIMES):
        return None
    x_train = np.asarray([row["router_features"] for row in train], dtype=np.float32)
    x_cal = np.asarray(
        [row["router_features"] for row in calibration], dtype=np.float32
    )
    model = _classifier(seed)
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_cal)
    model_classes = tuple(str(value) for value in model.classes_)
    calibrators = {
        name: fit_platt_calibrator(
            raw[:, model_classes.index(name)], np.asarray(y_cal == name, dtype=np.int8)
        )
        for name in FORWARD_REGIMES
    }
    return {
        "model": model,
        "model_classes": model_classes,
        "calibrators": calibrators,
        "priors": {name: float(np.mean(y_train == name)) for name in FORWARD_REGIMES},
    }


def _router_probabilities(
    rows: Sequence[Mapping[str, Any]], router: Mapping[str, Any]
) -> list[Mapping[str, float]]:
    unique = _unique_timestamps(rows)
    x_values = np.asarray([row["router_features"] for row in unique], dtype=np.float32)
    raw = router["model"].predict_proba(x_values)
    model_classes = tuple(router["model_classes"])
    by_timestamp = {}
    for index, row in enumerate(unique):
        values = {
            name: float(
                router["calibrators"][name].apply(
                    float(raw[index, model_classes.index(name)])
                )
            )
            for name in FORWARD_REGIMES
        }
        denominator = sum(values.values())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("V8 router produced invalid probabilities")
        by_timestamp[str(row["timestamp"])] = {
            name: value / denominator for name, value in values.items()
        }
    return [by_timestamp[str(row["timestamp"])] for row in rows]


def _router_metrics(
    rows: Sequence[Mapping[str, Any]], router: Mapping[str, Any]
) -> Mapping[str, Any]:
    unique = _unique_timestamps(rows)
    probabilities = _router_probabilities(unique, router)
    labels = [str(row["forward_regime"]) for row in unique]
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, values[label]))
                for values, label in zip(probabilities, labels)
            ]
        )
    )
    priors = router["priors"]
    prior_loss = -float(
        np.mean([math.log(max(epsilon, priors[label])) for label in labels])
    )
    predictions = [max(values, key=values.get) for values in probabilities]
    accuracy = float(
        np.mean([predicted == label for predicted, label in zip(predictions, labels)])
    )
    majority = max(FORWARD_REGIMES, key=lambda name: priors[name])
    majority_accuracy = float(np.mean([label == majority for label in labels]))
    return {
        "timestamps": len(unique),
        "log_loss": log_loss,
        "training_prior_log_loss": prior_loss,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "passed": log_loss < prior_loss and accuracy >= majority_accuracy,
    }


def _fit(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any] | None:
    if len(train) < 500 or len(calibration) < 200:
        return None
    x_train, x_cal = _x(train), _x(calibration)
    classifiers: dict[str, tuple[Any, CalibratorSpec]] = {}
    for offset, target in enumerate(CLASSIFIER_TARGETS):
        y_train = np.asarray([row[target] for row in train], dtype=np.int8)
        y_cal = np.asarray([row[target] for row in calibration], dtype=np.int8)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_cal)) < 2:
            return None
        model = _classifier(seed + offset)
        model.fit(x_train, y_train)
        calibrator = fit_platt_calibrator(model.predict_proba(x_cal)[:, 1], y_cal)
        classifiers[target] = (model, calibrator)
    regressors: dict[str, Any] = {}
    targets = {"mae_q90": "mae_fraction", "time_to_positive": "time_to_positive"}
    for offset, (name, target) in enumerate(targets.items(), start=10):
        model = _regressor(seed + offset, quantile=0.90 if name == "mae_q90" else None)
        model.fit(x_train, np.asarray([float(row[target]) for row in train]))
        regressors[name] = model
    profile_names = tuple(sorted(train[0]["profile_returns"]))
    for offset, profile in enumerate(profile_names, start=20):
        model = _regressor(seed + offset)
        model.fit(
            x_train,
            np.asarray([row["profile_returns"][profile]["stress"] for row in train]),
        )
        regressors[f"profile::{profile}"] = model
    router = _fit_router(train, calibration, seed + 1000)
    if router is None:
        return None
    return {
        "classifiers": classifiers,
        "regressors": regressors,
        "profiles": profile_names,
        "router": router,
    }


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = _x(rows)
    probabilities = {
        target: np.asarray(
            [
                calibrator.apply(float(value))
                for value in model.predict_proba(features)[:, 1]
            ]
        )
        for target, (model, calibrator) in bundle["classifiers"].items()
    }
    regressions = {
        name: np.asarray(model.predict(features), dtype=np.float64)
        for name, model in bundle["regressors"].items()
    }
    router = _router_probabilities(rows, bundle["router"])
    result = []
    for index, row in enumerate(rows):
        profile_predictions = {
            profile: float(regressions[f"profile::{profile}"][index])
            for profile in bundle["profiles"]
        }
        selected_profile = max(profile_predictions, key=profile_predictions.get)
        mae = max(0.0, float(regressions["mae_q90"][index]))
        timing = min(1.0, max(0.0, float(regressions["time_to_positive"][index])))
        clean = float(probabilities["clean_entry"][index])
        positive = float(probabilities["positive_stress_net"][index])
        late = float(probabilities["late_entry"][index])
        catastrophic = float(probabilities["catastrophic_stress_loss"][index])
        actual = row["profile_returns"][selected_profile]
        result.append(
            {
                **row,
                "clean_probability": clean,
                "positive_probability": positive,
                "late_probability": late,
                "catastrophic_probability": catastrophic,
                "expected_stress_net": profile_predictions[selected_profile],
                "mae_q90": mae,
                "predicted_time_to_positive": timing,
                "selected_profile": selected_profile,
                "selected_expected_net": actual["expected"],
                "selected_stress_net": actual["stress"],
                "selected_severe_net": actual["severe"],
                "router_probabilities": router[index],
                "v8_quality_score": tail_aware_quality_score(
                    clean_probability=clean,
                    positive_probability=positive,
                    late_probability=late,
                    catastrophic_probability=catastrophic,
                    expected_stress_net=profile_predictions[selected_profile],
                    mae_q90=mae,
                    time_to_positive=timing,
                ),
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


def _candidate_policies(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = _mapping(config["validation"], "validation")
    arrays = {
        "score": np.asarray([row["v8_quality_score"] for row in calibration]),
        "late": np.asarray([row["late_probability"] for row in calibration]),
        "catastrophic": np.asarray(
            [row["catastrophic_probability"] for row in calibration]
        ),
        "mae": np.asarray([row["mae_q90"] for row in calibration]),
    }
    return [
        {
            "minimum_score": float(np.quantile(arrays["score"], score_q)),
            "maximum_late_probability": float(np.quantile(arrays["late"], late_q)),
            "maximum_catastrophic_probability": float(
                np.quantile(arrays["catastrophic"], cat_q)
            ),
            "maximum_mae_q90": float(np.quantile(arrays["mae"], mae_q)),
            "maximum_selected_per_timestamp": int(
                validation["maximum_selected_per_timestamp"]
            ),
            "source": "CALIBRATION_ONLY",
            "quantiles": {
                "score": score_q,
                "late": late_q,
                "catastrophic": cat_q,
                "mae": mae_q,
            },
        }
        for score_q in validation["score_quantiles"]
        for late_q in validation["maximum_late_probability_quantiles"]
        for cat_q in validation["maximum_catastrophic_probability_quantiles"]
        for mae_q in validation["maximum_mae_quantiles"]
    ]


def _derive_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    minimum = int(config["validation"]["minimum_calibration_selections"])
    choices = []
    for policy in _candidate_policies(calibration, config):
        mask = select_tail_aware_cross_section(calibration, policy)
        selected = [row for row, keep in zip(calibration, mask) if keep]
        if len(selected) < minimum:
            continue
        metrics = tail_selection_metrics(
            selected, tail_quantile=float(config["trajectory"]["tail_quantile"])
        )
        choices.append((policy, metrics))
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
    rows: Sequence[Mapping[str, Any]], side: str
) -> tuple[list[dict[str, Any]], str]:
    matching = [row for row in rows if row["entry_brain_action"] == side]
    source = matching or list(rows)
    identity = "CURRENT_BRAIN" if matching else "UNFILTERED_SIDE_FALLBACK"
    return [
        {
            **row,
            "selected_profile": "CURRENT_TS",
            "selected_expected_net": row["profile_returns"]["CURRENT_TS"]["expected"],
            "selected_stress_net": row["profile_returns"]["CURRENT_TS"]["stress"],
            "selected_severe_net": row["profile_returns"]["CURRENT_TS"]["severe"],
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
    bundle = _fit(train, calibration, 20260821 + fold)
    if bundle is None or not test:
        return {
            "fold": fold,
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
            "fold": fold,
            "status": "NO_CALIBRATION_POLICY",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted_test = _predict(test, bundle)
    mask = select_tail_aware_cross_section(predicted_test, policy)
    selected = [row for row, keep in zip(predicted_test, mask) if keep]
    controls, control_identity = _control_rows(predicted_test, str(rows[0]["side"]))
    tail_quantile = float(config["trajectory"]["tail_quantile"])
    metrics = tail_selection_metrics(selected, tail_quantile=tail_quantile)
    control = tail_selection_metrics(controls, tail_quantile=tail_quantile)
    router = _router_metrics(test, bundle["router"])
    late_skill = binary_skill_metrics(
        [row["late_probability"] for row in predicted_test],
        [row["late_entry"] for row in predicted_test],
    )
    passed = bool(
        router["passed"]
        and late_skill["passed"]
        and fold_passes(
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
        "regime_router": router,
        "late_entry_detector": late_skill,
        "stress_net_bootstrap": bootstrap_mean_interval(
            [row["selected_stress_net"] for row in selected],
            samples=int(validation["bootstrap_samples"]),
            seed=20260821 + fold,
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
    evaluated = [value for value in folds if value["status"] == "EVALUATED"]
    passing = sum(bool(value["passed"]) for value in evaluated)
    router_skilled = sum(bool(value["regime_router"]["passed"]) for value in evaluated)
    late_skilled = sum(
        bool(value["late_entry_detector"]["passed"]) for value in evaluated
    )
    worst_non_negative = bool(
        evaluated
        and all(
            value["selected"]["mean_stress_net"] is not None
            and value["selected"]["mean_stress_net"] >= 0.0
            for value in evaluated
        )
    )
    validation = _mapping(config["validation"], "validation")
    primary = bool(
        len(evaluated) == int(validation["folds"])
        and passing >= int(validation["minimum_positive_folds"])
        and router_skilled >= int(config["forward_regime"]["minimum_skilled_folds"])
        and late_skilled >= int(config["late_entry_detector"]["minimum_skilled_folds"])
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
        "router_skilled_folds": router_skilled,
        "late_detector_skilled_folds": late_skilled,
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
        raise ValueError("V8 dataset hash mismatch")
    sides = {
        side: evaluate_side(load_side(dataset, side), config)
        for side in ("LONG", "SHORT")
    }
    passed = all(value["validation_pass"] for value in sides.values())
    result = {
        "schema_id": "aegis-tail-aware-entry-v8-validation-v1",
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
        default=Path("config/experiments/aegis_tail_aware_entry_v8_research.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tail_aware_entry_v8/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/tail_aware_entry_v8/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/tail_aware_entry_v8/validation.json")
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
                        "router_skilled_folds": value["router_skilled_folds"],
                        "late_detector_skilled_folds": value[
                            "late_detector_skilled_folds"
                        ],
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
