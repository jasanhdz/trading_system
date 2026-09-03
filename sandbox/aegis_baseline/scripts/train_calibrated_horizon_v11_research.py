#!/usr/bin/env python3
"""Validate calibrated horizon specialists and clean-entry V11."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from aegis.research.calibrated_horizon_v11 import attributed_utility
from aegis.research.calibrated_horizon_v11_training import (
    attribution_means,
    multiclass_ece,
    select_v11_cross_section,
    shrink_group_probabilities,
)
from aegis.research.competing_barrier_v10 import (
    BarrierContract,
    BarrierOutcome,
)
from aegis.research.competing_barrier_v10_training import fold_passes, utility_metrics
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

DIRECTION_CLASSES = ("LONG", "SHORT", "ABSTAIN")
OUTCOME_CLASSES = tuple(value.value for value in BarrierOutcome)
CLEAN_CLASSES = ("CLEAN", "NOT_CLEAN")


def load_side(path: Path, side: str) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = dict(_mapping(json.loads(line), f"dataset:{line_number}"))
            if source.get("side") != side:
                continue
            features = tuple(float(value) for value in source["v9_features"])
            direction = tuple(float(value) for value in source["v9_direction_features"])
            if not all(math.isfinite(value) for value in (*features, *direction)):
                raise ValueError("V11 dataset contains non-finite features")
            rows.append(
                {
                    **source,
                    "timestamp_value": datetime.fromisoformat(str(source["timestamp"])),
                    "features": features,
                    "direction_features": direction,
                    "direction_label": str(source["v10_direction_label"]),
                    "clean_label": "CLEAN" if source["v11_clean_entry_label"] else "NOT_CLEAN",
                    "regime": str(source["v11_causal_regime"]),
                    "outcomes": _mapping(source["v10_contract_outcomes"], "outcomes"),
                }
            )
    if not rows:
        raise ValueError(f"V11 dataset has no {side} rows")
    return rows


def _contracts(rows: Sequence[Mapping[str, Any]], severe_cost: float) -> tuple[BarrierContract, ...]:
    outcomes = rows[0]["outcomes"]
    contracts = tuple(
        BarrierContract(
            name=name,
            favorable_fraction=float(value["favorable_fraction"]),
            adverse_fraction=float(value["adverse_fraction"]),
            horizon_bars=int(value["horizon_bars"]),
            severe_cost_fraction=severe_cost,
        )
        for name, value in sorted(outcomes.items())
    )
    if len(contracts) != 9:
        raise ValueError("V11 requires nine V10 contracts")
    return contracts


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


def _x(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def _fit_calibrators(
    raw: np.ndarray,
    labels: np.ndarray,
    model_classes: Sequence[str],
    classes: Sequence[str],
    indices: np.ndarray,
    *,
    minimum_positive: int,
) -> Mapping[str, Any | None]:
    calibrators = {}
    for name in classes:
        target = np.asarray(labels[indices] == name, dtype=np.int8)
        if (
            name in model_classes
            and int(np.sum(target)) >= minimum_positive
            and int(len(target) - np.sum(target)) >= minimum_positive
        ):
            calibrators[name] = fit_platt_calibrator(
                raw[indices, model_classes.index(name)], target
            )
        else:
            calibrators[name] = None
    return calibrators


def _fit_multiclass(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    target: Callable[[Mapping[str, Any]], str],
    classes: Sequence[str],
    feature_key: str,
    config: Mapping[str, Any],
    seed: int,
    hierarchical: bool = True,
) -> Mapping[str, Any] | None:
    labels_train = np.asarray([target(row) for row in train])
    labels_calibration = np.asarray([target(row) for row in calibration])
    if len(set(labels_train)) < 2 or not set(labels_train).issubset(set(classes)):
        return None
    model = _classifier(seed)
    model.fit(_x(train, feature_key), labels_train)
    raw = model.predict_proba(_x(calibration, feature_key))
    model_classes = tuple(str(value) for value in model.classes_)
    all_indices = np.arange(len(calibration))
    global_calibrators = _fit_calibrators(
        raw,
        labels_calibration,
        model_classes,
        classes,
        all_indices,
        minimum_positive=1,
    )
    group_layers = {}
    calibration_config = config["calibration"]
    if hierarchical:
        definitions = {
            "SYMBOL": lambda row: str(row["symbol"]),
            "CAUSAL_REGIME": lambda row: str(row["regime"]),
        }
        for layer in calibration_config["hierarchical_groups"]:
            key = definitions[str(layer)]
            values = sorted({key(row) for row in calibration})
            for value in values:
                indices = np.asarray(
                    [index for index, row in enumerate(calibration) if key(row) == value]
                )
                if len(indices) < int(calibration_config["minimum_group_rows"]):
                    continue
                calibrators = _fit_calibrators(
                    raw,
                    labels_calibration,
                    model_classes,
                    classes,
                    indices,
                    minimum_positive=int(calibration_config["minimum_positive_class_rows"]),
                )
                if any(item is not None for item in calibrators.values()):
                    group_layers[f"{layer}::{value}"] = {
                        "calibrators": calibrators,
                        "rows": len(indices),
                    }
    denominator = len(labels_train) + len(classes)
    return {
        "model": model,
        "classes": tuple(classes),
        "model_classes": model_classes,
        "global_calibrators": global_calibrators,
        "group_layers": group_layers,
        "priors": {
            name: (float(np.sum(labels_train == name)) + 1.0) / denominator
            for name in classes
        },
        "feature_key": feature_key,
    }


def _normalized_probabilities(
    raw_row: np.ndarray,
    bundle: Mapping[str, Any],
    calibrators: Mapping[str, Any | None],
    fallback: Mapping[str, float] | None = None,
) -> Mapping[str, float]:
    values = {}
    model_classes = tuple(bundle["model_classes"])
    for name in bundle["classes"]:
        if calibrators[name] is None and fallback is not None:
            value = float(fallback[name])
        elif name not in model_classes:
            value = float(bundle["priors"][name])
        else:
            value = float(raw_row[model_classes.index(name)])
            calibrator = calibrators[name]
            if calibrator is not None:
                value = float(calibrator.apply(value))
        values[name] = max(1e-9, value)
    denominator = sum(values.values())
    return {name: value / denominator for name, value in values.items()}


def _probabilities(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[Mapping[str, float]]:
    raw = bundle["model"].predict_proba(_x(rows, str(bundle["feature_key"])))
    result = []
    shrinkage = int(config["calibration"]["shrinkage_rows"])
    for index, row in enumerate(rows):
        global_values = _normalized_probabilities(
            raw[index], bundle, bundle["global_calibrators"]
        )
        groups = []
        for identity in (
            f"SYMBOL::{row['symbol']}",
            f"CAUSAL_REGIME::{row['regime']}",
        ):
            group = bundle["group_layers"].get(identity)
            if group is not None:
                groups.append(
                    (
                        _normalized_probabilities(
                            raw[index],
                            bundle,
                            group["calibrators"],
                            fallback=global_values,
                        ),
                        int(group["rows"]),
                    )
                )
        result.append(
            shrink_group_probabilities(
                global_values, groups, shrinkage_rows=shrinkage
            )
        )
    return result


def _expanded(
    rows: Sequence[Mapping[str, Any]], contracts: Sequence[BarrierContract], horizon: int
) -> list[dict[str, Any]]:
    selected = [contract for contract in contracts if contract.horizon_bars == horizon]
    result = []
    for row in rows:
        for contract in selected:
            result.append(
                {
                    **row,
                    "features_with_barrier": (
                        *row["features"],
                        contract.favorable_fraction,
                        contract.favorable_fraction * 15.0,
                    ),
                    "contract_name": contract.name,
                    "outcome_label": str(row["outcomes"][contract.name]["outcome"]),
                }
            )
    return result


def _fit_bundle(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    if len(train) < 500 or len(calibration) < 200:
        return None
    severe = float(config["utility"]["severe_round_trip_fraction"])
    contracts = _contracts(train, severe)
    direction = _fit_multiclass(
        train,
        calibration,
        target=lambda row: str(row["direction_label"]),
        classes=DIRECTION_CLASSES,
        feature_key="direction_features",
        config=config,
        seed=seed,
    )
    clean = _fit_multiclass(
        train,
        calibration,
        target=lambda row: str(row["clean_label"]),
        classes=CLEAN_CLASSES,
        feature_key="features",
        config=config,
        seed=seed + 1,
    )
    if direction is None or clean is None:
        return None
    horizons = {}
    for offset, horizon in enumerate(config["models"]["horizon_specialists"]["horizons_bars"], start=10):
        expanded_train = _expanded(train, contracts, int(horizon))
        expanded_calibration = _expanded(calibration, contracts, int(horizon))
        fitted = _fit_multiclass(
            expanded_train,
            expanded_calibration,
            target=lambda row: str(row["outcome_label"]),
            classes=OUTCOME_CLASSES,
            feature_key="features_with_barrier",
            config=config,
            seed=seed + offset,
        )
        if fitted is None:
            return None
        horizons[int(horizon)] = fitted
    baselines = {}
    for offset, contract in enumerate(contracts, start=30):
        fitted = _fit_multiclass(
            train,
            calibration,
            target=lambda row, name=contract.name: str(row["outcomes"][name]["outcome"]),
            classes=OUTCOME_CLASSES,
            feature_key="features",
            config=config,
            seed=seed + offset,
            hierarchical=False,
        )
        if fitted is None:
            return None
        baselines[contract.name] = fitted
    return {
        "contracts": contracts,
        "direction": direction,
        "clean": clean,
        "horizons": horizons,
        "baselines": baselines,
    }


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not rows:
        return []
    direction = _probabilities(rows, bundle["direction"], config)
    clean = _probabilities(rows, bundle["clean"], config)
    contract_probabilities: dict[str, list[Mapping[str, float]]] = {}
    for horizon, horizon_bundle in bundle["horizons"].items():
        expanded = _expanded(rows, bundle["contracts"], int(horizon))
        expanded_probabilities = _probabilities(expanded, horizon_bundle, config)
        horizon_contracts = [
            contract for contract in bundle["contracts"] if contract.horizon_bars == horizon
        ]
        for contract_index, contract in enumerate(horizon_contracts):
            contract_probabilities[contract.name] = [
                expanded_probabilities[row_index * len(horizon_contracts) + contract_index]
                for row_index in range(len(rows))
            ]
    unknown_penalty = float(config["utility"]["unknown_penalty_fraction_of_adverse"])
    clean_bonus = float(config["utility"]["clean_probability_bonus_fraction_of_favorable"])
    result = []
    for index, row in enumerate(rows):
        choices = []
        for contract in bundle["contracts"]:
            probabilities = contract_probabilities[contract.name][index]
            attribution = attributed_utility(
                probabilities,
                contract,
                clean_probability=clean[index]["CLEAN"],
                unknown_penalty_fraction=unknown_penalty,
                clean_bonus_fraction=clean_bonus,
            )
            unknown = (
                probabilities[BarrierOutcome.SAME_BAR_AMBIGUOUS.value]
                + probabilities[BarrierOutcome.NEITHER_REACHED.value]
            )
            choices.append(
                (
                    attribution["total_utility"],
                    -unknown,
                    contract.name,
                    probabilities,
                    attribution,
                )
            )
        utility, negative_unknown, contract_name, probabilities, attribution = max(choices)
        actual = row["outcomes"][contract_name]
        result.append(
            {
                **row,
                "direction_probabilities": direction[index],
                "direction_probability": direction[index][str(row["side"])],
                "clean_probability": clean[index]["CLEAN"],
                "selected_contract": contract_name,
                "outcome_probabilities": probabilities,
                "unknown_probability": -negative_unknown,
                "utility_attribution": attribution,
                "predicted_utility": utility,
                "actual_utility": float(actual["realized_utility"]),
                "actual_outcome": str(actual["outcome"]),
            }
        )
    return result


def _skill(
    probabilities: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [math.log(max(epsilon, values[label])) for values, label in zip(probabilities, labels)]
        )
    )
    prior_loss = -float(
        np.mean([math.log(max(epsilon, bundle["priors"][label])) for label in labels])
    )
    predictions = [max(values, key=values.get) for values in probabilities]
    accuracy = float(np.mean([left == right for left, right in zip(predictions, labels)]))
    majority = max(bundle["priors"], key=bundle["priors"].get)
    majority_accuracy = float(np.mean([label == majority for label in labels]))
    ece = multiclass_ece(
        probabilities,
        labels,
        bins=int(config["calibration"]["expected_calibration_error_bins"]),
    )
    return {
        "count": len(labels),
        "log_loss": log_loss,
        "training_prior_log_loss": prior_loss,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "ece": ece,
        "supported_calibration_groups": len(bundle["group_layers"]),
        "passed": bool(
            log_loss < prior_loss
            and accuracy >= majority_accuracy
            and ece <= float(config["calibration"]["maximum_multiclass_ece"])
        ),
    }


def _components(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    direction_probabilities = _probabilities(rows, bundle["direction"], config)
    direction = _skill(
        direction_probabilities,
        [str(row["direction_label"]) for row in rows],
        bundle["direction"],
        config,
    )
    clean_probabilities = _probabilities(rows, bundle["clean"], config)
    clean = _skill(
        clean_probabilities,
        [str(row["clean_label"]) for row in rows],
        bundle["clean"],
        config,
    )
    horizons = {}
    for horizon, specialist in bundle["horizons"].items():
        expanded = _expanded(rows, bundle["contracts"], int(horizon))
        probabilities = _probabilities(expanded, specialist, config)
        labels = [str(row["outcome_label"]) for row in expanded]
        skill = dict(_skill(probabilities, labels, specialist, config))
        baseline_losses = []
        for contract in [item for item in bundle["contracts"] if item.horizon_bars == horizon]:
            baseline_probabilities = _probabilities(rows, bundle["baselines"][contract.name], config)
            baseline_labels = [str(row["outcomes"][contract.name]["outcome"]) for row in rows]
            baseline_losses.append(
                _skill(
                    baseline_probabilities,
                    baseline_labels,
                    bundle["baselines"][contract.name],
                    config,
                )["log_loss"]
            )
        skill["v10_control_mean_log_loss"] = float(np.mean(baseline_losses))
        skill["not_worse_than_v10_control"] = bool(
            skill["log_loss"] <= skill["v10_control_mean_log_loss"]
        )
        skill["passed"] = bool(skill["passed"] and skill["not_worse_than_v10_control"])
        horizons[str(horizon)] = skill
    skilled = sum(bool(value["passed"]) for value in horizons.values())
    return {
        "direction": direction,
        "clean_entry": clean,
        "horizon_specialists": {
            "horizons": horizons,
            "skilled": skilled,
            "required": int(config["validation"]["minimum_horizon_specialist_skilled_per_fold"]),
            "passed": skilled
            >= int(config["validation"]["minimum_horizon_specialist_skilled_per_fold"]),
        },
    }


def _split(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    calibration_times = sorted(
        {
            row["timestamp_value"]
            for row in rows
            if train_end + embargo < row["timestamp_value"] <= calibration_end
        }
    )
    if not calibration_times:
        return [], [], [], []
    fraction = float(config["validation"]["calibration_fraction_for_probability_fit"])
    split_time = calibration_times[min(len(calibration_times) - 1, int(len(calibration_times) * fraction))]
    train = [
        row
        for row in rows
        if row["timestamp_value"] <= train_end
        and (excluded_train_symbol is None or row["symbol"] != excluded_train_symbol)
    ]
    probability_calibration = [
        row
        for row in rows
        if train_end + embargo < row["timestamp_value"] <= split_time
        and (excluded_train_symbol is None or row["symbol"] != excluded_train_symbol)
    ]
    policy = [
        row
        for row in rows
        if split_time + embargo < row["timestamp_value"] <= calibration_end
        and (excluded_train_symbol is None or row["symbol"] != excluded_train_symbol)
    ]
    test = [
        row
        for row in rows
        if calibration_end + embargo < row["timestamp_value"] <= test_end
        and (test_symbol is None or row["symbol"] == test_symbol)
    ]
    return train, probability_calibration, policy, test


def _candidate_policies(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selection = config["selection"]
    arrays = {
        "utility": np.asarray([max(0.0, float(row["predicted_utility"])) for row in rows]),
        "direction": np.asarray([float(row["direction_probability"]) for row in rows]),
        "clean": np.asarray([float(row["clean_probability"]) for row in rows]),
        "unknown": np.asarray([float(row["unknown_probability"]) for row in rows]),
    }
    return [
        {
            "minimum_utility": max(0.0, float(np.quantile(arrays["utility"], uq))),
            "minimum_direction_probability": float(np.quantile(arrays["direction"], dq)),
            "minimum_clean_probability": float(np.quantile(arrays["clean"], cq)),
            "maximum_unknown_probability": float(np.quantile(arrays["unknown"], nq)),
            "maximum_selected_per_timestamp": int(selection["maximum_selected_per_timestamp"]),
            "source": "POLICY_WINDOW_ONLY",
            "quantiles": {"utility": uq, "direction": dq, "clean": cq, "unknown": nq},
        }
        for uq in selection["utility_quantiles"]
        for dq in selection["direction_probability_quantiles"]
        for cq in selection["clean_probability_quantiles"]
        for nq in selection["maximum_unknown_probability_quantiles"]
    ]


def _derive_policy(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    minimum = int(config["selection"]["minimum_policy_selections"])
    choices = []
    for policy in _candidate_policies(rows, config):
        mask = select_v11_cross_section(rows, policy)
        selected = [row for row, keep in zip(rows, mask) if keep]
        if len(selected) >= minimum:
            choices.append((policy, utility_metrics(selected)))
    if not choices:
        values = [float(row["predicted_utility"]) for row in rows]
        return {
            "eligible": False,
            "reason": "NO_POLICY_WITH_MINIMUM_SELECTIONS",
            "maximum_predicted_utility": max(values),
            "positive_predicted_utility_count": sum(value >= 0.0 for value in values),
            "minimum_required_selections": minimum,
            "policies_evaluated": 0,
        }
    policy, metrics = max(
        choices,
        key=lambda item: (item[1]["mean_utility"], item[1]["cvar"], item[1]["count"]),
    )
    return {**policy, "eligible": True, "policy_metrics": metrics, "policies_evaluated": len(choices)}


def _control(rows: Sequence[Mapping[str, Any]], primary: str) -> list[dict[str, Any]]:
    return [
        {**row, "actual_utility": float(row["outcomes"][primary]["realized_utility"])}
        for row in rows
    ]


def _protection_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {"count": 0, "mean_current_ts_stress": None, "mean_mae": None, "clean_rate": None}
    return {
        "count": len(rows),
        "mean_current_ts_stress": float(
            np.mean([float(row["v8_profile_cost_returns"]["CURRENT_TS"]["stress"]) for row in rows])
        ),
        "mean_mae": float(np.mean([float(row["mae_fraction"]) for row in rows])),
        "clean_rate": float(np.mean([row["clean_label"] == "CLEAN" for row in rows])),
        "entry_gate_influence": "NONE",
    }


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold: int,
    config: Mapping[str, Any],
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> Mapping[str, Any]:
    train, calibration, policy_rows, test = _split(
        rows,
        boundaries,
        config,
        excluded_train_symbol=excluded_train_symbol,
        test_symbol=test_symbol,
    )
    bundle = _fit_bundle(train, calibration, config, 20260824 + fold)
    if bundle is None or not policy_rows or not test:
        return {
            "fold": fold,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "probability_calibration": len(calibration),
            "policy": len(policy_rows),
            "test": len(test),
            "passed": False,
        }
    components = _components(test, bundle, config)
    predicted_policy = _predict(policy_rows, bundle, config)
    policy = _derive_policy(predicted_policy, config)
    predicted_test = _predict(test, bundle, config)
    primary = str(config["labels"]["clean_entry"]["source_contract"])
    control = utility_metrics(_control(test, primary))
    if not bool(policy["eligible"]):
        selected_rows: list[Mapping[str, Any]] = []
        selected = utility_metrics([])
        economic = False
        status = "NO_POSITIVE_POLICY"
    else:
        mask = select_v11_cross_section(predicted_test, policy)
        selected_rows = [row for row, keep in zip(predicted_test, mask) if keep]
        selected = utility_metrics(selected_rows)
        economic = fold_passes(
            selected,
            control,
            minimum_count=int(config["validation"]["minimum_test_selections_per_fold"]),
            minimum_payoff=float(config["validation"]["require_payoff_ratio_at_least"]),
            maximum_p95_gap_hours=float(config["validation"]["maximum_p95_opportunity_gap_hours"]),
        )
        status = "EVALUATED"
    passed = bool(
        components["direction"]["passed"]
        and components["clean_entry"]["passed"]
        and components["horizon_specialists"]["passed"]
        and economic
    )
    return {
        "fold": fold,
        "status": status,
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "probability_calibration": len(calibration),
        "policy_rows": len(policy_rows),
        "test": len(test),
        "policy": policy,
        "selected": selected,
        "control_identity": "UNFILTERED_PRIMARY_CONTRACT",
        "control": control,
        "components": components,
        "economic_gate": economic,
        "utility_attribution": attribution_means(selected_rows),
        "protection_diagnostics": _protection_diagnostics(selected_rows),
        "passed": passed,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def evaluate_side(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    boundaries = _fold_boundaries(sorted({row["timestamp_value"] for row in rows}))
    folds = [_evaluate_fold(rows, boundary, index + 1, config) for index, boundary in enumerate(boundaries)]
    evaluated = [fold for fold in folds if fold["status"] != "INSUFFICIENT_MODEL_DATA"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    direction = sum(bool(fold["components"]["direction"]["passed"]) for fold in evaluated)
    horizon = sum(bool(fold["components"]["horizon_specialists"]["passed"]) for fold in evaluated)
    clean = sum(bool(fold["components"]["clean_entry"]["passed"]) for fold in evaluated)
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
        and direction >= int(validation["minimum_direction_skilled_folds"])
        and horizon >= int(validation["minimum_horizon_specialist_skilled_folds"])
        and clean >= int(validation["minimum_clean_entry_skilled_folds"])
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
        passing_symbols = sum(bool(report.get("passed")) for report in reports.values())
        loso = {
            "status": "EVALUATED",
            "symbols": reports,
            "passing_symbols": passing_symbols,
            "required_symbols": int(validation["minimum_symbols_without_regression"]),
            "passed": passing_symbols >= int(validation["minimum_symbols_without_regression"]),
        }
    passed = primary and bool(loso["passed"])
    return {
        "rows": len(rows),
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "direction_skilled_folds": direction,
        "horizon_skilled_folds": horizon,
        "clean_entry_skilled_folds": clean,
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


def run(*, config_path: Path, dataset: Path, manifest_path: Path, output: Path) -> Mapping[str, Any]:
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    manifest = _mapping(json.loads(manifest_path.read_text()), "manifest")
    if sha256_file(dataset) != str(manifest["dataset_sha256"]):
        raise ValueError("V11 dataset hash mismatch")
    sides = {side: evaluate_side(load_side(dataset, side), config) for side in ("LONG", "SHORT")}
    passed = all(value["validation_pass"] for value in sides.values())
    result = {
        "schema_id": "aegis-calibrated-horizon-v11-validation-v1",
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
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_calibrated_horizon_v11_research.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/calibrated_horizon_v11/canonical_dataset.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/calibrated_horizon_v11/dataset_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/calibrated_horizon_v11/validation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = run(
        config_path=resolve(args.config),
        dataset=resolve(args.dataset),
        manifest_path=resolve(args.manifest),
        output=resolve(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
