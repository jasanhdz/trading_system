#!/usr/bin/env python3
"""Validate the preregistered joint direction/path V12 research policy."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aegis.research.calibrated_horizon_v11_training import multiclass_ece
from aegis.research.competing_barrier_v10 import BarrierContract, BarrierOutcome
from aegis.research.competing_barrier_v10_training import fold_passes, utility_metrics
from aegis.research.joint_path_v12 import (
    JointPathState,
    joint_path_state,
    joint_state_utility,
    path_quality_metrics,
    select_joint_cross_section,
)
from aegis.research.joint_path_v12_training import (
    assign_contracts_by_regime,
    assigned_contract_name,
)
from aegis.utils import Sha256HashProvider, sha256_file
from train_calibrated_horizon_v11_research import (
    _contracts,
    _fit_multiclass,
    _probabilities,
    load_side,
)
from train_long_entry_v21_shadow import _fold_boundaries, _mapping

JOINT_CLASSES = tuple(state.value for state in JointPathState)
REGIMES = (
    "RANGE_LOW_VOL",
    "RANGE_HIGH_VOL",
    "TREND_UP_LOW_VOL",
    "TREND_UP_HIGH_VOL",
    "TREND_DOWN_LOW_VOL",
    "TREND_DOWN_HIGH_VOL",
    "TRANSITION_LOW_VOL",
    "TRANSITION_HIGH_VOL",
)


def _contract_features(
    row: Mapping[str, Any], contract: BarrierContract
) -> tuple[float, ...]:
    regime = str(row["regime"])
    if regime not in REGIMES:
        raise ValueError("V12 encountered an unregistered regime")
    return (
        *tuple(float(value) for value in row["features"]),
        contract.favorable_fraction * 100.0,
        contract.horizon_bars / 24.0,
        *(1.0 if regime == name else 0.0 for name in REGIMES),
    )


def _expanded(
    rows: Sequence[Mapping[str, Any]], contracts: Sequence[BarrierContract]
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        for contract in contracts:
            outcome = str(row["outcomes"][contract.name]["outcome"])
            result.append(
                {
                    **row,
                    "contract_name": contract.name,
                    "contract_features": _contract_features(row, contract),
                    "joint_label": joint_path_state(
                        side=str(row["side"]),
                        direction_label=str(row["direction_label"]),
                        clean_entry=bool(row["v11_clean_entry_label"]),
                        outcome=outcome,
                    ),
                }
            )
    return result


def _fit_bundle(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    severe = float(config["utility"]["severe_round_trip_fraction"])
    contracts = _contracts(train, severe)
    expanded_train = _expanded(train, contracts)
    expanded_calibration = _expanded(calibration, contracts)
    joint = _fit_multiclass(
        expanded_train,
        expanded_calibration,
        target=lambda row: str(row["joint_label"]),
        classes=JOINT_CLASSES,
        feature_key="contract_features",
        config={
            "calibration": {
                "hierarchical_groups": [],
                "minimum_group_rows": 1,
                "minimum_positive_class_rows": 1,
                "shrinkage_rows": 1,
            }
        },
        seed=seed,
        hierarchical=False,
    )
    return None if joint is None else {"contracts": contracts, "joint": joint}


def _joint_probabilities(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, list[Mapping[str, float]]]:
    expanded = _expanded(rows, bundle["contracts"])
    probability_config = {
        "calibration": {
            "shrinkage_rows": 1,
            "expected_calibration_error_bins": int(
                config["calibration"]["expected_calibration_error_bins"]
            ),
        }
    }
    values = _probabilities(expanded, bundle["joint"], probability_config)
    width = len(bundle["contracts"])
    return {
        contract.name: [
            values[row_index * width + contract_index] for row_index in range(len(rows))
        ]
        for contract_index, contract in enumerate(bundle["contracts"])
    }


def _predict_assigned(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    assignment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    probabilities = _joint_probabilities(rows, bundle, config)
    utility_config = config["utility"]
    contracts = {contract.name: contract for contract in bundle["contracts"]}
    result = []
    for index, row in enumerate(rows):
        name = assigned_contract_name(row, assignment)
        contract = contracts[name]
        vector = probabilities[name][index]
        attribution = joint_state_utility(vector, contract, utility_config)
        actual = row["outcomes"][name]
        coherent = (
            vector[JointPathState.COHERENT_CLEAN_FAVORABLE.value]
            + vector[JointPathState.COHERENT_DIRTY_FAVORABLE.value]
        )
        result.append(
            {
                **row,
                "selected_contract": name,
                "joint_probabilities": vector,
                "coherent_probability": coherent,
                "adverse_probability": vector[JointPathState.ADVERSE_FIRST.value],
                "unknown_probability": (
                    vector[JointPathState.SAME_BAR_AMBIGUOUS.value]
                    + vector[JointPathState.UNRESOLVED_OR_DIRECTION_MISMATCH.value]
                ),
                "utility_attribution": attribution,
                "predicted_utility": attribution["total_utility"],
                "actual_utility": float(actual["realized_utility"]),
                "actual_outcome": str(actual["outcome"]),
            }
        )
    return result


def _joint_skill(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    expanded = _expanded(rows, bundle["contracts"])
    probabilities_by_contract = _joint_probabilities(rows, bundle, config)
    probabilities = [
        probabilities_by_contract[contract.name][index]
        for index in range(len(rows))
        for contract in bundle["contracts"]
    ]
    labels = [str(row["joint_label"]) for row in expanded]
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, values[label]))
                for values, label in zip(probabilities, labels)
            ]
        )
    )
    prior_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, bundle["joint"]["priors"][label]))
                for label in labels
            ]
        )
    )
    predictions = [max(values, key=values.get) for values in probabilities]
    accuracy = float(
        np.mean([left == right for left, right in zip(predictions, labels)])
    )
    majority = max(bundle["joint"]["priors"], key=bundle["joint"]["priors"].get)
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
        "passed": bool(
            log_loss < prior_loss
            and accuracy >= majority_accuracy
            and ece <= float(config["calibration"]["maximum_multiclass_ece"])
        ),
    }


def _split(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
    *,
    excluded_train_symbol: str | None = None,
    test_symbol: str | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]:
    train_end, policy_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    middle_times = sorted(
        {
            row["timestamp_value"]
            for row in rows
            if train_end + embargo < row["timestamp_value"] <= policy_end
        }
    )
    if len(middle_times) < 3:
        return [], [], [], [], []
    calibration_fraction = float(
        config["validation"]["probability_calibration_fraction"]
    )
    first_index = min(
        len(middle_times) - 2, int(len(middle_times) * calibration_fraction)
    )
    calibration_end = middle_times[first_index]
    policy_times = [
        value for value in middle_times if value > calibration_end + embargo
    ]
    if len(policy_times) < 2:
        return [], [], [], [], []
    assignment_fraction = float(
        config["validation"]["contract_assignment_fraction_of_policy"]
    )
    second_index = min(
        len(policy_times) - 1, int(len(policy_times) * assignment_fraction)
    )
    assignment_end = policy_times[second_index]
    allowed = (
        lambda row: excluded_train_symbol is None
        or row["symbol"] != excluded_train_symbol
    )
    train = [
        row for row in rows if row["timestamp_value"] <= train_end and allowed(row)
    ]
    calibration = [
        row
        for row in rows
        if train_end + embargo < row["timestamp_value"] <= calibration_end
        and allowed(row)
    ]
    assignment = [
        row
        for row in rows
        if calibration_end + embargo < row["timestamp_value"] <= assignment_end
        and allowed(row)
    ]
    policy = [
        row
        for row in rows
        if assignment_end + embargo < row["timestamp_value"] <= policy_end
        and allowed(row)
    ]
    test = [
        row
        for row in rows
        if policy_end + embargo < row["timestamp_value"] <= test_end
        and (test_symbol is None or row["symbol"] == test_symbol)
    ]
    return train, calibration, assignment, policy, test


def _candidate_policies(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    selection = config["selection"]
    arrays = {
        "utility": np.asarray(
            [max(0.0, float(row["predicted_utility"])) for row in rows]
        ),
        "coherent": np.asarray([float(row["coherent_probability"]) for row in rows]),
        "adverse": np.asarray([float(row["adverse_probability"]) for row in rows]),
        "unknown": np.asarray([float(row["unknown_probability"]) for row in rows]),
    }
    return [
        {
            "minimum_utility": max(0.0, float(np.quantile(arrays["utility"], uq))),
            "minimum_coherent_probability": float(np.quantile(arrays["coherent"], cq)),
            "maximum_adverse_probability": float(np.quantile(arrays["adverse"], aq)),
            "maximum_unknown_probability": float(np.quantile(arrays["unknown"], nq)),
            "maximum_selected_per_timestamp": int(
                selection["maximum_selected_per_timestamp"]
            ),
            "source": "THRESHOLD_POLICY_WINDOW_ONLY",
            "quantiles": {"utility": uq, "coherent": cq, "adverse": aq, "unknown": nq},
        }
        for uq in selection["utility_quantiles"]
        for cq in selection["coherent_probability_quantiles"]
        for aq in selection["adverse_probability_quantiles"]
        for nq in selection["unknown_probability_quantiles"]
    ]


def _derive_policy(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    minimum = int(config["selection"]["minimum_policy_selections"])
    choices = []
    for policy in _candidate_policies(rows, config):
        selected = [
            row
            for row, keep in zip(rows, select_joint_cross_section(rows, policy))
            if keep
        ]
        if len(selected) >= minimum:
            choices.append(
                (policy, utility_metrics(selected), path_quality_metrics(selected))
            )
    if not choices:
        values = [float(row["predicted_utility"]) for row in rows]
        return {
            "eligible": False,
            "reason": "NO_POLICY_WITH_MINIMUM_SELECTIONS",
            "maximum_predicted_utility": max(values),
            "positive_predicted_utility_count": sum(value >= 0.0 for value in values),
            "minimum_required_selections": minimum,
        }
    policy, metrics, quality = max(
        choices,
        key=lambda item: (
            item[1]["mean_utility"],
            item[1]["cvar"],
            item[2]["clean_rate"],
            item[1]["count"],
        ),
    )
    return {
        **policy,
        "eligible": True,
        "policy_metrics": metrics,
        "policy_path_quality": quality,
        "policies_evaluated": len(choices),
    }


def _control(rows: Sequence[Mapping[str, Any]], primary: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "actual_utility": float(row["outcomes"][primary]["realized_utility"]),
            "actual_outcome": str(row["outcomes"][primary]["outcome"]),
        }
        for row in rows
    ]


def _ranking_diagnostic(
    rows: Sequence[Mapping[str, Any]], *, count: int
) -> Mapping[str, Any]:
    """Describe top model rankings without granting selection authority."""

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["predicted_utility"]),
            -float(row["coherent_probability"]),
            float(row["adverse_probability"]),
            str(row["timestamp"]),
            str(row["symbol"]),
        ),
    )[:count]
    return {
        "role": "REPORT_ONLY_NOT_A_POLICY",
        "requested_count": count,
        "utility": utility_metrics(ranked),
        "path_quality": path_quality_metrics(ranked),
        "maximum_predicted_utility": (
            max(float(row["predicted_utility"]) for row in rows) if rows else None
        ),
        "positive_predicted_utility_count": sum(
            float(row["predicted_utility"]) >= 0.0 for row in rows
        ),
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
    train, calibration, assignment_rows, policy_rows, test = _split(
        rows,
        boundaries,
        config,
        excluded_train_symbol=excluded_train_symbol,
        test_symbol=test_symbol,
    )
    bundle = _fit_bundle(train, calibration, config, 20260830 + fold)
    if bundle is None or min(len(assignment_rows), len(policy_rows), len(test)) == 0:
        return {
            "fold": fold,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "calibration": len(calibration),
            "assignment": len(assignment_rows),
            "policy": len(policy_rows),
            "test": len(test),
            "passed": False,
        }
    assignment_config = config["contract_assignment"]
    assignment = assign_contracts_by_regime(
        assignment_rows,
        bundle["contracts"],
        minimum_group_rows=int(assignment_config["minimum_group_rows"]),
        minimum_contract_observations=int(
            assignment_config["minimum_contract_observations"]
        ),
    )
    policy_predictions = _predict_assigned(policy_rows, bundle, assignment, config)
    policy = _derive_policy(policy_predictions, config)
    test_predictions = _predict_assigned(test, bundle, assignment, config)
    primary = "ROE_10_H12"
    control_rows = _control(test, primary)
    control = utility_metrics(control_rows)
    control_quality = path_quality_metrics(control_rows)
    selected_rows: list[Mapping[str, Any]] = []
    if policy["eligible"]:
        selected_rows = [
            row
            for row, keep in zip(
                test_predictions, select_joint_cross_section(test_predictions, policy)
            )
            if keep
        ]
    selected = utility_metrics(selected_rows)
    selected_quality = path_quality_metrics(selected_rows)
    economic = bool(
        policy["eligible"]
        and fold_passes(
            selected,
            control,
            minimum_count=int(config["validation"]["minimum_test_selections_per_fold"]),
            minimum_payoff=float(config["validation"]["require_payoff_ratio_at_least"]),
            maximum_p95_gap_hours=float(
                config["validation"]["maximum_p95_opportunity_gap_hours"]
            ),
        )
    )
    quality = bool(
        selected_quality["count"]
        and float(selected_quality["clean_rate"])
        >= float(config["validation"]["minimum_clean_selected_rate"])
        and float(selected_quality["adverse_first_rate"])
        <= float(config["validation"]["maximum_adverse_first_rate"])
        and float(selected_quality["mean_mae_fraction"])
        <= float(control_quality["mean_mae_fraction"])
        - float(config["validation"]["minimum_mae_improvement_fraction"])
    )
    joint_skill = _joint_skill(test, bundle, config)
    return {
        "fold": fold,
        "status": "EVALUATED" if policy["eligible"] else "NO_POSITIVE_POLICY",
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "probability_calibration": len(calibration),
        "contract_assignment_rows": len(assignment_rows),
        "threshold_policy_rows": len(policy_rows),
        "test": len(test),
        "joint_state": joint_skill,
        "contract_assignment": assignment,
        "policy": policy,
        "selected": selected,
        "selected_path_quality": selected_quality,
        "control": control,
        "control_path_quality": control_quality,
        "top_30_ranking_diagnostic": _ranking_diagnostic(test_predictions, count=30),
        "economic_gate": economic,
        "path_quality_gate": quality,
        "passed": bool(joint_skill["passed"] and economic and quality),
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def evaluate_side(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    boundaries = _fold_boundaries(sorted({row["timestamp_value"] for row in rows}))
    folds = [
        _evaluate_fold(rows, boundary, index + 1, config)
        for index, boundary in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] != "INSUFFICIENT_MODEL_DATA"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    skilled = sum(bool(fold["joint_state"]["passed"]) for fold in evaluated)
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
        and skilled >= int(validation["minimum_joint_state_skilled_folds"])
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
            "passed": passing_symbols
            >= int(validation["minimum_symbols_without_regression"]),
        }
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "joint_state_skilled_folds": skilled,
        "worst_fold_non_negative": worst_non_negative,
        "primary_gate_passed": primary,
        "leave_one_symbol_out": loso,
        "passed": bool(primary and loso["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_joint_path_v12_research.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/joint_path_v12/validation.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    config_path = resolve(args.config)
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    authority = _mapping(config["authority"], "authority")
    for key, hash_key in (
        ("source_dataset", "source_dataset_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("source_validation", "source_validation_sha256"),
        ("source_config", "source_config_sha256"),
    ):
        path = root / str(authority[key])
        if sha256_file(path) != str(authority[hash_key]):
            raise ValueError(f"V12 authority hash mismatch: {path.name}")
    dataset = root / str(authority["source_dataset"])
    sides = {
        side: evaluate_side(load_side(dataset, side), config)
        for side in ("LONG", "SHORT")
    }
    result = {
        "schema_id": "aegis-joint-path-v12-validation-v1",
        "experiment_id": config["experiment_id"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path.resolve()),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "sides": sides,
        "verdict": (
            "PROMOTABLE_TO_SEPARATELY_AUTHORIZED_SHADOW"
            if all(report["passed"] for report in sides.values())
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "model_exported": False,
        "runtime_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
