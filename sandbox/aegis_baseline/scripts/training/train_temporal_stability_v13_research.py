#!/usr/bin/env python3
"""Validate preregistered temporal-consensus V13 entry research."""

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
from sklearn.ensemble import HistGradientBoostingRegressor

from aegis.research.calibrated_horizon_v11_training import multiclass_ece
from aegis.research.competing_barrier_v10 import BarrierContract
from aegis.research.competing_barrier_v10_training import fold_passes, utility_metrics
from aegis.research.joint_path_v12 import (
    JointPathState,
    joint_state_utility,
    path_quality_metrics,
)
from aegis.research.joint_path_v12_training import (
    assign_contracts_by_regime,
    assigned_contract_name,
)
from aegis.research.temporal_stability_v13 import (
    consensus_probabilities,
    conservative_mae,
    distribution_scores,
    fit_robust_distribution,
    select_temporal_cross_section,
)
from aegis.utils import Sha256HashProvider, sha256_file
from train_calibrated_horizon_v11_research import (
    _contracts,
    _fit_multiclass,
    _probabilities,
    load_side,
)
from train_joint_path_v12_research import (
    JOINT_CLASSES,
    _control,
    _expanded,
    _ranking_diagnostic,
    _split,
)
from train_long_entry_v21_shadow import _fold_boundaries, _mapping


def _recent_rows(
    rows: Sequence[Mapping[str, Any]], *, days: int, minimum_rows: int
) -> list[Mapping[str, Any]]:
    if not rows or days <= 0 or minimum_rows <= 0:
        return []
    cutoff = max(row["timestamp_value"] for row in rows) - timedelta(days=days)
    recent = [row for row in rows if row["timestamp_value"] >= cutoff]
    if len(recent) >= minimum_rows:
        return recent
    return list(rows[-minimum_rows:]) if len(rows) >= minimum_rows else []


def _fit_joint(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    contracts: Sequence[BarrierContract],
    *,
    seed: int,
) -> Mapping[str, Any] | None:
    return _fit_multiclass(
        _expanded(train, contracts),
        _expanded(calibration, contracts),
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


def _mae_model(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, seed: int
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(config["mae_risk"]["quantile"]),
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=4.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        np.asarray([row["features"] for row in rows], dtype=np.float32),
        np.asarray([float(row["mae_fraction"]) for row in rows], dtype=np.float64),
    )
    return model


def _fit_bundle(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    recent_config = config["models"]["recent"]
    recent = _recent_rows(
        train,
        days=int(recent_config["training_window_days"]),
        minimum_rows=int(recent_config["minimum_training_rows"]),
    )
    if len(train) < 500 or len(calibration) < 200 or not recent:
        return None
    contracts = _contracts(
        train, float(config["utility"]["severe_round_trip_fraction"])
    )
    historical = _fit_joint(train, calibration, contracts, seed=seed)
    recent_joint = _fit_joint(recent, calibration, contracts, seed=seed + 1)
    if historical is None or recent_joint is None:
        return None
    experts = {}
    expert_config = config["models"]["regime_expert"]
    if bool(expert_config["enabled"]):
        regimes = sorted({str(row["regime"]) for row in train})
        for offset, regime in enumerate(regimes, start=10):
            regime_train = [row for row in train if row["regime"] == regime]
            regime_calibration = [row for row in calibration if row["regime"] == regime]
            if len(regime_train) >= int(expert_config["minimum_training_rows"]) and len(
                regime_calibration
            ) >= int(expert_config["minimum_calibration_rows"]):
                fitted = _fit_joint(
                    regime_train,
                    regime_calibration,
                    contracts,
                    seed=seed + offset,
                )
                if fitted is not None:
                    experts[regime] = fitted
    distribution = fit_robust_distribution(
        [row["features"] for row in train],
        minimum_scale=float(config["distribution_gate"]["minimum_scale"]),
    )
    calibration_scores = distribution_scores(
        [row["features"] for row in calibration], distribution
    )
    distribution_threshold = float(
        np.quantile(
            calibration_scores,
            float(config["distribution_gate"]["calibration_quantile"]),
        )
    )
    return {
        "contracts": contracts,
        "historical": historical,
        "recent": recent_joint,
        "regime_experts": experts,
        "historical_mae": _mae_model(train, config, seed=seed + 30),
        "recent_mae": _mae_model(recent, config, seed=seed + 31),
        "distribution": distribution,
        "distribution_threshold": distribution_threshold,
        "recent_train_start": min(row["timestamp_value"] for row in recent).isoformat(),
        "recent_train_rows": len(recent),
    }


def _probability_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "calibration": {
            "shrinkage_rows": 1,
            "expected_calibration_error_bins": int(
                config["calibration"]["expected_calibration_error_bins"]
            ),
        }
    }


def _contract_probabilities(
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    contracts: Sequence[BarrierContract],
    config: Mapping[str, Any],
) -> Mapping[str, list[Mapping[str, float]]]:
    expanded = _expanded(rows, contracts)
    values = _probabilities(expanded, model, _probability_config(config))
    width = len(contracts)
    return {
        contract.name: [
            values[row_index * width + contract_index] for row_index in range(len(rows))
        ]
        for contract_index, contract in enumerate(contracts)
    }


def _expert_probabilities(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[tuple[int, str], Mapping[str, float]]:
    result = {}
    for regime, expert in bundle["regime_experts"].items():
        indexed = [
            (index, row) for index, row in enumerate(rows) if row["regime"] == regime
        ]
        if not indexed:
            continue
        subset = [row for _, row in indexed]
        values = _contract_probabilities(subset, expert, bundle["contracts"], config)
        for local_index, (global_index, _) in enumerate(indexed):
            for contract in bundle["contracts"]:
                result[(global_index, contract.name)] = values[contract.name][
                    local_index
                ]
    return result


def _predict(
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    assignment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    contracts = {contract.name: contract for contract in bundle["contracts"]}
    historical = _contract_probabilities(
        rows, bundle["historical"], bundle["contracts"], config
    )
    recent = _contract_probabilities(
        rows, bundle["recent"], bundle["contracts"], config
    )
    experts = _expert_probabilities(rows, bundle, config)
    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    predicted_mae = conservative_mae(
        bundle["historical_mae"].predict(features),
        bundle["recent_mae"].predict(features),
    )
    distribution = distribution_scores(features, bundle["distribution"])
    result = []
    for index, row in enumerate(rows):
        name = assigned_contract_name(row, assignment)
        consensus = consensus_probabilities(
            historical[name][index],
            recent[name][index],
            regime_expert=experts.get((index, name)),
            maximum_divergence=float(
                config["agreement"]["maximum_jensen_shannon_divergence"]
            ),
        )
        probabilities = consensus["probabilities"]
        contract = contracts[name]
        attribution = joint_state_utility(probabilities, contract, config["utility"])
        actual = row["outcomes"][name]
        result.append(
            {
                **row,
                "selected_contract": name,
                "joint_probabilities": probabilities,
                "coherent_probability": (
                    probabilities[JointPathState.COHERENT_CLEAN_FAVORABLE.value]
                    + probabilities[JointPathState.COHERENT_DIRTY_FAVORABLE.value]
                ),
                "adverse_probability": probabilities[
                    JointPathState.ADVERSE_FIRST.value
                ],
                "unknown_probability": (
                    probabilities[JointPathState.SAME_BAR_AMBIGUOUS.value]
                    + probabilities[
                        JointPathState.UNRESOLVED_OR_DIRECTION_MISMATCH.value
                    ]
                ),
                "temporal_consensus": bool(consensus["eligible"]),
                "historical_recent_match": bool(consensus["historical_recent_match"]),
                "regime_expert_match": bool(consensus["regime_expert_match"]),
                "consensus_divergence": float(consensus["maximum_divergence"]),
                "predictor_count": int(consensus["predictor_count"]),
                "distribution_score": float(distribution[index]),
                "in_distribution": bool(
                    distribution[index] <= bundle["distribution_threshold"]
                ),
                "predicted_mae_q90": float(predicted_mae[index]),
                "utility_attribution": attribution,
                "predicted_utility": float(attribution["total_utility"]),
                "actual_utility": float(actual["realized_utility"]),
                "actual_outcome": str(actual["outcome"]),
            }
        )
    return result


def _model_skill(
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    contracts: Sequence[BarrierContract],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    expanded = _expanded(rows, contracts)
    by_contract = _contract_probabilities(rows, model, contracts, config)
    probabilities = [
        by_contract[contract.name][index]
        for index in range(len(rows))
        for contract in contracts
    ]
    labels = [str(row["joint_label"]) for row in expanded]
    epsilon = 1e-12
    log_loss = -float(
        np.mean(
            [
                math.log(max(epsilon, vector[label]))
                for vector, label in zip(probabilities, labels)
            ]
        )
    )
    prior_loss = -float(
        np.mean([math.log(max(epsilon, model["priors"][label])) for label in labels])
    )
    predicted = [max(vector, key=vector.get) for vector in probabilities]
    accuracy = float(np.mean([left == right for left, right in zip(predicted, labels)]))
    majority = max(model["priors"], key=model["priors"].get)
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


def _candidate_policies(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    selection = config["selection"]
    mae_quantiles = config["mae_risk"]["maximum_mae_quantiles"]
    arrays = {
        "utility": np.asarray(
            [max(0.0, float(row["predicted_utility"])) for row in rows]
        ),
        "coherent": np.asarray([float(row["coherent_probability"]) for row in rows]),
        "adverse": np.asarray([float(row["adverse_probability"]) for row in rows]),
        "unknown": np.asarray([float(row["unknown_probability"]) for row in rows]),
        "mae": np.asarray([float(row["predicted_mae_q90"]) for row in rows]),
    }
    return [
        {
            "minimum_utility": max(0.0, float(np.quantile(arrays["utility"], uq))),
            "minimum_coherent_probability": float(np.quantile(arrays["coherent"], cq)),
            "maximum_adverse_probability": float(np.quantile(arrays["adverse"], aq)),
            "maximum_unknown_probability": float(np.quantile(arrays["unknown"], nq)),
            "maximum_predicted_mae": float(np.quantile(arrays["mae"], mq)),
            "maximum_selected_per_timestamp": int(
                selection["maximum_selected_per_timestamp"]
            ),
            "source": "V13_THRESHOLD_POLICY_WINDOW_ONLY",
            "quantiles": {
                "utility": uq,
                "coherent": cq,
                "adverse": aq,
                "unknown": nq,
                "mae": mq,
            },
        }
        for uq in selection["utility_quantiles"]
        for cq in selection["coherent_probability_quantiles"]
        for aq in selection["adverse_probability_quantiles"]
        for nq in selection["unknown_probability_quantiles"]
        for mq in mae_quantiles
    ]


def _derive_policy(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    minimum = int(config["selection"]["minimum_policy_selections"])
    choices = []
    for policy in _candidate_policies(rows, config):
        selected = [
            row
            for row, keep in zip(rows, select_temporal_cross_section(rows, policy))
            if keep
        ]
        if len(selected) >= minimum:
            choices.append(
                (policy, utility_metrics(selected), path_quality_metrics(selected))
            )
    if not choices:
        return {
            "eligible": False,
            "reason": "NO_TEMPORAL_POLICY_WITH_MINIMUM_SELECTIONS",
            "temporal_consensus_count": sum(
                bool(row["temporal_consensus"]) for row in rows
            ),
            "in_distribution_count": sum(bool(row["in_distribution"]) for row in rows),
            "joint_gate_count": sum(
                bool(row["temporal_consensus"] and row["in_distribution"])
                for row in rows
            ),
            "positive_predicted_utility_count": sum(
                float(row["predicted_utility"]) >= 0.0 for row in rows
            ),
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


def _temporal_diagnostic(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        row for row in rows if row["temporal_consensus"] and row["in_distribution"]
    ]
    return {
        "rows": len(rows),
        "historical_recent_match_rate": float(
            np.mean([bool(row["historical_recent_match"]) for row in rows])
        ),
        "regime_expert_match_rate": float(
            np.mean([bool(row["regime_expert_match"]) for row in rows])
        ),
        "consensus_rate": float(
            np.mean([bool(row["temporal_consensus"]) for row in rows])
        ),
        "in_distribution_rate": float(
            np.mean([bool(row["in_distribution"]) for row in rows])
        ),
        "joint_gate_rows": len(eligible),
        "mean_predicted_mae_q90": float(
            np.mean([float(row["predicted_mae_q90"]) for row in rows])
        ),
        "top_30_after_temporal_gates": _ranking_diagnostic(eligible, count=30),
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
    bundle = _fit_bundle(train, calibration, config, 20260910 + fold)
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
    policy_predictions = _predict(policy_rows, bundle, assignment, config)
    policy = _derive_policy(policy_predictions, config)
    test_predictions = _predict(test, bundle, assignment, config)
    control_rows = _control(test, "ROE_10_H12")
    control = utility_metrics(control_rows)
    control_quality = path_quality_metrics(control_rows)
    selected_rows: list[Mapping[str, Any]] = []
    if policy["eligible"]:
        selected_rows = [
            row
            for row, keep in zip(
                test_predictions,
                select_temporal_cross_section(test_predictions, policy),
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
        < float(control_quality["mean_mae_fraction"])
    )
    historical_skill = _model_skill(
        test, bundle["historical"], bundle["contracts"], config
    )
    recent_skill = _model_skill(test, bundle["recent"], bundle["contracts"], config)
    return {
        "fold": fold,
        "status": "EVALUATED" if policy["eligible"] else "NO_POSITIVE_TEMPORAL_POLICY",
        "boundaries": [value.isoformat() for value in boundaries],
        "train": len(train),
        "recent_train_rows": bundle["recent_train_rows"],
        "recent_train_start": bundle["recent_train_start"],
        "probability_calibration": len(calibration),
        "contract_assignment_rows": len(assignment_rows),
        "threshold_policy_rows": len(policy_rows),
        "test": len(test),
        "historical_model": historical_skill,
        "recent_model": recent_skill,
        "supported_regime_experts": sorted(bundle["regime_experts"]),
        "distribution_threshold": bundle["distribution_threshold"],
        "contract_assignment": assignment,
        "policy": policy,
        "selected": selected,
        "selected_path_quality": selected_quality,
        "control": control,
        "control_path_quality": control_quality,
        "temporal_diagnostic": _temporal_diagnostic(test_predictions),
        "economic_gate": economic,
        "path_quality_gate": quality,
        "passed": bool(
            historical_skill["passed"]
            and recent_skill["passed"]
            and economic
            and quality
        ),
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
    historical = sum(bool(fold["historical_model"]["passed"]) for fold in evaluated)
    recent = sum(bool(fold["recent_model"]["passed"]) for fold in evaluated)
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
        and historical >= int(validation["minimum_historical_skilled_folds"])
        and recent >= int(validation["minimum_recent_skilled_folds"])
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
        "historical_skilled_folds": historical,
        "recent_skilled_folds": recent,
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
        default=Path("config/experiments/aegis_temporal_stability_v13_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/temporal_stability_v13/validation.json"),
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
        ("source_v12_validation", "source_v12_validation_sha256"),
        ("source_v12_config", "source_v12_config_sha256"),
    ):
        path = root / str(authority[key])
        if sha256_file(path) != str(authority[hash_key]):
            raise ValueError(f"V13 authority hash mismatch: {path.name}")
    dataset = root / str(authority["source_dataset"])
    sides = {
        side: evaluate_side(load_side(dataset, side), config)
        for side in ("LONG", "SHORT")
    }
    result = {
        "schema_id": "aegis-temporal-stability-v13-validation-v1",
        "experiment_id": config["experiment_id"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path.resolve()),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "v12_validation_sha256": sha256_file(
            root / str(authority["source_v12_validation"])
        ),
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
