#!/usr/bin/env python3
"""Train and validate V7 side/archetype entry and protection specialists."""

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
    bootstrap_mean_interval,
)
from aegis.research.regime_entry_exit_v7_training import (
    V7_ABLATIONS,
    fold_passes,
    joint_quality_score,
    select_v7_cross_section,
    v7_ablation_score,
    v7_selection_metrics,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _mapping
from train_regime_aware_directional_v6_shadow import (
    _fit_regime_router,
    _regime_router_metrics,
)

PROFILE_NAMES = (
    "CURRENT_TS",
    "LOCK_AT_5_ROE",
    "LOCK_AT_10_ROE",
    "LOCK_AT_20_ROE",
)
CLASSIFIER_TARGETS = (
    "clean_entry",
    "positive_best_profile",
    "late_entry",
)


def load_side(path: Path, side: str) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = dict(_mapping(json.loads(line), f"dataset:{line_number}"))
            if source.get("side") != side:
                continue
            attribution = _mapping(
                source["trajectory_attribution"], "trajectory_attribution"
            )
            profiles = _mapping(source["protection_profiles"], "protection_profiles")
            profile_returns = {
                name: float(_mapping(profiles[name], name)["worst_net_return"])
                for name in PROFILE_NAMES
            }
            available = float(attribution["available_net_opportunity"])
            best_net = max(profile_returns.values())
            first_positive = source.get("first_positive_after_cost_bar")
            features = tuple(float(value) for value in source["v7_features"])
            if not features or not all(math.isfinite(value) for value in features):
                raise ValueError("V7 dataset contains invalid model features")
            rows.append(
                {
                    **source,
                    "timestamp_value": datetime.fromisoformat(str(source["timestamp"])),
                    "features": features,
                    "clean_entry": bool(attribution["clean_entry"]),
                    "late_entry": bool(attribution["late_entry"]),
                    "positive_best_profile": best_net > 0.0,
                    "time_to_positive": (
                        min(1.0, int(first_positive) / 24.0)
                        if first_positive is not None
                        else 1.0
                    ),
                    "best_capture_efficiency": (
                        min(1.0, max(0.0, best_net) / available)
                        if available > 0.0
                        else 0.0
                    ),
                    "profile_returns": profile_returns,
                }
            )
    if not rows:
        raise ValueError(f"V7 dataset has no {side} rows")
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


def _fit_archetype(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
    minimum_rows: int,
) -> Mapping[str, Any] | None:
    if len(train) < minimum_rows or len(calibration) < max(100, minimum_rows // 5):
        return None
    x_train = _x(train)
    x_calibration = _x(calibration)
    classifiers: dict[str, tuple[Any, CalibratorSpec]] = {}
    for offset, target in enumerate(CLASSIFIER_TARGETS):
        y_train = np.asarray([bool(row[target]) for row in train], dtype=np.int8)
        y_calibration = np.asarray(
            [bool(row[target]) for row in calibration], dtype=np.int8
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_calibration)) < 2:
            return None
        model = _classifier(seed + offset)
        model.fit(x_train, y_train)
        calibrator = fit_platt_calibrator(
            model.predict_proba(x_calibration)[:, 1], y_calibration
        )
        classifiers[target] = (model, calibrator)
    regressors: dict[str, Any] = {}
    targets = {
        "mae_q90": "mae_fraction",
        "time_to_positive": "time_to_positive",
        "capture_efficiency": "best_capture_efficiency",
    }
    for offset, (name, target) in enumerate(targets.items(), start=10):
        model = _regressor(seed + offset, quantile=0.90 if name == "mae_q90" else None)
        model.fit(
            x_train,
            np.asarray([float(row[target]) for row in train], dtype=np.float64),
        )
        regressors[name] = model
    for offset, profile in enumerate(PROFILE_NAMES, start=20):
        model = _regressor(seed + offset)
        model.fit(
            x_train,
            np.asarray(
                [float(row["profile_returns"][profile]) for row in train],
                dtype=np.float64,
            ),
        )
        regressors[f"profile::{profile}"] = model
    return {"classifiers": classifiers, "regressors": regressors}


def _fit(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    archetype_config = _mapping(config["archetypes"], "archetypes")
    identities = tuple(str(value) for value in archetype_config["identities"])
    minimum = int(archetype_config["minimum_training_rows"])
    specialists = {}
    for index, identity in enumerate(identities):
        bundle = _fit_archetype(
            [row for row in train if row["v7_archetype"] == identity],
            [row for row in calibration if row["v7_archetype"] == identity],
            seed + index * 100,
            minimum,
        )
        if bundle is not None:
            specialists[identity] = bundle
    router = _fit_regime_router(train, calibration, seed + 1000)
    if not specialists or router is None:
        return None
    return {"specialists": specialists, "regime_router": router}


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["v7_archetype"])].append((index, row))
    predicted: dict[int, dict[str, Any]] = {}
    abstained = 0
    for identity, indexed in groups.items():
        specialist = bundle["specialists"].get(identity)
        if specialist is None:
            abstained += len(indexed)
            continue
        values = [row for _, row in indexed]
        features = _x(values)
        outputs = {}
        for target, (model, calibrator) in specialist["classifiers"].items():
            outputs[target] = np.asarray(
                [
                    calibrator.apply(float(value))
                    for value in model.predict_proba(features)[:, 1]
                ],
                dtype=np.float64,
            )
        regressions = {
            name: np.asarray(model.predict(features), dtype=np.float64)
            for name, model in specialist["regressors"].items()
        }
        for local_index, (global_index, row) in enumerate(indexed):
            profile_predictions = {
                profile: float(regressions[f"profile::{profile}"][local_index])
                for profile in PROFILE_NAMES
            }
            selected_profile = max(profile_predictions, key=profile_predictions.get)
            expected_net = profile_predictions[selected_profile]
            mae = max(0.0, float(regressions["mae_q90"][local_index]))
            timing = min(
                1.0, max(0.0, float(regressions["time_to_positive"][local_index]))
            )
            capture = min(
                1.0,
                max(0.0, float(regressions["capture_efficiency"][local_index])),
            )
            clean = float(outputs["clean_entry"][local_index])
            positive = float(outputs["positive_best_profile"][local_index])
            late = float(outputs["late_entry"][local_index])
            actual_net = float(row["profile_returns"][selected_profile])
            available = float(
                row["trajectory_attribution"]["available_net_opportunity"]
            )
            actual_capture = (
                min(1.0, max(0.0, actual_net) / available) if available > 0.0 else 0.0
            )
            predicted[global_index] = {
                **row,
                "clean_probability": clean,
                "positive_probability": positive,
                "late_probability": late,
                "expected_profile_net": expected_net,
                "mae_q90": mae,
                "predicted_time_to_positive": timing,
                "predicted_capture_efficiency": capture,
                "profile_net_predictions": profile_predictions,
                "selected_profile": selected_profile,
                "selected_profile_net": actual_net,
                "selected_capture_efficiency": actual_capture,
                "v7_quality_score": joint_quality_score(
                    clean_probability=clean,
                    positive_probability=positive,
                    late_probability=late,
                    expected_profile_net=expected_net,
                    mae_q90=mae,
                    time_to_positive=timing,
                    capture_efficiency=capture,
                ),
            }
    return [predicted[index] for index in sorted(predicted)], abstained


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


def _control_rows(
    rows: Sequence[Mapping[str, Any]], side: str
) -> tuple[list[dict[str, Any]], str]:
    matching = [row for row in rows if row["entry_brain_action"] == side]
    source = matching or list(rows)
    identity = "CURRENT_BRAIN" if matching else "UNFILTERED_SIDE_FALLBACK"
    result = []
    for row in source:
        net = float(row["profile_returns"]["CURRENT_TS"])
        available = float(row["trajectory_attribution"]["available_net_opportunity"])
        result.append(
            {
                **row,
                "selected_profile": "CURRENT_TS",
                "selected_profile_net": net,
                "selected_capture_efficiency": (
                    min(1.0, max(0.0, net) / available) if available > 0.0 else 0.0
                ),
            }
        )
    return result, identity


def _candidate_policies(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    validation = _mapping(config["validation"], "validation")
    scores = np.asarray([float(row["v7_quality_score"]) for row in calibration])
    late = np.asarray([float(row["late_probability"]) for row in calibration])
    mae = np.asarray([float(row["mae_q90"]) for row in calibration])
    return [
        {
            "minimum_score": float(np.quantile(scores, score_quantile)),
            "maximum_late_probability": float(np.quantile(late, late_quantile)),
            "maximum_mae_q90": float(np.quantile(mae, mae_quantile)),
            "maximum_selected_per_timestamp": int(
                validation["maximum_selected_per_timestamp"]
            ),
            "source": "CALIBRATION_ONLY",
            "quantiles": {
                "score": float(score_quantile),
                "late": float(late_quantile),
                "mae": float(mae_quantile),
            },
        }
        for score_quantile in validation["score_quantiles"]
        for late_quantile in validation["maximum_late_probability_quantiles"]
        for mae_quantile in validation["maximum_mae_quantiles"]
    ]


def _derive_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    minimum = int(config["validation"]["minimum_calibration_selections"])
    choices = []
    for policy in _candidate_policies(calibration, config):
        mask = select_v7_cross_section(calibration, policy)
        selected = [row for row, keep in zip(calibration, mask) if keep]
        if len(selected) < minimum:
            continue
        metrics = v7_selection_metrics(selected)
        choices.append((policy, metrics))
    if not choices:
        return None
    policy, metrics = max(
        choices,
        key=lambda item: (
            float(item[1]["mean_net"]),
            float(item[1]["mean_capture_efficiency"]),
            -float(item[1]["mae_q90"]),
            int(item[1]["count"]),
        ),
    )
    return {
        **policy,
        "calibration_metrics": metrics,
        "policies_evaluated": len(choices),
    }


def _ablation_audit(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _mapping(config["ablations"], "ablations")
    variants = tuple(str(value) for value in raw["variants"])
    if set(variants) != set(V7_ABLATIONS):
        raise ValueError("V7 ablation identities are invalid")
    quantile = float(raw["calibration_score_quantile"])
    reports = {}
    for variant in variants:
        threshold = float(
            np.quantile(
                [v7_ablation_score(row, variant) for row in calibration], quantile
            )
        )
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in test:
            if v7_ablation_score(row, variant) >= threshold:
                groups[str(row["timestamp"])].append(row)
        selected = [
            max(
                values,
                key=lambda row: (
                    v7_ablation_score(row, variant),
                    -float(row["mae_q90"]),
                ),
            )
            for values in groups.values()
        ]
        reports[variant] = {
            "threshold_source": "CALIBRATION_ONLY_FIXED_QUANTILE",
            "threshold": threshold,
            "metrics": v7_selection_metrics(selected),
            "promotion_eligible": False,
        }
    return reports


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
    bundle = _fit(train, calibration, config, 20260820 + fold)
    if bundle is None or not test:
        return {
            "fold": fold,
            "status": "INSUFFICIENT_MODEL_DATA",
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
            "passed": False,
        }
    predicted_calibration, calibration_abstained = _predict(calibration, bundle)
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
    predicted_test, test_abstained = _predict(test, bundle)
    mask = select_v7_cross_section(predicted_test, policy)
    selected = [row for row, keep in zip(predicted_test, mask) if keep]
    side = str(rows[0]["side"])
    controls, control_identity = _control_rows(predicted_test, side)
    metrics = v7_selection_metrics(selected)
    control = v7_selection_metrics(controls)
    router = _regime_router_metrics(predicted_test, bundle["regime_router"], config)
    passed = bool(
        router["passed"]
        and fold_passes(
            metrics,
            control,
            minimum_count=int(validation["minimum_test_selections_per_fold"]),
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
        "trained_archetypes": sorted(bundle["specialists"]),
        "calibration_abstained": calibration_abstained,
        "test_abstained": test_abstained,
        "policy": policy,
        "selected": metrics,
        "control_identity": control_identity,
        "control": control,
        "regime_router": router,
        "diagnostic_ablations": _ablation_audit(
            predicted_calibration, predicted_test, config
        ),
        "net_bootstrap": bootstrap_mean_interval(
            [float(row["selected_profile_net"]) for row in selected],
            samples=int(validation["bootstrap_samples"]),
            seed=20260820 + fold,
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
    worst_non_negative = bool(
        evaluated
        and all(
            value["selected"]["mean_net"] is not None
            and float(value["selected"]["mean_net"]) >= 0.0
            for value in evaluated
        )
    )
    validation = _mapping(config["validation"], "validation")
    primary = bool(
        len(evaluated) == int(validation["folds"])
        and passing >= int(validation["minimum_positive_folds"])
        and router_skilled >= int(validation["minimum_router_skilled_folds"])
        and worst_non_negative
    )
    # LOSO is intentionally conditional: an already-failed primary hypothesis
    # must not consume compute or be mistaken for promotion evidence.
    loso: Mapping[str, Any] = {
        "status": "NOT_RUN_PRIMARY_GATE_FAILED",
        "passing_symbols": 0,
        "required_symbols": int(validation["minimum_symbols_without_regression"]),
        "passed": False,
    }
    if primary:
        reports = {}
        for index, symbol in enumerate(sorted({str(row["symbol"]) for row in rows})):
            reports[symbol] = _evaluate_fold(
                rows,
                boundaries[-1],
                100 + index,
                config,
                excluded_train_symbol=symbol,
                test_symbol=symbol,
            )
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
        raise ValueError("V7 dataset hash mismatch")
    sides = {
        side: evaluate_side(load_side(dataset, side), config)
        for side in ("LONG", "SHORT")
    }
    passed = all(bool(value["validation_pass"]) for value in sides.values())
    result = {
        "schema_id": "aegis-regime-entry-exit-v7-validation-v1",
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
        default=Path("config/experiments/aegis_regime_entry_exit_v7_research.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/regime_entry_exit_v7/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/regime_entry_exit_v7/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regime_entry_exit_v7/validation.json"),
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
                "validation_pass": result["validation_pass"],
                "sides": {
                    side: {
                        "passing_folds": value["passing_folds"],
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
