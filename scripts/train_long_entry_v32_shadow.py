#!/usr/bin/env python3
"""Validate pooled family-aware LONG v3.2 without relaxing v3.1 gates."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.long_entry_v3_shadow import LONG_V3_FEATURE_NAMES
from aegis.utils import sha256_file
from train_long_entry_v31_shadow import (
    FAMILIES,
    _derive_policy,
    _family_retirement,
    _fit_binary,
    _group_attribution,
    _mapping,
    _predict,
    _select,
    _selection_metrics,
    _specialist_metrics,
    _verify_protection_authority,
    build_datasets,
)
from train_long_entry_v21_shadow import _fold_boundaries


FAMILY_FEATURE_NAMES = tuple(f"candidate_family_{family.value}" for family in FAMILIES)


def _augment(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        one_hot = tuple(
            1.0 if row["candidate_family"] == family.value else 0.0
            for family in FAMILIES
        )
        updated = {**row, "features": (*row["features"], *one_hot)}
        if "opportunity_features" in row:
            updated["opportunity_features"] = (*row["opportunity_features"], *one_hot)
        result.append(updated)
    return result


def _pooled_bundles(
    opportunity_train: Sequence[Mapping[str, Any]],
    opportunity_calibration: Sequence[Mapping[str, Any]],
    execution_train: Sequence[Mapping[str, Any]],
    execution_calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Mapping[str, tuple[Any, Any]]] | None:
    weight = float(config["hard_negatives"]["training_weight"])
    fits = {
        "opportunity_probability": _fit_binary(
            opportunity_train,
            opportunity_calibration,
            target="target_before_stop",
            feature_field="features",
            seed=seed,
            negative_weight=weight,
        ),
        "timing_probability": _fit_binary(
            execution_train,
            execution_calibration,
            target="clean_fast_success",
            feature_field="features",
            seed=seed + 1,
            negative_weight=weight,
        ),
        "falling_knife_probability": _fit_binary(
            execution_train,
            execution_calibration,
            target="falling_knife",
            feature_field="features",
            seed=seed + 2,
            negative_weight=weight,
        ),
        "path_risk_probability": _fit_binary(
            execution_train,
            execution_calibration,
            target="catastrophic_path",
            feature_field="features",
            seed=seed + 3,
            negative_weight=weight,
        ),
    }
    if any(value is None for value in fits.values()):
        return None
    shared = {name: value for name, value in fits.items() if value is not None}
    return {family.value: shared for family in FAMILIES}


def _split(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
    *,
    excluded_symbol: str | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    allowed = lambda row: excluded_symbol is None or row["symbol"] != excluded_symbol
    opportunity_train = [
        row for row in opportunities if allowed(row) and row["timestamp"] <= train_end
    ]
    opportunity_calibration = [
        row
        for row in opportunities
        if allowed(row) and train_end + embargo < row["timestamp"] <= calibration_end
    ]
    execution_train = [
        row for row in executions if allowed(row) and row["timestamp"] <= train_end
    ]
    execution_calibration = [
        row
        for row in executions
        if allowed(row) and train_end + embargo < row["timestamp"] <= calibration_end
    ]
    if excluded_symbol is None:
        test = [
            row
            for row in executions
            if row["independent"]
            and calibration_end + embargo < row["timestamp"] <= test_end
        ]
    else:
        test = [
            row
            for row in executions
            if row["symbol"] == excluded_symbol
            and row["independent"]
            and calibration_end + embargo < row["timestamp"] <= test_end
        ]
    return (
        opportunity_train,
        opportunity_calibration,
        execution_train,
        execution_calibration,
        test,
    )


def _evaluate_fold(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    opportunity_train, opportunity_calibration, train, calibration_rows, test = _split(
        opportunities, executions, boundaries, config
    )
    bundles = _pooled_bundles(
        opportunity_train,
        opportunity_calibration,
        train,
        calibration_rows,
        config,
        20261300 + fold_id * 100,
    )
    if bundles is None or len(test) < 100:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "opportunity_train_rows": len(opportunity_train),
            "execution_train_rows": len(train),
            "calibration_rows": len(calibration_rows),
            "test_rows": len(test),
            "passed": False,
        }
    calibration = _predict(calibration_rows, bundles)
    predicted_test = _predict(test, bundles)
    policy = _derive_policy(calibration, config)
    selected = _select(predicted_test, policy)
    metrics = _selection_metrics(predicted_test, selected)
    minimum = int(config["validation"]["minimum_scoring_selections_per_fold"])
    passed = bool(
        policy["valid"]
        and metrics["selected_rows"] >= minimum
        and metrics["selected_protected_worst_net"] is not None
        and metrics["selected_protected_worst_net"] > 0.0
        and metrics["selected_mae"] < metrics["baseline_mae"]
        and metrics["selected_underwater_bars"] < metrics["baseline_underwater_bars"]
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"]
        <= float(config["ranking"]["maximum_p95_gap_hours"])
    )
    family_metrics = {}
    for family in FAMILIES:
        indices = [
            index
            for index, row in enumerate(predicted_test)
            if row["candidate_family"] == family.value
        ]
        family_metrics[family.value] = _selection_metrics(
            [predicted_test[index] for index in indices], selected[indices]
        )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "opportunity_train_rows": len(opportunity_train),
        "execution_train_rows": len(train),
        "calibration_rows": len(calibration_rows),
        "test_rows": len(test),
        "policy": policy,
        "specialist_metrics": _specialist_metrics(predicted_test),
        "metrics": metrics,
        "family_metrics": family_metrics,
        "passed": passed,
    }


def _loso(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    reports = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        opportunity_train, opportunity_calibration, train, calibration_rows, test = _split(
            opportunities,
            executions,
            boundaries,
            config,
            excluded_symbol=symbol,
        )
        bundles = _pooled_bundles(
            opportunity_train,
            opportunity_calibration,
            train,
            calibration_rows,
            config,
            20265000 + index * 100,
        )
        if bundles is None or len(test) < 30:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        calibration = _predict(calibration_rows, bundles)
        predicted = _predict(test, bundles)
        policy = _derive_policy(calibration, config)
        metrics = _selection_metrics(predicted, _select(predicted, policy))
        reports[symbol] = {
            "status": "EVALUATED",
            "test_rows": len(test),
            "generalized_without_regression": bool(
                policy["valid"]
                and metrics["selected_rows"] >= 5
                and metrics["selected_protected_worst_net"] is not None
                and metrics["selected_protected_worst_net"]
                >= metrics["baseline_protected_worst_net"]
                and metrics["selected_mae"] <= metrics["baseline_mae"]
            ),
            "metrics": metrics,
        }
    evaluated = [row for row in reports.values() if row["status"] == "EVALUATED"]
    passing = sum(bool(row["generalized_without_regression"]) for row in evaluated)
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def train_and_validate(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in opportunities})
    boundaries = _fold_boundaries(times)
    folds = [
        _evaluate_fold(opportunities, executions, fold, index + 1, config)
        for index, fold in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    retirement = _family_retirement(folds, config)
    eligible = [name for name, result in retirement.items() if result["passed"]]
    primary = bool(
        len(evaluated) == 4
        and passing >= int(config["validation"]["minimum_positive_folds"])
        and eligible
        and all(
            float(fold["metrics"]["selected_protected_worst_net"] or -1.0) >= 0.0
            for fold in evaluated
        )
    )
    loso = (
        _loso(opportunities, executions, boundaries[-1], config)
        if primary
        else {"status": "NOT_RUN_PRIMARY_WALK_FORWARD_GATE_FAILED", "passed": False}
    )
    passed = primary and bool(loso["passed"])
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "family_retirement": retirement,
        "eligible_families": eligible,
        "primary_walk_forward_pass": primary,
        "leave_one_symbol_out": loso,
        "validation_pass": passed,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_REVIEW"
            if passed
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v32_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v32_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    prereg = _mapping(yaml.safe_load(config_path.read_text()), "v32_config")
    if (
        prereg.get("schema_version")
        != "aegis-long-entry-v32-shadow-preregistration-v1"
        or prereg.get("mode") != "SHADOW"
        or prereg.get("selection_effect") != "NONE"
    ):
        raise SystemExit("AEGIS_LONG_V32_CONFIG_INVALID")
    base_path = root / str(prereg["base_preregistration"]["path"])
    if sha256_file(base_path) != str(prereg["base_preregistration"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V32_BASE_PREREGISTRATION_DRIFT")
    config = _mapping(yaml.safe_load(base_path.read_text()), "base_config")
    _verify_protection_authority(root, config)
    v3_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v3_shadow.yaml").read_text()
        ),
        "v3_config",
    )
    label_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text()
        ),
        "label_config",
    )
    opportunities, executions, source = build_datasets(
        root, config, v3_config, label_config
    )
    opportunities = _augment(opportunities)
    executions = _augment(executions)
    validation = train_and_validate(opportunities, executions, config)
    report = {
        "schema_id": "aegis-long-entry-v32-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "base_preregistration_sha256": sha256_file(base_path),
        "source": source,
        "feature_names": [*LONG_V3_FEATURE_NAMES, *FAMILY_FEATURE_NAMES],
        "execution_attribution": {
            "by_family": _group_attribution(
                executions, lambda row: str(row["candidate_family"])
            ),
            "by_symbol": _group_attribution(executions, lambda row: str(row["symbol"])),
        },
        "validation": validation,
        "deployment": {
            "selection_effect": "NONE",
            "shadow_runtime_enabled": False,
            "live_enabled": False,
            "model_exported": False,
            "exchange_authority": False,
        },
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "verdict": validation["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
