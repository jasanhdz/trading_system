#!/usr/bin/env python3
"""Train the preregistered LONG v2.2 specialist committee in Shadow."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.long_entry_specialists_shadow import LongArchetypeV2
from aegis.research.long_entry_v22_shadow import (
    LONG_V22_FEATURE_NAMES,
    long_v22_feature_vector,
    select_cross_section,
    specialist_committee_score,
)
from aegis.utils import sha256_file
from train_long_entry_v21_shadow import (
    MODELED_ARCHETYPES,
    _fit_classifier,
    _fold_boundaries,
    _probability,
    _selection_metrics,
    build_dataset,
)

TARGETS = {
    "direction_probability": "target_before_stop",
    "timing_probability": "clean_fast_success",
    "path_risk_probability": "catastrophic_path",
}


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def augment_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        axes = _mapping(row["regime_axes"], "regime_axes")
        result.append(
            {
                **row,
                "features": long_v22_feature_vector(row["features"], axes),
            }
        )
    return result


def _fit_bundles(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Mapping[str, tuple[Any, Any]]] | None:
    bundles: dict[str, dict[str, tuple[Any, Any]]] = {}
    for archetype_index, archetype in enumerate(MODELED_ARCHETYPES):
        train_group = [row for row in train if row["archetype"] == archetype.value]
        calibration_group = [
            row for row in calibration if row["archetype"] == archetype.value
        ]
        specialists: dict[str, tuple[Any, Any]] = {}
        for target_index, (output, target) in enumerate(TARGETS.items()):
            fitted = _fit_classifier(
                train_group,
                calibration_group,
                target,
                seed + archetype_index * 10 + target_index,
            )
            if fitted is None:
                return None
            specialists[output] = fitted
        bundles[archetype.value] = specialists
    return bundles


def _predict_rows(
    rows: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, tuple[Any, Any]]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    by_archetype: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_archetype[str(row["archetype"])].append((index, row))
    ordered: list[dict[str, Any] | None] = [None] * len(rows)
    for archetype, indexed in by_archetype.items():
        specialists = bundles.get(archetype)
        if specialists is None:
            continue
        group = [row for _, row in indexed]
        output_values = {
            output: _probability(model, calibrator, group)
            for output, (model, calibrator) in specialists.items()
        }
        for local_index, (source_index, row) in enumerate(indexed):
            direction = float(output_values["direction_probability"][local_index])
            timing = float(output_values["timing_probability"][local_index])
            path_risk = float(output_values["path_risk_probability"][local_index])
            ordered[source_index] = {
                **row,
                "direction_probability": direction,
                "timing_probability": timing,
                "path_risk_probability": path_risk,
                "committee_score": specialist_committee_score(
                    direction, timing, path_risk
                ),
            }
    predictions.extend(row for row in ordered if row is not None)
    return predictions


def _policy_selection(
    rows: Sequence[Mapping[str, Any]],
    global_policy: Mapping[str, Any],
    direction_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> np.ndarray:
    policies = direction_policies or {}
    eligible_rows = []
    source_indices = []
    for index, row in enumerate(rows):
        direction = str(row["regime_axes"]["direction"])
        policy = policies.get(direction, global_policy)
        if float(row["committee_score"]) >= float(policy["minimum_score"]) and float(
            row["path_risk_probability"]
        ) <= float(policy["maximum_path_risk"]):
            eligible_rows.append(row)
            source_indices.append(index)
    selected = np.zeros(len(rows), dtype=bool)
    if not eligible_rows:
        return selected
    ranked = select_cross_section(
        eligible_rows,
        minimum_score=0.0,
        maximum_path_risk=1.0,
        maximum_selected_per_timestamp=int(
            global_policy["maximum_selected_per_timestamp"]
        ),
    )
    for source_index, keep in zip(source_indices, ranked):
        selected[source_index] = keep
    return selected


def _policy_valid(metrics: Mapping[str, Any], minimum: int) -> bool:
    return bool(
        metrics["selected_rows"] >= minimum
        and metrics["selected_protected_worst_net"] is not None
        and metrics["selected_protected_worst_net"] > 0.0
        and metrics["selected_mae"] < metrics["baseline_mae"]
        and metrics["selected_underwater_bars"] < metrics["baseline_underwater_bars"]
    )


def _derive_policy(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    fixed_top_k: int | None = None,
) -> Mapping[str, Any]:
    ranking = _mapping(config["ranking"], "ranking")
    validation = _mapping(config["validation"], "validation")
    scores = np.asarray([float(row["committee_score"]) for row in rows])
    risks = np.asarray([float(row["path_risk_probability"]) for row in rows])
    top_values = (
        [fixed_top_k]
        if fixed_top_k is not None
        else [int(value) for value in ranking["maximum_selected_per_timestamp_grid"]]
    )
    choices = []
    for score_quantile in ranking["score_quantiles"]:
        minimum_score = float(
            np.quantile(scores, float(score_quantile), method="higher")
        )
        for risk_quantile in ranking["maximum_risk_quantiles"]:
            maximum_risk = float(
                np.quantile(risks, float(risk_quantile), method="lower")
            )
            for top_k in top_values:
                policy = {
                    "minimum_score": minimum_score,
                    "maximum_path_risk": maximum_risk,
                    "maximum_selected_per_timestamp": top_k,
                    "score_quantile": float(score_quantile),
                    "risk_quantile": float(risk_quantile),
                }
                metrics = _selection_metrics(rows, _policy_selection(rows, policy))
                choices.append(
                    {
                        **policy,
                        "metrics": metrics,
                        "valid": _policy_valid(
                            metrics,
                            int(validation["minimum_calibration_selections"]),
                        ),
                    }
                )
    valid = [choice for choice in choices if choice["valid"]]
    pool = valid or choices
    return max(
        pool,
        key=lambda choice: (
            float(choice["metrics"]["selected_protected_worst_net"] or -1.0),
            -float(choice["metrics"]["selected_mae"] or 1.0),
            int(choice["metrics"]["selected_rows"]),
        ),
    )


def _derive_direction_policies(
    rows: Sequence[Mapping[str, Any]],
    global_policy: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    minimum = int(config["regime_specialization"]["minimum_partition_rows"])
    policies = {}
    for direction in ("BULLISH", "NEUTRAL", "BEARISH"):
        subset = [row for row in rows if row["regime_axes"]["direction"] == direction]
        if len(subset) < minimum:
            continue
        candidate = _derive_policy(
            subset,
            config,
            fixed_top_k=int(global_policy["maximum_selected_per_timestamp"]),
        )
        if candidate["valid"]:
            policies[direction] = candidate
    if not policies:
        return {}
    global_metrics = _selection_metrics(rows, _policy_selection(rows, global_policy))
    partitioned_metrics = _selection_metrics(
        rows, _policy_selection(rows, global_policy, policies)
    )
    if not _policy_valid(
        partitioned_metrics,
        int(config["validation"]["minimum_calibration_selections"]),
    ):
        return {}
    if float(partitioned_metrics["selected_protected_worst_net"]) < float(
        global_metrics["selected_protected_worst_net"]
    ):
        return {}
    return policies


def _probability_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for probability, target in TARGETS.items():
        truth = np.asarray([bool(row[target]) for row in rows], dtype=int)
        predicted = np.asarray([float(row[probability]) for row in rows])
        result[probability] = {
            "prevalence": float(np.mean(truth)),
            "average_precision": float(average_precision_score(truth, predicted)),
        }
    return result


def _evaluate_fold(
    records: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    train = [row for row in records if row["timestamp"] <= train_end]
    calibration = [
        row
        for row in records
        if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    test = [
        row
        for row in records
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
    ]
    bundles = _fit_bundles(train, calibration, 20261000 + fold_id * 100)
    if bundles is None or len(test) < 200:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "passed": False,
        }
    predicted_calibration = _predict_rows(calibration, bundles)
    predicted_test = _predict_rows(test, bundles)
    global_policy = _derive_policy(predicted_calibration, config)
    direction_policies = _derive_direction_policies(
        predicted_calibration, global_policy, config
    )
    selected = _policy_selection(predicted_test, global_policy, direction_policies)
    metrics = _selection_metrics(predicted_test, selected)
    minimum = int(config["validation"]["minimum_scoring_selections_per_fold"])
    passed = bool(
        global_policy["valid"]
        and _policy_valid(metrics, minimum)
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"]
        <= float(config["validation"]["maximum_p95_gap_hours"])
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "global_policy": global_policy,
        "direction_policies": direction_policies,
        "specialist_metrics": _probability_metrics(predicted_test),
        "metrics": metrics,
        "passed": passed,
    }


def _group_attribution(
    rows: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]
) -> Mapping[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["independent"]:
            groups[key(row)].append(row)
    result = {}
    for identity, group in sorted(groups.items()):
        result[identity] = {
            "rows": len(group),
            "protected_worst_net": float(
                np.mean([row["protected_worst_net_return"] for row in group])
            ),
            "mae": float(np.mean([row["mae_fraction"] for row in group])),
            "time_underwater_bars": float(
                np.mean([row["time_underwater_bars"] for row in group])
            ),
            "direction_success_rate": float(
                np.mean([row["target_before_stop"] for row in group])
            ),
            "timing_success_rate": float(
                np.mean([row["clean_fast_success"] for row in group])
            ),
            "catastrophic_rate": float(
                np.mean([row["catastrophic_path"] for row in group])
            ),
        }
    return result


def _leave_one_symbol_out(
    records: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    reports = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        train = [
            row
            for row in records
            if row["symbol"] != symbol and row["timestamp"] <= train_end
        ]
        calibration = [
            row
            for row in records
            if row["symbol"] != symbol
            and train_end + embargo < row["timestamp"] <= calibration_end
        ]
        test = [
            row
            for row in records
            if row["symbol"] == symbol
            and row["independent"]
            and calibration_end + embargo < row["timestamp"] <= test_end
        ]
        bundles = _fit_bundles(train, calibration, 20262000 + index * 100)
        if bundles is None or len(test) < 50:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        predicted_calibration = _predict_rows(calibration, bundles)
        predicted_test = _predict_rows(test, bundles)
        policy = _derive_policy(predicted_calibration, config)
        selected = _policy_selection(predicted_test, policy)
        metrics = _selection_metrics(predicted_test, selected)
        no_regression = bool(
            metrics["selected_rows"] >= 10
            and metrics["selected_protected_worst_net"] is not None
            and metrics["selected_protected_worst_net"]
            >= metrics["baseline_protected_worst_net"]
            and metrics["selected_mae"] <= metrics["baseline_mae"]
        )
        reports[symbol] = {
            "status": "EVALUATED",
            "training_excluded_symbol": True,
            "test_rows": len(test),
            "policy_valid_without_symbol": bool(policy["valid"]),
            "metrics": metrics,
            "generalized_without_regression": no_regression,
        }
    evaluated = [row for row in reports.values() if row["status"] == "EVALUATED"]
    passing = sum(bool(row["generalized_without_regression"]) for row in evaluated)
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "method": "THREE_SPECIALISTS_PER_ARCHETYPE_REFIT_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def train_and_validate(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in records})
    boundaries = _fold_boundaries(times)
    folds = [
        _evaluate_fold(records, boundaries[index], index + 1, config)
        for index in range(len(boundaries))
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    primary_pass = len(evaluated) == 4 and passing >= int(
        config["validation"]["minimum_positive_folds"]
    )
    if primary_pass:
        loso = _leave_one_symbol_out(records, boundaries[-1], config)
    else:
        loso = {
            "status": "NOT_RUN_PRIMARY_WALK_FORWARD_GATE_FAILED",
            "passed": False,
        }
    validation_pass = primary_pass and bool(loso["passed"])
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "primary_walk_forward_pass": primary_pass,
        "leave_one_symbol_out": loso,
        "validation_pass": validation_pass,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_SHADOW_RUNTIME_REVIEW"
            if validation_pass
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v22_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v22_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    if (
        config.get("schema_version") != "aegis-long-entry-v22-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
        or config.get("automatic_live_promotion") is not False
    ):
        raise SystemExit("AEGIS_LONG_V22_CONFIG_INVALID")
    v21_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "v21_config",
    )
    database = root / str(config["source"]["base_database"])
    delta = root / str(config["source"]["public_delta"])
    source_records, source = build_dataset(database, delta, v21_config)
    records = augment_records(source_records)
    del source_records
    validation = train_and_validate(records, config)
    report = {
        "schema_id": "aegis-long-entry-v22-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": {
            **source,
            "v21_attribution": "data/long_entry_v21_shadow/attribution.json",
        },
        "feature_names": list(LONG_V22_FEATURE_NAMES),
        "feature_availability": config["features"]["availability_audit"],
        "row_level_attribution": {
            "by_symbol": _group_attribution(records, lambda row: str(row["symbol"])),
            "by_archetype": _group_attribution(
                records, lambda row: str(row["archetype"])
            ),
            "by_regime_direction": _group_attribution(
                records, lambda row: str(row["regime_axes"]["direction"])
            ),
            "by_regime_identity": _group_attribution(
                records, lambda row: str(row["regime"])
            ),
        },
        "validation": validation,
        "deployment": {
            "selection_effect": "NONE",
            "shadow_runtime_enabled": False,
            "live_enabled": False,
            "automatic_promotion": False,
            "exchange_authority": False,
        },
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "verdict": validation["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
