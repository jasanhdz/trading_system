#!/usr/bin/env python3
"""Run the preregistered LONG v5 multi-objective Shadow validation."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.long_entry_v3_shadow import HardNegativeType
from aegis.research.long_entry_v4_hybrid_shadow import LongTechnicalSetup
from aegis.research.long_entry_v5_multiobjective_shadow import (
    MultiObjectiveEstimate,
    classify_regime_evidence,
    multiobjective_score,
    time_to_profit_fraction,
)
from aegis.utils import sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _selection_metrics
from train_long_entry_v31_shadow import _fit_binary, _mapping, build_datasets
from train_long_entry_v4_shadow import (
    SETUP_FEATURE_NAMES,
    _augment_entry_records,
    _verify_protection_authority,
)


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    generator = np.random.default_rng(seed)
    return [rows[index] for index in generator.integers(0, len(rows), len(rows))]


def _regressor(seed: int, *, quantile: float | None = None) -> Any:
    settings = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 2.0,
        "early_stopping": False,
        "random_state": seed,
    }
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **settings)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **settings)


def _fit_regression_ensemble(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    target: str,
    members: int,
    seed: int,
    quantile: float | None = None,
) -> list[tuple[Any, float]] | None:
    if len(train) < 200 or len(calibration) < 80:
        return None
    x_calibration = np.asarray([row["features"] for row in calibration], dtype=np.float64)
    y_calibration = np.asarray([float(row[target]) for row in calibration])
    result = []
    for member in range(members):
        sampled = _bootstrap(train, seed + member)
        x_train = np.asarray([row["features"] for row in sampled], dtype=np.float64)
        y_train = np.asarray([float(row[target]) for row in sampled])
        weights = np.asarray(
            [
                2.0
                if row["hard_negative"] != HardNegativeType.NOT_HARD_NEGATIVE.value
                else 1.0
                for row in sampled
            ]
        )
        model = _regressor(seed + member, quantile=quantile).fit(
            x_train, y_train, sample_weight=weights
        )
        residual = y_calibration - model.predict(x_calibration)
        correction = (
            max(0.0, float(np.quantile(residual, quantile, method="higher")))
            if quantile is not None
            else float(np.median(residual))
        )
        result.append((model, correction))
    return result


def _fit_classifier_ensemble(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    members: int,
    seed: int,
) -> list[tuple[Any, Any]] | None:
    result = []
    for member in range(members):
        fit = _fit_binary(
            _bootstrap(train, seed + member),
            calibration,
            target="target_before_stop",
            feature_field="features",
            seed=seed + member,
            negative_weight=2.0,
        )
        if fit is None:
            return None
        result.append(fit)
    return result


def _fit_bundle(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    members = int(config["multiobjective_model"]["bootstrap_members"])
    fits = {
        "success": _fit_classifier_ensemble(
            train, calibration, members=members, seed=seed
        ),
        "net": _fit_regression_ensemble(
            train,
            calibration,
            target="protected_worst_net_return",
            members=members,
            seed=seed + 100,
        ),
        "mae": _fit_regression_ensemble(
            train,
            calibration,
            target="mae_fraction",
            members=members,
            seed=seed + 200,
            quantile=0.90,
        ),
        "time": _fit_regression_ensemble(
            train,
            calibration,
            target="observed_time_to_profit_fraction",
            members=members,
            seed=seed + 300,
        ),
    }
    return None if any(value is None for value in fits.values()) else fits


def _classifier_predictions(
    members: Sequence[tuple[Any, Any]], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    return np.asarray(
        [
            [calibrator.apply(float(value)) for value in model.predict_proba(matrix)[:, 1]]
            for model, calibrator in members
        ]
    )


def _regression_predictions(
    members: Sequence[tuple[Any, float]], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    return np.asarray([model.predict(matrix) + correction for model, correction in members])


def _predict(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    success = _classifier_predictions(bundle["success"], rows)
    net = _regression_predictions(bundle["net"], rows)
    mae = np.maximum(0.0, _regression_predictions(bundle["mae"], rows))
    timing = np.clip(_regression_predictions(bundle["time"], rows), 0.0, 1.0)
    z = float(
        config["uncertainty_and_abstention"]["lower_confidence_standard_deviations"]
    )
    predicted = []
    for index, row in enumerate(rows):
        estimate = MultiObjectiveEstimate(
            success_probability=float(np.mean(success[:, index])),
            expected_protected_net=float(np.mean(net[:, index])),
            mae_q90=float(np.mean(mae[:, index])),
            time_to_profit_fraction=float(np.mean(timing[:, index])),
            success_uncertainty=float(np.std(success[:, index])),
            net_uncertainty=float(np.std(net[:, index])),
            mae_uncertainty=float(np.std(mae[:, index])),
            time_uncertainty=float(np.std(timing[:, index])),
        )
        score = multiobjective_score(
            estimate,
            atr_fraction=float(row["atr_fraction"]),
            adverse_barrier_fraction=float(row["adverse_barrier_fraction"]),
            confidence_standard_deviations=z,
        )
        predicted.append(
            {
                **row,
                **estimate.__dict__,
                **score,
            }
        )
    return predicted


def _select(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> np.ndarray:
    eligible: defaultdict[datetime, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            float(row["committee_score"]) >= float(policy["minimum_score"])
            and float(row["normalized_uncertainty"]) <= float(policy["maximum_uncertainty"])
            and float(row["expected_net_lower_bound"]) > 0.0
            and float(row["mae_upper_bound"]) <= float(row["adverse_barrier_fraction"])
            and float(row["time_to_profit_upper_bound"])
            <= float(policy["maximum_time_to_profit_fraction"])
        ):
            eligible[row["timestamp"]].append((index, row))
    selected = np.zeros(len(rows), dtype=bool)
    for candidates in eligible.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["committee_score"]),
                float(item[1]["mae_upper_bound"]),
                str(item[1]["symbol"]),
            ),
        )
        for index, _ in ordered[: int(policy["maximum_selected_per_timestamp"])]:
            selected[index] = True
    return selected


def _derive_policy(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    contract = config["uncertainty_and_abstention"]
    scores = np.asarray([float(row["committee_score"]) for row in rows])
    uncertainty = np.asarray([float(row["normalized_uncertainty"]) for row in rows])
    choices = []
    for score_quantile in contract["score_quantiles"]:
        minimum_score = float(np.quantile(scores, float(score_quantile), method="higher"))
        for uncertainty_quantile in contract["uncertainty_quantiles"]:
            maximum_uncertainty = float(
                np.quantile(uncertainty, float(uncertainty_quantile), method="lower")
            )
            for top_k in config["ranking"]["maximum_selected_per_timestamp_grid"]:
                policy = {
                    "minimum_score": minimum_score,
                    "maximum_uncertainty": maximum_uncertainty,
                    "maximum_time_to_profit_fraction": float(
                        contract["maximum_time_to_profit_fraction"]
                    ),
                    "maximum_selected_per_timestamp": int(top_k),
                    "score_quantile": float(score_quantile),
                    "uncertainty_quantile": float(uncertainty_quantile),
                }
                metrics = _selection_metrics(rows, _select(rows, policy))
                valid = bool(
                    metrics["selected_rows"]
                    >= int(config["validation"]["minimum_calibration_selections"])
                    and metrics["selected_protected_worst_net"] is not None
                    and metrics["selected_protected_worst_net"] > 0.0
                )
                choices.append({**policy, "metrics": metrics, "valid": valid})
    valid = [choice for choice in choices if choice["valid"]]
    return max(
        valid or choices,
        key=lambda choice: (
            float(choice["metrics"]["selected_protected_worst_net"] or -1.0),
            -float(choice["metrics"]["selected_mae"] or 1.0),
            int(choice["metrics"]["selected_rows"]),
        ),
    )


def _model_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    truth = np.asarray([bool(row["target_before_stop"]) for row in rows], dtype=int)
    return {
        "success_average_precision": float(
            average_precision_score(truth, [row["success_probability"] for row in rows])
        ),
        "success_prevalence": float(np.mean(truth)),
        "protected_net_mae": float(
            mean_absolute_error(
                [row["protected_worst_net_return"] for row in rows],
                [row["expected_protected_net"] for row in rows],
            )
        ),
        "mae_q90_coverage": float(
            np.mean([row["mae_fraction"] <= row["mae_q90"] for row in rows])
        ),
        "time_to_profit_mae": float(
            mean_absolute_error(
                [row["observed_time_to_profit_fraction"] for row in rows],
                [row["time_to_profit_fraction"] for row in rows],
            )
        ),
        "mean_normalized_uncertainty": float(
            np.mean([row["normalized_uncertainty"] for row in rows])
        ),
    }


def _regime_audit(
    rows: Sequence[Mapping[str, Any]], selected: np.ndarray, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    baseline_target = float(np.mean([bool(row["target_before_stop"]) for row in rows]))
    baseline_mae = float(np.mean([float(row["mae_fraction"]) for row in rows]))
    minimum = int(config["regime_truth_audit"]["minimum_independent_candidates"])
    regimes = {}
    for regime in sorted({str(row["regime"]) for row in rows}):
        all_rows = [row for row in rows if row["regime"] == regime]
        chosen = [
            row
            for row, keep in zip(rows, selected)
            if keep and row["regime"] == regime
        ]
        regimes[regime] = {
            "all_candidates": classify_regime_evidence(
                all_rows,
                unconditional_target_rate=baseline_target,
                unconditional_mae=baseline_mae,
                minimum_rows=minimum,
            ),
            "selected_rows": len(chosen),
            "selected_protected_net": (
                float(np.mean([row["protected_worst_net_return"] for row in chosen]))
                if chosen
                else None
            ),
        }
    supported = [
        name
        for name, evidence in regimes.items()
        if evidence["all_candidates"]["supported"]
    ]
    return {
        "unconditional_target_rate": baseline_target,
        "unconditional_mae": baseline_mae,
        "regimes": regimes,
        "commercially_supported_regimes": supported,
    }


def _split(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
    *,
    excluded_symbol: str | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    train = [
        row
        for row in rows
        if row["timestamp"] <= train_end
        and (excluded_symbol is None or row["symbol"] != excluded_symbol)
    ]
    calibration = [
        row
        for row in rows
        if train_end + embargo < row["timestamp"] <= calibration_end
        and (excluded_symbol is None or row["symbol"] != excluded_symbol)
    ]
    test = [
        row
        for row in rows
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
        and (excluded_symbol is None or row["symbol"] == excluded_symbol)
    ]
    return train, calibration, test


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
    v4_fold: Mapping[str, Any],
) -> Mapping[str, Any]:
    train, calibration, test = _split(rows, boundaries, config)
    bundle = _fit_bundle(train, calibration, config, 20269000 + fold_id * 1000)
    if bundle is None or len(test) < 100:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "test_rows": len(test),
            "passed": False,
        }
    calibration_prediction = _predict(calibration, bundle, config)
    prediction = _predict(test, bundle, config)
    policy = _derive_policy(calibration_prediction, config)
    selected = _select(prediction, policy)
    metrics = _selection_metrics(prediction, selected)
    v4 = v4_fold["hybrid_metrics"]
    regime = _regime_audit(prediction, selected, config)
    passed = bool(
        policy["valid"]
        and metrics["selected_rows"]
        >= int(config["validation"]["minimum_scoring_selections_per_fold"])
        and metrics["selected_protected_worst_net"] is not None
        and metrics["selected_protected_worst_net"] > 0.0
        and metrics["selected_protected_worst_net"]
        > float(v4["selected_protected_worst_net"])
        and metrics["selected_mae"] < float(v4["selected_mae"])
        and metrics["selected_underwater_bars"]
        < float(v4["selected_underwater_bars"])
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"] <= float(config["ranking"]["maximum_p95_gap_hours"])
        and bool(regime["commercially_supported_regimes"])
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "policy": policy,
        "metrics": metrics,
        "v4_hybrid_control": v4,
        "multiobjective_metrics": _model_metrics(prediction),
        "regime_truth_audit": regime,
        "passed": passed,
    }


def _prospective_inventory(database: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    required = config["prospective_microstructure"]["required_sources"]
    if not database.is_file():
        return {"status": "PENDING_COLLECTION", "database_present": False, "tables": {}}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    fields = {
        "kline_microstructure": "open_time_ms",
        "funding_history": "funding_time_ms",
        "open_interest_recent": "timestamp_ms",
        "taker_ratio_recent": "timestamp_ms",
        "depth_snapshots": "transaction_time_ms",
    }
    tables = {}
    try:
        for table in required:
            field = fields[str(table)]
            rows = connection.execute(
                f"SELECT symbol, COUNT(*), MIN({field}), MAX({field}) FROM {table} GROUP BY symbol"
            ).fetchall()
            durations = [
                (int(maximum) - int(minimum)) / 86_400_000
                for _, _, minimum, maximum in rows
                if minimum is not None and maximum is not None
            ]
            tables[str(table)] = {
                "symbols": len(rows),
                "rows": sum(int(row[1]) for row in rows),
                "minimum_symbol_days": min(durations) if durations else 0.0,
            }
    finally:
        connection.close()
    minimum_days = int(config["prospective_microstructure"]["minimum_continuous_days"])
    minimum_symbols = int(config["prospective_microstructure"]["minimum_symbol_coverage"])
    ready = all(
        row["symbols"] >= minimum_symbols and row["minimum_symbol_days"] >= minimum_days
        for row in tables.values()
    )
    return {
        "status": "READY" if ready else "PENDING_COLLECTION",
        "database_present": True,
        "tables": tables,
        "minimum_continuous_days": minimum_days,
        "minimum_symbol_coverage": minimum_symbols,
    }


def _leave_one_symbol_out(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    reports = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        train, calibration, test = _split(
            rows, boundaries, config, excluded_symbol=symbol
        )
        bundle = _fit_bundle(train, calibration, config, 20270000 + index * 1000)
        if bundle is None or len(test) < 30:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        calibration_prediction = _predict(calibration, bundle, config)
        prediction = _predict(test, bundle, config)
        policy = _derive_policy(calibration_prediction, config)
        selected = _select(prediction, policy)
        selected_metrics = _selection_metrics(prediction, selected)
        baseline_metrics = _selection_metrics(
            prediction, np.ones(len(prediction), dtype=bool)
        )
        reports[symbol] = {
            "status": "EVALUATED",
            "test_rows": len(test),
            "selected_metrics": selected_metrics,
            "baseline_metrics": baseline_metrics,
            "generalized_without_regression": bool(
                policy["valid"]
                and selected_metrics["selected_rows"] >= 5
                and selected_metrics["selected_protected_worst_net"] is not None
                and selected_metrics["selected_protected_worst_net"]
                >= baseline_metrics["selected_protected_worst_net"]
                and selected_metrics["selected_mae"] <= baseline_metrics["selected_mae"]
            ),
        }
    evaluated = [row for row in reports.values() if row["status"] == "EVALUATED"]
    passing = sum(bool(row["generalized_without_regression"]) for row in evaluated)
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "method": "MULTIOBJECTIVE_REFIT_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v5_shadow.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/long_entry_v5_shadow/validation.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "v5_config")
    if (
        config.get("schema_version")
        != "aegis-long-entry-v5-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
    ):
        raise SystemExit("AEGIS_LONG_V5_CONFIG_INVALID")
    v4_config_path = root / str(config["frozen_v4_contract"]["path"])
    v4_evidence_path = root / str(config["frozen_v4_evidence"]["path"])
    if sha256_file(v4_config_path) != str(config["frozen_v4_contract"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V5_V4_CONTRACT_DRIFT")
    if sha256_file(v4_evidence_path) != str(config["frozen_v4_evidence"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V5_V4_EVIDENCE_DRIFT")
    v4_config = _mapping(yaml.safe_load(v4_config_path.read_text()), "v4_config")
    inherited_path = root / str(v4_config["inherited_entry_contract"]["path"])
    inherited = _mapping(yaml.safe_load(inherited_path.read_text()), "inherited")
    _verify_protection_authority(root, inherited)
    candidate_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v3_shadow.yaml").read_text()
        ),
        "candidate_config",
    )
    label_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text()
        ),
        "label_config",
    )
    raw_opportunities, raw_executions, source = build_datasets(
        root, inherited, candidate_config, label_config
    )
    opportunities, executions, setup_inventory = _augment_entry_records(
        raw_opportunities, raw_executions, inherited
    )
    for row in executions:
        row["observed_time_to_profit_fraction"] = time_to_profit_fraction(
            row, horizon_bars=int(row["horizon_bars"])
        )
    v4_report = json.loads(v4_evidence_path.read_text())
    times = sorted({row["timestamp"] for row in opportunities})
    boundaries = _fold_boundaries(times)
    folds = [
        _evaluate_fold(
            executions,
            boundary,
            index + 1,
            config,
            v4_report["tournament"]["entry_folds"][index],
        )
        for index, boundary in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    worst = min(
        (
            float(fold["metrics"]["selected_protected_worst_net"])
            for fold in evaluated
            if fold["metrics"]["selected_protected_worst_net"] is not None
        ),
        default=-1.0,
    )
    entry_pass = bool(
        len(evaluated) == 4
        and passing >= int(config["validation"]["minimum_positive_folds"])
        and worst >= 0.0
    )
    leave_one_symbol_out = (
        _leave_one_symbol_out(executions, boundaries[-1], config)
        if entry_pass
        else {"status": "NOT_RUN_PRIMARY_ENTRY_GATE_FAILED", "passed": False}
    )
    prospective = _prospective_inventory(
        root / str(config["prospective_microstructure"]["database"]), config
    )
    exit_pass = bool(v4_report["tournament"]["exit_primary_pass"])
    validation_pass = bool(
        entry_pass
        and exit_pass
        and leave_one_symbol_out["passed"]
        and prospective["status"] == "READY"
    )
    report = {
        "schema_id": "aegis-long-entry-v5-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "setup_inventory": setup_inventory,
        "feature_names": [
            *v4_report["entry_feature_names"][: -len(SETUP_FEATURE_NAMES)],
            *SETUP_FEATURE_NAMES,
        ],
        "multiobjective_targets": [
            "target_before_stop",
            "protected_worst_net_return",
            "mae_fraction_q90",
            "time_to_profit_fraction",
        ],
        "prospective_microstructure": prospective,
        "frozen_v4_control": {
            "path": str(v4_evidence_path.relative_to(root)),
            "sha256": sha256_file(v4_evidence_path),
            "entry_passing_folds": v4_report["tournament"]["entry_passing_folds"],
            "exit_passing_folds": v4_report["tournament"]["exit_passing_folds"],
        },
        "validation": {
            "folds": folds,
            "passing_folds": passing,
            "entry_primary_pass": entry_pass,
            "exit_primary_pass": exit_pass,
            "leave_one_symbol_out": leave_one_symbol_out,
            "prospective_data_ready": prospective["status"] == "READY",
            "validation_pass": validation_pass,
            "verdict": (
                "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_REVIEW"
                if validation_pass
                else "RESEARCH_ONLY_NOT_PROMOTABLE"
            ),
        },
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
    print(json.dumps({"output": str(output), "verdict": report["validation"]["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
