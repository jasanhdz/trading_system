#!/usr/bin/env python3
"""Run the preregistered LONG v4 technical/hybrid/exit Shadow tournament."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.long_entry_v3_shadow import HardNegativeType
from aegis.research.long_entry_v4_hybrid_shadow import (
    EXIT_STATE_FEATURE_NAMES,
    LongTechnicalSetup,
    clean_entry_label,
    exit_now_preferred_label,
    exit_state_feature_vector,
    hybrid_score,
    technical_setup,
)
from aegis.utils import sha256_file
from train_long_entry_v21_shadow import _fold_boundaries, _selection_metrics, _source_series
from train_long_entry_v31_shadow import (
    _derive_policy,
    _fit_binary,
    _group_attribution,
    _mapping,
    _select,
    _verify_protection_authority,
    build_datasets,
)


SETUPS = tuple(LongTechnicalSetup)
SETUP_FEATURE_NAMES = tuple(f"technical_setup_{setup.value}" for setup in SETUPS)


def _augment_entry_records(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    inherited: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    horizon_config = inherited["opportunity_stage"]
    mapped_opportunities = []
    mapped_executions = []
    excluded: Counter[str] = Counter()
    for row in opportunities:
        setup = technical_setup(str(row["candidate_family"]))
        if setup is None:
            excluded[str(row["candidate_family"])] += 1
            continue
        one_hot = tuple(1.0 if candidate is setup else 0.0 for candidate in SETUPS)
        mapped_opportunities.append(
            {
                **row,
                "setup_family": setup.value,
                "features": (*row["features"], *one_hot),
            }
        )
    for row in executions:
        setup = technical_setup(str(row["candidate_family"]))
        if setup is None:
            continue
        one_hot = tuple(1.0 if candidate is setup else 0.0 for candidate in SETUPS)
        horizon = int(
            horizon_config["family_horizon_bars"][str(row["candidate_family"])]
        )
        mapped_executions.append(
            {
                **row,
                "setup_family": setup.value,
                "features": (*row["features"], *one_hot),
                "opportunity_features": (*row["opportunity_features"], *one_hot),
                "clean_entry": clean_entry_label(row, horizon_bars=horizon),
                "horizon_bars": horizon,
            }
        )
    return mapped_opportunities, mapped_executions, {
        "opportunities": len(mapped_opportunities),
        "executions": len(mapped_executions),
        "excluded_candidates": dict(sorted(excluded.items())),
        "setup_counts": dict(
            sorted(Counter(row["setup_family"] for row in mapped_executions).items())
        ),
        "clean_entry_count": sum(bool(row["clean_entry"]) for row in mapped_executions),
    }


def _fit_hybrid(
    opportunity_train: Sequence[Mapping[str, Any]],
    opportunity_calibration: Sequence[Mapping[str, Any]],
    execution_train: Sequence[Mapping[str, Any]],
    execution_calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, tuple[Any, Any]] | None:
    weight = float(config["hybrid_meta_model"]["hard_negative_weight"])
    fits = {
        "opportunity_probability": _fit_binary(
            opportunity_train,
            opportunity_calibration,
            target="target_before_stop",
            feature_field="features",
            seed=seed,
            negative_weight=weight,
        ),
        "clean_entry_probability": _fit_binary(
            execution_train,
            execution_calibration,
            target="clean_entry",
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
    return {name: value for name, value in fits.items() if value is not None}


def _predict_hybrid(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, tuple[Any, Any]]
) -> list[dict[str, Any]]:
    probabilities = {}
    for output, (model, calibrator) in bundle.items():
        field = "opportunity_features" if output == "opportunity_probability" else "features"
        raw = model.predict_proba(
            np.asarray([row[field] for row in rows], dtype=np.float64)
        )[:, 1]
        probabilities[output] = np.asarray(
            [calibrator.apply(float(value)) for value in raw]
        )
    result = []
    for index, row in enumerate(rows):
        opportunity = float(probabilities["opportunity_probability"][index])
        clean = float(probabilities["clean_entry_probability"][index])
        falling = float(probabilities["falling_knife_probability"][index])
        risk = float(probabilities["path_risk_probability"][index])
        result.append(
            {
                **row,
                "opportunity_probability": opportunity,
                "timing_probability": clean,
                "clean_entry_probability": clean,
                "falling_knife_probability": falling,
                "path_risk_probability": risk,
                "committee_score": hybrid_score(opportunity, clean, falling, risk),
            }
        )
    return result


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
    test = [
        row
        for row in executions
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
        and (excluded_symbol is None or row["symbol"] == excluded_symbol)
    ]
    return (
        opportunity_train,
        opportunity_calibration,
        execution_train,
        execution_calibration,
        test,
    )


def _specialist_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    targets = {
        "opportunity_probability": "opportunity_target_before_stop",
        "clean_entry_probability": "clean_entry",
        "falling_knife_probability": "falling_knife",
        "path_risk_probability": "catastrophic_path",
    }
    return {
        output: {
            "prevalence": float(np.mean([bool(row[target]) for row in rows])),
            "average_precision": float(
                average_precision_score(
                    [bool(row[target]) for row in rows],
                    [float(row[output]) for row in rows],
                )
            ),
        }
        for output, target in targets.items()
    }


def _evaluate_entry_fold(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    opportunity_train, opportunity_calibration, train, calibration_rows, test = _split(
        opportunities, executions, boundaries, config
    )
    bundle = _fit_hybrid(
        opportunity_train,
        opportunity_calibration,
        train,
        calibration_rows,
        config,
        20266000 + fold_id * 100,
    )
    technical_metrics = _selection_metrics(test, np.ones(len(test), dtype=bool))
    if bundle is None or len(test) < 100:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "test_rows": len(test),
            "technical_metrics": technical_metrics,
            "passed": False,
        }
    calibration = _predict_hybrid(calibration_rows, bundle)
    predicted = _predict_hybrid(test, bundle)
    policy = _derive_policy(calibration, config)
    selected = _select(predicted, policy)
    hybrid_metrics = _selection_metrics(predicted, selected)
    minimum = int(config["validation"]["minimum_scoring_selections_per_fold"])
    passed = bool(
        policy["valid"]
        and hybrid_metrics["selected_rows"] >= minimum
        and hybrid_metrics["selected_protected_worst_net"] is not None
        and hybrid_metrics["selected_protected_worst_net"] > 0.0
        and hybrid_metrics["selected_protected_worst_net"]
        > technical_metrics["selected_protected_worst_net"]
        and hybrid_metrics["selected_mae"] < technical_metrics["selected_mae"]
        and hybrid_metrics["selected_underwater_bars"]
        < technical_metrics["selected_underwater_bars"]
        and hybrid_metrics["p95_gap_hours"] is not None
        and hybrid_metrics["p95_gap_hours"]
        <= float(config["ranking"]["maximum_p95_gap_hours"])
    )
    setup_metrics = {}
    for setup in SETUPS:
        indices = [
            index for index, row in enumerate(predicted) if row["setup_family"] == setup.value
        ]
        setup_metrics[setup.value] = _selection_metrics(
            [predicted[index] for index in indices], selected[indices]
        )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "test_rows": len(test),
        "policy": policy,
        "technical_metrics": technical_metrics,
        "hybrid_metrics": hybrid_metrics,
        "specialist_metrics": _specialist_metrics(predicted),
        "setup_metrics": setup_metrics,
        "passed": passed,
    }


def _exit_rows(
    root: Path,
    executions: Sequence[Mapping[str, Any]],
    inherited: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    source = inherited["source"]
    sampling = inherited["sampling"]
    candles, common, inventory = _source_series(
        root / str(source["base_database"]),
        root / str(source["public_candle_delta"]),
        lookback_days=int(source["lookback_days"]),
        history_bars=int(sampling["history_bars"]),
        horizon_bars=int(sampling["maximum_horizon_bars"]) + 1,
    )
    index_by_close = {
        candles[CANONICAL_SYMBOLS[0]][index].close_time: index
        for index in range(len(common))
    }
    checkpoints = [int(value) for value in config["exit_shadow_model"]["checkpoints_bars"]]
    cost = float(inherited["typescript_protection"]["round_trip_cost_fraction"])
    rows = []
    skipped_after_protection = 0
    for execution in executions:
        index = index_by_close.get(execution["timestamp"])
        if index is None:
            continue
        horizon = int(execution["horizon_bars"])
        future = candles[str(execution["symbol"])][index + 1 : index + 1 + horizon]
        alive_bars = min(
            int(result["bars_held"])
            for result in execution["protection_results"].values()
        )
        for checkpoint in checkpoints:
            if checkpoint >= alive_bars or checkpoint >= len(future):
                skipped_after_protection += 1
                continue
            observed = future[:checkpoint]
            state = exit_state_feature_vector(
                entry_price=float(execution["entry_price"]),
                observed=observed,
                horizon_bars=horizon,
                atr_fraction=float(execution["atr_fraction"]),
                round_trip_cost_fraction=cost,
            )
            current_net = float(state[1])
            continue_net = float(execution["protected_worst_net_return"])
            rows.append(
                {
                    "timestamp": observed[-1].close_time,
                    "symbol": execution["symbol"],
                    "setup_family": execution["setup_family"],
                    "checkpoint_bars": checkpoint,
                    "features": (*execution["features"], *state),
                    "hard_negative": execution["hard_negative"],
                    "exit_now_preferred": exit_now_preferred_label(
                        current_net_return=current_net,
                        continue_worst_protected_net=continue_net,
                    ),
                    "exit_advantage": current_net - continue_net,
                }
            )
    return rows, {
        **inventory,
        "rows": len(rows),
        "checkpoints": dict(sorted(Counter(row["checkpoint_bars"] for row in rows).items())),
        "target_prevalence": (
            float(np.mean([row["exit_now_preferred"] for row in rows])) if rows else None
        ),
        "skipped_at_or_after_protection_exit": skipped_after_protection,
    }


def _fit_exit(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[Any, Any] | None:
    return _fit_binary(
        train,
        calibration,
        target="exit_now_preferred",
        feature_field="features",
        seed=seed,
        negative_weight=1.0,
    )


def _exit_probability(bundle: tuple[Any, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    model, calibrator = bundle
    raw = model.predict_proba(
        np.asarray([row["features"] for row in rows], dtype=np.float64)
    )[:, 1]
    return np.asarray([calibrator.apply(float(value)) for value in raw])


def _exit_policy(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    choices = []
    minimum = int(config["exit_shadow_model"]["minimum_calibration_selections"])
    for quantile in config["exit_shadow_model"]["probability_quantiles"]:
        threshold = float(np.quantile(probabilities, float(quantile), method="higher"))
        selected = probabilities >= threshold
        advantages = [
            float(row["exit_advantage"]) for row, keep in zip(rows, selected) if keep
        ]
        choices.append(
            {
                "quantile": float(quantile),
                "minimum_probability": threshold,
                "selected_rows": len(advantages),
                "average_exit_advantage": float(np.mean(advantages)) if advantages else None,
                "valid": len(advantages) >= minimum and float(np.mean(advantages)) > 0.0,
            }
        )
    valid = [choice for choice in choices if choice["valid"]]
    return max(
        valid or choices,
        key=lambda choice: (
            float(choice["average_exit_advantage"] or -1.0),
            int(choice["selected_rows"]),
        ),
    )


def _evaluate_exit_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    train = [row for row in rows if row["timestamp"] <= train_end]
    calibration = [
        row for row in rows if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    test = [
        row for row in rows if calibration_end + embargo < row["timestamp"] <= test_end
    ]
    bundle = _fit_exit(train, calibration, 20267000 + fold_id)
    minimum_rows = int(config["exit_shadow_model"]["minimum_test_rows"])
    if bundle is None or len(test) < minimum_rows:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "test_rows": len(test),
            "passed": False,
        }
    calibration_probability = _exit_probability(bundle, calibration)
    test_probability = _exit_probability(bundle, test)
    policy = _exit_policy(calibration, calibration_probability, config)
    selected = test_probability >= float(policy["minimum_probability"])
    advantages = [
        float(row["exit_advantage"]) for row, keep in zip(test, selected) if keep
    ]
    truth = np.asarray([bool(row["exit_now_preferred"]) for row in test], dtype=int)
    prevalence = float(np.mean(truth))
    average_precision = float(average_precision_score(truth, test_probability))
    lift = average_precision - prevalence
    passed = bool(
        policy["valid"]
        and len(advantages) >= int(config["exit_shadow_model"]["minimum_test_selections"])
        and float(np.mean(advantages)) > 0.0
        and lift
        >= float(
            config["exit_shadow_model"]["minimum_average_precision_lift_over_prevalence"]
        )
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "test_rows": len(test),
        "policy": policy,
        "target_prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": lift,
        "selected_rows": len(advantages),
        "selected_average_exit_advantage": (
            float(np.mean(advantages)) if advantages else None
        ),
        "passed": passed,
    }


def _setup_retirement(
    folds: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = {}
    for setup in SETUPS:
        rows = [
            fold["setup_metrics"][setup.value]
            for fold in folds
            if fold["status"] == "EVALUATED"
        ]
        positive = sum(
            row["selected_rows"] >= 5
            and row["selected_protected_worst_net"] is not None
            and row["selected_protected_worst_net"] > 0.0
            and row["selected_mae"] < row["baseline_mae"]
            for row in rows
        )
        worst = min(
            (
                float(row["selected_protected_worst_net"])
                for row in rows
                if row["selected_protected_worst_net"] is not None
            ),
            default=-1.0,
        )
        passed = bool(len(rows) == 4 and positive >= 3 and worst >= 0.0)
        result[setup.value] = {
            "positive_folds": positive,
            "worst_fold_protected_net": worst,
            "status": "ELIGIBLE" if passed else "RETIRED",
            "passed": passed,
        }
    return result


def _leave_one_symbol_out(
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
        bundle = _fit_hybrid(
            opportunity_train,
            opportunity_calibration,
            train,
            calibration_rows,
            config,
            20268000 + index * 100,
        )
        if bundle is None or len(test) < 30:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        calibration = _predict_hybrid(calibration_rows, bundle)
        predicted = _predict_hybrid(test, bundle)
        policy = _derive_policy(calibration, config)
        hybrid_metrics = _selection_metrics(predicted, _select(predicted, policy))
        technical_metrics = _selection_metrics(
            predicted, np.ones(len(predicted), dtype=bool)
        )
        reports[symbol] = {
            "status": "EVALUATED",
            "test_rows": len(test),
            "technical_metrics": technical_metrics,
            "hybrid_metrics": hybrid_metrics,
            "generalized_without_regression": bool(
                policy["valid"]
                and hybrid_metrics["selected_rows"] >= 5
                and hybrid_metrics["selected_protected_worst_net"] is not None
                and hybrid_metrics["selected_protected_worst_net"]
                >= technical_metrics["selected_protected_worst_net"]
                and hybrid_metrics["selected_mae"] <= technical_metrics["selected_mae"]
            ),
        }
    evaluated = [row for row in reports.values() if row["status"] == "EVALUATED"]
    passing = sum(bool(row["generalized_without_regression"]) for row in evaluated)
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "method": "HYBRID_REFIT_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def run_tournament(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    exit_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in opportunities})
    boundaries = _fold_boundaries(times)
    entry_folds = [
        _evaluate_entry_fold(opportunities, executions, fold, index + 1, config)
        for index, fold in enumerate(boundaries)
    ]
    exit_folds = [
        _evaluate_exit_fold(exit_rows, fold, index + 1, config)
        for index, fold in enumerate(boundaries)
    ]
    entry_evaluated = [fold for fold in entry_folds if fold["status"] == "EVALUATED"]
    exit_evaluated = [fold for fold in exit_folds if fold["status"] == "EVALUATED"]
    entry_passing = sum(bool(fold["passed"]) for fold in entry_evaluated)
    exit_passing = sum(bool(fold["passed"]) for fold in exit_evaluated)
    retirement = _setup_retirement(entry_folds, config)
    eligible = [name for name, row in retirement.items() if row["passed"]]
    entry_primary = bool(
        len(entry_evaluated) == 4
        and entry_passing >= int(config["validation"]["minimum_positive_folds"])
        and eligible
    )
    exit_primary = len(exit_evaluated) == 4 and exit_passing >= 3
    loso = (
        _leave_one_symbol_out(opportunities, executions, boundaries[-1], config)
        if entry_primary
        else {
            "status": "NOT_RUN_PRIMARY_ENTRY_GATE_FAILED",
            "passed": False,
        }
    )
    validation_pass = entry_primary and exit_primary and bool(loso["passed"])
    return {
        "entry_folds": entry_folds,
        "exit_folds": exit_folds,
        "entry_passing_folds": entry_passing,
        "exit_passing_folds": exit_passing,
        "setup_retirement": retirement,
        "eligible_setups": eligible,
        "entry_primary_pass": entry_primary,
        "exit_primary_pass": exit_primary,
        "leave_one_symbol_out": loso,
        "validation_pass": validation_pass,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_REVIEW"
            if validation_pass
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v4_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v4_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "v4_config")
    if (
        config.get("schema_version") != "aegis-long-entry-v4-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
    ):
        raise SystemExit("AEGIS_LONG_V4_CONFIG_INVALID")
    inherited_path = root / str(config["inherited_entry_contract"]["path"])
    if sha256_file(inherited_path) != str(config["inherited_entry_contract"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V4_INHERITED_CONTRACT_DRIFT")
    control_path = root / str(config["frozen_ml_first_control"]["path"])
    if sha256_file(control_path) != str(config["frozen_ml_first_control"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V4_CONTROL_DRIFT")
    inherited = _mapping(yaml.safe_load(inherited_path.read_text()), "inherited")
    _verify_protection_authority(root, inherited)
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
    raw_opportunities, raw_executions, source = build_datasets(
        root, inherited, v3_config, label_config
    )
    opportunities, executions, setup_inventory = _augment_entry_records(
        raw_opportunities, raw_executions, inherited
    )
    exit_rows, exit_inventory = _exit_rows(root, executions, inherited, config)
    tournament = run_tournament(opportunities, executions, exit_rows, config)
    control = json.loads(control_path.read_text())
    report = {
        "schema_id": "aegis-long-entry-v4-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "setup_inventory": setup_inventory,
        "exit_inventory": exit_inventory,
        "entry_feature_names": [
            *control["feature_names"][:166],
            *SETUP_FEATURE_NAMES,
        ],
        "exit_state_feature_names": list(EXIT_STATE_FEATURE_NAMES),
        "technical_attribution": {
            "by_setup": _group_attribution(executions, lambda row: str(row["setup_family"])),
            "by_symbol": _group_attribution(executions, lambda row: str(row["symbol"])),
        },
        "frozen_ml_first_control": {
            "path": str(control_path.relative_to(root)),
            "sha256": sha256_file(control_path),
            "verdict": control["validation"]["verdict"],
            "passing_folds": control["validation"]["passing_folds"],
        },
        "tournament": tournament,
        "deployment": {
            "selection_effect": "NONE",
            "shadow_runtime_enabled": False,
            "live_enabled": False,
            "exit_runtime_authority": "NONE",
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
    print(json.dumps({"output": str(output), "verdict": tournament["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
