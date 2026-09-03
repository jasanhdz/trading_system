#!/usr/bin/env python3
"""Evaluate the preregistered V17 safety-gated pairwise ranker offline."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.directional_contract_v15 import contract_indices
from aegis.research.economic_ranker_v16 import pairwise_accuracy
from aegis.research.safety_gated_ranker_v17 import (
    gate_survivors,
    gate_thresholds,
    split_calibration,
)
from aegis.utils import Sha256HashProvider, sha256_file
from train_directional_contract_v15_research import (
    _fit_bundle,
    _mapping,
    _partition,
    _policy,
    _predict,
    _predictive_metrics,
    _selected,
    _selection_metrics,
)
from train_economic_ranker_v16_research import (
    _fit_ranker,
    _group_metrics,
    _load_v16,
    _rank_policy,
    _rank_selected,
    _ranked,
)


def _verify_authority(root: Path, config: Mapping[str, Any]) -> None:
    authority = _mapping(config["authority"], "authority")
    for path_key, hash_key in (
        ("source_dataset", "source_dataset_sha256"),
        ("source_v15_config", "source_v15_config_sha256"),
        ("source_v16_config", "source_v16_config_sha256"),
        ("source_v16_validation", "source_v16_validation_sha256"),
        ("source_v16_report", "source_v16_report_sha256"),
        ("source_v16_verdict", "source_v16_verdict_sha256"),
    ):
        path = root / str(authority[path_key])
        if sha256_file(path) != str(authority[hash_key]):
            raise BarrierResearchError(f"V17 authority mismatch: {path_key}")


def _finite_metric(value: Any, *, lower_default: float) -> float:
    return lower_default if value is None else float(value)


def _choose_gate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    calibration = _mapping(config["calibration"], "calibration")
    grid = _mapping(calibration["gate_grid"], "gate_grid")
    minimum_count = int(calibration["minimum_gate_survivors"])
    minimum_rate = float(calibration["minimum_gate_survivor_rate"])
    choices = []
    for clean_q, danger_q, mae_q in product(
        grid["minimum_clean_probability_quantiles"],
        grid["maximum_danger_probability_quantiles"],
        grid["maximum_mae_q90_quantiles"],
    ):
        thresholds = gate_thresholds(
            rows,
            clean_quantile=float(clean_q),
            danger_quantile=float(danger_q),
            mae_quantile=float(mae_q),
        )
        survivors = gate_survivors(rows, thresholds)
        metrics = _selection_metrics(survivors)
        rate = len(survivors) / len(rows)
        choices.append(
            {
                "quantiles": {
                    "minimum_clean_probability": float(clean_q),
                    "maximum_danger_probability": float(danger_q),
                    "maximum_mae_q90": float(mae_q),
                },
                "thresholds": thresholds,
                "survivors": len(survivors),
                "survivor_rate": rate,
                "metrics": metrics,
                "valid": len(survivors) >= minimum_count and rate >= minimum_rate,
            }
        )
    valid = [choice for choice in choices if choice["valid"]]
    if not valid:
        raise BarrierResearchError("V17 calibration cannot derive a valid gate")
    selected = max(
        valid,
        key=lambda choice: (
            -_finite_metric(
                choice["metrics"]["path_quality"]["adverse_first_rate"],
                lower_default=1.0,
            ),
            -_finite_metric(
                choice["metrics"]["path_quality"]["mean_mae_fraction"],
                lower_default=float("inf"),
            ),
            _finite_metric(
                choice["metrics"]["utility"]["mean_utility"],
                lower_default=float("-inf"),
            ),
            float(choice["survivor_rate"]),
        ),
    )
    return {**selected, "evaluated_policies": len(choices)}


def _choose_rank_policy(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    calibration = _mapping(config["calibration"], "calibration")
    rank_config = {
        "calibration": {"score_quantiles": calibration["rank_score_quantiles"]},
        "validation": {
            "minimum_selected_per_fold": calibration["minimum_rank_selected"]
        },
    }
    return _rank_policy(rows, rank_config)


def _control_v15(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    v15_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    policy = _policy(calibration, v15_config)
    selected = _selected(test, float(policy["minimum_score"]))
    return {
        "policy": policy,
        "predictive": _predictive_metrics(test),
        "selected": _selection_metrics(selected),
    }


def _control_v16(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    v16_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    policy = _rank_policy(calibration, v16_config)
    selected = _rank_selected(test, float(policy["minimum_score"]))
    return {"policy": policy, "selected": _selection_metrics(selected)}


def _candidate(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    gate_calibration, rank_calibration = split_calibration(calibration)
    gate = _choose_gate(gate_calibration, config)
    rank_survivors = gate_survivors(rank_calibration, gate["thresholds"])
    test_survivors = gate_survivors(test, gate["thresholds"])
    try:
        rank_policy = _choose_rank_policy(rank_survivors, config)
    except BarrierResearchError:
        return {
            "status": "CALIBRATION_INFEASIBLE",
            "gate_calibration_rows": len(gate_calibration),
            "rank_calibration_rows": len(rank_calibration),
            "gate": gate,
            "rank_calibration_survivors": len(rank_survivors),
            "rank_policy": None,
            "test_survivors": len(test_survivors),
            "test_survivor_rate": len(test_survivors) / len(test),
            "selected": _selection_metrics([]),
            "selected_by_regime": {},
            "selected_by_symbol": {},
        }
    selected = _rank_selected(test_survivors, float(rank_policy["minimum_score"]))
    return {
        "status": "EVALUATED",
        "gate_calibration_rows": len(gate_calibration),
        "rank_calibration_rows": len(rank_calibration),
        "gate": gate,
        "rank_calibration_survivors": len(rank_survivors),
        "rank_policy": rank_policy,
        "test_survivors": len(test_survivors),
        "test_survivor_rate": len(test_survivors) / len(test),
        "selected": _selection_metrics(selected),
        "selected_by_regime": _group_metrics(selected, "regime"),
        "selected_by_symbol": _group_metrics(selected, "symbol"),
    }


def _improvements(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    direct_ranker: bool,
    config: Mapping[str, Any],
) -> Mapping[str, bool]:
    candidate_selected = candidate["selected"]
    control_selected = control["selected"]
    cu = candidate_selected["utility"]
    co = control_selected["utility"]
    cp = candidate_selected["path_quality"]
    op = control_selected["path_quality"]
    result = {
        "mean_utility": cu["mean_utility"] is not None
        and co["mean_utility"] is not None
        and float(cu["mean_utility"]) > float(co["mean_utility"]),
        "cvar": cu["cvar"] is not None
        and co["cvar"] is not None
        and float(cu["cvar"]) >= float(co["cvar"]),
        "positive_rate": cu["positive_rate"] is not None
        and co["positive_rate"] is not None
        and float(cu["positive_rate"]) >= float(co["positive_rate"]),
        "adverse_first_rate": cp["adverse_first_rate"] is not None
        and op["adverse_first_rate"] is not None
        and float(cp["adverse_first_rate"]) <= float(op["adverse_first_rate"]),
        "mean_mae_fraction": cp["mean_mae_fraction"] is not None
        and op["mean_mae_fraction"] is not None
        and float(cp["mean_mae_fraction"]) <= float(op["mean_mae_fraction"]),
        "clean_rate": cp["clean_rate"] is not None
        and op["clean_rate"] is not None
        and float(cp["clean_rate"]) >= float(op["clean_rate"]),
    }
    if direct_ranker:
        validation = _mapping(config["validation"], "validation")
        candidate_count = int(cu["count"])
        control_count = int(co["count"])
        candidate_gap = cu["p95_gap_hours"]
        control_gap = co["p95_gap_hours"]
        result.update(
            {
                "selection_count": control_count > 0
                and candidate_count / control_count
                >= float(validation["minimum_selection_count_ratio_vs_v16"]),
                "opportunity_gap": candidate_gap is not None
                and control_gap is not None
                and float(candidate_gap)
                <= float(control_gap)
                * float(validation["maximum_p95_gap_ratio_vs_v16"]),
            }
        )
    return result


def train(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    _verify_authority(root, config)
    dataset = root / str(config["authority"]["source_dataset"])
    rows = _load_v16(dataset)
    v15_config = _mapping(
        yaml.safe_load(
            (root / str(config["authority"]["source_v15_config"])).read_text()
        ),
        "v15_config",
    )
    v16_config = _mapping(
        yaml.safe_load(
            (root / str(config["authority"]["source_v16_config"])).read_text()
        ),
        "v16_config",
    )
    minimum = int(config["validation"]["minimum_rows_per_partition"])
    reports = {}
    for side_offset, side in enumerate(("LONG", "SHORT")):
        population = [row for row in rows if row["side"] == side]
        indices = contract_indices(v15_config, side)
        folds = []
        for fold in config["validation"]["folds"]:
            train_rows, calibration, test = _partition(
                population, fold, int(config["validation"]["embargo_minutes"])
            )
            if min(len(train_rows), len(calibration), len(test)) < minimum:
                raise BarrierResearchError("V17 partition has insufficient rows")
            fold_id = int(fold["id"])
            safety = _fit_bundle(
                train_rows,
                indices,
                seed=2026081800 + side_offset * 100 + fold_id * 10,
            )
            calibration_predictions = _predict(calibration, safety)
            test_predictions = _predict(test, safety)
            ranker, pair_inventory = _fit_ranker(
                train_rows,
                indices,
                v16_config,
                seed=2026081900 + side_offset * 100 + fold_id,
            )
            calibration_ranked = _ranked(calibration_predictions, ranker, indices)
            test_ranked = _ranked(test_predictions, ranker, indices)
            control_v15 = _control_v15(
                calibration_predictions, test_predictions, v15_config
            )
            control_v16 = _control_v16(calibration_ranked, test_ranked, v16_config)
            candidate = _candidate(calibration_ranked, test_ranked, config)
            candidate["pairwise"] = pairwise_accuracy(
                test_ranked, [float(row["score"]) for row in test_ranked]
            )
            candidate["training_pairs"] = pair_inventory
            versus_v15 = _improvements(
                candidate, control_v15, direct_ranker=False, config=config
            )
            versus_v16 = _improvements(
                candidate, control_v16, direct_ranker=True, config=config
            )
            positive = (
                candidate["selected"]["utility"]["mean_utility"] is not None
                and float(candidate["selected"]["utility"]["mean_utility"]) > 0.0
            )
            folds.append(
                {
                    "fold": fold_id,
                    "train_rows": len(train_rows),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "control_v15": control_v15,
                    "control_v16": control_v16,
                    "candidate": candidate,
                    "positive_mean_utility": positive,
                    "versus_v15": versus_v15,
                    "versus_v16": versus_v16,
                    "fold_passed": positive
                    and all(versus_v15.values())
                    and all(versus_v16.values()),
                }
            )
        successful = sum(bool(fold["fold_passed"]) for fold in folds)
        reports[side] = {
            "feature_count": len(indices),
            "folds": folds,
            "successful_folds": successful,
            "retrospective_hypothesis_supported": successful
            >= int(config["validation"]["required_improvement_folds"]),
            "promotable": False,
        }
    supported = all(
        report["retrospective_hypothesis_supported"] for report in reports.values()
    )
    return {
        "schema_id": "aegis-safety-gated-ranker-v17-validation-v1",
        "experiment_id": str(config["experiment_id"]),
        "mode": "RESEARCH_ONLY",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "sides": reports,
        "retrospective_hypothesis_supported": supported,
        "future_holdout_required": True,
        "candidate_passed_for_promotion": False,
        "verdict": (
            "RESEARCH_HYPOTHESIS_SUPPORTED_FUTURE_HOLDOUT_REQUIRED"
            if supported
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "model_exported": False,
        "shadow_changed": False,
        "live_changed": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_safety_gated_ranker_v17_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/safety_gated_ranker_v17/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    result = train(root, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "output": str(output),
                "retrospective_hypothesis_supported": result[
                    "retrospective_hypothesis_supported"
                ],
                "verdict": result["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
