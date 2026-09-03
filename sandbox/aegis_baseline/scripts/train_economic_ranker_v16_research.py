#!/usr/bin/env python3
"""Evaluate V16 pairwise economic ranking against the V15 composite control."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.directional_contract_v15 import contract_indices
from aegis.research.economic_ranker_v16 import pairwise_accuracy, pairwise_examples
from aegis.utils import Sha256HashProvider, sha256_file
from train_directional_contract_v15_research import (
    _evaluate_contract,
    _load,
    _mapping,
    _partition,
    _selection_metrics,
)


def _verify_authority(root: Path, config: Mapping[str, Any]) -> None:
    authority = _mapping(config["authority"], "authority")
    for path_key, hash_key in (
        ("source_dataset", "source_dataset_sha256"),
        ("source_v15_config", "source_v15_config_sha256"),
        ("source_v15_validation", "source_v15_validation_sha256"),
        ("source_v15_report", "source_v15_report_sha256"),
        ("source_v15_verdict", "source_v15_verdict_sha256"),
    ):
        path = root / str(authority[path_key])
        if sha256_file(path) != str(authority[hash_key]):
            raise BarrierResearchError(f"V16 authority mismatch: {path_key}")


def _policy_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        **config,
        "validation": {
            **config["validation"],
            "score_quantiles": config["calibration"]["score_quantiles"],
        },
    }


def _load_v16(path: Path) -> list[dict[str, Any]]:
    base = _load(path)
    underwater: dict[tuple[str, str, str], int] = {}
    regimes: dict[tuple[str, str, str], str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            key = (str(source["timestamp"]), str(source["symbol"]), str(source["side"]))
            underwater[key] = int(source["time_underwater_bars"])
            regimes[key] = str(source["v11_causal_regime"])
    result = []
    for row in base:
        key = (str(row["timestamp"]), str(row["symbol"]), str(row["side"]))
        result.append(
            {
                **row,
                "time_underwater_bars": underwater[key],
                "regime": regimes[key],
            }
        )
    return result


def _fit_ranker(
    train: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, Mapping[str, int]]:
    matrix, labels, inventory = pairwise_examples(train, indices)
    if len(labels) < int(config["validation"]["minimum_training_pairs"]):
        raise BarrierResearchError("V16 has insufficient training pairs")
    settings = config["candidate"]
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=float(settings["alpha"]),
            max_iter=int(settings["maximum_iterations"]),
            tol=float(settings["tolerance"]),
            random_state=seed,
            shuffle=True,
        ),
    )
    model.fit(matrix, labels)
    return model, inventory


def _ranked(
    rows: Sequence[Mapping[str, Any]], model: Any, indices: Sequence[int]
) -> list[dict[str, Any]]:
    matrix = np.asarray(
        [[row["features"][index] for index in indices] for row in rows],
        dtype=np.float32,
    )
    scores = model.decision_function(matrix)
    if not np.isfinite(scores).all():
        raise BarrierResearchError("V16 produced non-finite ranking scores")
    return [{**row, "score": float(score)} for row, score in zip(rows, scores)]


def _group_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        result[value] = _selection_metrics(subset)
    return result


def _rank_selected(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["score"]) >= threshold:
            grouped[str(row["timestamp"])].append(row)
    selected = []
    for timestamp in sorted(grouped):
        selected.append(
            min(
                grouped[timestamp],
                key=lambda row: (-float(row["score"]), str(row["symbol"])),
            )
        )
    return selected


def _rank_policy(
    calibration: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    scores = np.asarray([float(row["score"]) for row in calibration])
    minimum = int(config["validation"]["minimum_selected_per_fold"])
    choices = []
    for quantile in config["calibration"]["score_quantiles"]:
        threshold = float(np.quantile(scores, float(quantile)))
        selected = _rank_selected(calibration, threshold)
        metrics = _selection_metrics(selected)
        utility = metrics["utility"]
        choices.append(
            {
                "quantile": float(quantile),
                "minimum_score": threshold,
                "metrics": metrics,
                "valid": int(utility["count"]) >= minimum
                and utility["mean_utility"] is not None
                and utility["cvar"] is not None,
            }
        )
    valid = [choice for choice in choices if choice["valid"]]
    if not valid:
        raise BarrierResearchError("V16 calibration cannot derive a valid rank policy")
    return max(
        valid,
        key=lambda choice: (
            float(choice["metrics"]["utility"]["mean_utility"]),
            float(choice["metrics"]["utility"]["cvar"]),
            -float(choice["metrics"]["path_quality"]["mean_mae_fraction"]),
        ),
    )


def _evaluate_ranker(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    model, pair_inventory = _fit_ranker(train, indices, config, seed=seed)
    calibration_ranked = _ranked(calibration, model, indices)
    policy = _rank_policy(calibration_ranked, config)
    test_ranked = _ranked(test, model, indices)
    selected = _rank_selected(test_ranked, float(policy["minimum_score"]))
    accuracy = pairwise_accuracy(
        test_ranked, [float(row["score"]) for row in test_ranked]
    )
    return {
        "feature_count": len(indices),
        "training_pairs": pair_inventory,
        "policy": policy,
        "pairwise": accuracy,
        "selected": _selection_metrics(selected),
        "selected_by_regime": _group_metrics(selected, "regime"),
        "selected_by_symbol": _group_metrics(selected, "symbol"),
    }


def _comparison(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> Mapping[str, bool]:
    candidate_utility = candidate["selected"]["utility"]
    control_utility = control["selected"]["utility"]
    candidate_path = candidate["selected"]["path_quality"]
    control_path = control["selected"]["path_quality"]
    return {
        "pairwise_skill": float(candidate["pairwise"]["accuracy"]) > 0.5,
        "positive_mean_utility": candidate_utility["mean_utility"] is not None
        and float(candidate_utility["mean_utility"]) > 0.0,
        "mean_utility": candidate_utility["mean_utility"] is not None
        and control_utility["mean_utility"] is not None
        and float(candidate_utility["mean_utility"])
        > float(control_utility["mean_utility"]),
        "cvar": candidate_utility["cvar"] is not None
        and control_utility["cvar"] is not None
        and float(candidate_utility["cvar"]) >= float(control_utility["cvar"]),
        "positive_rate": candidate_utility["positive_rate"] is not None
        and control_utility["positive_rate"] is not None
        and float(candidate_utility["positive_rate"])
        >= float(control_utility["positive_rate"]),
        "adverse_first": candidate_path["adverse_first_rate"] is not None
        and control_path["adverse_first_rate"] is not None
        and float(candidate_path["adverse_first_rate"])
        <= float(control_path["adverse_first_rate"]),
        "clean_rate": candidate_path["clean_rate"] is not None
        and control_path["clean_rate"] is not None
        and float(candidate_path["clean_rate"]) >= float(control_path["clean_rate"]),
        "mae": candidate_path["mean_mae_fraction"] is not None
        and control_path["mean_mae_fraction"] is not None
        and float(candidate_path["mean_mae_fraction"])
        <= float(control_path["mean_mae_fraction"]),
    }


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
    minimum = int(config["validation"]["minimum_rows_per_partition"])
    policy_config = _policy_config(config)
    sides = {}
    for side_offset, side in enumerate(("LONG", "SHORT")):
        population = [row for row in rows if row["side"] == side]
        indices = contract_indices(v15_config, side)
        folds = []
        for fold in config["validation"]["folds"]:
            train_rows, calibration, test = _partition(
                population, fold, int(config["validation"]["embargo_minutes"])
            )
            if min(len(train_rows), len(calibration), len(test)) < minimum:
                raise BarrierResearchError("V16 partition has insufficient rows")
            fold_id = int(fold["id"])
            control = _evaluate_contract(
                train_rows,
                calibration,
                test,
                indices,
                policy_config,
                seed=2026081600 + side_offset * 100 + fold_id * 10,
            )
            candidate = _evaluate_ranker(
                train_rows,
                calibration,
                test,
                indices,
                policy_config,
                seed=2026081700 + side_offset * 100 + fold_id,
            )
            folds.append(
                {
                    "fold": fold_id,
                    "train_rows": len(train_rows),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "control": control,
                    "candidate": candidate,
                    "comparison": _comparison(candidate, control),
                }
            )
        counts = {
            metric: sum(bool(fold["comparison"][metric]) for fold in folds)
            for metric in folds[0]["comparison"]
        }
        required = int(config["validation"]["required_improvement_folds"])
        retrospective_supported = all(value >= required for value in counts.values())
        sides[side] = {
            "feature_count": len(indices),
            "folds": folds,
            "successful_folds": counts,
            "retrospective_hypothesis_supported": retrospective_supported,
            "promotable": False,
        }
    supported = all(
        report["retrospective_hypothesis_supported"] for report in sides.values()
    )
    return {
        "schema_id": "aegis-economic-ranker-v16-validation-v1",
        "experiment_id": str(config["experiment_id"]),
        "mode": "RESEARCH_ONLY",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "sides": sides,
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
        default=Path("config/experiments/aegis_economic_ranker_v16_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/economic_ranker_v16/validation.json"),
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
