#!/usr/bin/env python3
"""Audit V14 feature information with purged walk-forward comparisons."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.competing_barrier_v10 import BarrierResearchError
from aegis.research.decomposed_entry_v9 import V9_FEATURE_NAMES
from aegis.research.feature_information_v14 import (
    TAKER_FLOW_FEATURE_NAMES,
    binary_probability_metrics,
    feature_families,
    positional_feature_names,
    quality_profile,
    quantile_pinball_loss,
    robust_shift,
)
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _mapping


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = _mapping(json.loads(line), f"dataset:{line_number}")
            outcome = str(
                _mapping(source["v10_contract_outcomes"], "contracts")["ROE_10_H12"][
                    "outcome"
                ]
            )
            base = tuple(float(value) for value in source["v9_features"])
            flow = tuple(float(value) for value in source["v14_taker_flow_features"])
            if len(base) != len(V9_FEATURE_NAMES) or len(flow) != len(
                TAKER_FLOW_FEATURE_NAMES
            ):
                raise BarrierResearchError("V14 feature width mismatch")
            values = (*base, *flow)
            if not all(math.isfinite(value) for value in values):
                raise BarrierResearchError("V14 encountered non-finite features")
            rows.append(
                {
                    "timestamp": _time(str(source["timestamp"])),
                    "symbol": str(source["symbol"]),
                    "side": str(source["side"]),
                    "independent": bool(source["independent"]),
                    "base": base,
                    "candidate": values,
                    "danger": outcome in {"ADVERSE_FIRST", "SAME_BAR_AMBIGUOUS"},
                    "clean": bool(source["v11_clean_entry_label"]),
                    "mae": float(source["mae_fraction"]),
                }
            )
    return sorted(rows, key=lambda row: (row["timestamp"], row["symbol"], row["side"]))


def _folds(times: Sequence[datetime]) -> list[tuple[datetime, datetime]]:
    fractions = ((0.45, 0.575), (0.575, 0.70), (0.70, 0.825), (0.825, 1.0))

    def at(value: float) -> datetime:
        return times[min(len(times) - 1, int((len(times) - 1) * value))]

    return [(at(train), at(test)) for train, test in fractions]


def _split(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime],
    *,
    embargo_minutes: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train_end, test_end = boundaries
    test_start = train_end + timedelta(minutes=embargo_minutes)
    return (
        [row for row in rows if row["timestamp"] <= train_end],
        [
            row
            for row in rows
            if test_start < row["timestamp"] <= test_end and row["independent"]
        ],
    )


def _classifier(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=250,
            solver="liblinear",
            random_state=seed,
        ),
    )


def _binary(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    target: str,
    *,
    seed: int,
) -> Mapping[str, Any]:
    x_train = np.asarray(
        [[row["candidate"][index] for index in indices] for row in train],
        dtype=np.float64,
    )
    x_test = np.asarray(
        [[row["candidate"][index] for index in indices] for row in test],
        dtype=np.float64,
    )
    y_train = np.asarray([bool(row[target]) for row in train], dtype=np.int8)
    y_test = np.asarray([bool(row[target]) for row in test], dtype=np.int8)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise BarrierResearchError(f"V14 {target} lacks both classes")
    model = _classifier(seed).fit(x_train, y_train)
    return binary_probability_metrics(y_test, model.predict_proba(x_test)[:, 1])


def _mae(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    seed: int,
) -> Mapping[str, Any]:
    x_train = np.asarray(
        [[row["candidate"][index] for index in indices] for row in train],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [[row["candidate"][index] for index in indices] for row in test],
        dtype=np.float32,
    )
    y_train = np.asarray([float(row["mae"]) for row in train], dtype=np.float64)
    y_test = np.asarray([float(row["mae"]) for row in test], dtype=np.float64)
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.9,
        learning_rate=0.05,
        max_iter=50,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=4.0,
        early_stopping=False,
        random_state=seed,
    ).fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "rows": len(test),
        "q90_pinball_loss": quantile_pinball_loss(y_test, predicted, quantile=0.9),
        "actual_mae_mean": float(np.mean(y_test)),
        "predicted_q90_mean": float(np.mean(predicted)),
    }


def _improvements(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    return {
        "log_loss": sum(
            float(right["log_loss"]) < float(left["log_loss"])
            for left, right in zip(baseline, candidate)
        ),
        "average_precision": sum(
            float(right["average_precision"]) > float(left["average_precision"])
            for left, right in zip(baseline, candidate)
        ),
    }


def _quality(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    source_names = (*V9_FEATURE_NAMES, *TAKER_FLOW_FEATURE_NAMES)
    names = positional_feature_names(source_names)
    matrix = np.asarray([row["candidate"] for row in rows], dtype=np.float64)
    inventory = _mapping(config["inventory"], "inventory")
    profile = dict(
        quality_profile(
            matrix,
            names,
            near_constant_std=float(inventory["near_constant_standard_deviation"]),
        )
    )
    split = len(matrix) // 4
    shifts = robust_shift(matrix[:split], matrix[-split:], names)
    correlation = np.corrcoef(matrix, rowvar=False)
    pairs = []
    threshold = float(inventory["duplicate_correlation_absolute"])
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = float(correlation[left, right])
            if math.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {"left": names[left], "right": names[right], "correlation": value}
                )
    profile.update(
        {
            "robust_shift_warning_threshold": float(
                inventory["drift_robust_shift_warning"]
            ),
            "robust_shift_warning_features": {
                name: value
                for name, value in shifts.items()
                if value >= float(inventory["drift_robust_shift_warning"])
            },
            "near_duplicate_pairs": pairs,
            "duplicate_source_names": sorted(
                {name for name in source_names if source_names.count(name) > 1}
            ),
        }
    )
    return profile


def train(config: Mapping[str, Any], dataset: Path) -> Mapping[str, Any]:
    rows = _load(dataset)
    names = (*V9_FEATURE_NAMES, *TAKER_FLOW_FEATURE_NAMES)
    families = feature_families()
    base_indices = tuple(range(len(V9_FEATURE_NAMES)))
    candidate_indices = tuple(range(len(names)))
    boundaries = _folds(sorted({row["timestamp"] for row in rows}))
    embargo = int(config["validation"]["embargo_minutes"])
    quality = _quality(rows, config)
    side_reports = {}
    for side_index, side in enumerate(("LONG", "SHORT")):
        population = [row for row in rows if row["side"] == side]
        folds = []
        for fold_id, boundary in enumerate(boundaries, start=1):
            train_rows, test_rows = _split(
                population, boundary, embargo_minutes=embargo
            )
            if min(len(train_rows), len(test_rows)) < int(
                config["validation"]["minimum_rows_per_fold"]
            ):
                raise BarrierResearchError("V14 fold has insufficient rows")
            report: dict[str, Any] = {
                "fold": fold_id,
                "train_end": boundary[0].isoformat(),
                "test_end": boundary[1].isoformat(),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "baseline": {},
                "candidate": {},
                "family_ablation_danger": {},
                "family_alone_danger": {},
            }
            for target_index, target in enumerate(("danger", "clean")):
                report["baseline"][target] = _binary(
                    train_rows,
                    test_rows,
                    base_indices,
                    target,
                    seed=2026081400 + side_index * 100 + fold_id * 10 + target_index,
                )
                report["candidate"][target] = _binary(
                    train_rows,
                    test_rows,
                    candidate_indices,
                    target,
                    seed=2026081400 + side_index * 100 + fold_id * 10 + target_index,
                )
            report["baseline"]["mae"] = _mae(
                train_rows,
                test_rows,
                base_indices,
                seed=2026081600 + side_index * 10 + fold_id,
            )
            report["candidate"]["mae"] = _mae(
                train_rows,
                test_rows,
                candidate_indices,
                seed=2026081600 + side_index * 10 + fold_id,
            )
            for family_index, (family, family_names) in enumerate(families.items()):
                family_name_set = set(family_names)
                family_indices = tuple(
                    index
                    for index, name in enumerate(V9_FEATURE_NAMES)
                    if name in family_name_set
                )
                without = tuple(
                    index for index in base_indices if index not in family_indices
                )
                report["family_ablation_danger"][family] = _binary(
                    train_rows,
                    test_rows,
                    without,
                    "danger",
                    seed=2026081800 + side_index * 100 + fold_id * 10 + family_index,
                )
                report["family_alone_danger"][family] = _binary(
                    train_rows,
                    test_rows,
                    family_indices,
                    "danger",
                    seed=2026082000 + side_index * 100 + fold_id * 10 + family_index,
                )
            folds.append(report)
        baseline_danger = [fold["baseline"]["danger"] for fold in folds]
        candidate_danger = [fold["candidate"]["danger"] for fold in folds]
        baseline_clean = [fold["baseline"]["clean"] for fold in folds]
        candidate_clean = [fold["candidate"]["clean"] for fold in folds]
        mae_wins = sum(
            float(fold["candidate"]["mae"]["q90_pinball_loss"])
            < float(fold["baseline"]["mae"]["q90_pinball_loss"])
            for fold in folds
        )
        family_summary = {}
        for family in families:
            removed = [fold["family_ablation_danger"][family] for fold in folds]
            alone = [fold["family_alone_danger"][family] for fold in folds]
            family_summary[family] = {
                "removal_improves_log_loss_folds": sum(
                    float(value["log_loss"]) < float(base["log_loss"])
                    for value, base in zip(removed, baseline_danger)
                ),
                "removal_improves_average_precision_folds": sum(
                    float(value["average_precision"]) > float(base["average_precision"])
                    for value, base in zip(removed, baseline_danger)
                ),
                "alone_mean_log_loss": float(
                    np.mean([value["log_loss"] for value in alone])
                ),
                "alone_mean_average_precision": float(
                    np.mean([value["average_precision"] for value in alone])
                ),
            }
        side_reports[side] = {
            "folds": folds,
            "candidate_danger_improvement_folds": _improvements(
                baseline_danger, candidate_danger
            ),
            "candidate_clean_improvement_folds": _improvements(
                baseline_clean, candidate_clean
            ),
            "candidate_mae_pinball_improvement_folds": mae_wins,
            "family_summary": family_summary,
        }
    required = int(config["validation"]["minimum_candidate_improvement_folds"])
    candidate_pass = all(
        report["candidate_danger_improvement_folds"]["log_loss"] >= required
        and report["candidate_danger_improvement_folds"]["average_precision"]
        >= required
        and report["candidate_clean_improvement_folds"]["log_loss"] >= required
        and report["candidate_mae_pinball_improvement_folds"] >= required
        for report in side_reports.values()
    )
    return {
        "schema_id": "aegis-feature-information-v14-validation-v1",
        "experiment_id": str(config["experiment_id"]),
        "mode": "RESEARCH_ONLY",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "episodes": len(rows) // 2,
        "feature_inventory": {
            "base_schema": "aegis-features-v2",
            "base_contract_features": 83,
            "v9_baseline_features": len(V9_FEATURE_NAMES),
            "candidate_features": len(names),
            "families": {name: list(values) for name, values in families.items()},
            "quality": quality,
        },
        "sides": side_reports,
        "candidate_taker_flow_passed": candidate_pass,
        "verdict": (
            "RESEARCH_CANDIDATE_ADMISSIBLE_NOT_PROMOTED"
            if candidate_pass
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "selection_effect": "NONE",
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
        default=Path("config/experiments/aegis_feature_information_v14_research.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/feature_information_v14/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/feature_information_v14/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    config = _mapping(yaml.safe_load(resolve(args.config).read_text()), "config")
    result = train(config, resolve(args.dataset))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "candidate_taker_flow_passed": result["candidate_taker_flow_passed"],
                "rows": result["rows"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
