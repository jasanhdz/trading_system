#!/usr/bin/env python3
"""Evaluate all non-empty combinations of the four LONG v5 heads."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aegis.research.long_entry_v51_ablation_shadow import (
    LongV5Head,
    ablation_factors,
    all_head_combinations,
    combination_identity,
)
from aegis.research.long_entry_v5_multiobjective_shadow import time_to_profit_fraction
from aegis.utils import sha256_file
from training.train_long_entry_v21_shadow import _fold_boundaries, _selection_metrics
from training.train_long_entry_v31_shadow import _mapping, build_datasets
from training.train_long_entry_v4_shadow import _augment_entry_records, _verify_protection_authority
from training.train_long_entry_v5_shadow import _fit_bundle, _predict, _split


def _with_combination(
    rows: Sequence[Mapping[str, Any]],
    heads: Sequence[LongV5Head],
    *,
    conservative: bool,
) -> list[dict[str, Any]]:
    return [
        {**row, **ablation_factors(row, heads, conservative=conservative)}
        for row in rows
    ]


def _head_constraints(row: Mapping[str, Any], heads: Sequence[LongV5Head]) -> bool:
    return bool(
        (LongV5Head.NET not in heads or float(row["expected_net_lower_bound"]) > 0.0)
        and (
            LongV5Head.MAE not in heads
            or float(row["mae_upper_bound"])
            <= float(row["adverse_barrier_fraction"])
        )
        and (
            LongV5Head.SPEED not in heads
            or float(row["time_to_profit_upper_bound"]) <= 0.50
        )
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    heads: Sequence[LongV5Head],
    *,
    apply_head_constraints: bool,
) -> np.ndarray:
    eligible: defaultdict[Any, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            (not apply_head_constraints or _head_constraints(row, heads))
            and float(row["committee_score"]) >= float(policy["minimum_score"])
            and float(row["normalized_uncertainty"])
            <= float(policy["maximum_uncertainty"])
        ):
            eligible[row["timestamp"]].append((index, row))
    selected = np.zeros(len(rows), dtype=bool)
    for candidates in eligible.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["committee_score"]),
                float(item[1]["normalized_uncertainty"]),
                str(item[1]["symbol"]),
            ),
        )
        for index, _ in ordered[: int(policy["maximum_selected_per_timestamp"])]:
            selected[index] = True
    return selected


def _policy(
    rows: Sequence[Mapping[str, Any]],
    heads: Sequence[LongV5Head],
    config: Mapping[str, Any],
    *,
    apply_head_constraints: bool,
) -> Mapping[str, Any]:
    ranking = config["ranking"]
    scores = np.asarray([float(row["committee_score"]) for row in rows])
    uncertainty = np.asarray([float(row["normalized_uncertainty"]) for row in rows])
    choices = []
    minimum = int(config["validation"]["minimum_calibration_selections"])
    for score_quantile in ranking["score_quantiles"]:
        score = float(np.quantile(scores, float(score_quantile), method="higher"))
        for uncertainty_quantile in ranking["uncertainty_quantiles"]:
            maximum_uncertainty = float(
                np.quantile(uncertainty, float(uncertainty_quantile), method="lower")
            )
            for top_k in ranking["maximum_selected_per_timestamp_grid"]:
                candidate = {
                    "minimum_score": score,
                    "maximum_uncertainty": maximum_uncertainty,
                    "maximum_selected_per_timestamp": int(top_k),
                    "score_quantile": float(score_quantile),
                    "uncertainty_quantile": float(uncertainty_quantile),
                }
                metrics = _selection_metrics(
                    rows,
                    _select(
                        rows,
                        candidate,
                        heads,
                        apply_head_constraints=apply_head_constraints,
                    ),
                )
                choices.append(
                    {
                        **candidate,
                        "metrics": metrics,
                        "enough_rows": metrics["selected_rows"] >= minimum,
                        "calibration_positive": bool(
                            metrics["selected_protected_worst_net"] is not None
                            and metrics["selected_protected_worst_net"] > 0.0
                        ),
                    }
                )
    enough = [choice for choice in choices if choice["enough_rows"]]
    return max(
        enough or choices,
        key=lambda choice: (
            float(choice["metrics"]["selected_protected_worst_net"] or -1.0),
            -float(choice["metrics"]["selected_mae"] or 1.0),
            int(choice["metrics"]["selected_rows"]),
        ),
    )


def _evaluate_fold(
    rows: Sequence[Mapping[str, Any]],
    boundaries: tuple[Any, Any, Any],
    fold_id: int,
    v5_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train, calibration, test = _split(rows, boundaries, v5_config)
    bundle = _fit_bundle(train, calibration, v5_config, 20271000 + fold_id * 1000)
    if bundle is None or len(test) < 100:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "test_rows": len(test),
            "combinations": {},
        }
    raw_calibration = _predict(calibration, bundle, v5_config)
    raw_test = _predict(test, bundle, v5_config)
    reports = {}
    sensitivity_reports = {}
    for heads in all_head_combinations():
        identity = combination_identity(heads)
        calibration_rows = _with_combination(
            raw_calibration, heads, conservative=True
        )
        test_rows = _with_combination(raw_test, heads, conservative=True)
        policy = _policy(
            calibration_rows, heads, config, apply_head_constraints=True
        )
        selected = _select(
            test_rows, policy, heads, apply_head_constraints=True
        )
        metrics = _selection_metrics(test_rows, selected)
        reports[identity] = {
            "heads": [head.value for head in heads],
            "policy": policy,
            "metrics": metrics,
            "test_evidence_sufficient": metrics["selected_rows"]
            >= int(config["validation"]["minimum_test_selections_for_fold_evidence"]),
        }
        sensitivity_calibration = _with_combination(
            raw_calibration, heads, conservative=False
        )
        sensitivity_test = _with_combination(raw_test, heads, conservative=False)
        sensitivity_policy = _policy(
            sensitivity_calibration,
            heads,
            config,
            apply_head_constraints=False,
        )
        sensitivity_selected = _select(
            sensitivity_test,
            sensitivity_policy,
            heads,
            apply_head_constraints=False,
        )
        sensitivity_metrics = _selection_metrics(
            sensitivity_test, sensitivity_selected
        )
        sensitivity_reports[identity] = {
            "heads": [head.value for head in heads],
            "analysis_class": "EXPLORATORY_POINT_ESTIMATE_DIAGNOSTIC_ONLY",
            "policy": sensitivity_policy,
            "metrics": sensitivity_metrics,
            "test_evidence_sufficient": sensitivity_metrics["selected_rows"]
            >= int(config["validation"]["minimum_test_selections_for_fold_evidence"]),
        }
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "combinations": reports,
        "point_estimate_sensitivity": sensitivity_reports,
    }


def _weighted(entries: Sequence[Mapping[str, Any]], field: str) -> float | None:
    selected = sum(int(row["metrics"]["selected_rows"]) for row in entries)
    if selected == 0:
        return None
    return sum(
        float(row["metrics"][field]) * int(row["metrics"]["selected_rows"])
        for row in entries
        if row["metrics"][field] is not None
    ) / selected


def _aggregate(
    folds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    field: str,
) -> Mapping[str, Any]:
    reports = {}
    for heads in all_head_combinations():
        identity = combination_identity(heads)
        entries = [
            fold[field][identity]
            for fold in folds
            if fold["status"] == "EVALUATED"
        ]
        sufficient = [row for row in entries if row["test_evidence_sufficient"]]
        nets = [
            float(row["metrics"]["selected_protected_worst_net"])
            for row in sufficient
            if row["metrics"]["selected_protected_worst_net"] is not None
        ]
        positive = sum(value > 0.0 for value in nets)
        worst = min(nets, default=None)
        robust = bool(
            len(sufficient) == 4
            and positive >= int(config["validation"]["minimum_positive_folds"])
            and worst is not None
            and worst >= 0.0
        )
        reports[identity] = {
            "heads": [head.value for head in heads],
            "folds_evaluated": len(entries),
            "folds_with_sufficient_test_evidence": len(sufficient),
            "positive_folds": positive,
            "worst_fold_protected_net": worst,
            "total_selected": sum(
                int(row["metrics"]["selected_rows"]) for row in entries
            ),
            "weighted_protected_net": _weighted(
                entries, "selected_protected_worst_net"
            ),
            "weighted_mae": _weighted(entries, "selected_mae"),
            "weighted_underwater_bars": _weighted(
                entries, "selected_underwater_bars"
            ),
            "weighted_target_before_stop": _weighted(
                entries, "selected_target_before_stop"
            ),
            "robust": robust,
        }
    ordered = sorted(
        reports,
        key=lambda identity: (
            reports[identity]["robust"],
            reports[identity]["positive_folds"],
            reports[identity]["worst_fold_protected_net"]
            if reports[identity]["worst_fold_protected_net"] is not None
            else -1.0,
            reports[identity]["weighted_protected_net"]
            if reports[identity]["weighted_protected_net"] is not None
            else -1.0,
            -(
                reports[identity]["weighted_mae"]
                if reports[identity]["weighted_mae"] is not None
                else 1.0
            ),
        ),
        reverse=True,
    )
    return {
        "combinations": reports,
        "ranking": ordered,
        "robust_combinations": [
            identity for identity in ordered if reports[identity]["robust"]
        ],
        "best_exploratory_hypothesis": ordered[0] if ordered else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v51_ablation_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v51_ablation_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "v51_config")
    if (
        config.get("schema_version")
        != "aegis-long-entry-v51-ablation-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
    ):
        raise SystemExit("AEGIS_LONG_V51_CONFIG_INVALID")
    v5_config_path = root / str(config["frozen_v5_contract"]["path"])
    v5_evidence_path = root / str(config["frozen_v5_evidence"]["path"])
    if sha256_file(v5_config_path) != str(config["frozen_v5_contract"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V51_V5_CONTRACT_DRIFT")
    if sha256_file(v5_evidence_path) != str(config["frozen_v5_evidence"]["sha256"]):
        raise SystemExit("AEGIS_LONG_V51_V5_EVIDENCE_DRIFT")
    v5_config = _mapping(yaml.safe_load(v5_config_path.read_text()), "v5_config")
    v4_config_path = root / str(v5_config["frozen_v4_contract"]["path"])
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
    opportunities, executions, inventory = _augment_entry_records(
        raw_opportunities, raw_executions, inherited
    )
    for row in executions:
        row["observed_time_to_profit_fraction"] = time_to_profit_fraction(
            row, horizon_bars=int(row["horizon_bars"])
        )
    boundaries = _fold_boundaries(sorted({row["timestamp"] for row in opportunities}))
    folds = [
        _evaluate_fold(executions, boundary, index + 1, v5_config, config)
        for index, boundary in enumerate(boundaries)
    ]
    aggregate = _aggregate(folds, config, field="combinations")
    point_estimate_sensitivity = _aggregate(
        folds, config, field="point_estimate_sensitivity"
    )
    report = {
        "schema_id": "aegis-long-entry-v51-ablation-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "analysis_class": "EXPLORATORY_NOT_PROMOTION_ELIGIBLE",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "setup_inventory": inventory,
        "combination_count": len(all_head_combinations()),
        "folds": folds,
        "aggregate": aggregate,
        "post_hoc_point_estimate_sensitivity": point_estimate_sensitivity,
        "verdict": (
            "EXPLORATORY_ROBUST_HYPOTHESIS_REQUIRES_NEW_HOLDOUT"
            if aggregate["robust_combinations"]
            else "NO_ROBUST_HEAD_COMBINATION_FOUND"
        ),
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
    print(json.dumps({"output": str(output), "verdict": report["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
