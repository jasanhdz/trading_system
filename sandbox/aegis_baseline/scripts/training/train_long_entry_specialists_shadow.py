#!/usr/bin/env python3
"""Train continuation/reversal LONG specialists with temporal validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from aegis.research.long_entry_specialists_shadow import (
    LONG_SPECIALIST_FEATURE_NAMES,
    LongArchetype,
    LongEntrySpecialistError,
    classify_long_archetype,
    long_specialist_feature_vector,
    mirror_short_path_as_long_outcome,
)
from aegis.training.competition import export_hist_gradient_boosting
from aegis.training.hybrid_directional import calibrator_mapping
from aegis.training.train import calibration_metrics, fit_platt_calibrator
from aegis.utils import Sha256HashProvider, canonical_json, sha256_file


def rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("LONG specialist timestamp is invalid")
    return parsed


def probabilities(model: Any, calibrator: Any, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)[:, 1]
    return np.asarray([calibrator.apply(float(value)) for value in raw])


def selection_metrics(
    records: list[Mapping[str, Any]],
    success_probability: np.ndarray,
    danger_probability: np.ndarray,
    success_threshold: float,
    danger_threshold: float,
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("LONG specialist metrics require rows")
    target = np.asarray([row["clean_fast_success"] for row in records], dtype=int)
    selected = (success_probability >= success_threshold) & (
        danger_probability <= danger_threshold
    )
    selected_rows = [row for row, keep in zip(records, selected) if keep]

    def mean(field: str, values: list[Mapping[str, Any]]) -> float | None:
        return float(np.mean([float(row[field]) for row in values])) if values else None

    ece, brier = calibration_metrics(success_probability, target)
    selected_times = [parse_time(str(row["timestamp"])) for row in selected_rows]
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(selected_times, selected_times[1:])
    ]
    has_both_classes = len(np.unique(target)) == 2
    return {
        "rows": len(records),
        "prevalence": float(np.mean(target)),
        "average_precision": (
            float(average_precision_score(target, success_probability))
            if np.any(target == 1)
            else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(target, success_probability))
            if has_both_classes
            else None
        ),
        "ece": ece,
        "brier": brier,
        "success_threshold": success_threshold,
        "danger_threshold": danger_threshold,
        "selected_rows": len(selected_rows),
        "selected_fraction": float(np.mean(selected)),
        "selected_success_rate": mean("clean_fast_success", selected_rows),
        "selected_danger_rate": mean("dangerous_entry", selected_rows),
        "selected_positive_net_rate": (
            float(
                np.mean(
                    [
                        float(row["net_return_after_costs"]) > 0.0
                        for row in selected_rows
                    ]
                )
            )
            if selected_rows
            else None
        ),
        "selected_mean_net_return_after_costs": mean(
            "net_return_after_costs", selected_rows
        ),
        "selected_mean_mae_fraction": mean("mae_fraction", selected_rows),
        "selected_mean_mfe_fraction": mean("mfe_fraction", selected_rows),
        "all_mean_net_return_after_costs": mean("net_return_after_costs", records),
        "all_mean_mae_fraction": mean("mae_fraction", records),
        "all_mean_mfe_fraction": mean("mfe_fraction", records),
        "maximum_selected_gap_hours": max(gaps) if gaps else None,
    }


def gates(metrics: Mapping[str, Any]) -> Mapping[str, bool]:
    return {
        "minimum_test_rows": int(metrics["rows"]) >= 50,
        "minimum_selected_rows": int(metrics["selected_rows"]) >= 10,
        "ranking_better_than_prevalence": float(metrics["average_precision"])
        > float(metrics["prevalence"]),
        "success_lift_at_least_five_points": metrics["selected_success_rate"]
        is not None
        and float(metrics["selected_success_rate"])
        >= float(metrics["prevalence"]) + 0.05,
        "selected_mae_lower": metrics["selected_mean_mae_fraction"] is not None
        and float(metrics["selected_mean_mae_fraction"])
        < float(metrics["all_mean_mae_fraction"]),
        "selected_net_positive": metrics["selected_mean_net_return_after_costs"]
        is not None
        and float(metrics["selected_mean_net_return_after_costs"]) > 0.0,
        "danger_rate_below_half": metrics["selected_danger_rate"] is not None
        and float(metrics["selected_danger_rate"]) < 0.5,
        "calibration_ece_bounded": float(metrics["ece"]) <= 0.10,
    }


def danger_metrics(
    records: list[Mapping[str, Any]],
    probability: np.ndarray,
    threshold: float,
) -> Mapping[str, Any]:
    target = np.asarray([row["dangerous_entry"] for row in records], dtype=int)
    safe = probability <= threshold
    safe_rows = [row for row, keep in zip(records, safe) if keep]
    ece, brier = calibration_metrics(probability, target)
    prevalence = float(np.mean(target))
    return {
        "rows": len(records),
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": (
            float(roc_auc_score(target, probability))
            if len(np.unique(target)) == 2
            else None
        ),
        "ece": ece,
        "brier": brier,
        "maximum_probability": threshold,
        "safe_rows": len(safe_rows),
        "safe_fraction": float(np.mean(safe)),
        "safe_danger_rate": (
            float(np.mean([row["dangerous_entry"] for row in safe_rows]))
            if safe_rows
            else None
        ),
        "safe_mean_mae_fraction": (
            float(np.mean([row["mae_fraction"] for row in safe_rows]))
            if safe_rows
            else None
        ),
        "all_mean_mae_fraction": float(
            np.mean([row["mae_fraction"] for row in records])
        ),
    }


def fit_classifier(
    train: list[Mapping[str, Any]],
    calibration: list[Mapping[str, Any]],
    target_name: str,
    *,
    seed: int,
) -> tuple[Any, Any]:
    x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
    y_train = np.asarray([row[target_name] for row in train], dtype=int)
    x_calibration = np.asarray(
        [row["features"] for row in calibration], dtype=np.float64
    )
    y_calibration = np.asarray([row[target_name] for row in calibration], dtype=int)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_calibration)) != 2:
        raise ValueError(f"LONG specialist target lacks both classes: {target_name}")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        early_stopping=False,
        random_state=seed,
    ).fit(x_train, y_train)
    calibrator = fit_platt_calibrator(
        model.predict_proba(x_calibration)[:, 1], y_calibration
    )
    return model, calibrator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("data/entry_quality_v3_feedback/feedback_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_specialists_shadow/validation.json"),
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    rejected = Counter()
    for row in rows(args.feedback):
        try:
            feature_values = row["feature_values"]
            if not isinstance(feature_values, Mapping):
                raise LongEntrySpecialistError("features missing")
            feature_vector = long_specialist_feature_vector(feature_values)
            archetype = classify_long_archetype(feature_values)
            outcome = mirror_short_path_as_long_outcome(
                observed=row["observed"], label=row["label"]
            )
            timestamp = str(row["signal_timestamp"])
            parse_time(timestamp)
        except (KeyError, TypeError, LongEntrySpecialistError, ValueError):
            rejected["invalid"] += 1
            continue
        records.append(
            {
                "timestamp": timestamp,
                "symbol": str(row["symbol"]),
                "archetype": archetype["archetype"],
                "non_overlapping_episode": row.get("non_overlapping_episode") is True,
                "features": feature_vector,
                **outcome,
            }
        )
    unique_times = sorted({parse_time(str(row["timestamp"])) for row in records})
    if len(unique_times) < 30:
        raise ValueError("LONG specialists have insufficient temporal coverage")
    train_end = unique_times[max(0, int(len(unique_times) * 0.60) - 1)]
    calibration_end = unique_times[max(1, int(len(unique_times) * 0.80) - 1)]
    calibration_start = train_end + timedelta(minutes=60)
    test_start = calibration_end + timedelta(minutes=60)

    train_all = [
        row for row in records if parse_time(str(row["timestamp"])) <= train_end
    ]
    calibration_all = [
        row
        for row in records
        if calibration_start <= parse_time(str(row["timestamp"])) <= calibration_end
    ]
    test_all = [
        row
        for row in records
        if row["non_overlapping_episode"]
        and parse_time(str(row["timestamp"])) >= test_start
    ]
    danger_model, danger_calibrator = fit_classifier(
        train_all, calibration_all, "dangerous_entry", seed=20260810
    )
    danger_calibration_probability = probabilities(
        danger_model,
        danger_calibrator,
        np.asarray([row["features"] for row in calibration_all]),
    )
    danger_threshold = float(
        np.quantile(danger_calibration_probability, 0.50, method="higher")
    )
    exported_danger = export_hist_gradient_boosting(
        danger_model,
        "long-entry-danger-shared-shadow",
        LONG_SPECIALIST_FEATURE_NAMES,
        classifier=True,
    )

    x_test_all = np.asarray([row["features"] for row in test_all])
    danger_test_probability = probabilities(danger_model, danger_calibrator, x_test_all)
    shared_danger_metrics = danger_metrics(
        test_all, danger_test_probability, danger_threshold
    )
    shared_danger_gates = {
        "minimum_test_rows": int(shared_danger_metrics["rows"]) >= 50,
        "minimum_safe_rows": int(shared_danger_metrics["safe_rows"]) >= 20,
        "ranking_better_than_prevalence": float(
            shared_danger_metrics["average_precision"]
        )
        > float(shared_danger_metrics["prevalence"]),
        "safe_group_has_lower_danger_rate": shared_danger_metrics["safe_danger_rate"]
        is not None
        and float(shared_danger_metrics["safe_danger_rate"])
        < float(shared_danger_metrics["prevalence"]),
        "safe_group_has_lower_mae": shared_danger_metrics["safe_mean_mae_fraction"]
        is not None
        and float(shared_danger_metrics["safe_mean_mae_fraction"])
        < float(shared_danger_metrics["all_mean_mae_fraction"]),
        "calibration_ece_bounded": float(shared_danger_metrics["ece"]) <= 0.10,
    }
    danger_export_probability = np.asarray(
        [danger_calibrator.apply(exported_danger.evaluate(row)) for row in x_test_all]
    )
    danger_parity_error = float(
        np.max(np.abs(danger_export_probability - danger_test_probability))
    )
    if danger_parity_error > 1e-10:
        raise ValueError("LONG danger export parity failed")

    specialists: dict[str, Any] = {}
    overall_pass = all(shared_danger_gates.values())
    for index, archetype in enumerate(
        (LongArchetype.TREND_CONTINUATION, LongArchetype.CONFIRMED_REVERSAL)
    ):
        train = [row for row in train_all if row["archetype"] == archetype.value]
        calibration = [
            row for row in calibration_all if row["archetype"] == archetype.value
        ]
        test = [row for row in test_all if row["archetype"] == archetype.value]
        if min(len(train), len(calibration), len(test)) < 20:
            specialists[archetype.value] = {
                "status": "INSUFFICIENT_ARCHETYPE_EVIDENCE",
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "validation_pass": False,
            }
            overall_pass = False
            continue
        try:
            model, calibrator = fit_classifier(
                train, calibration, "clean_fast_success", seed=20260811 + index
            )
        except ValueError as exc:
            specialists[archetype.value] = {
                "status": "TARGET_CLASS_EVIDENCE_INSUFFICIENT",
                "reason": str(exc),
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "validation_pass": False,
            }
            overall_pass = False
            continue
        x_calibration = np.asarray([row["features"] for row in calibration])
        success_calibration_probability = probabilities(
            model, calibrator, x_calibration
        )
        success_threshold = float(
            np.quantile(success_calibration_probability, 0.80, method="higher")
        )
        x_test = np.asarray([row["features"] for row in test])
        success_test_probability = probabilities(model, calibrator, x_test)
        danger_test_probability = probabilities(danger_model, danger_calibrator, x_test)
        metrics = selection_metrics(
            test,
            success_test_probability,
            danger_test_probability,
            success_threshold,
            danger_threshold,
        )
        midpoint = len(test) // 2
        first_half = selection_metrics(
            test[:midpoint],
            success_test_probability[:midpoint],
            danger_test_probability[:midpoint],
            success_threshold,
            danger_threshold,
        )
        second_half = selection_metrics(
            test[midpoint:],
            success_test_probability[midpoint:],
            danger_test_probability[midpoint:],
            success_threshold,
            danger_threshold,
        )
        validation_gates = dict(gates(metrics))
        validation_gates["both_temporal_halves_select"] = all(
            int(value["selected_rows"]) >= 3 for value in (first_half, second_half)
        )
        validation_gates["both_temporal_halves_positive"] = all(
            value["selected_mean_net_return_after_costs"] is not None
            and float(value["selected_mean_net_return_after_costs"]) > 0.0
            for value in (first_half, second_half)
        )
        validation_pass = all(validation_gates.values())
        overall_pass &= validation_pass
        exported = export_hist_gradient_boosting(
            model,
            f"long-entry-{archetype.value.lower()}-shadow",
            LONG_SPECIALIST_FEATURE_NAMES,
            classifier=True,
        )
        exported_probability = np.asarray(
            [calibrator.apply(exported.evaluate(row)) for row in x_test]
        )
        parity_error = float(
            np.max(np.abs(exported_probability - success_test_probability))
        )
        if parity_error > 1e-10:
            raise ValueError(f"LONG {archetype.value} export parity failed")
        specialists[archetype.value] = {
            "status": "TRAINED_SHADOW_ONLY",
            "model": exported.to_payload(),
            "calibrator": calibrator_mapping(calibrator),
            "success_threshold": success_threshold,
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "test_metrics": metrics,
            "test_first_half": first_half,
            "test_second_half": second_half,
            "gates": validation_gates,
            "validation_pass": validation_pass,
            "export_parity_max_absolute_error": parity_error,
        }

    payload: dict[str, Any] = {
        "schema_id": "aegis-long-entry-specialists-shadow-validation-v1",
        "mode": "SHADOW",
        "feature_names": list(LONG_SPECIALIST_FEATURE_NAMES),
        "feature_count": len(LONG_SPECIALIST_FEATURE_NAMES),
        "archetype_contract": {
            "continuation": LongArchetype.TREND_CONTINUATION.value,
            "reversal": LongArchetype.CONFIRMED_REVERSAL.value,
            "routing": "CAUSAL_RESEARCH_ONLY",
        },
        "outcome_contract": {
            "target": "LONG_FAVORABLE_PEAK_WITHIN_6_BARS_BEFORE_ADVERSE_PEAK",
            "favorable_fraction": 0.003,
            "adverse_fraction": 0.003,
            "fast_bars": 6,
            "round_trip_cost_fraction": 0.001,
            "source": "MIRRORED_SAME_PRICE_PATH_EXTREMA",
        },
        "source": {
            "feedback": str(args.feedback),
            "feedback_sha256": sha256_file(args.feedback),
        },
        "dataset": {
            "rows": len(records),
            "non_overlapping_validation_rows": sum(
                bool(row["non_overlapping_episode"]) for row in records
            ),
            "rejected": dict(rejected),
            "archetype_counts": dict(
                sorted(Counter(str(row["archetype"]) for row in records).items())
            ),
            "train_end": train_end.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "calibration_end": calibration_end.isoformat(),
            "test_start": test_start.isoformat(),
            "embargo_minutes": 60,
            "overlap_policy": (
                "OVERLAPPING_ROWS_TRAIN_AND_CALIBRATION_NON_OVERLAPPING_TEST_ONLY"
            ),
        },
        "shared_danger_model": {
            "model": exported_danger.to_payload(),
            "calibrator": calibrator_mapping(danger_calibrator),
            "maximum_probability": danger_threshold,
            "threshold_semantics": "CALIBRATION_MEDIAN_RESEARCH_ONLY",
            "test_metrics": shared_danger_metrics,
            "gates": shared_danger_gates,
            "validation_pass": all(shared_danger_gates.values()),
            "export_parity_max_absolute_error": danger_parity_error,
        },
        "specialists": specialists,
        "validation_pass": overall_pass,
        "promotion_state": (
            "ELIGIBLE_FOR_ADDITIONAL_SHADOW_OBSERVATION"
            if overall_pass
            else "NOT_PROMOTABLE_REQUIRES_MORE_EVIDENCE_OR_REDESIGN"
        ),
        "live_selection_effect": "NONE",
        "typescript_guards_changed": False,
        "typescript_sizing_changed": False,
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "content_hash": payload["content_hash"],
                "dataset": payload["dataset"],
                "specialists": {
                    name: {
                        key: value.get(key)
                        for key in (
                            "status",
                            "test_metrics",
                            "gates",
                            "validation_pass",
                        )
                    }
                    for name, value in specialists.items()
                },
                "shared_danger_model": {
                    "test_metrics": shared_danger_metrics,
                    "gates": shared_danger_gates,
                    "validation_pass": all(shared_danger_gates.values()),
                },
                "validation_pass": overall_pass,
                "promotion_state": payload["promotion_state"],
                "exchange_authority": False,
                "exchange_mutations": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
