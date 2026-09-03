#!/usr/bin/env python3
"""Train and validate side-specific clean-entry path models in Shadow only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from aegis.research.entry_methodology_v2_shadow import (
    ENTRY_PATH_MODEL_FEATURE_NAMES,
    EntryMethodologyV2Policy,
    entry_path_model_features,
    label_clean_entry_path,
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
        raise ValueError("entry path timestamp is invalid")
    return parsed


def calibrated_probabilities(raw: np.ndarray, calibrator: Any) -> np.ndarray:
    return np.asarray(
        [calibrator.apply(float(value)) for value in raw], dtype=np.float64
    )


def outcome_metrics(
    records: list[Mapping[str, Any]], probabilities: np.ndarray, threshold: float
) -> Mapping[str, Any]:
    if not records or len(records) != len(probabilities):
        raise ValueError("entry path metric inputs are invalid")
    labels = np.asarray([bool(row["fast_edge_success"]) for row in records], dtype=int)
    selected = probabilities >= threshold
    selected_records = [row for row, keep in zip(records, selected) if keep]

    def mean(field: str, values: list[Mapping[str, Any]]) -> float | None:
        if not values:
            return None
        return float(np.mean([float(row[field]) for row in values]))

    ece, brier = calibration_metrics(probabilities, labels)
    result: dict[str, Any] = {
        "rows": len(records),
        "prevalence": float(np.mean(labels)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "ece": ece,
        "brier": brier,
        "threshold": threshold,
        "selected_rows": len(selected_records),
        "selected_fraction": float(np.mean(selected)),
        "selected_fast_edge_rate": mean("fast_edge_success", selected_records),
        "selected_clean_path_rate": mean("clean_path_success", selected_records),
        "selected_positive_net_rate": (
            float(
                np.mean(
                    [
                        float(row["net_return_after_costs"]) > 0.0
                        for row in selected_records
                    ]
                )
            )
            if selected_records
            else None
        ),
        "selected_mean_net_return_after_costs": mean(
            "net_return_after_costs", selected_records
        ),
        "selected_mean_mae_fraction": mean("mae_fraction", selected_records),
        "selected_mean_mfe_fraction": mean("mfe_fraction", selected_records),
        "selected_mean_underwater_bars": mean("underwater_bars", selected_records),
        "all_mean_net_return_after_costs": mean("net_return_after_costs", records),
        "all_mean_mae_fraction": mean("mae_fraction", records),
        "all_mean_mfe_fraction": mean("mfe_fraction", records),
        "all_mean_underwater_bars": mean("underwater_bars", records),
    }
    return result


def promotion_gates(metrics: Mapping[str, Any]) -> Mapping[str, bool]:
    selected_fast = metrics.get("selected_fast_edge_rate")
    selected_net = metrics.get("selected_mean_net_return_after_costs")
    selected_mae = metrics.get("selected_mean_mae_fraction")
    return {
        "minimum_test_rows": int(metrics["rows"]) >= 100,
        "minimum_selected_rows": int(metrics["selected_rows"]) >= 20,
        "ranking_better_than_prevalence": float(metrics["average_precision"])
        > float(metrics["prevalence"]),
        "selected_fast_edge_lift": selected_fast is not None
        and float(selected_fast) >= float(metrics["prevalence"]) + 0.03,
        "selected_mae_not_worse": selected_mae is not None
        and float(selected_mae) <= float(metrics["all_mean_mae_fraction"]),
        "selected_net_positive": selected_net is not None and float(selected_net) > 0.0,
        "calibration_ece_bounded": float(metrics["ece"]) <= 0.10,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/hybrid_directional_live_experiment/decisions.jsonl"),
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("data/hybrid_directional_shadow/signals.jsonl"),
    )
    parser.add_argument(
        "--intelligence",
        type=Path,
        default=Path("data/entry_intelligence_shadow/signals.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/entry_methodology_v2_shadow/meta_model_validation.json"),
    )
    args = parser.parse_args()
    policy = EntryMethodologyV2Policy()

    signal_by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows(args.signals):
        signal_by_symbol[str(row["symbol"])].append(row)
    signal_position: dict[tuple[str, str], int] = {}
    for symbol, values in signal_by_symbol.items():
        values.sort(key=lambda row: str(row["market_timestamp"]))
        for index, row in enumerate(values):
            signal_position[(str(row["market_timestamp"]), symbol)] = index

    intelligence = {
        (str(row["market_timestamp"]), str(row["symbol"])): row
        for row in rows(args.intelligence)
    }
    candidates: list[dict[str, Any]] = []
    rejected_incomplete = 0
    for candidate in rows(args.decisions):
        timestamp = str(candidate["market_timestamp"])
        symbol = str(candidate["symbol"])
        side = str(candidate["side"])
        position = signal_position.get((timestamp, symbol))
        context = intelligence.get((timestamp, symbol))
        symbol_signals = signal_by_symbol.get(symbol, [])
        if (
            context is None
            or position is None
            or position + policy.horizon_bars >= len(symbol_signals)
        ):
            rejected_incomplete += 1
            continue
        try:
            features = entry_path_model_features(
                side=side,
                prediction=candidate,
                confirmation=candidate.get("confirmation", {}),
                confirmation_features=candidate.get("confirmation_features", {}),
                entry_intelligence=context,
            )
        except ValueError:
            rejected_incomplete += 1
            continue
        future = symbol_signals[position + 1 : position + 1 + policy.horizon_bars]
        label = label_clean_entry_path(
            side=side,
            entry_price=float(future[0]["market_bar"]["open"]),
            future_bars=[row["market_bar"] for row in future],
            policy=policy,
        )
        candidates.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "features": features,
                **label,
            }
        )

    non_overlapping: list[dict[str, Any]] = []
    last_kept: dict[tuple[str, str], datetime] = {}
    for record in sorted(candidates, key=lambda row: str(row["timestamp"])):
        timestamp = parse_time(str(record["timestamp"]))
        key = (str(record["symbol"]), str(record["side"]))
        if key in last_kept and timestamp - last_kept[key] < timedelta(minutes=60):
            continue
        non_overlapping.append(record)
        last_kept[key] = timestamp
    unique_times = sorted(
        {parse_time(str(row["timestamp"])) for row in non_overlapping}
    )
    if len(unique_times) < 30:
        raise ValueError("entry path meta-model has insufficient temporal coverage")
    train_end = unique_times[max(0, int(len(unique_times) * 0.60) - 1)]
    calibration_end = unique_times[max(1, int(len(unique_times) * 0.80) - 1)]
    calibration_start = train_end + timedelta(minutes=60)
    test_start = calibration_end + timedelta(minutes=60)

    models: dict[str, Any] = {}
    overall_pass = True
    for side in ("LONG", "SHORT"):
        side_rows = [row for row in non_overlapping if row["side"] == side]
        train = [
            row for row in side_rows if parse_time(str(row["timestamp"])) <= train_end
        ]
        calibration = [
            row
            for row in side_rows
            if calibration_start <= parse_time(str(row["timestamp"])) <= calibration_end
        ]
        test = [
            row for row in side_rows if parse_time(str(row["timestamp"])) >= test_start
        ]
        if min(len(train), len(calibration), len(test)) < 40:
            raise ValueError(f"entry path meta-model has insufficient {side} rows")
        x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
        y_train = np.asarray([row["fast_edge_success"] for row in train], dtype=int)
        x_calibration = np.asarray(
            [row["features"] for row in calibration], dtype=np.float64
        )
        y_calibration = np.asarray(
            [row["fast_edge_success"] for row in calibration], dtype=int
        )
        x_test = np.asarray([row["features"] for row in test], dtype=np.float64)
        if len(np.unique(y_train)) != 2 or len(np.unique(y_calibration)) != 2:
            raise ValueError(f"entry path meta-model lacks both {side} classes")
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260809,
        ).fit(x_train, y_train)
        calibration_raw = model.predict_proba(x_calibration)[:, 1]
        calibrator = fit_platt_calibrator(calibration_raw, y_calibration)
        calibration_probabilities = calibrated_probabilities(
            calibration_raw, calibrator
        )
        threshold = float(np.quantile(calibration_probabilities, 0.80, method="higher"))
        test_raw = model.predict_proba(x_test)[:, 1]
        test_probabilities = calibrated_probabilities(test_raw, calibrator)
        metrics = outcome_metrics(test, test_probabilities, threshold)
        midpoint = len(test) // 2
        first_half = outcome_metrics(
            test[:midpoint], test_probabilities[:midpoint], threshold
        )
        second_half = outcome_metrics(
            test[midpoint:], test_probabilities[midpoint:], threshold
        )
        gates = dict(promotion_gates(metrics))
        gates["both_temporal_halves_have_selection"] = (
            int(first_half["selected_rows"]) >= 5
            and int(second_half["selected_rows"]) >= 5
        )
        gates["both_temporal_halves_non_negative"] = all(
            value.get("selected_mean_net_return_after_costs") is not None
            and float(value["selected_mean_net_return_after_costs"]) >= 0.0
            for value in (first_half, second_half)
        )
        side_pass = all(gates.values())
        overall_pass &= side_pass
        exported = export_hist_gradient_boosting(
            model,
            f"entry-path-fast-success-{side.lower()}-shadow",
            ENTRY_PATH_MODEL_FEATURE_NAMES,
            classifier=True,
        )
        exported_probabilities = np.asarray(
            [calibrator.apply(exported.evaluate(row)) for row in x_test],
            dtype=np.float64,
        )
        parity_error = float(
            np.max(np.abs(exported_probabilities - test_probabilities))
        )
        if parity_error > 1e-10:
            raise ValueError(f"entry path {side} export parity failed")
        models[side] = {
            "model": exported.to_payload(),
            "calibrator": calibrator_mapping(calibrator),
            "selection_threshold": threshold,
            "selection_threshold_semantics": "CALIBRATION_TOP_QUINTILE_RESEARCH_ONLY",
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "train_prevalence": float(np.mean(y_train)),
            "calibration_prevalence": float(np.mean(y_calibration)),
            "test_metrics": metrics,
            "test_first_half": first_half,
            "test_second_half": second_half,
            "gates": gates,
            "validation_pass": side_pass,
            "export_parity_max_absolute_error": parity_error,
        }

    payload: dict[str, Any] = {
        "schema_id": "aegis-entry-path-meta-model-shadow-validation-v1",
        "mode": "SHADOW",
        "target": "CLEAN_FAST_SUCCESS",
        "target_semantics": (
            "FAVORABLE_0_3_PERCENT_WITHIN_6_BARS_BEFORE_ADVERSE_0_3_PERCENT"
        ),
        "feature_names": list(ENTRY_PATH_MODEL_FEATURE_NAMES),
        "feature_count": len(ENTRY_PATH_MODEL_FEATURE_NAMES),
        "source": {
            "decisions": str(args.decisions),
            "decisions_sha256": sha256_file(args.decisions),
            "signals": str(args.signals),
            "signals_sha256": sha256_file(args.signals),
            "intelligence": str(args.intelligence),
            "intelligence_sha256": sha256_file(args.intelligence),
        },
        "dataset": {
            "causal_candidate_rows": len(candidates),
            "non_overlapping_rows": len(non_overlapping),
            "rejected_incomplete_rows": rejected_incomplete,
            "embargo_minutes": 60,
            "train_end": train_end.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "calibration_end": calibration_end.isoformat(),
            "test_start": test_start.isoformat(),
        },
        "models": models,
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
                "models": {
                    side: {
                        "validation_pass": value["validation_pass"],
                        "test_metrics": value["test_metrics"],
                        "gates": value["gates"],
                    }
                    for side, value in models.items()
                },
                "validation_pass": payload["validation_pass"],
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
