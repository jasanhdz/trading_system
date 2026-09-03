#!/usr/bin/env python3
"""Train the preregistered selective LONG v3 committee in research Shadow."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import DeterministicFeaturePipeline
from aegis.research.long_entry_v21_shadow import (
    atr_normalized_long_outcome,
    factorized_regime,
    multitimeframe_long_features,
    protected_long_utility,
)
from aegis.research.long_entry_v22_shadow import (
    long_v22_feature_vector,
    specialist_committee_score,
)
from aegis.research.long_entry_v3_shadow import (
    LONG_V3_FEATURE_NAMES,
    HardNegativeType,
    LongCandidateFamily,
    MicrostructureBar,
    classify_hard_negative,
    classify_long_v3_candidate,
    long_v3_feature_vector,
    microstructure_feature_vector,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import sha256_file
from train_long_entry_archetypes_v2_shadow import _snapshot
from train_long_entry_v21_shadow import (
    _atr_contract,
    _classifier,
    _fold_boundaries,
    _probability,
    _protection,
    _selection_metrics,
    _source_series,
)
from train_long_entry_v22_shadow import (
    _derive_policy,
    _group_attribution,
    _policy_selection,
    _policy_valid,
)


CANDIDATE_FAMILIES = tuple(
    value for value in LongCandidateFamily if value is not LongCandidateFamily.NONE
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


def _verify_protection_authority(root: Path, config: Mapping[str, Any]) -> None:
    authority = _mapping(config["typescript_protection"], "typescript_protection")
    for path_field, hash_field in (
        ("gate_source", "gate_source_sha256"),
        ("guardian_source", "guardian_source_sha256"),
        ("runtime_config", "runtime_config_sha256"),
        ("replay_source", "replay_source_sha256"),
    ):
        path = root / str(authority[path_field])
        if not path.is_file() or sha256_file(path) != str(authority[hash_field]):
            raise RuntimeError("AEGIS_LONG_V3_TYPESCRIPT_PROTECTION_AUTHORITY_DRIFT")


def _load_microstructure(
    database: Path, common: Sequence[datetime]
) -> tuple[
    Mapping[str, Sequence[MicrostructureBar | None]],
    Mapping[str, tuple[Sequence[int], Sequence[float]]],
    Mapping[str, Any],
]:
    if not database.is_file():
        raise RuntimeError("AEGIS_LONG_V3_PUBLIC_MICROSTRUCTURE_MISSING")
    common_ms = [int(value.timestamp() * 1000) for value in common]
    start_ms, end_ms = common_ms[0], common_ms[-1]
    bars: dict[str, list[MicrostructureBar | None]] = {}
    funding: dict[str, tuple[list[int], list[float]]] = {}
    coverage: dict[str, Any] = {}
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        for symbol in CANONICAL_SYMBOLS:
            micro_rows = {
                int(timestamp): (float(quote), int(trades), float(taker))
                for timestamp, quote, trades, taker in connection.execute(
                    "SELECT open_time_ms, quote_volume, trade_count, taker_buy_base "
                    "FROM kline_microstructure WHERE symbol=? AND open_time_ms "
                    "BETWEEN ? AND ? ORDER BY open_time_ms",
                    (symbol, start_ms, end_ms),
                )
            }
            symbol_bars: list[MicrostructureBar | None] = []
            for timestamp, source_bar in zip(common_ms, common):
                raw = micro_rows.get(timestamp)
                if raw is None:
                    symbol_bars.append(None)
                else:
                    symbol_bars.append(
                        MicrostructureBar(
                            quote_volume=raw[0],
                            trade_count=raw[1],
                            taker_buy_base=raw[2],
                            base_volume=0.0,
                        )
                    )
            bars[symbol] = symbol_bars
            funding_rows = [
                (int(timestamp), float(rate))
                for timestamp, rate in connection.execute(
                    "SELECT funding_time_ms, funding_rate FROM funding_history "
                    "WHERE symbol=? AND funding_time_ms<=? ORDER BY funding_time_ms",
                    (symbol, end_ms),
                )
            ]
            funding[symbol] = (
                [row[0] for row in funding_rows],
                [row[1] for row in funding_rows],
            )
            present = sum(value is not None for value in symbol_bars)
            coverage[symbol] = {
                "aligned_candles": len(common),
                "kline_microstructure_rows": present,
                "kline_microstructure_fraction": present / len(common),
                "funding_rows": len(funding_rows),
            }
        recent_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "open_interest_recent",
                "taker_ratio_recent",
                "depth_snapshots",
            )
        }
        manifest_row = connection.execute(
            "SELECT value FROM collection_manifest WHERE key='latest'"
        ).fetchone()
    finally:
        connection.close()
    manifest = json.loads(manifest_row[0]) if manifest_row else None
    return bars, funding, {
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "database_open_mode": "READ_ONLY",
        "coverage": coverage,
        "prospective_only_row_counts": recent_counts,
        "collection_manifest": manifest,
    }


def _with_base_volume(
    bar: MicrostructureBar | None, base_volume: float
) -> MicrostructureBar | None:
    if bar is None:
        return None
    return MicrostructureBar(
        quote_volume=bar.quote_volume,
        trade_count=bar.trade_count,
        taker_buy_base=bar.taker_buy_base,
        base_volume=base_volume,
    )


def _funding_at(
    series: tuple[Sequence[int], Sequence[float]],
    timestamp_ms: int,
    maximum_age_ms: int,
) -> tuple[float, float] | None:
    times, rates = series
    index = bisect.bisect_right(times, timestamp_ms) - 1
    if index < 0 or timestamp_ms - times[index] > maximum_age_ms:
        return None
    current = rates[index]
    previous = rates[index - 1] if index > 0 else current
    return current, previous


def build_dataset(
    root: Path, config: Mapping[str, Any], label_config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    source = _mapping(config["source"], "source")
    sampling = _mapping(config["sampling"], "sampling")
    history_bars = int(sampling["history_bars"])
    horizon_bars = int(sampling["horizon_bars"])
    stride_bars = int(sampling["stride_bars"])
    independent_stride = int(sampling["independent_test_stride_bars"])
    candles, common, source_inventory = _source_series(
        root / str(source["base_database"]),
        root / str(source["public_candle_delta"]),
        lookback_days=int(source["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=horizon_bars,
    )
    micro_bars, funding, micro_inventory = _load_microstructure(
        root / str(source["public_microstructure_database"]), common
    )
    minimum_micro = float(
        config["public_data_contract"]["minimum_kline_microstructure_coverage"]
    )
    if any(
        row["kline_microstructure_fraction"] < minimum_micro
        for row in micro_inventory["coverage"].values()
    ):
        raise RuntimeError("AEGIS_LONG_V3_MICROSTRUCTURE_COVERAGE_FAILED")

    pipeline = DeterministicFeaturePipeline()
    atr_contract = _atr_contract(label_config)
    protection = _protection(label_config)
    utility_config = _mapping(label_config["utility"], "utility")
    candidate_config = _mapping(config["candidate_builder"], "candidate_builder")
    maximum_funding_age = int(
        float(config["public_data_contract"]["maximum_forward_fill_funding_hours"])
        * 3600
        * 1000
    )
    records: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    feature_ready = 0
    funding_ready = 0
    evaluation_number = 0
    independent_every = max(1, math.ceil(independent_stride / stride_bars))

    for index in range(history_bars - 1, len(common) - horizon_bars, stride_bars):
        batch = pipeline.transform(_snapshot(candles, index, 96))
        timestamp = candles[CANONICAL_SYMBOLS[0]][index].close_time
        timestamp_ms = int(timestamp.timestamp() * 1000)
        independent = evaluation_number % independent_every == 0
        for row in batch.rows:
            raw_micro = [
                _with_base_volume(
                    micro_bars[row.symbol][position],
                    candles[row.symbol][position].volume,
                )
                for position in range(index - 23, index + 1)
            ]
            if any(value is None for value in raw_micro):
                continue
            funding_values = _funding_at(
                funding[row.symbol], timestamp_ms, maximum_funding_age
            )
            if funding_values is None:
                continue
            funding_ready += 1
            base_features = dict(zip(batch.feature_names, row.raw_values))
            history = candles[row.symbol][index - history_bars + 1 : index + 1]
            v21_vector, context = multitimeframe_long_features(
                base_features, history, pipeline=pipeline
            )
            regime = factorized_regime(base_features, context)
            v22_vector = long_v22_feature_vector(v21_vector, regime)
            micro_vector = microstructure_feature_vector(
                tuple(value for value in raw_micro if value is not None),
                return_1=float(base_features["ret_1"]),
                atr_fraction=float(base_features["atr_12"]),
                funding_rate_last=funding_values[0],
                funding_rate_previous=funding_values[1],
            )
            micro_values = dict(
                zip(
                    LONG_V3_FEATURE_NAMES[len(v22_vector) :],
                    micro_vector,
                )
            )
            feature_ready += 1
            candidate = classify_long_v3_candidate(
                base=base_features,
                context=context,
                micro=micro_values,
                history=history,
                config=candidate_config,
            )
            family = str(candidate["family"])
            family_counts[family] += 1
            if not candidate["is_candidate"]:
                continue
            future = candles[row.symbol][index + 1 : index + 1 + horizon_bars]
            outcome = atr_normalized_long_outcome(
                signal=candles[row.symbol][index],
                future=future,
                atr_fraction=float(base_features["atr_12"]),
                contract=atr_contract,
            )
            utility = protected_long_utility(
                history=history,
                future=future,
                outcome=outcome,
                protection=protection,
                mae_penalty_weight=float(utility_config["mae_penalty_weight"]),
                underwater_bar_penalty_fraction=float(
                    utility_config["underwater_bar_penalty_fraction"]
                ),
                catastrophic_mae_atr_multiple=float(
                    utility_config["catastrophic_mae_atr_multiple"]
                ),
            )
            hard_negative = classify_hard_negative(family, {**outcome, **utility})
            hard_negative_counts[hard_negative.value] += 1
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": row.symbol,
                    "archetype": family,
                    "candidate_family": family,
                    "candidate_evidence": candidate["evidence"][family],
                    "regime": regime["identity"],
                    "regime_axes": regime,
                    "independent": independent,
                    "features": long_v3_feature_vector(v22_vector, micro_vector),
                    "hard_negative": hard_negative.value,
                    **outcome,
                    **utility,
                }
            )
        evaluation_number += 1

    source_population = evaluation_number * len(CANONICAL_SYMBOLS)
    candidate_fraction = len(records) / feature_ready if feature_ready else 0.0
    minimum_fraction = float(candidate_config["minimum_population_fraction"])
    maximum_fraction = float(candidate_config["maximum_population_fraction"])
    return records, {
        **source_inventory,
        "microstructure": micro_inventory,
        "evidence_start": common[history_bars - 1].isoformat(),
        "evidence_end": common[-1 - horizon_bars].isoformat(),
        "evaluated_snapshots": evaluation_number,
        "source_population_rows": source_population,
        "feature_ready_rows": feature_ready,
        "funding_ready_rows": funding_ready,
        "candidate_rows": len(records),
        "candidate_fraction_of_feature_ready": candidate_fraction,
        "candidate_prevalence_gate": minimum_fraction
        <= candidate_fraction
        <= maximum_fraction,
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
        "feature_schema": pipeline.schema_version,
        "model_feature_count": len(LONG_V3_FEATURE_NAMES),
        "entry_rule": "NEXT_BAR_OPEN",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def _fit_classifier_weighted(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    target: str,
    seed: int,
    hard_negative_weight: float,
) -> tuple[Any, Any] | None:
    y_train = np.asarray([bool(row[target]) for row in train], dtype=int)
    y_calibration = np.asarray([bool(row[target]) for row in calibration], dtype=int)
    if (
        len(train) < 200
        or len(calibration) < 80
        or len(np.unique(y_train)) != 2
        or len(np.unique(y_calibration)) != 2
    ):
        return None
    weights = np.asarray(
        [
            hard_negative_weight
            if row["hard_negative"] != HardNegativeType.NOT_HARD_NEGATIVE.value
            else 1.0
            for row in train
        ],
        dtype=np.float64,
    )
    model = _classifier(seed).fit(
        np.asarray([row["features"] for row in train], dtype=np.float64),
        y_train,
        sample_weight=weights,
    )
    raw = model.predict_proba(
        np.asarray([row["features"] for row in calibration], dtype=np.float64)
    )[:, 1]
    return model, fit_platt_calibrator(raw, y_calibration)


def _fit_bundles(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Mapping[str, tuple[Any, Any]]] | None:
    bundles: dict[str, dict[str, tuple[Any, Any]]] = {}
    weight = float(config["hard_negatives"]["training_weight"])
    for family_index, family in enumerate(CANDIDATE_FAMILIES):
        train_group = [row for row in train if row["archetype"] == family.value]
        calibration_group = [
            row for row in calibration if row["archetype"] == family.value
        ]
        specialists: dict[str, tuple[Any, Any]] = {}
        for target_index, (output, target) in enumerate(TARGETS.items()):
            fitted = _fit_classifier_weighted(
                train_group,
                calibration_group,
                target,
                seed + family_index * 10 + target_index,
                weight,
            )
            if fitted is None:
                return None
            specialists[output] = fitted
        bundles[family.value] = specialists
    return bundles


def _predict_rows(
    rows: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, tuple[Any, Any]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["archetype"])].append((index, row))
    ordered: list[dict[str, Any] | None] = [None] * len(rows)
    for family, indexed in groups.items():
        specialists = bundles.get(family)
        if specialists is None:
            continue
        group = [row for _, row in indexed]
        predictions = {
            output: _probability(model, calibrator, group)
            for output, (model, calibrator) in specialists.items()
        }
        for local_index, (source_index, row) in enumerate(indexed):
            direction = float(predictions["direction_probability"][local_index])
            timing = float(predictions["timing_probability"][local_index])
            risk = float(predictions["path_risk_probability"][local_index])
            ordered[source_index] = {
                **row,
                "direction_probability": direction,
                "timing_probability": timing,
                "path_risk_probability": risk,
                "committee_score": specialist_committee_score(direction, timing, risk),
            }
    return [row for row in ordered if row is not None]


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
    bundles = _fit_bundles(train, calibration, config, 20261100 + fold_id * 100)
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
    policy = _derive_policy(predicted_calibration, config)
    selected = _policy_selection(predicted_test, policy)
    metrics = _selection_metrics(predicted_test, selected)
    minimum = int(config["validation"]["minimum_scoring_selections_per_fold"])
    passed = bool(
        policy["valid"]
        and _policy_valid(metrics, minimum)
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"]
        <= float(config["ranking"]["maximum_p95_gap_hours"])
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "policy": policy,
        "specialist_metrics": _probability_metrics(predicted_test),
        "metrics": metrics,
        "passed": passed,
    }


def _leave_one_symbol_out(
    records: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    reports: dict[str, Any] = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
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
        bundles = _fit_bundles(
            train, calibration, config, 20263000 + symbol_index * 100
        )
        if bundles is None or len(test) < 50:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        predicted_calibration = _predict_rows(calibration, bundles)
        predicted_test = _predict_rows(test, bundles)
        policy = _derive_policy(predicted_calibration, config)
        selected = _policy_selection(predicted_test, policy)
        metrics = _selection_metrics(predicted_test, selected)
        no_regression = bool(
            policy["valid"]
            and metrics["selected_rows"] >= 10
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
        "method": "THREE_SPECIALISTS_PER_FAMILY_REFIT_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def train_and_validate(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    prevalence_gate: bool,
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in records})
    if len(times) < 1000:
        return {
            "folds": [],
            "primary_walk_forward_pass": False,
            "leave_one_symbol_out": {"status": "NOT_RUN_INSUFFICIENT_COVERAGE"},
            "validation_pass": False,
            "verdict": "RESEARCH_ONLY_NOT_PROMOTABLE",
        }
    boundaries = _fold_boundaries(times)
    folds = [
        _evaluate_fold(records, fold, index + 1, config)
        for index, fold in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    primary = bool(
        prevalence_gate
        and len(evaluated) == 4
        and passing >= int(config["validation"]["minimum_positive_folds"])
        and all(
            float(fold["metrics"]["selected_protected_worst_net"] or -1.0) >= 0.0
            for fold in evaluated
        )
    )
    loso = (
        _leave_one_symbol_out(records, boundaries[-1], config)
        if primary
        else {
            "status": "NOT_RUN_PRIMARY_WALK_FORWARD_GATE_FAILED",
            "passed": False,
        }
    )
    validation_pass = primary and bool(loso.get("passed"))
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "candidate_prevalence_gate": prevalence_gate,
        "primary_walk_forward_pass": primary,
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
        default=Path("config/experiments/aegis_long_entry_v3_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v3_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    if (
        config.get("schema_version")
        != "aegis-long-entry-v3-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
        or config.get("automatic_live_promotion") is not False
    ):
        raise SystemExit("AEGIS_LONG_V3_CONFIG_INVALID")
    _verify_protection_authority(root, config)
    label_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text()
        ),
        "label_config",
    )
    records, source = build_dataset(root, config, label_config)
    validation = train_and_validate(
        records, config, bool(source["candidate_prevalence_gate"])
    )
    report = {
        "schema_id": "aegis-long-entry-v3-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "feature_names": list(LONG_V3_FEATURE_NAMES),
        "row_level_attribution": {
            "by_symbol": _group_attribution(records, lambda row: str(row["symbol"])),
            "by_candidate_family": _group_attribution(
                records, lambda row: str(row["candidate_family"])
            ),
            "by_hard_negative": _group_attribution(
                records, lambda row: str(row["hard_negative"])
            ),
            "by_regime": _group_attribution(records, lambda row: str(row["regime"])),
        },
        "validation": validation,
        "deployment": {
            "selection_effect": "NONE",
            "shadow_runtime_enabled": False,
            "live_enabled": False,
            "automatic_promotion": False,
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
