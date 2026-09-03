#!/usr/bin/env python3
"""Train the preregistered multitimeframe LONG v2.1 utility ranker."""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score

from aegis.config import CANONICAL_SYMBOLS
from aegis.domain import Candle
from aegis.features import DeterministicFeaturePipeline
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.long_entry_specialists_shadow import (
    LongArchetypeV2,
    classify_long_archetype_v2,
)
from aegis.research.long_entry_v21_shadow import (
    AtrPathContract,
    LONG_V21_FEATURE_NAMES,
    atr_normalized_long_outcome,
    factorized_regime,
    multitimeframe_long_features,
    protected_long_utility,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import sha256_file
from train_long_entry_archetypes_v2_shadow import (
    _db_symbol,
    _load_symbol,
    _parse_timestamp,
    _snapshot,
)

MODELED_ARCHETYPES = tuple(
    value for value in LongArchetypeV2 if value is not LongArchetypeV2.OTHER
)


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _load_delta(path: Path) -> dict[str, dict[datetime, Candle]]:
    result: dict[str, dict[datetime, Candle]] = {
        symbol: {} for symbol in CANONICAL_SYMBOLS
    }
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), f"delta:{line_number}")
            symbol = str(row["symbol"])
            if symbol not in result or row.get("timeframe") != "5m":
                raise ValueError("LONG v2.1 candle delta identity is invalid")
            open_time = datetime.fromtimestamp(
                int(row["open_time_ms"]) / 1000.0, timezone.utc
            )
            result[symbol][open_time] = Candle(
                open_time=open_time,
                close_time=open_time + timedelta(minutes=5),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                is_closed=True,
                source="BINANCE_USDM_PUBLIC_KLINES_GET_ISOLATED_DELTA",
            )
    return result


def _source_series(
    database: Path,
    delta_path: Path,
    *,
    lookback_days: int,
    history_bars: int,
    horizon_bars: int,
) -> tuple[dict[str, list[Candle]], list[datetime], Mapping[str, Any]]:
    delta = _load_delta(delta_path)
    delta_latest = [max(rows) for rows in delta.values() if rows]
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        database_latest = _parse_timestamp(
            str(
                connection.execute(
                    "SELECT MIN(latest) FROM (SELECT MAX(timestamp) AS latest "
                    "FROM ohlcv_data WHERE timeframe='5m' GROUP BY symbol)"
                ).fetchone()[0]
            )
        )
        end = (
            min(delta_latest)
            if len(delta_latest) == len(CANONICAL_SYMBOLS)
            else database_latest
        )
        start = end - timedelta(
            days=lookback_days, minutes=5 * (history_bars + horizon_bars)
        )
        merged = {
            symbol: _load_symbol(connection, symbol, start, end)
            for symbol in CANONICAL_SYMBOLS
        }
    finally:
        connection.close()
    for symbol in CANONICAL_SYMBOLS:
        merged[symbol].update(
            {
                timestamp: candle
                for timestamp, candle in delta[symbol].items()
                if start <= timestamp <= end
            }
        )
    common = sorted(set.intersection(*(set(rows) for rows in merged.values())))
    if len(common) <= history_bars + horizon_bars:
        raise RuntimeError("AEGIS_LONG_V21_INSUFFICIENT_ALIGNED_CANDLES")
    candles = {
        symbol: [merged[symbol][timestamp] for timestamp in common]
        for symbol in CANONICAL_SYMBOLS
    }
    return (
        candles,
        common,
        {
            "base_database": str(database.resolve()),
            "base_database_open_mode": "READ_ONLY",
            "base_database_size_bytes": database.stat().st_size,
            "base_database_latest_common": database_latest.isoformat(),
            "public_delta": str(delta_path.resolve()),
            "public_delta_sha256": (
                sha256_file(delta_path) if delta_path.is_file() else None
            ),
            "public_delta_rows": sum(len(rows) for rows in delta.values()),
            "latest_common_candle": common[-1].isoformat(),
            "aligned_candles": len(common),
        },
    )


def _atr_contract(config: Mapping[str, Any]) -> AtrPathContract:
    raw = _mapping(config["atr_label"], "atr_label")
    return AtrPathContract(
        favorable_atr_multiple=float(raw["favorable_atr_multiple"]),
        adverse_atr_multiple=float(raw["adverse_atr_multiple"]),
        favorable_floor_fraction=float(raw["favorable_floor_fraction"]),
        adverse_floor_fraction=float(raw["adverse_floor_fraction"]),
        favorable_ceiling_fraction=float(raw["favorable_ceiling_fraction"]),
        adverse_ceiling_fraction=float(raw["adverse_ceiling_fraction"]),
        fast_success_bars=int(raw["fast_success_bars"]),
        round_trip_cost_fraction=float(raw["round_trip_cost_fraction"]),
    )


def _protection(config: Mapping[str, Any]) -> TsProtectionConfig:
    raw = _mapping(config["typescript_protection"], "typescript_protection")
    return TsProtectionConfig(
        leverage=float(raw["leverage"]),
        hard_stop_roe=float(raw["hard_stop_roe"]),
        take_profit_roe=float(raw["take_profit_roe"]),
        break_even_trigger_roe=float(raw["break_even_trigger_roe"]),
        break_even_offset_fraction=float(raw["break_even_offset_fraction"]),
        trailing_activation_roe=float(raw["trailing_activation_roe"]),
        trailing_callback_roe=float(raw["trailing_callback_roe"]),
        use_atr_trailing=bool(raw["use_atr_trailing"]),
        atr_period=int(raw["atr_period"]),
        atr_multiplier=float(raw["atr_multiplier"]),
        round_trip_cost_fraction=float(raw["round_trip_cost_fraction"]),
    )


def build_dataset(
    database: Path, delta_path: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    source = _mapping(config["source"], "source")
    sampling = _mapping(config["sampling"], "sampling")
    utility_config = _mapping(config["utility"], "utility")
    history_bars = int(sampling["history_bars"])
    horizon_bars = int(sampling["horizon_bars"])
    stride_bars = int(sampling["stride_bars"])
    independent_stride = int(sampling["independent_test_stride_bars"])
    candles, common, inventory = _source_series(
        database,
        delta_path,
        lookback_days=int(source["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=horizon_bars,
    )
    pipeline = DeterministicFeaturePipeline()
    contract = _atr_contract(config)
    protection = _protection(config)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    evaluation_number = 0
    independent_every = max(1, math.ceil(independent_stride / stride_bars))
    for index in range(history_bars - 1, len(common) - horizon_bars, stride_bars):
        batch = pipeline.transform(_snapshot(candles, index, 96))
        timestamp = candles[CANONICAL_SYMBOLS[0]][index].close_time
        independent = evaluation_number % independent_every == 0
        for row in batch.rows:
            base_features = dict(zip(batch.feature_names, row.raw_values))
            archetype = str(classify_long_archetype_v2(base_features)["archetype"])
            counts[archetype] += 1
            if archetype == LongArchetypeV2.OTHER.value:
                continue
            history = candles[row.symbol][index - history_bars + 1 : index + 1]
            future = candles[row.symbol][index + 1 : index + 1 + horizon_bars]
            vector, context = multitimeframe_long_features(
                base_features, history, pipeline=pipeline
            )
            regime = factorized_regime(base_features, context)
            outcome = atr_normalized_long_outcome(
                signal=candles[row.symbol][index],
                future=future,
                atr_fraction=float(base_features["atr_12"]),
                contract=contract,
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
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": row.symbol,
                    "archetype": archetype,
                    "regime": regime["identity"],
                    "regime_axes": regime,
                    "independent": independent,
                    "features": vector,
                    **outcome,
                    **utility,
                }
            )
        evaluation_number += 1
    return records, {
        **inventory,
        "evidence_start": common[history_bars - 1].isoformat(),
        "evidence_end": common[-1 - horizon_bars].isoformat(),
        "evaluated_snapshots": evaluation_number,
        "routed_records": len(records),
        "archetype_counts": dict(sorted(counts.items())),
        "feature_schema": pipeline.schema_version,
        "base_feature_count": len(pipeline.feature_names),
        "model_feature_count": len(LONG_V21_FEATURE_NAMES),
        "higher_timeframes": ["ROLLING_15M", "ROLLING_1H"],
        "entry_rule": "NEXT_BAR_OPEN",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def _fit_classifier(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    target: str,
    seed: int,
) -> tuple[Any, Any] | None:
    y_train = np.asarray([bool(row[target]) for row in train], dtype=int)
    y_cal = np.asarray([bool(row[target]) for row in calibration], dtype=int)
    if (
        len(train) < 200
        or len(calibration) < 80
        or len(np.unique(y_train)) != 2
        or len(np.unique(y_cal)) != 2
    ):
        return None
    model = _classifier(seed).fit(
        np.asarray([row["features"] for row in train], dtype=np.float64), y_train
    )
    calibrator = fit_platt_calibrator(
        model.predict_proba(
            np.asarray([row["features"] for row in calibration], dtype=np.float64)
        )[:, 1],
        y_cal,
    )
    return model, calibrator


def _probability(
    model: Any, calibrator: Any, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    raw = model.predict_proba(
        np.asarray([row["features"] for row in rows], dtype=np.float64)
    )[:, 1]
    return np.asarray([calibrator.apply(float(value)) for value in raw])


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return float(np.mean([float(row[field]) for row in rows])) if rows else None


def _selection_metrics(
    rows: Sequence[Mapping[str, Any]], selected: np.ndarray
) -> Mapping[str, Any]:
    chosen = [row for row, keep in zip(rows, selected) if keep]
    times = sorted(row["timestamp"] for row in chosen)
    gaps = [
        (right - left).total_seconds() / 3600.0 for left, right in zip(times, times[1:])
    ]
    return {
        "rows": len(rows),
        "selected_rows": len(chosen),
        "selected_fraction": float(np.mean(selected)),
        "baseline_protected_worst_net": _mean(rows, "protected_worst_net_return"),
        "selected_protected_worst_net": _mean(chosen, "protected_worst_net_return"),
        "baseline_utility": _mean(rows, "utility_target"),
        "selected_utility": _mean(chosen, "utility_target"),
        "baseline_mae": _mean(rows, "mae_fraction"),
        "selected_mae": _mean(chosen, "mae_fraction"),
        "baseline_underwater_bars": _mean(rows, "time_underwater_bars"),
        "selected_underwater_bars": _mean(chosen, "time_underwater_bars"),
        "baseline_target_before_stop": _mean(rows, "target_before_stop"),
        "selected_target_before_stop": _mean(chosen, "target_before_stop"),
        "baseline_catastrophic_rate": _mean(rows, "catastrophic_path"),
        "selected_catastrophic_rate": _mean(chosen, "catastrophic_path"),
        "p95_gap_hours": float(np.quantile(gaps, 0.95)) if gaps else None,
        "maximum_gap_hours": max(gaps) if gaps else None,
        "symbol_counts": dict(sorted(Counter(row["symbol"] for row in chosen).items())),
        "regime_counts": dict(sorted(Counter(row["regime"] for row in chosen).items())),
    }


def _derive_policy(
    rows: Sequence[Mapping[str, Any]],
    utility_prediction: np.ndarray,
    danger_probability: np.ndarray,
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    minimum = int(validation["minimum_calibration_selections"])
    choices = []
    for utility_quantile in validation["utility_threshold_quantiles"]:
        utility_threshold = float(
            np.quantile(utility_prediction, float(utility_quantile), method="higher")
        )
        for danger_quantile in validation["maximum_danger_quantiles"]:
            danger_threshold = float(
                np.quantile(danger_probability, float(danger_quantile), method="lower")
            )
            selected = (utility_prediction >= utility_threshold) & (
                danger_probability <= danger_threshold
            )
            metrics = _selection_metrics(rows, selected)
            valid = (
                metrics["selected_rows"] >= minimum
                and metrics["selected_protected_worst_net"] is not None
                and metrics["selected_protected_worst_net"] > 0.0
                and metrics["selected_mae"] < metrics["baseline_mae"]
                and metrics["selected_underwater_bars"]
                < metrics["baseline_underwater_bars"]
            )
            choices.append(
                {
                    "minimum_predicted_utility": utility_threshold,
                    "maximum_danger_probability": danger_threshold,
                    "utility_quantile": float(utility_quantile),
                    "danger_quantile": float(danger_quantile),
                    "metrics": metrics,
                    "valid": valid,
                }
            )
    valid = [choice for choice in choices if choice["valid"]]
    pool = valid or choices
    return max(
        pool,
        key=lambda choice: (
            float(choice["metrics"]["selected_protected_worst_net"] or -1.0),
            float(choice["metrics"]["selected_utility"] or -1.0),
            int(choice["metrics"]["selected_rows"]),
        ),
    )


def _fold_boundaries(
    times: Sequence[datetime],
) -> list[tuple[datetime, datetime, datetime]]:
    fractions = (
        (0.40, 0.50, 0.60),
        (0.50, 0.60, 0.70),
        (0.60, 0.70, 0.80),
        (0.70, 0.80, 0.90),
    )

    def at(fraction: float) -> datetime:
        return times[min(len(times) - 1, int(len(times) * fraction))]

    return [
        (at(train), at(calibration), at(test)) for train, calibration, test in fractions
    ]


def _evaluate_fold(
    records: Sequence[Mapping[str, Any]],
    archetype: LongArchetypeV2,
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    population = [row for row in records if row["archetype"] == archetype.value]
    train = [row for row in population if row["timestamp"] <= train_end]
    calibration = [
        row
        for row in population
        if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    test = [
        row
        for row in population
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
    ]
    danger_fit = _fit_classifier(
        train, calibration, "catastrophic_path", 20260860 + fold_id
    )
    success_fit = _fit_classifier(
        train, calibration, "target_before_stop", 20260870 + fold_id
    )
    if danger_fit is None or success_fit is None or len(test) < 50:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "passed": False,
        }
    x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
    x_cal = np.asarray([row["features"] for row in calibration], dtype=np.float64)
    x_test = np.asarray([row["features"] for row in test], dtype=np.float64)
    utility_model = _regressor(20260880 + fold_id).fit(
        x_train, np.asarray([row["utility_target"] for row in train])
    )
    danger_model, danger_calibrator = danger_fit
    success_model, success_calibrator = success_fit
    utility_cal = utility_model.predict(x_cal)
    danger_cal = _probability(danger_model, danger_calibrator, calibration)
    validation = _mapping(config["validation"], "validation")
    global_policy = _derive_policy(calibration, utility_cal, danger_cal, validation)
    minimum_regime = int(
        config["regime_calibration"]["minimum_regime_calibration_rows"]
    )
    regime_policies: dict[str, Any] = {}
    for regime in sorted({str(row["regime"]) for row in calibration}):
        indices = [
            index for index, row in enumerate(calibration) if row["regime"] == regime
        ]
        if len(indices) < minimum_regime:
            continue
        regime_policies[regime] = _derive_policy(
            [calibration[index] for index in indices],
            utility_cal[indices],
            danger_cal[indices],
            validation,
        )
    utility_test = utility_model.predict(x_test)
    danger_test = _probability(danger_model, danger_calibrator, test)
    success_test = _probability(success_model, success_calibrator, test)
    selected_values = []
    for index, row in enumerate(test):
        policy = regime_policies.get(str(row["regime"]), global_policy)
        selected_values.append(
            utility_test[index] >= float(policy["minimum_predicted_utility"])
            and danger_test[index] <= float(policy["maximum_danger_probability"])
        )
    selected = np.asarray(selected_values, dtype=bool)
    metrics = _selection_metrics(test, selected)
    target = np.asarray([bool(row["target_before_stop"]) for row in test], dtype=int)
    metrics = {
        **metrics,
        "target_prevalence": float(np.mean(target)),
        "target_average_precision": float(
            average_precision_score(target, success_test)
        ),
        "utility_prediction_correlation": float(
            np.corrcoef(
                utility_test,
                np.asarray([row["utility_target"] for row in test]),
            )[0, 1]
        ),
    }
    minimum_test = int(validation["minimum_scoring_selections_per_fold"])
    passed = (
        bool(global_policy["valid"])
        and metrics["selected_rows"] >= minimum_test
        and metrics["selected_protected_worst_net"] is not None
        and metrics["selected_protected_worst_net"] > 0.0
        and metrics["selected_mae"] < metrics["baseline_mae"]
        and metrics["selected_underwater_bars"] < metrics["baseline_underwater_bars"]
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"] <= float(validation["maximum_p95_gap_hours"])
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "global_policy": global_policy,
        "regime_policy_count": len(regime_policies),
        "valid_regime_policy_count": sum(
            bool(policy["valid"]) for policy in regime_policies.values()
        ),
        "metrics": metrics,
        "passed": passed,
    }


def _leave_symbol_out(
    records: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    archetype_index = {
        value.value: index for index, value in enumerate(MODELED_ARCHETYPES)
    }

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        one_hot = [0.0] * len(MODELED_ARCHETYPES)
        one_hot[archetype_index[str(row["archetype"])]] = 1.0
        return (*row["features"], *one_hot)

    reports: dict[str, Any] = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        train = [
            {**row, "features": vector(row)}
            for row in records
            if row["symbol"] != symbol and row["timestamp"] <= train_end
        ]
        calibration = [
            {**row, "features": vector(row)}
            for row in records
            if row["symbol"] != symbol
            and train_end + embargo < row["timestamp"] <= calibration_end
        ]
        test = [
            {**row, "features": vector(row)}
            for row in records
            if row["symbol"] == symbol
            and row["independent"]
            and calibration_end + embargo < row["timestamp"] <= test_end
        ]
        danger_fit = _fit_classifier(
            train, calibration, "catastrophic_path", 20260920 + symbol_index
        )
        if danger_fit is None or len(test) < 50:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
        x_cal = np.asarray([row["features"] for row in calibration], dtype=np.float64)
        x_test = np.asarray([row["features"] for row in test], dtype=np.float64)
        utility_model = _regressor(20260940 + symbol_index).fit(
            x_train, np.asarray([row["utility_target"] for row in train])
        )
        danger_model, danger_calibrator = danger_fit
        utility_cal = utility_model.predict(x_cal)
        danger_cal = _probability(danger_model, danger_calibrator, calibration)
        policy = _derive_policy(
            calibration, utility_cal, danger_cal, config["validation"]
        )
        utility_test = utility_model.predict(x_test)
        danger_test = _probability(danger_model, danger_calibrator, test)
        selected = (utility_test >= float(policy["minimum_predicted_utility"])) & (
            danger_test <= float(policy["maximum_danger_probability"])
        )
        metrics = _selection_metrics(test, selected)
        no_regression = (
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
        "method": "TRAIN_AND_CALIBRATE_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def train_and_validate(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in records})
    if len(times) < 2000:
        raise RuntimeError("AEGIS_LONG_V21_INSUFFICIENT_TEMPORAL_COVERAGE")
    boundaries = _fold_boundaries(times)
    archetypes: dict[str, Any] = {}
    for archetype in MODELED_ARCHETYPES:
        folds = [
            _evaluate_fold(records, archetype, fold, fold_id, config)
            for fold_id, fold in enumerate(boundaries, start=1)
        ]
        evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
        passing = sum(bool(fold["passed"]) for fold in evaluated)
        archetypes[archetype.value] = {
            "folds": folds,
            "evaluated_folds": len(evaluated),
            "passing_folds": passing,
            "passed": len(evaluated) == 4
            and passing >= int(config["validation"]["minimum_positive_folds"]),
        }
    loso = _leave_symbol_out(records, boundaries[-1], config)
    passed_archetypes = [name for name, row in archetypes.items() if row["passed"]]
    eligible = bool(passed_archetypes) and bool(loso["passed"])
    return {
        "archetypes": archetypes,
        "passed_archetypes": passed_archetypes,
        "leave_one_symbol_out": loso,
        "validation_pass": eligible,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_SHADOW_RUNTIME_REVIEW"
            if eligible
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v21_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v21_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    if (
        config.get("schema_version") != "aegis-long-entry-v21-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
        or config.get("automatic_live_promotion") is not False
    ):
        raise SystemExit("AEGIS_LONG_V21_CONFIG_INVALID")
    database = root / str(config["source"]["base_database"])
    delta_path = root / str(config["source"]["public_delta"])
    records, source = build_dataset(database, delta_path, config)
    validation = train_and_validate(records, config)
    report = {
        "schema_id": "aegis-long-entry-v21-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "feature_names": list(LONG_V21_FEATURE_NAMES),
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
