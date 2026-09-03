#!/usr/bin/env python3
"""Train and validate four LONG entry archetypes on real local 5m paths."""

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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import Candle, MarketSnapshot, PortfolioContext, SymbolSeries
from aegis.features import DeterministicFeaturePipeline
from aegis.research.long_entry_specialists_shadow import (
    LONG_SPECIALIST_FEATURE_NAMES,
    LongArchetypeV2,
    classify_long_archetype_v2,
    exact_long_path_outcome,
    long_specialist_feature_vector,
)
from aegis.training.train import fit_platt_calibrator

MODELED_ARCHETYPES = tuple(
    value for value in LongArchetypeV2 if value is not LongArchetypeV2.OTHER
)
ARCHETYPE_FEATURE_NAMES = tuple(value.value for value in MODELED_ARCHETYPES)


def _db_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_symbol(
    connection: sqlite3.Connection,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict[datetime, Candle]:
    rows = connection.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv_data WHERE symbol=? AND timeframe='5m' "
        "AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
        (
            _db_symbol(symbol),
            start.replace(tzinfo=None).isoformat(sep=" "),
            end.replace(tzinfo=None).isoformat(sep=" "),
        ),
    ).fetchall()
    result: dict[datetime, Candle] = {}
    for raw_timestamp, open_, high, low, close, volume in rows:
        timestamp = _parse_timestamp(str(raw_timestamp))
        result[timestamp] = Candle(
            open_time=timestamp,
            close_time=timestamp + timedelta(minutes=5),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            is_closed=True,
            source="LOCAL_BINANCE_CANDLE_DB_READ_ONLY",
        )
    return result


def _regime(features: Mapping[str, float]) -> str:
    breadth = float(features["market_breadth_6"])
    direction = float(features["market_direction_6"])
    btc_short_stack = float(features["btc_trend_proxy"]) > 0.5
    if float(features["high_vol_regime_proxy"]) > 0.5:
        return "HIGH_VOLATILITY"
    if breadth >= 0.60 and direction > 0.0 and not btc_short_stack:
        return "BULL_TREND"
    if breadth <= 0.40 and direction < 0.0 and btc_short_stack:
        return "BEAR_TREND"
    return "RANGE_OR_TRANSITION"


def _snapshot(
    candles: Mapping[str, Sequence[Candle]], index: int, history_bars: int
) -> MarketSnapshot:
    closed_at = candles[CANONICAL_SYMBOLS[0]][index].close_time
    return MarketSnapshot(
        closed_at=closed_at,
        timeframe="5m",
        symbol_set_hash=CANONICAL_SYMBOL_SET_HASH,
        series=tuple(
            SymbolSeries(
                symbol=symbol,
                candles=tuple(candles[symbol][index - history_bars + 1 : index + 1]),
                last_confirmed_close=closed_at,
            )
            for symbol in CANONICAL_SYMBOLS
        ),
        portfolio=PortfolioContext(
            available_slots=len(CANONICAL_SYMBOLS), operational_time=closed_at
        ),
    )


def build_dataset(
    database: Path,
    *,
    lookback_days: int,
    stride_bars: int,
    history_bars: int,
    horizon_bars: int,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        latest_row = connection.execute(
            "SELECT MIN(latest) FROM ("
            "SELECT MAX(timestamp) AS latest FROM ohlcv_data "
            "WHERE timeframe='5m' GROUP BY symbol)",
        ).fetchone()
        if latest_row is None or latest_row[0] is None:
            raise RuntimeError("AEGIS_LONG_V2_NO_LOCAL_CANDLES")
        end = _parse_timestamp(str(latest_row[0]))
        start = end - timedelta(
            days=lookback_days, minutes=5 * (history_bars + horizon_bars)
        )
        by_timestamp = {
            symbol: _load_symbol(connection, symbol, start, end)
            for symbol in CANONICAL_SYMBOLS
        }
    finally:
        connection.close()

    common = sorted(set.intersection(*(set(rows) for rows in by_timestamp.values())))
    if len(common) <= history_bars + horizon_bars:
        raise RuntimeError("AEGIS_LONG_V2_INSUFFICIENT_ALIGNED_CANDLES")
    candles = {
        symbol: [by_timestamp[symbol][timestamp] for timestamp in common]
        for symbol in CANONICAL_SYMBOLS
    }
    pipeline = DeterministicFeaturePipeline()
    records: list[dict[str, Any]] = []
    archetype_counts: Counter[str] = Counter()
    evaluation_number = 0
    for index in range(history_bars - 1, len(common) - horizon_bars, stride_bars):
        batch = pipeline.transform(_snapshot(candles, index, history_bars))
        timestamp = candles[CANONICAL_SYMBOLS[0]][index].close_time
        independent = evaluation_number % max(1, math.ceil(12 / stride_bars)) == 0
        for row in batch.rows:
            values = dict(zip(batch.feature_names, row.raw_values))
            classification = classify_long_archetype_v2(values)
            archetype = str(classification["archetype"])
            archetype_counts[archetype] += 1
            if archetype == LongArchetypeV2.OTHER.value:
                continue
            symbol_candles = candles[row.symbol]
            outcome = exact_long_path_outcome(
                entry_price=symbol_candles[index].close,
                future_candles=symbol_candles[index + 1 : index + 1 + horizon_bars],
            )
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": row.symbol,
                    "archetype": archetype,
                    "regime": _regime(values),
                    "independent": independent,
                    "features": long_specialist_feature_vector(values),
                    **outcome,
                }
            )
        evaluation_number += 1
    if not records:
        raise RuntimeError("AEGIS_LONG_V2_NO_ROUTED_SETUPS")
    return records, {
        "database": str(database.resolve()),
        "database_open_mode": "READ_ONLY",
        "database_size_bytes": database.stat().st_size,
        "evidence_start": common[history_bars - 1].isoformat().replace("+00:00", "Z"),
        "evidence_end": common[-1 - horizon_bars].isoformat().replace("+00:00", "Z"),
        "database_latest_common_candle": common[-1].isoformat().replace("+00:00", "Z"),
        "aligned_candles": len(common),
        "evaluated_snapshots": evaluation_number,
        "routed_records": len(records),
        "archetype_counts_all_snapshots": dict(sorted(archetype_counts.items())),
        "feature_schema": pipeline.schema_version,
        "feature_count": len(pipeline.feature_names),
        "model_feature_count": len(LONG_SPECIALIST_FEATURE_NAMES),
        "stride_bars": stride_bars,
        "history_bars": history_bars,
        "horizon_bars": horizon_bars,
        "entry_semantics": "SIGNAL_CANDLE_CLOSE",
    }


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def _regressor(
    seed: int, *, quantile: float | None = None
) -> HistGradientBoostingRegressor:
    parameters: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 2.0,
        "early_stopping": False,
        "random_state": seed,
    }
    if quantile is not None:
        parameters.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**parameters)


def _fit_probability(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    target: str,
    *,
    seed: int,
) -> tuple[Any, Any] | None:
    y_train = np.asarray([bool(row[target]) for row in train], dtype=int)
    y_cal = np.asarray([bool(row[target]) for row in calibration], dtype=int)
    if len(train) < 100 or len(calibration) < 40:
        return None
    if len(np.unique(y_train)) != 2 or len(np.unique(y_cal)) != 2:
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


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    success_probability: np.ndarray,
    selected: np.ndarray,
) -> Mapping[str, Any]:
    labels = np.asarray([bool(row["clean_fast_success"]) for row in rows], dtype=int)
    selected_rows = [row for row, keep in zip(rows, selected) if keep]
    by_regime: dict[str, Any] = {}
    for regime in sorted({str(row["regime"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["regime"] == regime]
        regime_selected = [rows[index] for index in indices if selected[index]]
        by_regime[regime] = {
            "rows": len(indices),
            "selected": len(regime_selected),
            "baseline_net": _mean(
                [rows[index] for index in indices], "net_return_after_costs"
            ),
            "selected_net": _mean(regime_selected, "net_return_after_costs"),
            "selected_mae": _mean(regime_selected, "mae_fraction"),
        }
    return {
        "rows": len(rows),
        "prevalence": float(np.mean(labels)),
        "average_precision": (
            float(average_precision_score(labels, success_probability))
            if np.any(labels == 1)
            else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(labels, success_probability))
            if len(np.unique(labels)) == 2
            else None
        ),
        "selected_rows": len(selected_rows),
        "selected_fraction": float(np.mean(selected)),
        "selected_success_rate": _mean(selected_rows, "clean_fast_success"),
        "selected_danger_rate": _mean(selected_rows, "dangerous_entry"),
        "selected_mean_mae": _mean(selected_rows, "mae_fraction"),
        "selected_mean_mfe": _mean(selected_rows, "mfe_fraction"),
        "selected_mean_underwater_bars": _mean(selected_rows, "time_underwater_bars"),
        "selected_mean_net": _mean(selected_rows, "net_return_after_costs"),
        "baseline_danger_rate": _mean(rows, "dangerous_entry"),
        "baseline_mean_mae": _mean(rows, "mae_fraction"),
        "baseline_mean_net": _mean(rows, "net_return_after_costs"),
        "by_regime": by_regime,
    }


def _fold_boundaries(times: Sequence[datetime]) -> list[tuple[datetime, ...]]:
    fractions = (
        (0.40, 0.50, 0.60),
        (0.50, 0.60, 0.70),
        (0.60, 0.70, 0.80),
        (0.70, 0.80, 0.90),
    )

    def at(fraction: float) -> datetime:
        return times[min(len(times) - 1, int(len(times) * fraction))]

    return [(times[0], at(train), at(cal), at(test)) for train, cal, test in fractions]


def _evaluate_archetype_fold(
    records: Sequence[Mapping[str, Any]],
    archetype: LongArchetypeV2,
    boundaries: tuple[datetime, ...],
    fold_id: int,
) -> Mapping[str, Any]:
    _, train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=60)
    selected_archetype = [row for row in records if row["archetype"] == archetype.value]
    train = [row for row in selected_archetype if row["timestamp"] <= train_end]
    calibration = [
        row
        for row in selected_archetype
        if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    test = [
        row
        for row in selected_archetype
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
    ]
    success_fit = _fit_probability(
        train, calibration, "clean_fast_success", seed=20260820 + fold_id
    )
    danger_fit = _fit_probability(
        train, calibration, "dangerous_entry", seed=20260830 + fold_id
    )
    if success_fit is None or danger_fit is None or len(test) < 20:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "passed": False,
        }
    success_model, success_calibrator = success_fit
    danger_model, danger_calibrator = danger_fit
    x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
    x_cal = np.asarray([row["features"] for row in calibration], dtype=np.float64)
    x_test = np.asarray([row["features"] for row in test], dtype=np.float64)
    q90_model = _regressor(20260840 + fold_id, quantile=0.90).fit(
        x_train, np.asarray([row["mae_fraction"] for row in train])
    )
    net_model = _regressor(20260850 + fold_id).fit(
        x_train, np.asarray([row["net_return_after_costs"] for row in train])
    )
    success_cal = _probability(success_model, success_calibrator, calibration)
    danger_cal = _probability(danger_model, danger_calibrator, calibration)
    q90_cal = q90_model.predict(x_cal)
    success_threshold = float(np.quantile(success_cal, 0.75, method="higher"))
    danger_threshold = float(np.quantile(danger_cal, 0.40, method="lower"))
    q90_threshold = float(np.quantile(q90_cal, 0.50, method="lower"))
    success_test = _probability(success_model, success_calibrator, test)
    danger_test = _probability(danger_model, danger_calibrator, test)
    q90_test = q90_model.predict(x_test)
    net_test = net_model.predict(x_test)
    selected = (
        (success_test >= success_threshold)
        & (danger_test <= danger_threshold)
        & (q90_test <= q90_threshold)
        & (net_test > 0.0)
    )
    metrics = _metrics(test, success_test, selected)
    passed = (
        metrics["rows"] >= 50
        and metrics["selected_rows"] >= 10
        and metrics["selected_success_rate"] is not None
        and metrics["selected_success_rate"] > metrics["prevalence"]
        and metrics["selected_danger_rate"] < metrics["baseline_danger_rate"]
        and metrics["selected_mean_mae"] < metrics["baseline_mean_mae"]
        and metrics["selected_mean_net"] > 0.0
    )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "thresholds_from_calibration_only": {
            "minimum_success_probability": success_threshold,
            "maximum_danger_probability": danger_threshold,
            "maximum_predicted_q90_mae": q90_threshold,
            "minimum_predicted_net_return": 0.0,
        },
        "metrics": metrics,
        "passed": passed,
    }


def _leave_symbol_out(
    records: Sequence[Mapping[str, Any]], boundaries: tuple[datetime, ...]
) -> Mapping[str, Any]:
    _, train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=60)
    reports: dict[str, Any] = {}
    archetype_index = {
        value.value: index for index, value in enumerate(MODELED_ARCHETYPES)
    }

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        one_hot = [0.0] * len(MODELED_ARCHETYPES)
        one_hot[archetype_index[str(row["archetype"])]] = 1.0
        return (*row["features"], *one_hot)

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
        fitted = _fit_probability(
            train, calibration, "clean_fast_success", seed=20260900 + symbol_index
        )
        if fitted is None or len(test) < 20:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        model, calibrator = fitted
        calibration_probability = _probability(model, calibrator, calibration)
        threshold = float(np.quantile(calibration_probability, 0.80, method="higher"))
        probability = _probability(model, calibrator, test)
        selected = probability >= threshold
        metrics = _metrics(test, probability, selected)
        reports[symbol] = {
            "status": "EVALUATED",
            "test_rows": len(test),
            "training_excluded_symbol": True,
            "threshold_from_non_symbol_calibration": threshold,
            "metrics": metrics,
            "generalized_without_regression": (
                metrics["average_precision"] >= metrics["prevalence"]
                and metrics["selected_success_rate"] is not None
                and metrics["selected_success_rate"] >= metrics["prevalence"]
            ),
        }
    evaluated = [value for value in reports.values() if value["status"] == "EVALUATED"]
    passing = sum(bool(value["generalized_without_regression"]) for value in evaluated)
    return {
        "method": "TRAIN_AND_CALIBRATE_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= 7,
    }


def train_and_validate(records: list[dict[str, Any]]) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in records})
    if len(times) < 1000:
        raise RuntimeError("AEGIS_LONG_V2_INSUFFICIENT_TEMPORAL_COVERAGE")
    fold_reports: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    boundaries = _fold_boundaries(times)
    for fold_id, fold in enumerate(boundaries, start=1):
        for archetype in MODELED_ARCHETYPES:
            fold_reports[archetype.value].append(
                _evaluate_archetype_fold(records, archetype, fold, fold_id)
            )
    summaries: dict[str, Any] = {}
    for archetype, reports in fold_reports.items():
        evaluated = [row for row in reports if row["status"] == "EVALUATED"]
        passing = sum(bool(row["passed"]) for row in evaluated)
        summaries[archetype] = {
            "folds": reports,
            "evaluated_folds": len(evaluated),
            "passing_folds": passing,
            "promotion_gate_passed": len(evaluated) == 4 and passing >= 3,
        }
    loso = _leave_symbol_out(records, boundaries[-1])
    promotable = any(
        value["promotion_gate_passed"] for value in summaries.values()
    ) and bool(loso["passed"])
    return {
        "archetypes": summaries,
        "leave_one_symbol_out": loso,
        "any_archetype_passed_temporal_gate": any(
            value["promotion_gate_passed"] for value in summaries.values()
        ),
        "all_generalization_gates_passed": promotable,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_SHADOW_RUNTIME_REVIEW"
            if promotable
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/binance_candles.db")
    )
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--stride-bars", type=int, default=6)
    parser.add_argument("--history-bars", type=int, default=96)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_archetypes_v2_shadow/validation.json"),
    )
    args = parser.parse_args()
    if (
        min(args.lookback_days, args.stride_bars, args.history_bars, args.horizon_bars)
        <= 0
    ):
        parser.error("all numeric parameters must be positive")
    root = Path(__file__).resolve().parents[1]
    database = args.database if args.database.is_absolute() else root / args.database
    output = args.output if args.output.is_absolute() else root / args.output
    records, source = build_dataset(
        database,
        lookback_days=args.lookback_days,
        stride_bars=args.stride_bars,
        history_bars=args.history_bars,
        horizon_bars=args.horizon_bars,
    )
    validation = train_and_validate(records)
    report = {
        "schema_id": "aegis-long-entry-archetypes-v2-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "SHADOW",
        "source": source,
        "labels": {
            "clean_fast_success": "LONG +0.3% BARRIER FIRST WITHIN 6 BARS",
            "danger": "LONG -0.3% BARRIER FIRST OR SAME-BAR AMBIGUITY",
            "mae": "EXACT MAXIMUM ADVERSE EXCURSION OVER 12 FUTURE 5M BARS",
            "net_return": "TERMINAL LONG RETURN MINUS 0.1% ROUND-TRIP COST",
            "future_fields_used_as_features": False,
        },
        "archetypes": [value.value for value in MODELED_ARCHETYPES],
        "training": {
            "method": "PURGED_EXPANDING_WALK_FORWARD",
            "fold_count": 4,
            "embargo_minutes": 60,
            "independent_test_sampling_minutes": 60,
            "objectives": [
                "CLEAN_FAST_SUCCESS_CLASSIFIER",
                "DANGER_CLASSIFIER",
                "Q90_MAE_REGRESSOR",
                "NET_RETURN_REGRESSOR",
            ],
            "threshold_source": "CALIBRATION_BLOCK_ONLY",
        },
        "validation": validation,
        "deployment": {
            "selection_effect": "NONE",
            "operational_import": False,
            "live_enabled": False,
            "shadow_runtime_enabled": False,
            "reason": "REQUIRES_ALL_PREREGISTERED_GENERALIZATION_GATES_AND_SEPARATE_REVIEW",
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
