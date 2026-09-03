"""Offline-only fitting for the preregistered Committee V2.1 risk model."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import CANONICAL_SYMBOLS, load_brain_config
from ..domain import (
    Candle,
    FeedQuality,
    MarketSnapshot,
    PortfolioContext,
    SymbolSeries,
)
from ..live_decision import CurrentBrainEngine
from ..training.run_state import atomic_write_json
from ..utils import canonical_json, sha256_file, to_primitive
from .committee_v21_shadow import (
    ARTIFACT_SCHEMA,
    CommitteeV21Contract,
    CommitteeV21ShadowError,
    basis_term_names,
    committee_v21_basis_from_stats,
    committee_v21_observation,
    load_committee_v21_contract,
)
from .regime_v2 import FactorizedRegimeAnalyzer, RegimeV2Observation
from .shadow_runtime import load_entry_quality_v2_config


class CommitteeV21FitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitteeV21FitConfig:
    path: Path
    contract: CommitteeV21Contract
    database: Path
    database_sha256: str
    table: str
    duplicate_candle_policy: str
    minimum_history_bars: int
    training_start: datetime
    training_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    diagnostic_start: datetime
    diagnostic_end: datetime
    regularization_c: float
    maximum_iterations: int
    random_seed: int
    calibration_bins: int
    retention_quantile: float
    artifact_path: Path
    private_fit_root: Path


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommitteeV21FitError(f"{identity} must be a mapping")
    return value


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _finite(value: Any, identity: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CommitteeV21FitError(f"{identity} is non-finite")
    return parsed


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_committee_v21_fit_config(
    path: Path,
    *,
    repo_root: Path,
) -> CommitteeV21FitConfig:
    root = repo_root.resolve()
    resolved = path.resolve()
    contract = load_committee_v21_contract(resolved)
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "preregistration",
        )
        data = _mapping(payload["data"], "data")
        training = _mapping(data["training"], "training")
        calibration = _mapping(data["calibration"], "calibration")
        diagnostic = _mapping(data["diagnostic_only"], "diagnostic")
        model = _mapping(payload["model"], "model")
        calibration_contract = _mapping(
            payload["calibration"],
            "calibration_contract",
        )
        artifacts = _mapping(payload["artifacts"], "artifacts")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_CONFIG_INVALID") from exc
    database = _resolve(root, data["database"])
    database_hash = str(data["database_sha256"])
    train_start = _timestamp(training["start_utc"])
    train_end = _timestamp(training["end_utc"])
    calibration_start = _timestamp(calibration["start_utc"])
    calibration_end = _timestamp(calibration["end_utc"])
    diagnostic_start = _timestamp(diagnostic["start_utc"])
    diagnostic_end = _timestamp(diagnostic["end_utc"])
    private_root = _resolve(root, artifacts["private_fit_root"])
    data_root = (root / "data").resolve()
    if (
        sha256_file(database) != database_hash
        or not train_start <= train_end < calibration_start <= calibration_end
        or not calibration_end < diagnostic_start <= diagnostic_end
        or diagnostic.get("promotion_use") != "PROHIBITED"
        or data.get("duplicate_candle_policy")
        != "MINIMUM_ROWID_FIRST_PERSISTED_OBSERVATION"
        or data.get("duplicate_candle_audit_required") is not True
        or private_root != data_root
        and data_root not in private_root.parents
    ):
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_AUTHORITY_INVALID")
    regularization = _finite(model["regularization_c"], "regularization_c")
    maximum_iterations = int(model["maximum_iterations"])
    bins = int(calibration_contract["expected_calibration_error_bins"])
    retention = _finite(
        calibration_contract["retention_quantile"],
        "retention_quantile",
    )
    if (
        regularization <= 0.0
        or maximum_iterations < 100
        or bins < 2
        or not 0.0 < retention < 1.0
    ):
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_HYPERPARAMETER_INVALID")
    return CommitteeV21FitConfig(
        path=resolved,
        contract=contract,
        database=database,
        database_sha256=database_hash,
        table=str(data["table"]),
        duplicate_candle_policy=str(data["duplicate_candle_policy"]),
        minimum_history_bars=int(data["minimum_history_bars"]),
        training_start=train_start,
        training_end=train_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        diagnostic_start=diagnostic_start,
        diagnostic_end=diagnostic_end,
        regularization_c=regularization,
        maximum_iterations=maximum_iterations,
        random_seed=int(model["random_seed"]),
        calibration_bins=bins,
        retention_quantile=retention,
        artifact_path=_resolve(root, artifacts["model"]),
        private_fit_root=private_root,
    )


def _database_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _read_candles(
    config: CommitteeV21FitConfig,
) -> tuple[Mapping[str, Mapping[datetime, Candle]], Mapping[str, int]]:
    interval = timedelta(minutes=5)
    warmup = max(config.minimum_history_bars, 288)
    query_start = config.training_start - interval * (
        config.minimum_history_bars + warmup
    )
    query_end = config.diagnostic_end + interval * config.contract.horizon_bars
    connection = sqlite3.connect(f"file:{config.database}?mode=ro", uri=True)
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if config.table not in table_names:
            raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_TABLE_MISSING")
        result: dict[str, dict[datetime, Candle]] = {}
        duplicate_identities = 0
        conflicting_ohlc = 0
        conflicting_volume = 0
        for symbol in CANONICAL_SYMBOLS:
            rows = connection.execute(
                f"SELECT source.timestamp, source.open, source.high, "
                f"source.low, source.close, source.volume "
                f"FROM {config.table} AS source "
                f"JOIN (SELECT timestamp, MIN(rowid) AS selected_rowid "
                f"FROM {config.table} "
                "WHERE symbol = ? AND timeframe = '5m' "
                "AND timestamp >= ? AND timestamp <= ? "
                "GROUP BY timestamp) AS selected "
                "ON source.rowid = selected.selected_rowid "
                "ORDER BY source.timestamp",
                (
                    _database_symbol(symbol),
                    query_start.replace(tzinfo=None).isoformat(sep=" "),
                    query_end.replace(tzinfo=None).isoformat(sep=" "),
                ),
            ).fetchall()
            candles = {}
            for row in rows:
                open_time = _timestamp(row[0])
                if open_time in candles:
                    raise CommitteeV21FitError(
                        "AEGIS_COMMITTEE_V21_FIT_DUPLICATE_CANDLE"
                    )
                candles[open_time] = Candle(
                    open_time,
                    open_time + interval,
                    _finite(row[1], "open"),
                    _finite(row[2], "high"),
                    _finite(row[3], "low"),
                    _finite(row[4], "close"),
                    _finite(row[5], "volume"),
                    True,
                    "LOCAL_READ_ONLY_OHLCV_FIT",
                    open_time.isoformat(),
                )
            result[symbol] = candles
            duplicate_rows = connection.execute(
                f"SELECT COUNT(*), "
                "SUM(CASE WHEN maximum_open <> minimum_open "
                "OR maximum_high <> minimum_high "
                "OR maximum_low <> minimum_low "
                "OR maximum_close <> minimum_close THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN maximum_volume <> minimum_volume "
                "THEN 1 ELSE 0 END) "
                "FROM ("
                f"SELECT timestamp, COUNT(*) AS count, "
                "MIN(open) AS minimum_open, MAX(open) AS maximum_open, "
                "MIN(high) AS minimum_high, MAX(high) AS maximum_high, "
                "MIN(low) AS minimum_low, MAX(low) AS maximum_low, "
                "MIN(close) AS minimum_close, MAX(close) AS maximum_close, "
                "MIN(volume) AS minimum_volume, MAX(volume) AS maximum_volume "
                f"FROM {config.table} "
                "WHERE symbol = ? AND timeframe = '5m' "
                "AND timestamp >= ? AND timestamp <= ? "
                "GROUP BY timestamp HAVING count > 1)",
                (
                    _database_symbol(symbol),
                    query_start.replace(tzinfo=None).isoformat(sep=" "),
                    query_end.replace(tzinfo=None).isoformat(sep=" "),
                ),
            ).fetchone()
            duplicate_identities += int(duplicate_rows[0] or 0)
            conflicting_ohlc += int(duplicate_rows[1] or 0)
            conflicting_volume += int(duplicate_rows[2] or 0)
        return result, {
            "duplicate_identities": duplicate_identities,
            "conflicting_ohlc_identities": conflicting_ohlc,
            "conflicting_volume_identities": conflicting_volume,
        }
    finally:
        connection.close()


def _snapshot(
    candles: Mapping[str, Mapping[datetime, Candle]],
    *,
    closed_at: datetime,
    history_bars: int,
    symbol_set_hash: str,
) -> MarketSnapshot:
    interval = timedelta(minutes=5)
    return MarketSnapshot(
        closed_at,
        "5m",
        symbol_set_hash,
        tuple(
            SymbolSeries(
                symbol,
                tuple(
                    candles[symbol][closed_at - interval * offset]
                    for offset in range(history_bars, 0, -1)
                ),
                closed_at,
                FeedQuality(),
            )
            for symbol in CANONICAL_SYMBOLS
        ),
        PortfolioContext(
            available_slots=1,
            operational_time=closed_at,
        ),
    )


def _outcome(
    candles: Mapping[str, Mapping[datetime, Candle]],
    *,
    symbol: str,
    signal_timestamp: datetime,
    horizon_bars: int,
    cost: float,
) -> tuple[float, float, float]:
    interval = timedelta(minutes=5)
    history = candles[symbol]
    entry = history[signal_timestamp - interval].close
    future = tuple(
        history[signal_timestamp + interval * index] for index in range(horizon_bars)
    )
    gross = (entry - future[-1].close) / entry
    return (
        gross - cost,
        max(0.0, (max(candle.high for candle in future) - entry) / entry),
        max(0.0, (entry - min(candle.low for candle in future)) / entry),
    )


def _split_name(
    config: CommitteeV21FitConfig,
    timestamp: datetime,
) -> str | None:
    if config.training_start <= timestamp <= config.training_end:
        return "TRAINING"
    if config.calibration_start <= timestamp <= config.calibration_end:
        return "CALIBRATION"
    if config.diagnostic_start <= timestamp <= config.diagnostic_end:
        return "DIAGNOSTIC_ONLY"
    return None


def build_committee_v21_dataset(
    config: CommitteeV21FitConfig,
    *,
    repo_root: Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    candles, duplicate_audit = _read_candles(config)
    interval = timedelta(minutes=5)
    warmup_start = config.training_start - interval * 288
    timestamps = []
    current = warmup_start
    while current <= config.diagnostic_end:
        valid = all(
            all(
                current - interval * offset in candles[symbol]
                for offset in range(config.minimum_history_bars, 0, -1)
            )
            and (
                current < config.training_start
                or all(
                    current + interval * index in candles[symbol]
                    for index in range(config.contract.horizon_bars)
                )
            )
            for symbol in CANONICAL_SYMBOLS
        )
        if valid:
            timestamps.append(current)
        current += interval
    engine = CurrentBrainEngine()
    engine.initialize()
    brain = load_brain_config(repo_root / "config")
    entry_config = load_entry_quality_v2_config(
        repo_root / "config/entry_quality_v2.yaml",
        repo_root=repo_root,
    )
    regime = FactorizedRegimeAnalyzer(entry_config.regime_settings)
    rows = []
    for index, closed_at in enumerate(timestamps, start=1):
        batch = engine.evaluate_replay(
            _snapshot(
                candles,
                closed_at=closed_at,
                history_bars=config.minimum_history_bars,
                symbol_set_hash=brain.universe.symbol_set_hash,
            )
        )
        split = _split_name(config, closed_at)
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(batch["results"][symbol], symbol)
            features = _mapping(result["research_features"], "research_features")
            observed_regime = regime.observe(
                RegimeV2Observation(
                    symbol=symbol,
                    timestamp=closed_at,
                    market_direction_6=float(features["market_direction_6"]),
                    range_mean_24=float(features["range_mean_24"]),
                    range_expansion=float(features["range_expansion"]),
                    chop_12=float(features["chop_12"]),
                    trend_strength_12=float(features["trend_strength_12"]),
                )
            )
            if split is None or not bool(result["selected"]):
                continue
            candidate = _mapping(result["candidate"], "candidate")
            if not str(candidate["side"]).endswith("SHORT"):
                raise CommitteeV21FitError(
                    "AEGIS_COMMITTEE_V21_NON_SHORT_CONTROL_SELECTION"
                )
            net, mae, mfe = _outcome(
                candles,
                symbol=symbol,
                signal_timestamp=closed_at,
                horizon_bars=config.contract.horizon_bars,
                cost=config.contract.round_trip_cost_fraction,
            )
            observation = committee_v21_observation(
                result,
                symbol=symbol,
                primary_overlay={
                    "regime": to_primitive(observed_regime),
                },
            )
            rows.append(
                {
                    "split": split,
                    "symbol": symbol,
                    "signal_timestamp": _iso(closed_at),
                    "observation": dict(observation),
                    "net_return_fraction": net,
                    "mae_fraction": mae,
                    "mfe_fraction": mfe,
                    "adverse_label": int(net <= 0.0),
                    "feature_vector_hash": result["feature_vector_hash"],
                }
            )
        if progress is not None and (index == len(timestamps) or index % 250 == 0):
            progress(index, len(timestamps))
    if not rows:
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_DATASET_EMPTY")
    return tuple(rows), duplicate_audit


def _stats(
    contract: CommitteeV21Contract,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    means = {}
    scales = {}
    for name in contract.numeric_terms:
        values = [
            _finite(_mapping(row["observation"], "observation")[name], name)
            for row in rows
        ]
        means[name] = statistics.fmean(values)
        scale = statistics.pstdev(values)
        scales[name] = scale if scale > 0.0 else 1.0
    return means, scales


def _matrix(
    contract: CommitteeV21Contract,
    means: Mapping[str, float],
    scales: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    terms = basis_term_names(contract)
    return np.asarray(
        [
            [
                committee_v21_basis_from_stats(
                    contract,
                    means,
                    scales,
                    _mapping(row["observation"], "observation"),
                )[term]
                for term in terms
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CommitteeV21FitError("cannot compute empty quantile")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _ece(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bins: int,
) -> tuple[float, tuple[Mapping[str, Any], ...]]:
    total = len(labels)
    if total == 0:
        return math.nan, ()
    records = []
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(probabilities)
            if lower <= probability < upper
            or index == bins - 1
            and probability == upper
        ]
        if not members:
            records.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_probability": None,
                    "observed_adverse_rate": None,
                }
            )
            continue
        mean_probability = statistics.fmean(probabilities[pos] for pos in members)
        observed = statistics.fmean(labels[pos] for pos in members)
        error += len(members) / total * abs(mean_probability - observed)
        records.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_probability": mean_probability,
                "observed_adverse_rate": observed,
            }
        )
    return error, tuple(records)


def _performance(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    returns = [float(row["net_return_fraction"]) for row in rows]
    maes = [float(row["mae_fraction"]) for row in rows]
    return {
        "episodes": len(rows),
        "mean_net_return_fraction": (statistics.fmean(returns) if returns else None),
        "win_rate": (
            sum(value > 0.0 for value in returns) / len(returns) if returns else None
        ),
        "mean_mae_fraction": statistics.fmean(maes) if maes else None,
        "p90_mae_fraction": _quantile(maes, 0.90) if maes else None,
        "worst_decile_mean_net_return_fraction": (
            statistics.fmean(sorted(returns)[: max(1, math.ceil(len(returns) * 0.10))])
            if returns
            else None
        ),
    }


def _evaluate_split(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    *,
    threshold: float,
    bins: int,
) -> Mapping[str, Any]:
    labels = [int(row["adverse_label"]) for row in rows]
    retained = [
        row for row, probability in zip(rows, probabilities) if probability <= threshold
    ]
    waited = [
        row for row, probability in zip(rows, probabilities) if probability > threshold
    ]
    ece, calibration_bins = _ece(labels, probabilities, bins)
    base_rate = statistics.fmean(labels)
    risk_bands = {}
    for name, lower, upper in (
        ("LOW", 0.0, 1.0 / 3.0),
        ("MEDIUM", 1.0 / 3.0, 2.0 / 3.0),
        ("HIGH", 2.0 / 3.0, 1.0),
    ):
        indices = [
            index
            for index, value in enumerate(probabilities)
            if lower <= value < upper or name == "HIGH" and value == upper
        ]
        risk_bands[name] = {
            "episodes": len(indices),
            "mean_probability": (
                statistics.fmean(probabilities[index] for index in indices)
                if indices
                else None
            ),
            "observed_adverse_rate": (
                statistics.fmean(labels[index] for index in indices)
                if indices
                else None
            ),
        }
    return {
        "episodes": len(rows),
        "adverse_base_rate": base_rate,
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(set(labels)) == 2
            else None
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "constant_base_rate_brier": float(
            brier_score_loss(labels, [base_rate] * len(labels))
        ),
        "expected_calibration_error": ece,
        "calibration_bins": calibration_bins,
        "risk_bands": risk_bands,
        "retained_coverage": len(retained) / len(rows),
        "control": _performance(rows),
        "retained": _performance(retained),
        "waited": _performance(waited),
        "policy_mean_net_return_fraction": (
            math.fsum(float(row["net_return_fraction"]) for row in retained) / len(rows)
        ),
        "mean_paired_delta_fraction": (
            -math.fsum(float(row["net_return_fraction"]) for row in waited) / len(rows)
        ),
    }


def fit_committee_v21(
    config: CommitteeV21FitConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: datetime | None = None,
    duplicate_audit: Mapping[str, int] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    training = [row for row in rows if row["split"] == "TRAINING"]
    calibration = [row for row in rows if row["split"] == "CALIBRATION"]
    diagnostic = [row for row in rows if row["split"] == "DIAGNOSTIC_ONLY"]
    if min(len(training), len(calibration), len(diagnostic)) == 0 or any(
        len({int(row["adverse_label"]) for row in split}) != 2
        for split in (training, calibration, diagnostic)
    ):
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_CLASS_COVERAGE_INVALID")
    means, scales = _stats(config.contract, training)
    train_x = _matrix(config.contract, means, scales, training)
    train_y = np.asarray(
        [int(row["adverse_label"]) for row in training],
        dtype=np.int64,
    )
    model = LogisticRegression(
        C=config.regularization_c,
        class_weight="balanced",
        max_iter=config.maximum_iterations,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    model.fit(train_x, train_y)
    if int(model.n_iter_[0]) >= config.maximum_iterations:
        raise CommitteeV21FitError("AEGIS_COMMITTEE_V21_FIT_DID_NOT_CONVERGE")
    calibration_x = _matrix(
        config.contract,
        means,
        scales,
        calibration,
    )
    calibration_y = np.asarray(
        [int(row["adverse_label"]) for row in calibration],
        dtype=np.int64,
    )
    calibration_logits = model.decision_function(calibration_x).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1_000_000.0,
        class_weight=None,
        max_iter=config.maximum_iterations,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    calibrator.fit(calibration_logits, calibration_y)
    slope = float(calibrator.coef_[0][0])
    calibration_intercept = float(calibrator.intercept_[0])

    def probabilities(population: Sequence[Mapping[str, Any]]) -> list[float]:
        matrix = _matrix(config.contract, means, scales, population)
        logits = model.decision_function(matrix)
        calibrated_logits = slope * logits + calibration_intercept
        calibrated = []
        for value in calibrated_logits:
            scalar = float(value)
            if scalar >= 0.0:
                decay = math.exp(-scalar)
                probability = 1.0 / (1.0 + decay)
            else:
                growth = math.exp(scalar)
                probability = growth / (1.0 + growth)
            calibrated.append(
                min(
                    config.contract.probability_maximum,
                    max(config.contract.probability_minimum, probability),
                )
            )
        return calibrated

    def raw_probabilities(
        population: Sequence[Mapping[str, Any]],
    ) -> list[float]:
        matrix = _matrix(config.contract, means, scales, population)
        return [float(value) for value in model.predict_proba(matrix)[:, 1]]

    calibration_probabilities = probabilities(calibration)
    threshold = _quantile(
        calibration_probabilities,
        config.retention_quantile,
    )
    terms = basis_term_names(config.contract)
    artifact = {
        "schema_id": ARTIFACT_SCHEMA,
        "model_id": "aegis-specialized-committee-v21-risk-v1",
        "contract_sha256": config.contract.sha256,
        "feature_schema": "aegis-features-v2",
        "feature_count": 83,
        "objective": "P_NET_RETURN_12_BARS_LESS_THAN_OR_EQUAL_TO_ZERO",
        "standardization": {
            "source": "TRAINING_SPLIT_ONLY",
            "means": means,
            "scales": scales,
        },
        "basis_terms": list(terms),
        "logistic": {
            "family": "L2_LOGISTIC_WITH_EXPLICIT_INTERACTIONS",
            "regularization_c": config.regularization_c,
            "class_weight": "balanced",
            "intercept": float(model.intercept_[0]),
            "coefficients": {
                name: float(value) for name, value in zip(terms, model.coef_[0])
            },
            "iterations": int(model.n_iter_[0]),
        },
        "calibration": {
            "method": "PLATT_LOGISTIC",
            "source": "CALIBRATION_SPLIT_ONLY",
            "slope": slope,
            "intercept": calibration_intercept,
        },
        "threshold": {
            "source": "CALIBRATION_RISK_DISTRIBUTION_Q70",
            "retention_quantile": config.retention_quantile,
            "calibrated_risk_probability": threshold,
        },
        "provenance": {
            "database_sha256": config.database_sha256,
            "training_start_utc": _iso(config.training_start),
            "training_end_utc": _iso(config.training_end),
            "calibration_start_utc": _iso(config.calibration_start),
            "calibration_end_utc": _iso(config.calibration_end),
            "diagnostic_start_utc": _iso(config.diagnostic_start),
            "diagnostic_end_utc": _iso(config.diagnostic_end),
            "training_episodes": len(training),
            "calibration_episodes": len(calibration),
            "diagnostic_episodes": len(diagnostic),
            "generated_at_utc": _iso(generated_at or datetime.now(timezone.utc)),
        },
        "runtime_training": False,
        "exchange_authority": False,
        "automatic_promotion": False,
    }
    split_reports = {}
    for name, population in (
        ("training", training),
        ("calibration", calibration),
        ("diagnostic_only", diagnostic),
    ):
        split_report = dict(
            _evaluate_split(
                population,
                probabilities(population),
                threshold=threshold,
                bins=config.calibration_bins,
            )
        )
        labels = [int(row["adverse_label"]) for row in population]
        raw = raw_probabilities(population)
        split_report["base_model_roc_auc"] = float(roc_auc_score(labels, raw))
        split_report["calibrated_model_roc_auc"] = split_report["roc_auc"]
        split_reports[name] = split_report
    report = {
        "schema_id": "aegis-specialized-committee-v21-fit-report-v1",
        "experiment_id": config.contract.experiment_id,
        "preregistration_sha256": config.contract.sha256,
        "artifact_path": str(config.artifact_path),
        "dataset": {
            "database_sha256": config.database_sha256,
            "duplicate_candle_policy": config.duplicate_candle_policy,
            "duplicate_candle_audit": dict(duplicate_audit or {}),
            "total_selected_episodes": len(rows),
            "training_episodes": len(training),
            "calibration_episodes": len(calibration),
            "diagnostic_only_episodes": len(diagnostic),
        },
        "model": {
            "basis_term_count": len(terms),
            "iterations": int(model.n_iter_[0]),
            "calibration_slope": slope,
            "calibration_intercept": calibration_intercept,
            "calibrated_risk_threshold": threshold,
        },
        "splits": split_reports,
        "diagnostic_promotion_use": "PROHIBITED",
        "prospective_shadow_required": True,
        "runtime_training": False,
        "automatic_promotion": False,
        "exchange_mutations": 0,
    }
    return artifact, report


def write_committee_v21_fit_outputs(
    config: CommitteeV21FitConfig,
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
) -> Mapping[str, str]:
    artifact_sha256 = atomic_write_json(
        config.artifact_path,
        artifact,
        immutable=True,
    )
    config.private_fit_root.mkdir(parents=True, exist_ok=True)
    os.chmod(config.private_fit_root, 0o700)
    report_path = config.private_fit_root / "fit_report.json"
    dataset_manifest = config.private_fit_root / "dataset_manifest.json"
    report_payload = {
        **dict(report),
        "artifact_sha256": artifact_sha256,
    }
    report_sha256 = atomic_write_json(
        report_path,
        report_payload,
        immutable=False,
    )
    os.chmod(report_path, 0o600)
    manifest_sha256 = atomic_write_json(
        dataset_manifest,
        {
            "schema_id": "aegis-specialized-committee-v21-dataset-manifest-v1",
            "preregistration_sha256": config.contract.sha256,
            "database_path": str(config.database),
            "database_sha256": config.database_sha256,
            "raw_dataset_persisted": False,
            "exchange_mutations": 0,
        },
        immutable=False,
    )
    os.chmod(dataset_manifest, 0o600)
    return {
        "artifact_path": str(config.artifact_path),
        "artifact_sha256": artifact_sha256,
        "fit_report_path": str(report_path),
        "fit_report_sha256": report_sha256,
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": manifest_sha256,
    }


def report_summary(report: Mapping[str, Any]) -> str:
    diagnostic = _mapping(
        _mapping(report["splits"], "splits")["diagnostic_only"],
        "diagnostic_only",
    )
    return canonical_json(
        {
            "experiment_id": report["experiment_id"],
            "diagnostic_episodes": diagnostic["episodes"],
            "diagnostic_roc_auc": diagnostic["roc_auc"],
            "diagnostic_brier_score": diagnostic["brier_score"],
            "diagnostic_expected_calibration_error": diagnostic[
                "expected_calibration_error"
            ],
            "diagnostic_retained_coverage": diagnostic["retained_coverage"],
            "diagnostic_mean_paired_delta_fraction": diagnostic[
                "mean_paired_delta_fraction"
            ],
            "promotion_use": "PROHIBITED",
            "exchange_mutations": 0,
        }
    )
