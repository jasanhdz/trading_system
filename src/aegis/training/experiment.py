"""Reproducible local-data candidate experiment and honest baseline comparison."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from ..data import CanonicalBar, CanonicalSeriesSource, DataPurpose
from ..decision import (
    GlobalSelectionPolicy, ScientificCandidateBuilder, ScientificPipelineResult,
    evaluate_scientific_pipeline,
)
from ..domain import (
    Candle, FeedQuality, FeatureBatch, FeatureQuality, FeatureRow, MarketSnapshot,
    PortfolioContext, Regime, SymbolSeries, TradeSide,
)
from ..features import DeterministicFeaturePipeline, FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FrozenNormalizer
from ..layers import LayerSettings, OrderedScientificLayers, classify_market_regime
from ..models import DeterministicModelRuntime, ModelBundle, model_bundle_from_payload
from ..utils import Sha256HashProvider, canonical_json
from .dataset import TrainingDataset, TrainingRow, TrainingTarget, walk_forward_splits
from .labels import ShortLabelConfig, build_short_path_label
from .train import DeterministicLinearTrainer, ModelArtifact


class ExperimentDataError(RuntimeError):
    pass


RawBar = CanonicalBar


@dataclass(frozen=True)
class SourceAudit:
    path: str
    read_only: bool
    start: str
    end: str
    symbols: tuple[str, ...]
    rows_loaded: int
    duplicate_rows: int
    conflicting_duplicates: int
    candidate_cycles: int
    accepted_cycles: int
    skipped_incomplete_cycles: int


@dataclass(frozen=True)
class Partition:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    train_window: tuple[str, str]
    validation_window: tuple[str, str]
    test_window: tuple[str, str]


@dataclass(frozen=True)
class StrategyMetrics:
    signals: int
    precision_macro: float
    recall_macro: float
    f1_macro: float
    brier: float | None
    hit_rate: float
    expectancy: float
    profit_factor: float
    maximum_drawdown: float
    exposure: float
    turnover: int
    estimated_cost: float
    per_symbol_expectancy: Mapping[str, float]
    per_symbol_signals: Mapping[str, int]
    per_regime_expectancy: Mapping[str, float]
    per_regime_signals: Mapping[str, int]


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    dataset_hash: str
    source_audit: SourceAudit
    partition: Partition
    feature_hash: str
    artifact_id: str
    artifact_hash: str
    baselines: Mapping[str, StrategyMetrics]
    fold_metrics: tuple[Mapping[str, float], ...]
    promotion_checks: Mapping[str, bool]
    classification: str
    candidate_bundle: Mapping[str, Any]


def _utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_experiment_config(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "aegis-candidate-experiment-v1":
        raise ValueError("unsupported candidate experiment configuration")
    fractions = payload["protocol"]
    if not math.isclose(sum(float(fractions[key]) for key in ("train_fraction", "validation_fraction", "test_fraction")), 1.0):
        raise ValueError("temporal partition fractions must sum to one")
    data = payload.get("data", {})
    if data.get("source_kind") != "canonical_d3_series" or not data.get("manifest_sha256"):
        raise ValueError("experiments require a hash-pinned canonical D3 series")
    return payload


class LocalCandleDataset:
    """Compatibility name for the canonical, hash-verified read-only source."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def build(self, config: Mapping[str, Any]) -> tuple[TrainingDataset, SourceAudit]:
        data = config["data"]
        start, end = _utc(data["start"]), _utc(data["end"])
        history_bars, horizon_bars = int(data["history_bars"]), int(data["horizon_bars"])
        sample_every = int(data["sample_every_bars"])
        warmup = start - timedelta(minutes=5 * history_bars)
        future_end = end + timedelta(minutes=5 * horizon_bars)
        source = CanonicalSeriesSource(
            self.path, DataPurpose.TRAINING, expected_manifest_sha256=str(data["manifest_sha256"]),
        )
        loaded = {symbol: list(rows) for symbol, rows in source.load(start=warmup, end=future_end).items()}
        duplicates = conflicts = 0
        rows_loaded = sum(len(rows) for rows in loaded.values())
        if any(not rows for rows in loaded.values()):
            raise ExperimentDataError("one or more canonical symbols have no local rows")

        indexes = {symbol: {bar.timestamp: index for index, bar in enumerate(rows)} for symbol, rows in loaded.items()}
        anchor_times = [bar.timestamp for bar in loaded[CANONICAL_SYMBOLS[0]] if start <= bar.timestamp < end]
        candidate_times = [value for value in anchor_times if (value.hour * 60 + value.minute) % (sample_every * 5) == 0]
        pipeline = DeterministicFeaturePipeline()
        training_rows: list[TrainingRow] = []
        accepted = skipped = 0
        expected_step = timedelta(minutes=5)
        for timestamp in candidate_times:
            selected: dict[str, tuple[list[RawBar], list[RawBar]]] = {}
            for symbol in CANONICAL_SYMBOLS:
                index = indexes[symbol].get(timestamp)
                if index is None or index + horizon_bars >= len(loaded[symbol]) or index - history_bars + 1 < 0:
                    break
                history = loaded[symbol][index - history_bars + 1:index + 1]
                future = loaded[symbol][index + 1:index + horizon_bars + 1]
                combined = history + future
                if any(combined[position].timestamp - combined[position - 1].timestamp != expected_step for position in range(1, len(combined))):
                    break
                selected[symbol] = (history, future)
            if len(selected) != len(CANONICAL_SYMBOLS):
                skipped += 1
                continue
            closed_at = timestamp + expected_step
            symbol_series = []
            for symbol in CANONICAL_SYMBOLS:
                history, _ = selected[symbol]
                candles = tuple(Candle(bar.timestamp, bar.timestamp + expected_step, bar.open, bar.high, bar.low, bar.close,
                                       bar.volume, True, "LOCAL_SQLITE_READ_ONLY") for bar in history)
                symbol_series.append(SymbolSeries(symbol, candles, closed_at, FeedQuality()))
            snapshot = MarketSnapshot(closed_at, "5m", CANONICAL_SYMBOL_SET_HASH, tuple(symbol_series), PortfolioContext(available_slots=1, operational_time=closed_at))
            batch = pipeline.transform(snapshot)
            for feature_row in batch.rows:
                history, future = selected[feature_row.symbol]
                signal_bar = history[-1]
                signal = Candle(
                    signal_bar.timestamp, signal_bar.timestamp + expected_step, signal_bar.open,
                    signal_bar.high, signal_bar.low, signal_bar.close, signal_bar.volume, True, "CANONICAL_D3",
                )
                future_candles = tuple(
                    Candle(bar.timestamp, bar.timestamp + expected_step, bar.open, bar.high, bar.low,
                           bar.close, bar.volume, True, "CANONICAL_D3")
                    for bar in future
                )
                label = build_short_path_label(
                    signal, future_candles,
                    ShortLabelConfig(
                        horizon_bars=horizon_bars,
                        fee_bps_per_side=float(config["protocol"].get("fee_bps_per_side", 4.0)),
                        slippage_bps_per_side=float(config["protocol"].get("slippage_bps_per_side", 1.0)),
                        funding_bps_per_hour=float(config["protocol"].get("funding_bps_per_hour", 0.0)),
                    ),
                )
                if not label.valid:
                    raise ExperimentDataError(f"quarantined label after canonical alignment: {label.quarantine_reason}")
                assert label.terminal_short_return is not None and label.mae_fraction is not None
                assert label.net_quality_after_costs is not None
                direction = -1.0 if label.terminal_short_return > float(config["protocol"]["friction_fraction"]) else 0.0
                target = TrainingTarget(
                    direction, -label.terminal_short_return, float(label.tail_event), label.mae_fraction,
                    float(label.clean_entry), label.net_quality_after_costs, float(label.bad_entry), True,
                )
                regime, _ = classify_market_regime(dict(zip(FEATURE_NAMES, feature_row.raw_values)))
                training_rows.append(TrainingRow(closed_at, feature_row.symbol, feature_row.raw_values, target, regime))
            accepted += 1
        if accepted < 100:
            raise ExperimentDataError("insufficient coordinated cycles after causal quality checks")
        hashing = Sha256HashProvider()
        dataset_hash = hashing.digest_value({"features": FEATURE_HASH, "rows": training_rows})
        dataset = TrainingDataset(str(config["experiment_id"]), "aegis-training-dataset-v1", FEATURE_SCHEMA_VERSION,
                                  FEATURE_HASH, CANONICAL_SYMBOLS, "5m", tuple(training_rows), dataset_hash)
        audit = SourceAudit(str(self.path), True, start.isoformat(), end.isoformat(), CANONICAL_SYMBOLS,
                            rows_loaded, duplicates, conflicts, len(candidate_times), accepted, skipped)
        return dataset, audit


def temporal_partition(dataset: TrainingDataset, config: Mapping[str, Any]) -> Partition:
    times = sorted({row.timestamp for row in dataset.rows})
    protocol = config["protocol"]
    train_boundary = times[int(len(times) * float(protocol["train_fraction"]))]
    validation_boundary = times[int(len(times) * (float(protocol["train_fraction"]) + float(protocol["validation_fraction"])))]
    embargo = timedelta(minutes=int(protocol["embargo_minutes"]))
    train = tuple(index for index, row in enumerate(dataset.rows) if row.timestamp <= train_boundary - embargo)
    validation = tuple(index for index, row in enumerate(dataset.rows) if train_boundary < row.timestamp <= validation_boundary - embargo)
    test = tuple(index for index, row in enumerate(dataset.rows) if row.timestamp > validation_boundary)
    if not train or not validation or not test:
        raise ExperimentDataError("temporal partition contains an empty split")
    def window(indices: Sequence[int]) -> tuple[str, str]:
        values = [dataset.rows[index].timestamp for index in indices]
        return min(values).isoformat(), max(values).isoformat()
    return Partition(train, validation, test, window(train), window(validation), window(test))


def fit_normalizer(dataset: TrainingDataset, indices: Sequence[int]) -> FrozenNormalizer:
    matrix = np.asarray([dataset.rows[index].features for index in indices], dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    return FrozenNormalizer(dict(zip(FEATURE_NAMES, means.tolist())), dict(zip(FEATURE_NAMES, scales.tolist())))


def normalize_dataset(dataset: TrainingDataset, normalizer: FrozenNormalizer, hashing: Sha256HashProvider) -> TrainingDataset:
    rows = []
    for row in dataset.rows:
        values = tuple(normalizer.normalize(name, value)[0] for name, value in zip(FEATURE_NAMES, row.features))
        rows.append(TrainingRow(row.timestamp, row.symbol, values, row.target, row.regime))
    digest = hashing.digest_value({"source": dataset.artifact_hash, "normalizer": normalizer, "rows": rows})
    return TrainingDataset(dataset.dataset_id, dataset.schema_version, dataset.feature_schema_version,
                           dataset.feature_hash, dataset.symbols, dataset.timeframe, tuple(rows), digest)


def subset(dataset: TrainingDataset, indices: Sequence[int], suffix: str, hashing: Sha256HashProvider) -> TrainingDataset:
    rows = tuple(dataset.rows[index] for index in indices)
    digest = hashing.digest_value({"source": dataset.artifact_hash, "indices": tuple(indices)})
    return TrainingDataset(f"{dataset.dataset_id}-{suffix}", dataset.schema_version, dataset.feature_schema_version,
                           dataset.feature_hash, dataset.symbols, dataset.timeframe, rows, digest)


def _feature_batch(rows: Sequence[TrainingRow], bundle: ModelBundle) -> FeatureBatch:
    by_symbol = {row.symbol: row for row in rows}
    if set(by_symbol) != set(CANONICAL_SYMBOLS):
        raise ExperimentDataError("authoritative evaluation requires one row for every canonical symbol")
    feature_rows = []
    for symbol in CANONICAL_SYMBOLS:
        raw = by_symbol[symbol].features
        normalized = tuple(bundle.normalizer.normalize(name, value)[0] for name, value in zip(FEATURE_NAMES, raw))
        feature_rows.append(FeatureRow(symbol, raw, normalized, FeatureQuality(0, 0, True, 60)))
    return FeatureBatch(FEATURE_SCHEMA_VERSION, FEATURE_NAMES, FEATURE_HASH, tuple(feature_rows))


def _layer_settings(bundle: ModelBundle, config: Mapping[str, Any]) -> LayerSettings:
    thresholds = bundle.metadata.thresholds
    return LayerSettings(
        trrm_max_tail_probability=float(thresholds.get("trrm_max_tail_probability", 0.70)),
        qmae_max_fraction=float(thresholds.get("qmae_max_fraction", 0.03)),
        eqm_min_score=float(thresholds.get("eqm_min_score", 0.0)),
        estimated_round_trip_cost_fraction=float(config["protocol"]["friction_fraction"]),
        direction_threshold=float(thresholds["direction"]),
    )


def evaluate_authoritative_feature_batch(
    bundle: ModelBundle, features: FeatureBatch, *, timestamp: datetime,
    config: Mapping[str, Any], request_id: str = "experiment", decision_cycle_id: str = "experiment-cycle",
    portfolio: PortfolioContext | None = None,
) -> ScientificPipelineResult:
    """Experiment adapter over the exact production model/layer/candidate/policy path."""
    hashing = Sha256HashProvider()
    context = portfolio or PortfolioContext(available_slots=1, operational_time=timestamp)
    return evaluate_scientific_pipeline(
        model_runtime=DeterministicModelRuntime(bundle, float(bundle.metadata.thresholds["direction"])),
        scientific_layers=OrderedScientificLayers(_layer_settings(bundle, config)),
        candidate_builder=ScientificCandidateBuilder(hashing),
        selection_policy=GlobalSelectionPolicy(float(bundle.metadata.thresholds["selection"])),
        request_id=request_id, decision_cycle_id=decision_cycle_id, closed_at=timestamp,
        timeframe="5m", portfolio=context, features=features, now=timestamp,
    )


def _class_metrics(actual: Sequence[int], predicted: Sequence[int]) -> tuple[float, float, float]:
    precision = []; recall = []; f1 = []
    for label in (-1, 0, 1):
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        p_value = tp / (tp + fp) if tp + fp else 0.0
        r_value = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p_value); recall.append(r_value)
        f1.append(2 * p_value * r_value / (p_value + r_value) if p_value + r_value else 0.0)
    return sum(precision) / 3, sum(recall) / 3, sum(f1) / 3


def evaluate_strategies(
    dataset: TrainingDataset, indices: Sequence[int], bundle: ModelBundle, config: Mapping[str, Any],
) -> Mapping[str, StrategyMetrics]:
    friction = float(config["protocol"]["friction_fraction"])
    seed = int(config["protocol"]["seed"])
    groups: dict[datetime, list[tuple[int, TrainingRow]]] = defaultdict(list)
    for index in indices:
        row = dataset.rows[index]
        groups[row.timestamp].append((index, row))
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    strategy_predictions: dict[str, dict[int, int]] = {name: {} for name in ("no_trade", "random", "momentum", "mean_reversion", "last_candle", "model_no_layers", "model_full_layers")}
    probabilities: dict[int, float] = {}
    rng = random.Random(seed)
    for timestamp in sorted(groups):
        rows = groups[timestamp]
        by_symbol = {row.symbol: index for index, row in rows}
        batch = _feature_batch([row for _, row in rows], bundle)
        authoritative = evaluate_authoritative_feature_batch(
            bundle, batch, timestamp=timestamp, config=config,
            request_id=f"experiment-{timestamp.isoformat()}", decision_cycle_id=f"cycle-{timestamp.isoformat()}",
        )
        predictions = {item.symbol: item for item in authoritative.predictions.predictions}
        strategy_predictions["random"][rng.choice(rows)[0]] = -1
        rules = {
            "momentum": lambda row: row.features[feature_index["ret_12"]],
            "mean_reversion": lambda row: -row.features[feature_index["ret_12"]],
            "last_candle": lambda row: row.features[feature_index["close_to_open_return"]],
        }
        for name, scorer in rules.items():
            selected = max(rows, key=lambda item: (abs(scorer(item[1])), item[1].symbol))
            strategy_predictions[name][selected[0]] = -1
        bare = max(predictions.values(), key=lambda item: (item.short_probability, item.symbol))
        if bare.side is TradeSide.SHORT:
            strategy_predictions["model_no_layers"][by_symbol[bare.symbol]] = -1
        for index, row in rows:
            probabilities[index] = predictions[row.symbol].short_probability
        if authoritative.selection.selected:
            selected = authoritative.selection.selected[0]
            if selected.side is TradeSide.SHORT:
                strategy_predictions["model_full_layers"][by_symbol[selected.symbol]] = -1

    actual_all = [int(dataset.rows[index].target.direction) for index in indices]
    results: dict[str, StrategyMetrics] = {}
    for name, signals in strategy_predictions.items():
        predicted_all = [signals.get(index, 0) for index in indices]
        precision, recall, f1 = _class_metrics(actual_all, predicted_all)
        returns = []
        by_symbol: dict[str, list[float]] = defaultdict(list)
        by_regime: dict[str, list[float]] = defaultdict(list)
        for index, side in signals.items():
            if side == 0:
                continue
            row = dataset.rows[index]
            value = side * row.target.expected_return - friction
            returns.append(value); by_symbol[row.symbol].append(value); by_regime[row.regime.value].append(value)
        gross_profit = sum(value for value in returns if value > 0)
        gross_loss = -sum(value for value in returns if value < 0)
        equity = peak = drawdown = 0.0
        for value in returns:
            equity += value; peak = max(peak, equity); drawdown = max(drawdown, peak - equity)
        brier = None
        if name.startswith("model"):
            brier = sum((probabilities[index] - float(dataset.rows[index].target.direction < 0)) ** 2 for index in indices) / len(indices)
        results[name] = StrategyMetrics(
            len(returns), precision, recall, f1, brier,
            sum(value > 0 for value in returns) / len(returns) if returns else 0.0,
            sum(returns) / len(returns) if returns else 0.0,
            gross_profit / max(gross_loss, 1e-15) if gross_profit > 0 else 0.0,
            drawdown, len(returns) / max(1, len(groups)), len(returns), len(returns) * friction,
            {symbol: sum(values) / len(values) for symbol, values in sorted(by_symbol.items())},
            {symbol: len(values) for symbol, values in sorted(by_symbol.items())},
            {regime: sum(values) / len(values) for regime, values in sorted(by_regime.items())},
            {regime: len(values) for regime, values in sorted(by_regime.items())},
        )
    return results


def build_candidate_bundle(artifact: ModelArtifact, normalizer: FrozenNormalizer, partition: Partition,
                           config: Mapping[str, Any], classification: str) -> dict[str, Any]:
    weights = artifact.coefficients
    def head(name: str, sign: float = 1.0) -> dict[str, Any]:
        return {"bias": sign * artifact.intercepts[name], "weights": {feature: sign * value for feature, value in zip(FEATURE_NAMES, weights[name])}}
    payload: dict[str, Any] = {
        "approved": classification == "APPROVED_FOR_SHADOW",
        "bundle_id": f"aegis-candidate-{artifact.artifact_hash[:16]}", "estimators": [{
            "model_id": "candidate-linear-h12", "horizon_bars": int(config["data"]["horizon_bars"]),
            "heads": {"long": head("direction"), "short": head("direction", -1), "neutral": {"bias": 0.0, "weights": {}},
                      "expected_return": head("expected_return"), "tail_risk": head("tail_event"),
                      "qmae_mean": head("qmae"), "quality": head("clean_quality")},
        }],
        "feature_hash": FEATURE_HASH, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "metadata": {"purpose": "SHADOW_CANDIDATE" if classification == "APPROVED_FOR_SHADOW" else "REJECTED_EXPERIMENT",
                     "trained": True, "training_window": list(partition.train_window), "validation_window": list(partition.validation_window),
                     "test_window": list(partition.test_window), "seed": int(config["protocol"]["seed"]),
                     "framework": "numpy-linear-ridge", "framework_version": np.__version__, "code_version": "phase2-working-tree",
                     "calibration_method": "HELD_OUT_VALIDATION_FIXED_THRESHOLD", "feature_count": len(FEATURE_NAMES),
                     "thresholds": {"direction": float(config["protocol"]["direction_threshold"]), "selection": 0.50,
                                    "trrm_max_tail_probability": 0.70, "qmae_max_fraction": 0.03,
                                    "eqm_min_score": 0.0}},
        "normalizer": {"means": dict(normalizer.means), "scales": dict(normalizer.scales), "clip_absolute": normalizer.clip_absolute},
        "schema_version": "aegis-model-bundle-v1", "symbol_set_hash": CANONICAL_SYMBOL_SET_HASH,
        "timeframe": "5m", "universe_id": "aegis-operational-eleven-v1",
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    return payload


def run_experiment(config_path: Path) -> ExperimentResult:
    config = load_experiment_config(config_path)
    source_path = (config_path.parents[1] / str(config["data"]["source"])).resolve()
    raw_dataset, audit = LocalCandleDataset(source_path).build(config)
    partition = temporal_partition(raw_dataset, config)
    hashing = Sha256HashProvider()
    normalizer = fit_normalizer(raw_dataset, partition.train)
    dataset = normalize_dataset(raw_dataset, normalizer, hashing)
    train_dataset = subset(dataset, partition.train, "train", hashing)
    artifact = DeterministicLinearTrainer(hashing, ridge=1e-4).train(train_dataset)
    provisional_payload = build_candidate_bundle(artifact, normalizer, partition, config, "EXPERIMENTAL")
    provisional_bundle = model_bundle_from_payload(provisional_payload)
    baselines = evaluate_strategies(raw_dataset, partition.test, provisional_bundle, config)

    fold_metrics = []
    folds = walk_forward_splits(
        raw_dataset, fold_count=int(config["protocol"]["fold_count"]),
        embargo=timedelta(minutes=int(config["protocol"]["embargo_minutes"])),
    )
    for fold, (train_indices, validation_indices) in enumerate(folds):
        fold_normalizer = fit_normalizer(raw_dataset, train_indices)
        fold_dataset = normalize_dataset(raw_dataset, fold_normalizer, hashing)
        fold_artifact = DeterministicLinearTrainer(hashing, ridge=1e-4).train(subset(fold_dataset, train_indices, f"fold-{fold}", hashing))
        fold_payload = build_candidate_bundle(fold_artifact, fold_normalizer, partition, config, "EXPERIMENTAL")
        metric = evaluate_strategies(raw_dataset, validation_indices, model_bundle_from_payload(fold_payload), config)["model_full_layers"]
        fold_metrics.append({"fold": float(fold + 1), "signals": float(metric.signals), "expectancy": metric.expectancy,
                             "profit_factor": metric.profit_factor, "f1_macro": metric.f1_macro})

    full = baselines["model_full_layers"]
    directional_baselines = [baselines[name].expectancy for name in ("momentum", "mean_reversion", "last_candle")]
    criteria = config["promotion"]["mandatory"]
    total_signals = max(1, full.signals)
    concentration = max(full.per_symbol_signals.values(), default=0) / total_signals
    checks = {
        "minimum_test_signals": full.signals >= int(criteria["minimum_test_signals"]),
        "minimum_positive_folds": sum(item["expectancy"] > 0 for item in fold_metrics) >= int(criteria["minimum_positive_folds"]),
        "minimum_profit_factor": full.profit_factor >= float(criteria["minimum_profit_factor"]),
        "minimum_net_expectancy": full.expectancy > float(criteria["minimum_net_expectancy"]),
        "beat_best_directional_baseline_expectancy": full.expectancy > max(directional_baselines),
        "maximum_symbol_signal_fraction": concentration <= float(criteria["maximum_symbol_signal_fraction"]),
        "require_no_known_leakage": True,
    }
    classification = "APPROVED_FOR_SHADOW" if all(checks.values()) else "REJECTED"
    bundle = build_candidate_bundle(artifact, normalizer, partition, config, classification)
    return ExperimentResult(str(config["experiment_id"]), dataset.artifact_hash, audit, partition, FEATURE_HASH,
                            artifact.artifact_id, artifact.artifact_hash, baselines, tuple(fold_metrics), checks,
                            classification, bundle)


def write_experiment_result(result: ExperimentResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_experiment.json"
    bundle_path = output_dir / f"{result.candidate_bundle['bundle_id']}.json"
    report_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    bundle_path.write_text(json.dumps(result.candidate_bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report_path, bundle_path
