"""Train and validate a genuine LONG opportunity model on purged folds."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalSeriesSource, DataPurpose
from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.dataset import (
    TrainingDataset,
    load_and_build_e2_hourly_long_dataset,
)
from aegis.training.experiment import fit_normalizer
from aegis.training.labels import LONG_LABEL_SCHEMA_VERSION
from aegis.training.long_opportunity import (
    LongOpportunityArtifact,
    LongOpportunityTrainingContract,
    LongOpportunityTrainingRow,
    fit_long_opportunity_model,
    write_long_opportunity_artifact,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("LONG split timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _indices(
    dataset: TrainingDataset,
    start: datetime,
    end: datetime,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, row in enumerate(dataset.rows)
        if start <= row.timestamp <= end and row.target.label_valid
    )


def _rows(
    dataset: TrainingDataset,
    indices: Sequence[int],
    normalizer,
) -> tuple[LongOpportunityTrainingRow, ...]:
    return tuple(
        LongOpportunityTrainingRow(
            timestamp=dataset.rows[index].timestamp,
            symbol=dataset.rows[index].symbol,
            features=tuple(
                normalizer.normalize(name, value)[0]
                for name, value in zip(FEATURE_NAMES, dataset.rows[index].features)
            ),
            clean_opportunity=dataset.rows[index].target.clean_quality >= 0.5,
            mae_fraction=dataset.rows[index].target.qmae,
        )
        for index in indices
    )


def _probability_on_normalized(
    artifact: LongOpportunityArtifact,
    row: LongOpportunityTrainingRow,
) -> float:
    learner = artifact.learner
    if learner.tree_ensemble is not None:
        raw = learner.tree_ensemble.evaluate(row.features)
    else:
        logit = learner.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(learner.coefficients, row.features)
        )
        raw = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
    return learner.calibrator.apply(row.symbol, raw)


def _selected_indices(
    rows: Sequence[LongOpportunityTrainingRow],
    probabilities: Sequence[float],
    threshold: float,
) -> tuple[int, ...]:
    by_timestamp: dict[datetime, list[int]] = {}
    for index, row in enumerate(rows):
        if probabilities[index] >= threshold:
            by_timestamp.setdefault(row.timestamp, []).append(index)
    return tuple(
        max(indices, key=lambda index: (probabilities[index], rows[index].symbol))
        for _, indices in sorted(by_timestamp.items())
    )


def _symbol_percentile_scores(
    calibration_rows: Sequence[LongOpportunityTrainingRow],
    calibration_probabilities: Sequence[float],
    target_rows: Sequence[LongOpportunityTrainingRow],
    target_probabilities: Sequence[float],
) -> tuple[float, ...]:
    references: dict[str, np.ndarray] = {}
    for symbol in sorted({row.symbol for row in calibration_rows}):
        references[symbol] = np.sort(
            np.asarray(
                [
                    probability
                    for row, probability in zip(
                        calibration_rows, calibration_probabilities
                    )
                    if row.symbol == symbol
                ],
                dtype=np.float64,
            )
        )
    scores = []
    for row, probability in zip(target_rows, target_probabilities):
        reference = references.get(row.symbol)
        if reference is None or not len(reference):
            raise ValueError("LONG symbol calibration reference is missing")
        rank = int(np.searchsorted(reference, probability, side="right"))
        scores.append(rank / len(reference))
    return tuple(scores)


def _selection_metrics(
    dataset: TrainingDataset,
    source_indices: Sequence[int],
    selected: Sequence[int],
    *,
    round_trip_cost_fraction: float,
) -> Mapping[str, Any]:
    dataset_indices = [source_indices[index] for index in selected]
    rows = [dataset.rows[index] for index in dataset_indices]
    net_returns = [
        row.target.expected_return - round_trip_cost_fraction for row in rows
    ]
    symbols = [row.symbol for row in rows]
    timestamps = sorted(row.timestamp for row in rows)
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(timestamps, timestamps[1:])
    ]
    counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
    return {
        "signals": len(rows),
        "mean_net_expectancy": statistics.fmean(net_returns) if net_returns else 0.0,
        "win_rate": (
            sum(value > 0.0 for value in net_returns) / len(net_returns)
            if net_returns
            else 0.0
        ),
        "mean_mae": (
            statistics.fmean(row.target.qmae for row in rows) if rows else 0.0
        ),
        "symbol_counts": counts,
        "symbol_concentration": (
            max(counts.values()) / len(rows) if rows else 1.0
        ),
        "median_gap_hours": statistics.median(gaps) if gaps else None,
        "p95_gap_hours": (
            float(np.quantile(np.asarray(gaps), 0.95)) if gaps else None
        ),
        "maximum_gap_hours": max(gaps) if gaps else None,
    }


def _derive_threshold(
    dataset: TrainingDataset,
    calibration_indices: Sequence[int],
    rows: Sequence[LongOpportunityTrainingRow],
    probabilities: Sequence[float],
    config: Mapping[str, Any],
    *,
    round_trip_cost_fraction: float,
) -> tuple[float, Mapping[str, Any], bool]:
    candidates = sorted(float(value) for value in config["probability_quantiles"])
    evaluated = []
    for threshold in candidates:
        selected = _selected_indices(rows, probabilities, threshold)
        metrics = _selection_metrics(
            dataset,
            calibration_indices,
            selected,
            round_trip_cost_fraction=round_trip_cost_fraction,
        )
        valid = (
            metrics["signals"] >= int(config["minimum_calibration_selections"])
            and metrics["mean_net_expectancy"] > 0.0
            and metrics["mean_mae"] <= float(config["maximum_mean_mae"])
        )
        evaluated.append((threshold, metrics, valid))
    valid_rows = [item for item in evaluated if item[2]]
    pool = valid_rows or evaluated
    threshold, metrics, valid = max(
        pool,
        key=lambda item: (
            float(item[1]["mean_net_expectancy"]),
            -float(item[1]["mean_mae"]),
            int(item[1]["signals"]),
        ),
    )
    return threshold, metrics, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_candidate_l1.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = _mapping(yaml.safe_load(config_path.read_text()), "LONG config")
    if (
        config.get("schema_version")
        != "aegis-long-candidate-preregistration-v1"
        or config.get("side") != "LONG"
        or config.get("label_schema_version") != LONG_LABEL_SCHEMA_VERSION
    ):
        raise SystemExit("AEGIS_LONG_TRAINING_CONFIG_INVALID")
    if config["lockbox"]["access"] != "FORBIDDEN":
        raise SystemExit("AEGIS_LONG_LOCKBOX_ACCESS_PROHIBITED")

    source_config = _mapping(config["source"], "source")
    source = CanonicalSeriesSource(
        Path(str(source_config["path"])),
        DataPurpose.TRAINING,
        expected_manifest_sha256=str(source_config["manifest_sha256"]),
    )
    build = load_and_build_e2_hourly_long_dataset(source, config)
    dataset = build.dataset
    if (
        dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or dataset.feature_hash != FEATURE_HASH
        or dataset.symbols != CANONICAL_SYMBOLS
    ):
        raise SystemExit("AEGIS_LONG_FEATURE_AUTHORITY_MISMATCH")

    training = _mapping(config["training"], "training")
    threshold_config = _mapping(config["threshold_derivation"], "threshold")
    candidates = _mapping(training["candidates"], "candidates")
    fold_reports: dict[str, list[Mapping[str, Any]]] = {
        str(candidate): [] for candidate in candidates
    }
    fold_artifacts: dict[tuple[str, int], LongOpportunityArtifact] = {}
    cost = 0.001
    for fold in config["fold_protocol"]["folds"]:
        fold_id = int(fold["id"])
        train_indices = _indices(
            dataset, _utc(fold["train_start"]), _utc(fold["train_end"])
        )
        calibration_indices = _indices(
            dataset,
            _utc(fold["calibration_start"]),
            _utc(fold["calibration_end"]),
        )
        scoring_indices = _indices(
            dataset, _utc(fold["scoring_start"]), _utc(fold["scoring_end"])
        )
        normalizer = fit_normalizer(dataset, train_indices)
        blocks = {
            "train": _rows(dataset, train_indices, normalizer),
            "calibration": _rows(dataset, calibration_indices, normalizer),
            "scoring": _rows(dataset, scoring_indices, normalizer),
        }
        for candidate_id, parameters in candidates.items():
            contract = LongOpportunityTrainingContract(
                schema_version="aegis-long-opportunity-training-contract-v1",
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_hash=FEATURE_HASH,
                maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
                model_candidate_id=str(candidate_id),
                model_parameters=_mapping(parameters, str(candidate_id)),
                seed=int(training["seed"]) + fold_id,
                minimum_embargo_minutes=int(
                    config["fold_protocol"]["embargo_minutes"]
                ),
                minimum_symbol_calibration_rows=int(
                    training["minimum_symbol_calibration_rows"]
                ),
                symbol_calibration_shrinkage_rows=int(
                    training["symbol_calibration_shrinkage_rows"]
                ),
            )
            artifact = fit_long_opportunity_model(
                blocks["train"],
                blocks["calibration"],
                blocks["scoring"],
                contract,
                normalizer=normalizer,
            )
            calibration_probabilities = [
                _probability_on_normalized(artifact, row)
                for row in blocks["calibration"]
            ]
            calibration_percentiles = _symbol_percentile_scores(
                blocks["calibration"],
                calibration_probabilities,
                blocks["calibration"],
                calibration_probabilities,
            )
            threshold, calibration_metrics, threshold_valid = _derive_threshold(
                dataset,
                calibration_indices,
                blocks["calibration"],
                calibration_percentiles,
                threshold_config,
                round_trip_cost_fraction=cost,
            )
            scoring_probabilities = [
                _probability_on_normalized(artifact, row)
                for row in blocks["scoring"]
            ]
            scoring_percentiles = _symbol_percentile_scores(
                blocks["calibration"],
                calibration_probabilities,
                blocks["scoring"],
                scoring_probabilities,
            )
            scoring_selection = _selected_indices(
                blocks["scoring"], scoring_percentiles, threshold
            )
            scoring_metrics = _selection_metrics(
                dataset,
                scoring_indices,
                scoring_selection,
                round_trip_cost_fraction=cost,
            )
            fold_reports[str(candidate_id)].append(
                {
                    "fold_id": fold_id,
                    "threshold": threshold,
                    "threshold_derived_from": "CALIBRATION_ONLY",
                    "cross_symbol_selection_space": (
                        "CALIBRATION_BLOCK_WITHIN_SYMBOL_PERCENTILE"
                    ),
                    "calibration_threshold_valid": threshold_valid,
                    "calibration_selection": calibration_metrics,
                    "classification": asdict(artifact.scoring_metrics),
                    "scoring_selection": scoring_metrics,
                }
            )
            fold_artifacts[(str(candidate_id), fold_id)] = artifact

    promotion = _mapping(config["promotion"], "promotion")
    candidate_summaries = {}
    for candidate_id, folds in fold_reports.items():
        expectancies = [
            float(fold["scoring_selection"]["mean_net_expectancy"])
            for fold in folds
        ]
        signals = sum(int(fold["scoring_selection"]["signals"]) for fold in folds)
        positive = sum(value > 0.0 for value in expectancies)
        concentrations = [
            float(fold["scoring_selection"]["symbol_concentration"])
            for fold in folds
        ]
        checks = {
            "minimum_total_scoring_signals": signals
            >= int(promotion["minimum_total_scoring_signals"]),
            "minimum_positive_folds": positive
            >= int(promotion["minimum_positive_folds"]),
            "minimum_worst_fold_expectancy": min(expectancies)
            > float(promotion["minimum_worst_fold_expectancy"]),
            "minimum_mean_expectancy": statistics.fmean(expectancies)
            > float(promotion["minimum_mean_expectancy"]),
            "average_precision_lift": all(
                float(fold["classification"]["average_precision"])
                - float(fold["classification"]["positive_rate"])
                >= float(
                    promotion[
                        "minimum_average_precision_lift_over_prevalence"
                    ]
                )
                for fold in folds
            ),
            "maximum_ece": all(
                float(fold["classification"]["ece"])
                <= float(promotion["maximum_ece_each_fold"])
                for fold in folds
            ),
            "maximum_brier": all(
                float(fold["classification"]["brier"])
                <= float(promotion["maximum_brier_each_fold"])
                for fold in folds
            ),
            "maximum_symbol_concentration": all(
                value <= float(promotion["maximum_symbol_concentration"])
                for value in concentrations
            ),
            "calibration_thresholds_valid": all(
                bool(fold["calibration_threshold_valid"]) for fold in folds
            ),
        }
        candidate_summaries[candidate_id] = {
            "signals": signals,
            "positive_folds": positive,
            "mean_expectancy": statistics.fmean(expectancies),
            "worst_fold_expectancy": min(expectancies),
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = [
        candidate
        for candidate, summary in candidate_summaries.items()
        if summary["passed"]
    ]
    ranking_pool = passed or list(candidate_summaries)
    winner = max(
        ranking_pool,
        key=lambda candidate: (
            float(candidate_summaries[candidate]["worst_fold_expectancy"]),
            float(candidate_summaries[candidate]["mean_expectancy"]),
            int(candidate_summaries[candidate]["signals"]),
        ),
    )
    ready = bool(passed)
    outputs = _mapping(config["outputs"], "outputs")
    artifact_path = root / str(outputs["artifact_path"])
    manifest_path = root / str(outputs["manifest_path"])
    report_path = root / str(outputs["validation_report_path"])
    readiness_path = root / str(outputs["readiness_record_path"])
    report = {
        "schema_id": "aegis-entry-quality-v3-long-validation-v1",
        "experiment_id": config["experiment_id"],
        "dataset_sha256": dataset.artifact_hash,
        "dataset_rows": dataset.row_count,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "label_schema": LONG_LABEL_SCHEMA_VERSION,
        "candidates": fold_reports,
        "candidate_summaries": candidate_summaries,
        "winner": winner,
        "promotion_ready": ready,
        "lockbox_accessed": False,
        "automatic_live_activation": False,
    }
    atomic_write_json(report_path, report)
    write_long_opportunity_artifact(artifact_path, fold_artifacts[(winner, 4)])
    manifest = {
        "schema_id": "aegis-entry-quality-v3-long-artifact-manifest-v1",
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": dataset.artifact_hash,
        "artifact_path": str(artifact_path.relative_to(root)),
        "artifact_sha256": sha256_file(artifact_path),
        "winner": winner,
        "source_fold": 4,
        "promotion_ready": ready,
        "runtime_mode": "SHADOW",
        "owner_live_switch_required": True,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        readiness_path,
        {
            "schema_id": "aegis-entry-quality-v3-long-readiness-v1",
            "state": (
                "SHADOW_EVIDENCE_REQUIRED"
                if ready
                else "OFFLINE_VALIDATION_FAILED"
            ),
            "promotion_ready": ready,
            "runtime_mode": "SHADOW",
            "automatic_live_activation": False,
            "owner_live_switch_required": True,
            "artifact_sha256": manifest["artifact_sha256"],
            "checks": candidate_summaries[winner]["checks"],
        },
    )
    print(
        yaml.safe_dump(
            {
                "winner": winner,
                "promotion_ready": ready,
                "summary": candidate_summaries[winner],
                "artifact": str(artifact_path.relative_to(root)),
            },
            sort_keys=True,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
