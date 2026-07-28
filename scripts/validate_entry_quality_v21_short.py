"""Train and validate the V2.1 SHORT challenger without Live authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalSeriesSource, DataPurpose
from aegis.domain import (
    FeatureBatch,
    FeatureQuality,
    FeatureRow,
    Regime,
    TradeSide,
)
from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.prospective.model_qualification import load_qualified_candidate
from aegis.research.directional_challenger import (
    DirectionalEvidenceRow,
    DirectionalSelectionContract,
    derive_selection_policy,
    scoring_policy_passes,
    select_one_per_timestamp,
    selection_metrics,
    within_symbol_percentiles,
    within_symbol_percentiles_with_global_fallback,
)
from aegis.research.meta_entry import (
    MetaEntryCounterfactual,
    assess_counterfactual_predictions,
    counterfactual_mapping,
    favorable_entry,
)
from aegis.training.dataset import (
    TrainingDataset,
    load_and_build_e2_hourly_dataset,
)
from aegis.training.experiment import (
    evaluate_authoritative_feature_batch,
    fit_normalizer,
)
from aegis.training.labels import SHORT_LABEL_SCHEMA_VERSION
from aegis.training.run_state import atomic_write_json
from aegis.training.short_opportunity import (
    ShortOpportunityArtifact,
    ShortOpportunityTrainingContract,
    ShortOpportunityTrainingRow,
    fit_short_opportunity_model,
    write_short_opportunity_artifact,
)
from aegis.utils import canonical_json, sha256_file


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("split timestamp must be timezone-aware")
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


def _training_rows(
    dataset: TrainingDataset,
    indices: Sequence[int],
    normalizer,
    *,
    label_mode: str = "CLEAN_ENTRY",
    round_trip_cost_fraction: float = 0.001,
    maximum_acceptable_mae: float = 0.00275,
) -> tuple[ShortOpportunityTrainingRow, ...]:
    def label(index: int) -> bool:
        target = dataset.rows[index].target
        if label_mode == "CLEAN_ENTRY":
            return target.clean_quality >= 0.5
        if label_mode == "POSITIVE_NET_LOW_MAE":
            return favorable_entry(
                net_return=(-target.expected_return - round_trip_cost_fraction),
                mae=target.qmae,
                bad_entry=target.bad_entry >= 0.5,
                maximum_mae=maximum_acceptable_mae,
            )
        if label_mode == "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS":
            return (-target.expected_return - round_trip_cost_fraction) > 0.0
        raise ValueError("unsupported meta-entry label mode")

    return tuple(
        ShortOpportunityTrainingRow(
            timestamp=dataset.rows[index].timestamp,
            symbol=dataset.rows[index].symbol,
            features=tuple(
                normalizer.normalize(name, value)[0]
                for name, value in zip(
                    FEATURE_NAMES,
                    dataset.rows[index].features,
                )
            ),
            clean_opportunity=label(index),
            mae_fraction=dataset.rows[index].target.qmae,
        )
        for index in indices
    )


def _current_brain_candidate_indices(
    dataset: TrainingDataset,
    indices: Sequence[int],
    bundle,
    *,
    round_trip_cost_fraction: float,
) -> tuple[int, ...]:
    grouped: dict[datetime, list[int]] = defaultdict(list)
    for index in indices:
        grouped[dataset.rows[index].timestamp].append(index)
    selected = []
    for timestamp, group in sorted(grouped.items()):
        if len(group) != len(CANONICAL_SYMBOLS):
            raise ValueError("current-brain candidate replay requires complete cycles")
        pipeline = evaluate_authoritative_feature_batch(
            bundle,
            _feature_batch(dataset, group, bundle),
            timestamp=timestamp,
            config={"protocol": {"friction_fraction": round_trip_cost_fraction}},
            request_id=f"meta-entry-candidate-{timestamp.isoformat()}",
            decision_cycle_id=(f"meta-entry-candidate-cycle-{timestamp.isoformat()}"),
        )
        for candidate in pipeline.selection.selected:
            if candidate.side is not TradeSide.SHORT:
                continue
            selected.append(
                next(
                    index
                    for index in group
                    if dataset.rows[index].symbol == candidate.symbol
                )
            )
    return tuple(selected)


def _probability_on_normalized(
    artifact: ShortOpportunityArtifact,
    row: ShortOpportunityTrainingRow,
) -> float:
    if artifact.tree_ensemble is not None:
        raw = artifact.tree_ensemble.evaluate(row.features)
    else:
        logit = artifact.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(artifact.coefficients, row.features)
        )
        raw = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
    return artifact.calibrator.apply(row.symbol, raw)


def _evidence_rows(
    dataset: TrainingDataset,
    indices: Sequence[int],
    probabilities: Sequence[float],
    *,
    round_trip_cost_fraction: float,
) -> tuple[DirectionalEvidenceRow, ...]:
    if len(indices) != len(probabilities):
        raise ValueError("evidence probabilities are incomplete")
    rows = []
    for dataset_index, probability in zip(indices, probabilities):
        source = dataset.rows[dataset_index]
        rows.append(
            DirectionalEvidenceRow(
                timestamp=source.timestamp,
                symbol=source.symbol,
                score=float(probability),
                net_return=(-source.target.expected_return - round_trip_cost_fraction),
                mae=source.target.qmae,
                bad_entry=source.target.bad_entry >= 0.5,
                regime=source.regime.value,
            )
        )
    return tuple(rows)


def _feature_batch(
    dataset: TrainingDataset,
    indices: Sequence[int],
    bundle,
) -> FeatureBatch:
    rows = [dataset.rows[index] for index in indices]
    by_symbol = {row.symbol: row for row in rows}
    if set(by_symbol) != set(CANONICAL_SYMBOLS):
        raise ValueError("current-brain replay requires all canonical symbols")
    feature_rows = []
    for symbol in CANONICAL_SYMBOLS:
        raw = by_symbol[symbol].features
        normalized = tuple(
            bundle.normalizer.normalize(name, value)[0]
            for name, value in zip(FEATURE_NAMES, raw)
        )
        feature_rows.append(
            FeatureRow(
                symbol,
                raw,
                normalized,
                FeatureQuality(0, 0, True, 288),
            )
        )
    return FeatureBatch(
        FEATURE_SCHEMA_VERSION,
        FEATURE_NAMES,
        FEATURE_HASH,
        tuple(feature_rows),
    )


def _current_brain_control(
    dataset: TrainingDataset,
    indices: Sequence[int],
    bundle,
    *,
    round_trip_cost_fraction: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_block_minutes: int,
) -> Mapping[str, Any]:
    grouped: dict[datetime, list[int]] = defaultdict(list)
    for index in indices:
        grouped[dataset.rows[index].timestamp].append(index)
    evidence = _evidence_rows(
        dataset,
        indices,
        [0.0] * len(indices),
        round_trip_cost_fraction=round_trip_cost_fraction,
    )
    local_index = {
        dataset_index: offset for offset, dataset_index in enumerate(indices)
    }
    selected_offsets = []
    for timestamp, group in sorted(grouped.items()):
        pipeline = evaluate_authoritative_feature_batch(
            bundle,
            _feature_batch(dataset, group, bundle),
            timestamp=timestamp,
            config={"protocol": {"friction_fraction": round_trip_cost_fraction}},
            request_id=f"v21-control-{timestamp.isoformat()}",
            decision_cycle_id=f"v21-control-cycle-{timestamp.isoformat()}",
        )
        for candidate in pipeline.selection.selected:
            if candidate.side is not TradeSide.SHORT:
                continue
            dataset_index = next(
                index
                for index in group
                if dataset.rows[index].symbol == candidate.symbol
            )
            selected_offsets.append(local_index[dataset_index])
    return asdict(
        selection_metrics(
            evidence,
            selected_offsets,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_block_minutes=bootstrap_block_minutes,
        )
    )


def _selection_contract(config: Mapping[str, Any]) -> DirectionalSelectionContract:
    return DirectionalSelectionContract(
        schema_version=str(config["schema_version"]),
        probability_quantiles=tuple(
            float(value) for value in config["probability_quantiles"]
        ),
        minimum_calibration_selections=int(config["minimum_calibration_selections"]),
        minimum_scoring_selections=int(config["minimum_scoring_selections"]),
        maximum_mean_mae=float(config["maximum_mean_mae"]),
        maximum_symbol_concentration=float(config["maximum_symbol_concentration"]),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_block_minutes=int(config.get("bootstrap_block_minutes", 720)),
        minimum_calibration_blocks=int(config.get("minimum_calibration_blocks", 10)),
        minimum_scoring_blocks=int(config.get("minimum_scoring_blocks", 10)),
    )


def _variant_regimes(variant: str) -> tuple[str, ...]:
    if variant == "MODEL_ONLY":
        return ()
    if variant == "BEAR_TREND_CONFIRMED":
        return (Regime.BEAR_TREND.value,)
    raise ValueError("unsupported V2.1 selection variant")


def _live_feedback_state(
    root: Path,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    report_path = root / str(config["feedback_report_path"])
    if not report_path.is_file():
        return {
            "state": "MISSING",
            "non_overlapping_episodes": 0,
            "selected_outcomes": 0,
            "passed": False,
        }
    report = _mapping(json.loads(report_path.read_text()), "feedback report")
    readiness = _mapping(report["training_readiness"], "feedback readiness")
    dataset_path = Path(str(report["dataset_path"]))
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    selected = 0
    if dataset_path.is_file():
        with dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _mapping(json.loads(line), "feedback row")
                challenger = _mapping(row["challenger"], "challenger")
                selected += int(bool(challenger["selected"]))
    observed = int(readiness["observed_non_overlapping_episodes"])
    passed = observed >= int(
        config["minimum_non_overlapping_episodes"]
    ) and selected >= int(config["minimum_challenger_selected_outcomes"])
    return {
        "state": "PASS" if passed else "COLLECTING",
        "non_overlapping_episodes": observed,
        "required_non_overlapping_episodes": int(
            config["minimum_non_overlapping_episodes"]
        ),
        "selected_outcomes": selected,
        "required_selected_outcomes": int(
            config["minimum_challenger_selected_outcomes"]
        ),
        "passed": passed,
    }


def _score_live_counterfactuals(
    root: Path,
    config: Mapping[str, Any],
    artifact: ShortOpportunityArtifact,
    calibration_evidence: Sequence[DirectionalEvidenceRow],
    policy,
    *,
    maximum_acceptable_mae: float,
) -> Mapping[str, Any]:
    dataset_path = root / str(config["feedback_dataset_path"])
    output_path = root / str(config["predictions_path"])
    if not dataset_path.is_file():
        return {
            "state": "MISSING_FEEDBACK_DATASET",
            "rows": 0,
            "exchange_mutations": 0,
        }
    source_rows = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _mapping(json.loads(line), "live feedback row")
            if bool(_mapping(row["control"], "control")["selected"]):
                source_rows.append(row)
    evidence = []
    probabilities = []
    for row in source_rows:
        features = _mapping(row["feature_values"], "feature_values")
        probability = artifact.probability(
            str(row["symbol"]),
            [float(features[name]) for name in FEATURE_NAMES],
        )
        probabilities.append(probability)
        observed = _mapping(row["observed"], "observed")
        label = _mapping(row["label"], "label")
        regime = _mapping(
            _mapping(row["challenger"], "challenger")["regime"],
            "regime",
        )
        evidence.append(
            DirectionalEvidenceRow(
                timestamp=_utc(str(row["signal_timestamp"])),
                symbol=str(row["symbol"]),
                score=probability,
                net_return=float(observed["net_return_fraction"]),
                mae=float(observed["mae_fraction"]),
                bad_entry=bool(label["bad_entry"]),
                regime=(
                    f"{regime['direction']}|{regime['volatility']}|"
                    f"{regime['structure']}"
                ),
            )
        )
    percentiles, fallback_symbols = within_symbol_percentiles_with_global_fallback(
        calibration_evidence,
        evidence,
    )
    selected = set(
        select_one_per_timestamp(
            evidence,
            percentiles,
            threshold=policy.threshold,
            allowed_regimes=policy.allowed_regimes,
        )
    )
    predictions = []
    for index, (source, row, probability, percentile) in enumerate(
        zip(source_rows, evidence, probabilities, percentiles)
    ):
        label = _mapping(source["label"], "label")
        prediction = MetaEntryCounterfactual(
            event_id=str(source["event_id"]),
            symbol=row.symbol,
            selected=index in selected,
            favorable=favorable_entry(
                net_return=row.net_return,
                mae=row.mae,
                bad_entry=bool(label["bad_entry"]),
                maximum_mae=maximum_acceptable_mae,
            ),
            net_return=row.net_return,
            mae=row.mae,
            probability=probability,
            percentile=percentile,
            actual_trade=source.get("actual_trade") is not None,
        )
        predictions.append(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for prediction in predictions:
            output.write(canonical_json(counterfactual_mapping(prediction)) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, output_path)
    return {
        "state": "COUNTERFACTUAL_COMPLETE",
        "threshold": policy.threshold,
        "threshold_source": "FOLD4_CALIBRATION_ONLY",
        "assessment": assess_counterfactual_predictions(predictions),
        "predictions_path": str(output_path.relative_to(root)),
        "predictions_sha256": sha256_file(output_path),
        "global_percentile_fallback_symbols": list(fallback_symbols),
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_entry_quality_v21_short_challenger.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _mapping(yaml.safe_load(config_path.read_text()), "V2.1 config")
    schema_version = str(config.get("schema_version"))
    if (
        schema_version
        not in {
            "aegis-entry-quality-v21-short-preregistration-v1",
            "aegis-entry-quality-v22-short-preregistration-v1",
            "aegis-meta-entry-v3-short-preregistration-v1",
            "aegis-short-profitability-semantics-v1-preregistration",
        }
        or config.get("side") != "SHORT"
        or config.get("runtime_authority") != "SHADOW_ONLY"
        or bool(config.get("automatic_live_activation"))
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_CONFIG_INVALID")

    source_config = _mapping(config["source"], "source")
    preregistration_path = root / str(source_config["preregistration_path"])
    if sha256_file(preregistration_path) != str(
        source_config["preregistration_sha256"]
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_PREREGISTRATION_DRIFT")
    preregistration = _mapping(
        yaml.safe_load(preregistration_path.read_text()),
        "source preregistration",
    )
    lockbox = _mapping(preregistration["lockbox"], "source lockbox")
    if (
        preregistration.get("label_schema_version") != SHORT_LABEL_SCHEMA_VERSION
        or not bool(lockbox.get("final_test_is_lockbox"))
        or int(lockbox.get("additional_lockbox_budget", -1)) != 0
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_LABEL_AUTHORITY_MISMATCH")
    source = CanonicalSeriesSource(
        Path(str(source_config["canonical_series_path"])),
        DataPurpose.TRAINING,
        expected_manifest_sha256=str(source_config["canonical_manifest_sha256"]),
    )
    dataset = load_and_build_e2_hourly_dataset(source, preregistration).dataset
    if (
        dataset.artifact_hash != str(source_config["expected_dataset_sha256"])
        or dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or dataset.feature_hash != FEATURE_HASH
        or dataset.symbols != CANONICAL_SYMBOLS
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_DATASET_AUTHORITY_MISMATCH")

    brain_path = root / str(source_config["qualified_brain_path"])
    if sha256_file(brain_path) != str(source_config["qualified_brain_sha256"]):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_BRAIN_AUTHORITY_MISMATCH")
    current_brain = load_qualified_candidate(brain_path).source

    training = _mapping(config["training"], "training")
    population = _mapping(
        config.get(
            "population",
            {
                "source": "ALL_DIRECTIONAL_ROWS",
                "label": "CLEAN_ENTRY",
            },
        ),
        "population",
    )
    population_source = str(population["source"])
    label_mode = str(population["label"])
    selection_config = _mapping(config["selection"], "selection")
    promotion = _mapping(config["promotion"], "promotion")
    contract = _selection_contract(selection_config)
    cost = float(training["round_trip_cost_fraction"])
    candidates = _mapping(training["candidates"], "candidates")
    variants = tuple(str(value) for value in selection_config["variants"])
    fold_reports: dict[str, list[Mapping[str, Any]]] = {
        f"{candidate}|{variant}": [] for candidate in candidates for variant in variants
    }
    fold_artifacts: dict[tuple[str, int], ShortOpportunityArtifact] = {}
    fold_calibration_evidence: dict[
        tuple[str, int], tuple[DirectionalEvidenceRow, ...]
    ] = {}
    fold_policies = {}
    candidate_population = (
        set(
            _current_brain_candidate_indices(
                dataset,
                tuple(range(len(dataset.rows))),
                current_brain,
                round_trip_cost_fraction=cost,
            )
        )
        if population_source == "CURRENT_BRAIN_SELECTED_SHORT_CANDIDATES"
        else None
    )
    if population_source not in {
        "ALL_DIRECTIONAL_ROWS",
        "CURRENT_BRAIN_SELECTED_SHORT_CANDIDATES",
    }:
        raise SystemExit("AEGIS_META_ENTRY_POPULATION_INVALID")

    folds = preregistration["fold_protocol"]["folds"]
    if len(folds) != int(promotion["fold_count"]):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V21_FOLD_COUNT_MISMATCH")
    for fold in folds:
        fold_id = int(fold["id"])
        train_indices = _indices(
            dataset,
            _utc(str(fold["train_start"])),
            _utc(str(fold["train_end"])),
        )
        calibration_indices = _indices(
            dataset,
            _utc(str(fold["calibration_start"])),
            _utc(str(fold["calibration_end"])),
        )
        scoring_indices = _indices(
            dataset,
            _utc(str(fold["scoring_start"])),
            _utc(str(fold["scoring_end"])),
        )
        if candidate_population is not None:
            train_indices = tuple(
                index for index in train_indices if index in candidate_population
            )
            calibration_indices = tuple(
                index for index in calibration_indices if index in candidate_population
            )
            scoring_indices = tuple(
                index for index in scoring_indices if index in candidate_population
            )
        normalizer = fit_normalizer(dataset, train_indices)
        blocks = {
            "train": _training_rows(
                dataset,
                train_indices,
                normalizer,
                label_mode=label_mode,
                round_trip_cost_fraction=cost,
                maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
            ),
            "calibration": _training_rows(
                dataset,
                calibration_indices,
                normalizer,
                label_mode=label_mode,
                round_trip_cost_fraction=cost,
                maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
            ),
            "scoring": _training_rows(
                dataset,
                scoring_indices,
                normalizer,
                label_mode=label_mode,
                round_trip_cost_fraction=cost,
                maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
            ),
        }
        if candidate_population is None:
            control_metrics = _current_brain_control(
                dataset,
                scoring_indices,
                current_brain,
                round_trip_cost_fraction=cost,
                bootstrap_resamples=contract.bootstrap_resamples,
                bootstrap_seed=contract.bootstrap_seed + fold_id,
                bootstrap_block_minutes=contract.bootstrap_block_minutes,
            )
        else:
            control_evidence = _evidence_rows(
                dataset,
                scoring_indices,
                [0.0] * len(scoring_indices),
                round_trip_cost_fraction=cost,
            )
            control_metrics = asdict(
                selection_metrics(
                    control_evidence,
                    tuple(range(len(control_evidence))),
                    bootstrap_resamples=contract.bootstrap_resamples,
                    bootstrap_seed=contract.bootstrap_seed + fold_id,
                    bootstrap_block_minutes=contract.bootstrap_block_minutes,
                )
            )
        for candidate_id, parameters in candidates.items():
            training_contract = ShortOpportunityTrainingContract(
                schema_version="aegis-short-opportunity-training-contract-v1",
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_hash=FEATURE_HASH,
                maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
                model_candidate_id=str(candidate_id),
                model_parameters=_mapping(parameters, str(candidate_id)),
                seed=int(training["seed"]) + fold_id,
                minimum_embargo_minutes=int(
                    preregistration["fold_protocol"]["embargo_minutes"]
                ),
                minimum_symbol_calibration_rows=int(
                    training["minimum_symbol_calibration_rows"]
                ),
                symbol_calibration_shrinkage_rows=int(
                    training["symbol_calibration_shrinkage_rows"]
                ),
                probability_semantics=(
                    "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS"
                    if label_mode == "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS"
                    else "CLEAN_ENTRY_LOW_MAE_H12"
                ),
            )
            artifact = fit_short_opportunity_model(
                blocks["train"],
                blocks["calibration"],
                blocks["scoring"],
                training_contract,
                normalizer=normalizer,
            )
            fold_artifacts[(str(candidate_id), fold_id)] = artifact
            calibration_probabilities = tuple(
                _probability_on_normalized(artifact, row)
                for row in blocks["calibration"]
            )
            scoring_probabilities = tuple(
                _probability_on_normalized(artifact, row) for row in blocks["scoring"]
            )
            calibration_evidence = _evidence_rows(
                dataset,
                calibration_indices,
                calibration_probabilities,
                round_trip_cost_fraction=cost,
            )
            fold_calibration_evidence[(str(candidate_id), fold_id)] = (
                calibration_evidence
            )
            scoring_evidence = _evidence_rows(
                dataset,
                scoring_indices,
                scoring_probabilities,
                round_trip_cost_fraction=cost,
            )
            calibration_percentiles = within_symbol_percentiles(
                calibration_evidence,
                calibration_evidence,
            )
            scoring_percentiles = within_symbol_percentiles(
                calibration_evidence,
                scoring_evidence,
            )
            for variant in variants:
                regimes = _variant_regimes(variant)
                policy = derive_selection_policy(
                    calibration_evidence,
                    calibration_percentiles,
                    contract,
                    allowed_regimes=regimes,
                )
                fold_policies[(str(candidate_id), variant, fold_id)] = policy
                selected = select_one_per_timestamp(
                    scoring_evidence,
                    scoring_percentiles,
                    threshold=policy.threshold,
                    allowed_regimes=regimes,
                )
                metrics = selection_metrics(
                    scoring_evidence,
                    selected,
                    bootstrap_resamples=contract.bootstrap_resamples,
                    bootstrap_seed=contract.bootstrap_seed + fold_id,
                    bootstrap_block_minutes=contract.bootstrap_block_minutes,
                )
                fold_reports[f"{candidate_id}|{variant}"].append(
                    {
                        "fold_id": fold_id,
                        "threshold": policy.threshold,
                        "threshold_source": (
                            "CALIBRATION_ONLY_WITHIN_SYMBOL_PERCENTILE"
                        ),
                        "allowed_regimes": list(regimes),
                        "calibration_valid": policy.calibration_valid,
                        "calibration_selection": asdict(policy.calibration_metrics),
                        "classification": asdict(artifact.scoring_metrics),
                        "scoring_selection": asdict(metrics),
                        "scoring_policy_passed": scoring_policy_passes(
                            metrics, contract
                        ),
                        "current_brain_control": control_metrics,
                        "expectancy_delta_vs_control": (
                            metrics.mean_net_expectancy
                            - float(control_metrics["mean_net_expectancy"])
                        ),
                        "mae_delta_vs_control": (
                            metrics.mean_mae - float(control_metrics["mean_mae"])
                        ),
                    }
                )

    summaries = {}
    for identity, reports in fold_reports.items():
        candidate_id, variant = identity.split("|", 1)
        expectancies = [
            float(report["scoring_selection"]["mean_net_expectancy"])
            for report in reports
        ]
        signals = sum(int(report["scoring_selection"]["signals"]) for report in reports)
        passing_folds = sum(bool(report["scoring_policy_passed"]) for report in reports)
        positive_folds = sum(value > 0.0 for value in expectancies)
        symbols = {
            symbol
            for report in reports
            for symbol in report["scoring_selection"]["symbol_counts"]
        }
        checks = {
            "minimum_total_scoring_signals": signals
            >= int(promotion["minimum_total_scoring_signals"]),
            "minimum_positive_folds": positive_folds
            >= int(promotion["minimum_positive_folds"]),
            "minimum_passing_folds": passing_folds
            >= int(promotion["minimum_passing_folds"]),
            "minimum_worst_fold_expectancy": min(expectancies)
            > float(promotion["minimum_worst_fold_expectancy"]),
            "minimum_mean_expectancy": statistics.fmean(expectancies)
            > float(promotion["minimum_mean_expectancy"]),
            "average_precision_lift": all(
                float(report["classification"]["average_precision"])
                - float(report["classification"]["positive_rate"])
                >= float(promotion["minimum_average_precision_lift_over_prevalence"])
                for report in reports
            ),
            "maximum_ece": all(
                float(report["classification"]["ece"])
                <= float(promotion["maximum_ece_each_fold"])
                for report in reports
            ),
            "maximum_brier": all(
                float(report["classification"]["brier"])
                <= float(promotion["maximum_brier_each_fold"])
                for report in reports
            ),
            "all_symbols_observed": (
                symbols == set(CANONICAL_SYMBOLS)
                if bool(promotion["require_all_symbols_observed"])
                else True
            ),
            "calibration_thresholds_valid": all(
                bool(report["calibration_valid"]) for report in reports
            ),
        }
        summaries[identity] = {
            "candidate_id": candidate_id,
            "variant": variant,
            "signals": signals,
            "positive_folds": positive_folds,
            "passing_folds": passing_folds,
            "mean_expectancy": statistics.fmean(expectancies),
            "worst_fold_expectancy": min(expectancies),
            "checks": checks,
            "passed": all(checks.values()),
        }

    passed = [identity for identity, value in summaries.items() if value["passed"]]
    pool = passed or list(summaries)
    winner = max(
        pool,
        key=lambda identity: (
            bool(summaries[identity]["passed"]),
            int(summaries[identity]["passing_folds"]),
            float(summaries[identity]["worst_fold_expectancy"]),
            float(summaries[identity]["mean_expectancy"]),
            int(summaries[identity]["signals"]),
        ),
    )
    offline_ready = bool(passed)
    live_feedback = _live_feedback_state(
        root,
        _mapping(config["live_shadow_evidence"], "live shadow evidence"),
    )
    ready = offline_ready and bool(live_feedback["passed"])
    outputs = _mapping(config["outputs"], "outputs")
    artifact_path = root / str(outputs["artifact_path"])
    report_path = root / str(outputs["validation_report_path"])
    readiness_path = root / str(outputs["readiness_record_path"])
    winning_candidate = str(summaries[winner]["candidate_id"])
    write_short_opportunity_artifact(
        artifact_path,
        fold_artifacts[(winning_candidate, 4)],
    )
    version = (
        "profitability-v1"
        if "profitability-semantics" in schema_version
        else (
            "meta-v3"
            if "meta-entry-v3" in schema_version
            else "v22" if "v22" in schema_version else "v21"
        )
    )
    counterfactual = None
    if "counterfactual" in config:
        winning_variant = str(summaries[winner]["variant"])
        counterfactual = _score_live_counterfactuals(
            root,
            _mapping(config["counterfactual"], "counterfactual"),
            fold_artifacts[(winning_candidate, 4)],
            fold_calibration_evidence[(winning_candidate, 4)],
            fold_policies[(winning_candidate, winning_variant, 4)],
            maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
        )
    report = {
        "schema_id": f"aegis-entry-quality-{version}-short-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "runtime_authority": "SHADOW_ONLY",
        "automatic_live_activation": False,
        "dataset_sha256": dataset.artifact_hash,
        "dataset_rows": dataset.row_count,
        "population_source": population_source,
        "candidate_population_rows": (
            len(candidate_population)
            if candidate_population is not None
            else dataset.row_count
        ),
        "label_mode": label_mode,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "label_schema": SHORT_LABEL_SCHEMA_VERSION,
        "selection_contract": asdict(contract),
        "folds": fold_reports,
        "summaries": summaries,
        "winner": winner,
        "offline_validation_ready": offline_ready,
        "live_shadow_evidence": live_feedback,
        "promotion_ready": ready,
        "live_counterfactual": counterfactual,
        "exchange_mutations": 0,
    }
    atomic_write_json(report_path, report)
    state = (
        "READY_FOR_OWNER_REVIEW"
        if ready
        else (
            "LIVE_SHADOW_EVIDENCE_REQUIRED"
            if offline_ready
            else "OFFLINE_VALIDATION_FAILED"
        )
    )
    atomic_write_json(
        readiness_path,
        {
            "schema_id": f"aegis-entry-quality-{version}-short-readiness-v1",
            "state": state,
            "winner": winner,
            "offline_validation_ready": offline_ready,
            "live_shadow_evidence": live_feedback,
            "promotion_ready": ready,
            "runtime_mode": "SHADOW",
            "automatic_live_activation": False,
            "owner_promotion_required": True,
            "artifact_path": str(artifact_path.relative_to(root)),
            "artifact_sha256": sha256_file(artifact_path),
            "validation_report_path": str(report_path.relative_to(root)),
            "validation_report_sha256": sha256_file(report_path),
            "exchange_mutations": 0,
        },
    )
    for private_path in (artifact_path, report_path, readiness_path):
        os.chmod(private_path, 0o600)
    print(
        json.dumps(
            {
                "state": state,
                "winner": winner,
                "summary": summaries[winner],
                "live_shadow_evidence": live_feedback,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
