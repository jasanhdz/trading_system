"""Train the hash-pinned Entry Quality V2 SHORT opportunity artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalSeriesSource, DataPurpose
from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.dataset import TrainingDataset, load_and_build_e2_hourly_dataset
from aegis.training.competition import load_scientific_competition_contract
from aegis.training.experiment import fit_normalizer
from aegis.training.labels import SHORT_LABEL_SCHEMA_VERSION, ShortLabelConfig
from aegis.training.short_opportunity import (
    ShortOpportunityTrainingContract,
    ShortOpportunityTrainingRow,
    fit_short_opportunity_model,
    write_short_opportunity_artifact,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("training split timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


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
) -> tuple[ShortOpportunityTrainingRow, ...]:
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
            clean_opportunity=dataset.rows[index].target.clean_quality >= 0.5,
            mae_fraction=dataset.rows[index].target.qmae,
        )
        for index in indices
    )


def _load(path: Path) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_entry_quality_v2_short_opportunity.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = _load(config_path)
    if config.get("schema_version") != "aegis-entry-quality-v2-training-v1":
        raise SystemExit("AEGIS_ENTRY_QUALITY_V2_TRAINING_CONFIG_INVALID")
    source_config = _mapping(config["source"], "source")
    preregistration_path = (
        root / str(source_config["preregistration_path"])
    ).resolve()
    preregistration = _load(preregistration_path)
    source = CanonicalSeriesSource(
        Path(str(source_config["canonical_series_path"])),
        DataPurpose.TRAINING,
        expected_manifest_sha256=str(
            source_config["canonical_manifest_sha256"]
        ),
    )
    build = load_and_build_e2_hourly_dataset(source, preregistration)
    dataset = build.dataset
    if dataset.artifact_hash != str(source_config["expected_dataset_sha256"]):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V2_DATASET_HASH_MISMATCH")
    if (
        dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or dataset.feature_hash != FEATURE_HASH
        or dataset.symbols != CANONICAL_SYMBOLS
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V2_FEATURE_AUTHORITY_MISMATCH")

    split = _mapping(config["split"], "split")
    train_indices = _indices(
        dataset,
        _utc(str(split["train_start"])),
        _utc(str(split["train_end"])),
    )
    calibration_indices = _indices(
        dataset,
        _utc(str(split["calibration_start"])),
        _utc(str(split["calibration_end"])),
    )
    scoring_indices = _indices(
        dataset,
        _utc(str(split["scoring_start"])),
        _utc(str(split["scoring_end"])),
    )
    normalizer = fit_normalizer(dataset, train_indices)
    training = _mapping(config["training"], "training")
    label_config = ShortLabelConfig()
    if (
        training.get("label_schema") != SHORT_LABEL_SCHEMA_VERSION
        or training.get("label_field") != "clean_quality"
        or float(training["maximum_acceptable_mae"])
        != label_config.clean_mae_fraction
    ):
        raise SystemExit("AEGIS_ENTRY_QUALITY_V2_LABEL_AUTHORITY_MISMATCH")
    model = _mapping(config["model"], "model")
    competition_path = (root / str(model["competition_contract_path"])).resolve()
    competition = load_scientific_competition_contract(
        competition_path,
        expected_sha256=str(model["competition_contract_sha256"]),
    )
    candidate_id = str(model["candidate_id"])
    contract = ShortOpportunityTrainingContract(
        schema_version="aegis-short-opportunity-training-contract-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        maximum_acceptable_mae=float(training["maximum_acceptable_mae"]),
        model_candidate_id=candidate_id,
        model_parameters=competition.parameters(candidate_id),
        seed=int(training["seed"]),
        minimum_embargo_minutes=int(split["embargo_minutes"]),
        minimum_symbol_calibration_rows=int(
            training["minimum_symbol_calibration_rows"]
        ),
        symbol_calibration_shrinkage_rows=int(
            training["symbol_calibration_shrinkage_rows"]
        ),
    )
    blocks = {
        "train": _rows(dataset, train_indices, normalizer),
        "calibration": _rows(dataset, calibration_indices, normalizer),
        "scoring": _rows(dataset, scoring_indices, normalizer),
    }
    class_balance = {
        name: {
            "rows": len(rows),
            "positive_rows": sum(
                row.clean_opportunity
                and row.mae_fraction <= contract.maximum_acceptable_mae
                for row in rows
            ),
        }
        for name, rows in blocks.items()
    }
    for values in class_balance.values():
        values["positive_rate"] = (
            values["positive_rows"] / values["rows"]
            if values["rows"]
            else 0.0
        )
    print(yaml.safe_dump({"class_balance": class_balance}, sort_keys=True))
    artifact = fit_short_opportunity_model(
        blocks["train"],
        blocks["calibration"],
        blocks["scoring"],
        contract,
        normalizer=normalizer,
    )

    outputs = _mapping(config["outputs"], "outputs")
    artifact_path = (root / str(outputs["artifact_path"])).resolve()
    manifest_path = (root / str(outputs["manifest_path"])).resolve()
    readiness_path = (root / str(outputs["readiness_record_path"])).resolve()
    write_short_opportunity_artifact(artifact_path, artifact)
    artifact_sha256 = sha256_file(artifact_path)

    readiness = _mapping(config["technical_readiness"], "technical_readiness")
    metrics = artifact.scoring_metrics
    checks = {
        "minimum_scoring_rows": (
            metrics.row_count >= int(readiness["minimum_scoring_rows"])
        ),
        "average_precision_lift": (
            metrics.average_precision - metrics.positive_rate
            >= float(
                readiness[
                    "minimum_average_precision_lift_over_prevalence"
                ]
            )
        ),
        "maximum_ece": metrics.ece <= float(readiness["maximum_ece"]),
        "maximum_brier": (
            metrics.brier <= float(readiness["maximum_brier"])
        ),
        "all_symbols": (
            set(metrics.per_symbol) == set(CANONICAL_SYMBOLS)
            if bool(readiness["require_all_symbols"])
            else True
        ),
    }
    technically_ready = all(checks.values())
    manifest = {
        "schema_id": "aegis-entry-quality-v2-short-opportunity-manifest-v1",
        "experiment_id": config["experiment_id"],
        "training_config_path": str(config_path.relative_to(root)),
        "training_config_sha256": sha256_file(config_path),
        "preregistration_path": str(preregistration_path.relative_to(root)),
        "preregistration_sha256": sha256_file(preregistration_path),
        "canonical_manifest_sha256": source_config[
            "canonical_manifest_sha256"
        ],
        "dataset_sha256": dataset.artifact_hash,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "feature_count": len(FEATURE_NAMES),
        "split_identity": split["identity"],
        "train_rows": len(train_indices),
        "calibration_rows": len(calibration_indices),
        "scoring_rows": len(scoring_indices),
        "normalizer_fit_block": "TRAIN_ONLY",
        "label_schema": SHORT_LABEL_SCHEMA_VERSION,
        "label_field": "clean_quality",
        "competition_contract_path": str(competition_path.relative_to(root)),
        "competition_contract_sha256": competition.physical_sha256,
        "class_balance": class_balance,
        "contract": asdict(contract),
        "scoring_metrics": asdict(metrics),
        "technical_readiness_checks": checks,
        "technical_readiness": technically_ready,
        "artifact_path": str(artifact_path.relative_to(root)),
        "artifact_sha256": artifact_sha256,
        "automatic_live_activation": False,
    }
    atomic_write_json(manifest_path, manifest)
    readiness_record = {
        "schema_id": "aegis-entry-quality-v2-live-readiness-v1",
        "state": (
            "LIVE_READY_NOT_ACTIVE"
            if technically_ready
            else "NOT_READY_FOR_LIVE_SWITCH"
        ),
        "current_mode": "SHADOW",
        "owner_live_switch_required": True,
        "automatic_live_activation": False,
        "artifact_path": str(artifact_path.relative_to(root)),
        "artifact_sha256": artifact_sha256,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_file(manifest_path),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "technical_readiness_checks": checks,
        "shadow_evidence_requirement": {
            "minimum_non_overlapping_episodes": 300,
            "current_state": "COLLECTING",
            "blocks_current_shadow_mode": False,
        },
        "exchange_mutations": 0,
    }
    atomic_write_json(readiness_path, readiness_record)
    print(
        yaml.safe_dump(
            {
                "artifact_path": str(artifact_path.relative_to(root)),
                "artifact_sha256": artifact_sha256,
                "readiness_record_path": str(readiness_path.relative_to(root)),
                "readiness_record_sha256": sha256_file(readiness_path),
                "technical_readiness": technically_ready,
                "checks": checks,
            },
            sort_keys=True,
        )
    )
    return 0 if technically_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
