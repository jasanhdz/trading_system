"""Deterministic qualification and Shadow-only sealing for a trained model."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from aegis.config import CANONICAL_SYMBOLS
from aegis.domain import FeatureBatch, FeatureQuality, FeatureRow, ModelPredictions
from aegis.features import feature_contract
from aegis.models import DeterministicModelRuntime, ModelBundle, load_model_bundle
from aegis.utils import Sha256HashProvider, canonical_json, sha256_file, to_primitive


QUALIFICATION_VERSION = "aegis-prospective-model-qualification-v1"
CANDIDATE_IDENTITY = "aegis-prospective-shadow-candidate-v1"
CANDIDATE_BUNDLE_SCHEMA = "aegis-prospective-shadow-model-bundle-v1"
CANDIDATE_BUNDLE_VERSION = "1"
CANDIDATE_MODE = "PROSPECTIVE_SHADOW_CANDIDATE"
APPROVAL_SCOPE = "PROSPECTIVE_SHADOW_ONLY"
SOURCE_BUNDLE_ID = "aegis-short-candidate-e3-experimental"
SOURCE_BUNDLE_SHA256 = "386742c20d74a3b67d47cd95629c646195472e05e9e8d136587d40989a82e3d1"
SOURCE_CONTENT_HASH = "941f6b462812d1779f53bec3aba35741719116ecf37392b380f62207fface0ba"
FEATURE_SCHEMA = "aegis-features-v2"
FEATURE_HASH = "2dc278b4353585fe22503233187e12832cabfd67e2a2e58f4cd683ee6f3b9454"
LABEL_CONTRACT_SHA256 = "d1cbd83874d9823be2db9931052818d36a32ebbfde2625e83b0cf7403ab1e66d"
DATASET_MANIFEST_SHA256 = "6c2e97c8ac7bb28a167c0a0783dab9b27ebff69e1ecf34e7052869e9944c5a1c"
DATASET_IDENTITY = "1ffd0eaf07515d3a1a5fd6363f09c2d8ffe1e1f3925989486dee398e25b8c294"
FOLD_MANIFEST_SHA256 = "c2bf619fd3372583119b0d7ad7808609e72e8c926705a29349e33f8b018398a7"
TRAINING_CODE_COMMIT = "aea3437e0a969aa72ba6adb2331ca6e87020c7ad"
FROZEN_SEED = 20260718
TIMEFRAME = "5m"


class ModelQualificationError(RuntimeError):
    pass


class ModelMode(str, Enum):
    OFFLINE_REFERENCE = "OFFLINE_REFERENCE"
    PROSPECTIVE_SHADOW_CANDIDATE = CANDIDATE_MODE


@dataclass(frozen=True)
class QualifiedCandidateBundle:
    payload: Mapping[str, Any]
    source: ModelBundle

    @property
    def model_identity(self) -> str:
        return str(self.payload["model_identity"])

    @property
    def model_artifact_hash(self) -> str:
        return str(self.payload["model_artifact"]["sha256"])


def _load_json(path: Path, failure_code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelQualificationError(failure_code) from exc
    if not isinstance(value, Mapping):
        raise ModelQualificationError(failure_code)
    return value


def _validated_source(source_path: Path, expected_sha256: str = SOURCE_BUNDLE_SHA256) -> ModelBundle:
    if not source_path.is_file():
        raise ModelQualificationError("PROSPECTIVE_MODEL_ARTIFACT_MISSING")
    if sha256_file(source_path) != expected_sha256:
        raise ModelQualificationError("PROSPECTIVE_MODEL_ARTIFACT_HASH_MISMATCH")
    try:
        source = load_model_bundle(source_path, expected_bundle_id=SOURCE_BUNDLE_ID)
    except (OSError, ValueError) as exc:
        raise ModelQualificationError("PROSPECTIVE_MODEL_ARTIFACT_INVALID") from exc
    if (
        not source.metadata.trained
        or source.content_hash != SOURCE_CONTENT_HASH
        or source.feature_schema_version != FEATURE_SCHEMA
        or source.feature_hash != FEATURE_HASH
        or source.metadata.seed != FROZEN_SEED
        or source.metadata.purpose != "PHASE_E_E3_PRE_LOCKBOX_VALIDATION"
    ):
        raise ModelQualificationError("PROSPECTIVE_MODEL_LINEAGE_MISMATCH")
    return source


def candidate_bundle_payload(source_path: Path, protocol_path: Path) -> Mapping[str, Any]:
    source = _validated_source(source_path)
    protocol_hash = sha256_file(protocol_path)
    unsigned: dict[str, Any] = {
        "approval_scope": APPROVAL_SCOPE,
        "approved": True,
        "approved_for_live": False,
        "bundle_version": CANDIDATE_BUNDLE_VERSION,
        "feature_contract": {
            "feature_count": source.metadata.feature_count,
            "schema_id": source.feature_schema_version,
            "sha256": source.feature_hash,
        },
        "label_contract": {
            "historical_training_schema_id": "aegis-labels-short-v4",
            "prospective_contract_sha256": LABEL_CONTRACT_SHA256,
        },
        "lifecycle_state": "SHADOW_APPROVED",
        "model_artifact": {
            "content_hash": source.content_hash,
            "path": source_path.as_posix(),
            "sha256": SOURCE_BUNDLE_SHA256,
            "source_bundle_id": source.bundle_id,
        },
        "model_identity": CANDIDATE_IDENTITY,
        "model_mode": CANDIDATE_MODE,
        "qualification": {
            "protocol_path": protocol_path.as_posix(),
            "protocol_sha256": protocol_hash,
            "version": QUALIFICATION_VERSION,
        },
        "schema_id": CANDIDATE_BUNDLE_SCHEMA,
        "training_lineage": {
            "dataset_identity": DATASET_IDENTITY,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fold_manifest_sha256": FOLD_MANIFEST_SHA256,
            "seed": FROZEN_SEED,
            "training_code_commit": TRAINING_CODE_COMMIT,
        },
        "trained": True,
    }
    return {**unsigned, "content_hash": Sha256HashProvider().digest_value(unsigned)}


def seal_candidate_bundle(source_path: Path, protocol_path: Path, output_path: Path) -> str:
    payload = candidate_bundle_payload(source_path, protocol_path)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    if output_path.exists():
        if output_path.read_bytes() != encoded:
            raise ModelQualificationError("PROSPECTIVE_MODEL_BUNDLE_OUTPUT_CONFLICT")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encoded)
    return sha256_file(output_path)


def load_qualified_candidate(bundle_path: Path) -> QualifiedCandidateBundle:
    payload = _load_json(bundle_path, "PROSPECTIVE_MODEL_BUNDLE_INVALID")
    unsigned = dict(payload)
    claimed = str(unsigned.pop("content_hash", ""))
    if claimed != Sha256HashProvider().digest_value(unsigned):
        raise ModelQualificationError("PROSPECTIVE_MODEL_BUNDLE_HASH_MISMATCH")
    required = {
        "approval_scope", "approved", "approved_for_live", "bundle_version", "content_hash",
        "feature_contract", "label_contract", "lifecycle_state", "model_artifact", "model_identity",
        "model_mode", "qualification", "schema_id", "trained", "training_lineage",
    }
    if set(payload) != required or payload.get("schema_id") != CANDIDATE_BUNDLE_SCHEMA:
        raise ModelQualificationError("PROSPECTIVE_MODEL_BUNDLE_SCHEMA_MISMATCH")
    if (
        payload.get("model_identity") != CANDIDATE_IDENTITY
        or payload.get("model_mode") != CANDIDATE_MODE
        or payload.get("approval_scope") != APPROVAL_SCOPE
        or payload.get("approved") is not True
        or payload.get("trained") is not True
        or payload.get("approved_for_live") is not False
        or payload.get("lifecycle_state") != "SHADOW_APPROVED"
    ):
        raise ModelQualificationError("PROSPECTIVE_MODEL_APPROVAL_INVALID")
    artifact = payload.get("model_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != SOURCE_BUNDLE_SHA256:
        raise ModelQualificationError("PROSPECTIVE_MODEL_ARTIFACT_HASH_MISMATCH")
    source = _validated_source(Path(str(artifact.get("path", ""))))
    if artifact.get("content_hash") != source.content_hash:
        raise ModelQualificationError("PROSPECTIVE_MODEL_ARTIFACT_HASH_MISMATCH")
    return QualifiedCandidateBundle(payload, source)


@dataclass(frozen=True)
class QualifiedShadowModelRuntime:
    candidate: QualifiedCandidateBundle

    def predict(self, features: FeatureBatch, *, timeframe: str) -> ModelPredictions:
        if timeframe != TIMEFRAME:
            raise ModelQualificationError("PROSPECTIVE_MODEL_INTERVAL_UNSUPPORTED")
        if any(row.symbol not in CANONICAL_SYMBOLS for row in features.rows):
            raise ModelQualificationError("PROSPECTIVE_MODEL_SYMBOL_UNSUPPORTED")
        if features.feature_names != feature_contract(FEATURE_SCHEMA)[0]:
            raise ModelQualificationError("PROSPECTIVE_MODEL_FEATURE_ORDER_MISMATCH")
        return DeterministicModelRuntime(self.candidate.source).predict(features)


def synthetic_feature_batch(symbols: Sequence[str] = ("ADAUSDT", "BTCUSDT", "ETHUSDT")) -> FeatureBatch:
    names, digest = feature_contract(FEATURE_SCHEMA)
    rows = []
    for row_index, symbol in enumerate(symbols, 1):
        values = tuple(((index % 13) - 6) * 0.03125 + row_index * 0.0078125 for index in range(len(names)))
        rows.append(FeatureRow(symbol, values, values, FeatureQuality(0, 0, True, 96)))
    return FeatureBatch(FEATURE_SCHEMA, names, digest, tuple(rows))


def _prediction_digest(value: ModelPredictions) -> str:
    return Sha256HashProvider().digest_value(to_primitive(value))


def qualify_inference(candidate: QualifiedCandidateBundle) -> Mapping[str, Any]:
    runtime = QualifiedShadowModelRuntime(candidate)
    batch = synthetic_feature_batch()
    started = time.perf_counter_ns()
    first = runtime.predict(batch, timeframe=TIMEFRAME)
    elapsed_ns = time.perf_counter_ns() - started
    second = runtime.predict(batch, timeframe=TIMEFRAME)
    if first != second:
        raise ModelQualificationError("PROSPECTIVE_MODEL_INFERENCE_NONDETERMINISTIC")
    singles = tuple(
        prediction
        for row in batch.rows
        for prediction in runtime.predict(
            FeatureBatch(batch.schema_version, batch.feature_names, batch.feature_hash, (row,)),
            timeframe=TIMEFRAME,
        ).predictions
    )
    if first.predictions != singles:
        raise ModelQualificationError("PROSPECTIVE_MODEL_BATCH_SINGLE_MISMATCH")
    scalar_outputs = tuple(
        value
        for item in first.predictions
        for value in (
            item.long_probability, item.short_probability, item.neutral_probability,
            item.expected_return, item.tail_risk_probability, item.qmae_mean,
            item.quality_probability, item.uncertainty,
        )
    )
    if not scalar_outputs or not all(math.isfinite(value) for value in scalar_outputs):
        raise ModelQualificationError("PROSPECTIVE_MODEL_OUTPUT_NONFINITE")
    signatures = {
        (item.long_probability, item.short_probability, item.neutral_probability, item.expected_return)
        for item in first.predictions
    }
    if len(signatures) <= 1:
        raise ModelQualificationError("PROSPECTIVE_MODEL_OUTPUT_DEGENERATE")
    return {
        "batch_single_agreement": "PASS",
        "finite_output": "PASS",
        "inference_elapsed_ns": elapsed_ns,
        "model_artifact_sha256": candidate.model_artifact_hash,
        "model_identity": candidate.model_identity,
        "non_degeneracy": "PASS",
        "output_digest": _prediction_digest(first),
        "prediction_count": len(first.predictions),
        "repeated_inference": "BYTE_IDENTICAL",
        "schema_id": "aegis-prospective-model-inference-validation-v1",
        "validation_fixture": "PREACTIVATION_NON_COHORT_SYNTHETIC",
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Seal and validate the prospective Shadow candidate")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    seal_candidate_bundle(args.source, args.protocol, args.bundle)
    candidate = load_qualified_candidate(args.bundle)
    report = qualify_inference(candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

