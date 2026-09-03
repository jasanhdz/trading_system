from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.domain import FeatureBatch, FeatureQuality, FeatureRow
from aegis.prospective.model_qualification import (
    CANDIDATE_IDENTITY,
    FEATURE_SCHEMA,
    ModelQualificationError,
    QualifiedShadowModelRuntime,
    load_qualified_candidate,
    qualify_inference,
    run_full_brain_smoke,
    seal_candidate_bundle,
    synthetic_feature_batch,
)
from aegis.utils import Sha256HashProvider, canonical_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/experiments/e3_validation_official/attempt_1/aegis-short-candidate-e3/runs/d742d9bc0ae867bb/experimental_bundle.json"
PROTOCOL = ROOT / "reports/governance/aegis_prospective_validation/model_qualification/aegis_model_qualification_protocol_v1.md"


@pytest.fixture(scope="module")
def candidate_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("qualified-model") / "candidate.json"
    seal_candidate_bundle(SOURCE, PROTOCOL, output)
    return output


def test_sealed_candidate_is_distinct_trained_shadow_only_and_deterministic(candidate_path: Path) -> None:
    original = candidate_path.read_bytes()
    seal_candidate_bundle(SOURCE, PROTOCOL, candidate_path)
    candidate = load_qualified_candidate(candidate_path)
    assert candidate_path.read_bytes() == original
    assert candidate.model_identity == CANDIDATE_IDENTITY
    assert candidate.payload["trained"] is True
    assert candidate.payload["approved"] is True
    assert candidate.payload["approval_scope"] == "PROSPECTIVE_SHADOW_ONLY"
    assert candidate.payload["approved_for_live"] is False
    assert candidate.payload["model_identity"] != "aegis-offline-reference-v1"


def test_candidate_hash_corruption_and_conflicting_output_fail_closed(candidate_path: Path, tmp_path: Path) -> None:
    payload = json.loads(candidate_path.read_text())
    payload["model_identity"] = "corrupted"
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(canonical_json(payload) + "\n")
    with pytest.raises(ModelQualificationError, match="PROSPECTIVE_MODEL_BUNDLE_HASH_MISMATCH"):
        load_qualified_candidate(corrupted)
    conflicting = tmp_path / "conflicting.json"
    conflicting.write_text("conflict\n")
    with pytest.raises(ModelQualificationError, match="PROSPECTIVE_MODEL_BUNDLE_OUTPUT_CONFLICT"):
        seal_candidate_bundle(SOURCE, PROTOCOL, conflicting)


def test_inference_is_finite_non_degenerate_deterministic_and_batch_equivalent(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    seal_candidate_bundle(SOURCE, PROTOCOL, output)
    result = qualify_inference(load_qualified_candidate(output))
    assert result["finite_output"] == "PASS"
    assert result["non_degeneracy"] == "PASS"
    assert result["repeated_inference"] == "BYTE_IDENTICAL"
    assert result["batch_single_agreement"] == "PASS"


def test_full_brain_smoke_is_deterministic_and_preactivation_only(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    seal_candidate_bundle(SOURCE, PROTOCOL, output)
    result = run_full_brain_smoke(load_qualified_candidate(output))
    assert result["full_brain_replay"] == "BYTE_IDENTICAL"
    assert result["event_classification"] == "PREACTIVATION_NON_COHORT"
    assert result["private_requests"] == 0
    assert result["real_orders"] == 0
    assert result["persistent_service_started"] is False


def test_runtime_rejects_unsupported_symbol_interval_and_feature_contract(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    seal_candidate_bundle(SOURCE, PROTOCOL, output)
    runtime = QualifiedShadowModelRuntime(load_qualified_candidate(output))
    batch = synthetic_feature_batch(("ADAUSDT",))
    with pytest.raises(ModelQualificationError, match="PROSPECTIVE_MODEL_INTERVAL_UNSUPPORTED"):
        runtime.predict(batch, timeframe="1m")
    bad_symbol = synthetic_feature_batch(("UNKNOWNUSDT",))
    with pytest.raises(ModelQualificationError, match="PROSPECTIVE_MODEL_SYMBOL_UNSUPPORTED"):
        runtime.predict(bad_symbol, timeframe="5m")
    reordered = FeatureBatch(
        FEATURE_SCHEMA,
        tuple(reversed(batch.feature_names)),
        batch.feature_hash,
        (
            FeatureRow(
                "ADAUSDT",
                tuple(reversed(batch.rows[0].raw_values)),
                tuple(reversed(batch.rows[0].normalized_values)),
                FeatureQuality(0, 0, True, 96),
            ),
        ),
    )
    with pytest.raises(ModelQualificationError, match="PROSPECTIVE_MODEL_FEATURE_ORDER_MISMATCH"):
        runtime.predict(reordered, timeframe="5m")


def test_missing_nonfinite_and_wrong_feature_count_fail_before_inference() -> None:
    batch = synthetic_feature_batch(("ADAUSDT",))
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        FeatureBatch(
            batch.schema_version,
            batch.feature_names,
            batch.feature_hash,
            (FeatureRow("ADAUSDT", (), (), FeatureQuality(0, 0, True, 96)),),
        )
    invalid = list(batch.rows[0].normalized_values)
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite feature"):
        FeatureBatch(
            batch.schema_version,
            batch.feature_names,
            batch.feature_hash,
            (FeatureRow("ADAUSDT", batch.rows[0].raw_values, tuple(invalid), FeatureQuality(0, 0, True, 96)),),
        )


def test_bundle_content_hash_is_canonical(candidate_path: Path) -> None:
    payload = json.loads(candidate_path.read_text())
    claimed = payload.pop("content_hash")
    assert claimed == Sha256HashProvider().digest_value(payload)
