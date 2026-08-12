import math
from copy import deepcopy
from pathlib import Path

import pytest

from aegis.utils import Sha256HashProvider
from aegis.v17_research_artifact import (
    FrozenLinearModel,
    V17ResearchArtifact,
    V17ResearchArtifactError,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_linear_probability_and_raw_score_are_exact() -> None:
    payload = {
        "schema_id": "aegis-v17-frozen-linear-v1",
        "feature_names": ["a", "b"],
        "means": [1.0, 2.0],
        "scales": [2.0, 4.0],
        "coefficients": [0.5, -1.0],
        "intercept": 0.25,
        "output": "RAW_SCORE",
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    raw = FrozenLinearModel.from_payload(payload)
    assert raw.evaluate([3.0, 6.0]) == -0.25
    probability_payload = {**payload, "output": "PROBABILITY"}
    probability_payload.pop("content_hash")
    probability_payload["content_hash"] = Sha256HashProvider().digest_value(
        probability_payload
    )
    probability = FrozenLinearModel.from_payload(probability_payload)
    assert math.isclose(
        probability.evaluate([3.0, 6.0]),
        1.0 / (1.0 + math.exp(0.25)),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_frozen_linear_rejects_missing_or_wrong_hash_and_schema() -> None:
    payload = {
        "schema_id": "aegis-v17-frozen-linear-v1",
        "feature_names": ["a"],
        "means": [0.0],
        "scales": [1.0],
        "coefficients": [1.0],
        "intercept": 0.0,
        "output": "RAW_SCORE",
    }
    with pytest.raises(V17ResearchArtifactError, match="HASH"):
        FrozenLinearModel.from_payload(payload)
    payload["content_hash"] = "0" * 64
    with pytest.raises(V17ResearchArtifactError, match="HASH"):
        FrozenLinearModel.from_payload(payload)
    payload.pop("content_hash")
    payload["schema_id"] = "wrong"
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    with pytest.raises(V17ResearchArtifactError, match="SCHEMA"):
        FrozenLinearModel.from_payload(payload)


def test_research_artifact_loads_and_rejects_nested_model_tampering(tmp_path) -> None:
    source = ROOT / "config/bundles/aegis-v17-research-artifact-v1.json"
    artifact = V17ResearchArtifact.load(source)
    assert artifact.promotion_authority is False
    payload = __import__("json").loads(source.read_text(encoding="utf-8"))
    tampered = deepcopy(payload)
    tampered["sides"]["SHORT"]["models"]["ranker"]["coefficients"][0] += 1.0
    unsigned = dict(tampered)
    unsigned.pop("content_hash")
    tampered["content_hash"] = Sha256HashProvider().digest_value(unsigned)
    target = tmp_path / "tampered.json"
    target.write_text(__import__("json").dumps(tampered), encoding="utf-8")
    with pytest.raises(V17ResearchArtifactError, match="LINEAR_MODEL_HASH"):
        V17ResearchArtifact.load(target)
