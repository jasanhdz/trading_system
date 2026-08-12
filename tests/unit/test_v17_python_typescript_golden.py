import json
from pathlib import Path

import pytest

from aegis.utils import Sha256HashProvider, sha256_file
from aegis.v17_feature_contract import V17FeatureVector
from aegis.v17_research_artifact import V17ResearchArtifact


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "config/bundles/aegis-v17-research-artifact-v1.json"
GOLDEN = (
    ROOT
    / "binance-futures-bot-ts/src/challengers/fixtures/v17-golden-dataset.json"
)


def test_python_replays_the_typescript_golden_dataset_exactly() -> None:
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    claimed = unsigned.pop("content_hash")
    assert claimed == Sha256HashProvider().digest_value(unsigned)
    assert payload["artifact_sha256"] == sha256_file(ARTIFACT)
    assert payload["closed_historical_events"] is True
    assert payload["event_count"] == 22

    artifact = V17ResearchArtifact.load(ARTIFACT)
    for event in payload["events"]:
        raw = event["feature_vector"]
        vector = V17FeatureVector(
            side=raw["side"],
            schema_version=raw["schemaVersion"],
            schema_hash=raw["schemaHash"],
            names=tuple(raw["names"]),
            values=tuple(float(value) for value in raw["values"]),
            dtype=raw["dtype"],
        )
        actual = artifact.sides[event["side"]].score(vector)
        expected = event["python"]
        for field in (
            "clean_probability",
            "danger_probability",
            "mae_q90",
            "rank_score",
        ):
            assert actual[field] == pytest.approx(expected[field], abs=1e-12)
        assert actual["selected"] == expected["selected"]
        assert actual["policy_status"] == expected["policy_status"]
