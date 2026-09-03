import json
from pathlib import Path

import pytest

from aegis.models import ModelBundleError, load_model_bundle
from aegis.utils import Sha256HashProvider


def _bundle_payload() -> dict:
    path = Path(__file__).parents[2] / "config" / "bundles" / "aegis-offline-reference-v1.json"
    return json.loads(path.read_text())


def _write_bundle(tmp_path: Path, payload: dict) -> Path:
    unsigned = dict(payload); unsigned.pop("content_hash", None)
    payload["content_hash"] = Sha256HashProvider().digest_value(unsigned)
    path = tmp_path / "bundle.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reference_bundle_is_explicitly_untrained_reference_only() -> None:
    path = Path(__file__).parents[2] / "config" / "bundles" / "aegis-offline-reference-v1.json"
    bundle = load_model_bundle(path)
    assert bundle.approved is False
    assert bundle.metadata.trained is False
    assert bundle.metadata.purpose == "OFFLINE_REFERENCE_ONLY"


@pytest.mark.parametrize("mutation", ["checksum", "missing_head", "unknown_feature", "wrong_count", "bad_scale", "wrong_schema"])
def test_corrupt_or_incompatible_bundles_fail_closed(tmp_path: Path, mutation: str) -> None:
    payload = _bundle_payload()
    if mutation == "checksum":
        payload["estimators"][0]["heads"]["long"]["bias"] = 99
        path = tmp_path / "bundle.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "missing_head":
        del payload["estimators"][0]["heads"]["quality"]; path = _write_bundle(tmp_path, payload)
    elif mutation == "unknown_feature":
        payload["estimators"][0]["heads"]["long"]["weights"]["future_label"] = 1; path = _write_bundle(tmp_path, payload)
    elif mutation == "wrong_count":
        payload["metadata"]["feature_count"] = 40; path = _write_bundle(tmp_path, payload)
    elif mutation == "bad_scale":
        payload["normalizer"] = {"means": {"ret_1": 0}, "scales": {"ret_1": 0}, "clip_absolute": 12}; path = _write_bundle(tmp_path, payload)
    else:
        payload["schema_version"] = "unknown"; path = _write_bundle(tmp_path, payload)
    with pytest.raises(ModelBundleError):
        load_model_bundle(path)
