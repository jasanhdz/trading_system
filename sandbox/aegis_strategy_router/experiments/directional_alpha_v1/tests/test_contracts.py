from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from directional_alpha_v1.features import assert_allowlist, feature_hash  # noqa: E402


DATASET = EXPERIMENT / "artifacts/dataset_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_opportunity_artifact_is_unchanged_and_not_retrained() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    model = REPOSITORY / config["opportunity_model"]["artifact"]
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    assert sha256(model) == config["opportunity_model"]["artifact_sha256"]
    assert manifest["opportunity_model_sha256"] == config["opportunity_model"]["artifact_sha256"]
    assert manifest["opportunity_retrained"] is False


def test_train_threshold_is_reused_for_later_splits() -> None:
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    threshold = manifest["opportunity_thresholds_from_train_scores_only"]["0.9"]
    development = pd.read_parquet(DATASET / "development_labeled.parquet")
    for split in ("TRAIN", "CALIBRATION", "VALIDATION"):
        values = development.loc[development.split.eq(split)]
        assert values.opportunity_top_90.equals(values.opportunity_score_frozen.ge(threshold))


def test_dataset_is_causal_symmetric_and_independent_of_aegis() -> None:
    development = pd.read_parquet(DATASET / "development_labeled.parquet")
    assert (pd.to_datetime(development.max_feature_available_at, utc=True) <= pd.to_datetime(development.decision_at, utc=True)).all()
    assert set(development.groupby("market_state_group_id").side.apply(frozenset)) == {frozenset({"LONG", "SHORT"})}
    forbidden = ("aegis", "candidate_strategy", "committee", "phase2")
    assert not [column for column in development if any(token in column.lower() for token in forbidden)]


def test_holdout_is_feature_only_and_sealed() -> None:
    holdout = pd.read_parquet(DATASET / "final_holdout_features_sealed.parquet")
    assert not [column for column in holdout if column.startswith("target__")]
    assert holdout.label_state.eq("SEALED").all()


def test_directional_feature_hash_is_deterministic() -> None:
    dictionary = json.loads((DATASET / "feature_dictionary.json").read_text())
    columns = [item["name"] for item in dictionary["features"]]
    development = pd.read_parquet(DATASET / "development_labeled.parquet").head(100)
    calculated = [feature_hash(row, columns) for row in development[columns].to_dict("records")]
    assert calculated == development.directional_feature_hash.tolist()


def test_allowlist_fails_closed_for_outcomes_and_aegis() -> None:
    assert_allowlist(["feature__directional_flow__sequence__flow_acceleration"])
    with pytest.raises(ValueError, match="LEAKAGE"):
        assert_allowlist(["feature__future_mfe"])
    with pytest.raises(ValueError, match="LEAKAGE"):
        assert_allowlist(["feature__aegis_confidence"])


def test_validation_support_failure_cannot_promote_experiment() -> None:
    result = json.loads((EXPERIMENT / "artifacts/run_01/result.json").read_text())
    assert result["support"]["validation"] is False
    assert result["flags"]["VALIDATION_SUPPORT_SUFFICIENT"] is False
    assert result["flags"]["DIRECTIONAL_ALPHA_PROMISING"] is False
    assert result["flags"]["FINAL_HOLDOUT_OPENED"] is False
    assert result["flags"]["READY_FOR_SHADOW"] is False
    assert result["flags"]["READY_FOR_LIVE"] is False
