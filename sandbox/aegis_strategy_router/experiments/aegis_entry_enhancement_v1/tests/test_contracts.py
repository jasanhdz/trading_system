from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from aegis_entry_enhancement_v1.dataset import signal_timestamp  # noqa: E402
from aegis_entry_enhancement_v1.evaluation import policy_masks  # noqa: E402

DATASET = EXPERIMENT / "artifacts/dataset_v1"
RUN = EXPERIMENT / "artifacts/run_01"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trade_id_decodes_exact_policy_decision_timestamp() -> None:
    assert signal_timestamp("AEGIS-TURBO-SUIUSDT-20260701-001807-489") == pd.Timestamp("2026-07-01T00:18:07.489Z")


def test_frozen_module_hashes_match_and_retraining_is_disabled() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    for name in ("opportunity", "directional"):
        spec = config["frozen_modules"][name]
        assert sha256(REPOSITORY / spec["artifact"]) == spec["sha256"]
        assert spec["retrain"] is False


def test_dataset_is_causal_and_side_is_immutable() -> None:
    frame = pd.read_parquet(DATASET / "development_labeled.parquet")
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    assert (pd.to_datetime(frame.max_feature_available_at, utc=True) <= pd.to_datetime(frame.signal_timestamp, utc=True)).all()
    assert manifest["aegis_side_changed"] is False
    assert set(frame.side) == {"LONG", "SHORT"}


def test_holdout_remains_unlabeled_and_sealed() -> None:
    holdout = pd.read_parquet(DATASET / "final_holdout_features_sealed.parquet")
    result = json.loads((RUN / "result.json").read_text())
    assert not [column for column in holdout if column.startswith("target__")]
    assert result["final_holdout_state"] == "SEALED_NOT_OPENED"
    assert result["flags"]["FINAL_HOLDOUT_OPENED"] is False


def test_policy_masks_are_deterministic_and_do_not_mutate_side() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    frame = pd.read_parquet(DATASET / "development_labeled.parquet").loc[lambda value: value.split.eq("VALIDATION")]
    original_side = frame.side.copy()
    first = policy_masks(frame, config)
    second = policy_masks(frame, config)
    assert all(first[name].equals(second[name]) for name in first)
    assert frame.side.equals(original_side)


def test_zero_coverage_cannot_claim_value() -> None:
    result = json.loads((RUN / "result.json").read_text())
    assert result["primary"]["coverage"] == 0.0
    assert result["flags"]["COMBINED_GATE_ADDS_VALUE"] is False
    assert result["flags"]["NET_EXPECTANCY_IMPROVED_VS_AEGIS"] is False
    assert result["verdict"] == "AEGIS_ENTRY_ENHANCEMENT_NO_VALUE"


def test_validation_has_no_long_evidence_and_cannot_promote() -> None:
    frame = pd.read_parquet(DATASET / "development_labeled.parquet")
    validation = frame.loc[frame.split.eq("VALIDATION")]
    result = json.loads((RUN / "result.json").read_text())
    assert validation.side.eq("SHORT").all()
    assert result["flags"]["LONG_SUBGROUP_IMPROVED"] is False
    assert result["flags"]["VALIDATION_SUPPORT_SUFFICIENT"] is False
    assert result["flags"]["READY_FOR_PROSPECTIVE_OBSERVATION"] is False


def test_experiment_has_no_financial_mutation_capability() -> None:
    prohibited = ("createorder", "cancelorder", "setleverage", "positionrisk", "api_secret")
    text = "\n".join(path.read_text(errors="ignore").lower() for path in (EXPERIMENT / "src").rglob("*.py"))
    assert not [token for token in prohibited if token in text]
