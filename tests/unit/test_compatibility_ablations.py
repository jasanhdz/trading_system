from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.compat_replay.ablations import (
    DEV_END,
    E2_LAST_OPEN,
    FROZEN_AXES,
    STAGE_DEFINITIONS,
    StageDefinition,
    _canonical_scientific,
    _base_h12,
    _capacity_matrix,
    _select,
    _timestamp_key,
)
from scripts.diagnostics.compat_replay.manifests import digest, sha256_file
from scripts.diagnostics.compat_replay.schemas import ReplayConfig, STAGES


ROOT = Path(__file__).resolve().parents[2]
REPLAY_ROOT = ROOT / "scripts/diagnostics/compat_replay"


def test_ablation_stages_are_closed_and_change_exactly_one_axis() -> None:
    assert tuple(STAGE_DEFINITIONS) == STAGES[1:]
    assert {definition.changed_axis for definition in STAGE_DEFINITIONS.values()} == {
        "sampling", "features", "model_capacity", "runtime_selection", "eqm_population",
    }
    for definition in STAGE_DEFINITIONS.values():
        definition.validate()
        assert definition.parent == "STAGE_0"
        assert definition.changed_axis not in definition.frozen_axes
        assert set(definition.frozen_axes) == set(FROZEN_AXES) - {definition.changed_axis}


def test_multi_axis_ablation_fails_closed() -> None:
    invalid = StageDefinition("STAGE_X", "STAGE_0", "features+sampling", FROZEN_AXES, "FORBIDDEN")
    with pytest.raises(ValueError, match="exactly one axis"):
        invalid.validate()


def test_replay_config_rejects_extra_stages_and_keys(tmp_path: Path) -> None:
    payload = (REPLAY_ROOT / "replay_v1.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "extra-stage.yaml"
    changed.write_text(payload.replace("STAGE_5]", "STAGE_5, STAGE_6]"), encoding="utf-8")
    with pytest.raises(ValueError, match="closed ordered list"):
        ReplayConfig.load(changed)
    changed.write_text(payload + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level keys"):
        ReplayConfig.load(changed)


def test_base_population_cannot_cross_dev_boundary(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "id.timestamp": ["2026-04-26T23:55:00", "2026-04-27T00:00:00"],
        "id.symbol": ["BTCUSDT", "BTCUSDT"], "id.horizon": [12, 12],
    }).to_csv(path, index=False)
    result = _base_h12(path)
    assert result["_ts"].max() <= DEV_END
    assert len(result) == 1


def test_hourly_e2_boundary_is_close_time_based() -> None:
    assert E2_LAST_OPEN + pd.Timedelta(minutes=5) == pd.Timestamp("2026-04-26 23:00:00")


def test_historical_selection_is_deterministic() -> None:
    timestamps = pd.date_range("2024-07-12", periods=2200, freq="5min")
    data = pd.DataFrame({
        "_ts": timestamps, "id.timestamp": timestamps, "id.symbol": ["BTCUSDT"] * len(timestamps),
        "id.horizon": [12] * len(timestamps),
    })
    tail = np.linspace(0.0, 1.0, len(data)); scores = np.sin(np.arange(len(data)))
    first, first_counts = _select(data, tail, scores, "historical")
    second, second_counts = _select(data, tail, scores, "historical")
    assert first_counts == second_counts
    pd.testing.assert_frame_equal(first, second)
    assert digest(first.assign(ts=first["ts"].astype(str)).to_dict("records")) == digest(
        second.assign(ts=second["ts"].astype(str)).to_dict("records")
    )


def test_scientific_hash_normalization_is_explicitly_limited_to_1e_12() -> None:
    left = {"score": 0.050266081257980044, "key": "BTCUSDT"}
    right = {"score": 0.05026608125798004, "key": "BTCUSDT"}
    assert digest(_canonical_scientific(left)) == digest(_canonical_scientific(right))
    assert digest(_canonical_scientific({"score": 0.1})) != digest(_canonical_scientific({"score": 0.10000000001}))


def test_overlap_timestamp_key_normalizes_equivalent_utc_encodings() -> None:
    assert _timestamp_key("2025-04-16 02:55:00") == _timestamp_key("2025-04-16T02:55:00Z")


def test_e2_capacity_ablation_is_frozen_to_executed_values() -> None:
    capacities = _capacity_matrix()
    assert capacities["trrm_random_forest"]["e2"] == {
        "n_estimators": 80, "max_depth": 8, "min_samples_leaf": 8, "class_weight": None,
    }
    assert capacities["eqm_extra_trees"]["e2"]["n_estimators"] == 80
    assert capacities["qmae_hgb"]["e2"]["max_iter"] == 80


def test_stage_zero_evidence_remains_exact_control() -> None:
    stage = json.loads((ROOT / "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_0.json").read_text())
    assert stage["passed"] is True
    assert stage["trades"] == 688
    assert stage["trade_overlap"] == 1.0
    assert stage["maximum_common_trade_net_error"] <= 1e-9


def test_attempt_manifest_is_written_before_scientific_execution() -> None:
    source = (REPLAY_ROOT / "ablations.py").read_text(encoding="utf-8")
    manifest_write = source.index('atomic_write(output / "manifest.json", manifest)')
    first_stage_execution = source.index('if stage == "STAGE_1":', manifest_write)
    assert manifest_write < first_stage_execution


def test_manifest_contract_records_operational_timestamp_and_normalized_hash() -> None:
    source = (REPLAY_ROOT / "ablations.py").read_text(encoding="utf-8")
    assert '"created_at": datetime.now(timezone.utc).isoformat()' in source
    assert '"scientific_manifest_hash": digest(scientific)' in source


def test_ablation_code_is_diagnostic_and_does_not_import_phase_e() -> None:
    text = (REPLAY_ROOT / "ablations.py").read_text(encoding="utf-8")
    assert "aegis.training.phase_e" not in text
    assert "PhaseE" not in text
    assert "LockboxLease" not in text
    assert "CANDIDATE" not in text
    assert "SystemFreeze" not in text
    assert "SelectionPolicy" not in text


def test_governance_hashes_and_lockbox_are_unchanged() -> None:
    expected = {
        "config/experiments/aegis_short_candidate_e1.yaml": "2604df3461ca891db05b6d877ab2e6373eac8fc7b7412d08571b2644f351ae39",
        "config/experiments/aegis_short_candidate_e2.yaml": "759a7c87d2afaec2f6ede7b6451154a287ea5527e00036e040beb26c32732b8e",
        "config/experiments/aegis_short_candidate_e3.yaml": "281e27f93f8be0f9c4fbe78673d55f1a4f3391aa421c3fcaf90823113dc81e62",
        "config/scientific_competition_v1.yaml": "6eb6e89b42ec518ddc7b277cec1ec7df22dd5f5b2fdbb78341d9885aaf27988c",
        "config/scientific_competition_v2.yaml": "70c889223b1466ed3e0817a63e7cfafb5b0966bf7c4d3eb3d06544c70c097f79",
    }
    assert {path: sha256_file(ROOT / path) for path in expected} == expected
    lockbox = json.loads((ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    assert (lockbox["status"], lockbox["consumed_queries"], lockbox["maximum_queries_total"]) == (
        "NOT_CONSUMED", [], 1,
    )
