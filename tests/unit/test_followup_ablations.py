from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.diagnostics.compat_replay.followup_ablations import (
    E2_THRESHOLD,
    STAGE_1B_PREREGISTRATION,
    STAGE_1B_SHA256,
    STAGE_4B_PREREGISTRATION,
    STAGE_4B_SHA256,
    FollowupProtocolError,
    _historical_label_functions,
    _load_preregistration,
    _rank_survivors,
    _select_stage_4b_variant,
    _validate_stage_1b,
    _validate_stage_4b,
)
from scripts.diagnostics.compat_replay.manifests import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_followup_preregistrations_are_hash_bound_and_closed() -> None:
    stage_1b = _load_preregistration(STAGE_1B_PREREGISTRATION, STAGE_1B_SHA256)
    stage_4b = _load_preregistration(STAGE_4B_PREREGISTRATION, STAGE_4B_SHA256)
    _validate_stage_1b(stage_1b)
    _validate_stage_4b(stage_4b)
    assert stage_1b["evaluation"]["deterministic_runs"] == 2
    assert stage_4b["closed_variant_ids"] == ["STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"]


def test_changed_followup_preregistration_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "stage_1b.yaml"
    changed.write_text(STAGE_1B_PREREGISTRATION.read_text().replace("minimum_trades_each_fold: 100", "minimum_trades_each_fold: 99"))
    with pytest.raises(FollowupProtocolError, match="hash mismatch"):
        _load_preregistration(changed, STAGE_1B_SHA256)


def test_historical_label_source_reconstructs_frozen_short_v4_math() -> None:
    config_type, metrics_fn, clean_fn = _historical_label_functions()
    close = np.array([100.0] + [99.9 - index * 0.05 for index in range(12)])
    high = close + 0.05
    low = close - 0.05
    metrics = metrics_fn(high=high, low=low, close=close, entry_index=0, horizon=12, config=config_type(horizon=12))
    assert metrics["sample_complete"] is True
    assert metrics["mfe_roe_proxy"] > 0.08
    assert metrics["net_quality_after_costs"] == pytest.approx(
        metrics["mfe_roe_proxy"] - metrics["mae_roe_proxy"] - 0.02,
    )
    assert clean_fn(metrics, config_type(horizon=12)) in {0, 1}


def test_rank_veto_is_exactly_thirty_percent_and_deterministic() -> None:
    indices = np.arange(10)
    tail = np.array([0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4, 0.9, 0.0])
    first = _rank_survivors(indices, tail)
    second = _rank_survivors(indices, tail)
    assert first.tolist() == second.tolist() == [9, 1, 3, 5, 7, 6, 4]


def _selection_fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    timestamps = pd.date_range("2024-01-01", periods=2200, freq="5min")
    data = pd.DataFrame({
        "_ts": timestamps,
        "id.timestamp": timestamps,
        "id.symbol": np.resize(np.array(["ADAUSDT", "BTCUSDT"]), len(timestamps)),
        "id.horizon": 12,
    })
    tail = np.linspace(0.0, 1.0, len(data))
    scores = np.linspace(-0.01, 0.02, len(data))
    return data, tail, scores


@pytest.mark.parametrize("variant", ["STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"])
def test_stage_4b_variants_are_deterministic_and_emit_only_short_candidates(variant: str) -> None:
    data, tail, scores = _selection_fixture()
    first, first_counts = _select_stage_4b_variant(data, tail, scores, variant)
    second, second_counts = _select_stage_4b_variant(data, tail, scores, variant)
    pd.testing.assert_frame_equal(first, second)
    assert first_counts == second_counts
    assert set(first["symbol"]) <= {"ADAUSDT", "BTCUSDT"}


def test_stage_4b_c_uses_the_frozen_e2_threshold_after_rank_veto() -> None:
    payload = yaml.safe_load(STAGE_4B_PREREGISTRATION.read_text())
    variant = payload["variants"][2]
    assert variant["threshold"]["value"] == E2_THRESHOLD
    assert variant["threshold"]["application_order"] == "AFTER_VETO_BEFORE_RANKING"
    assert variant["changed_axis"] == "threshold"


def test_followup_runner_has_no_phase_e_lockbox_or_production_side_effect_path() -> None:
    source = (ROOT / "scripts/diagnostics/compat_replay/followup_ablations.py").read_text()
    assert "aegis.training.phase_e" not in source
    assert "LockboxLease" not in source
    assert "DecisionFreeze" not in source
    assert "SelectionPolicy" not in source
    assert "createOrder" not in source


def test_frozen_governance_and_lockbox_remain_unchanged() -> None:
    expected = {
        "config/experiments/aegis_short_candidate_e1.yaml": "2604df3461ca891db05b6d877ab2e6373eac8fc7b7412d08571b2644f351ae39",
        "config/experiments/aegis_short_candidate_e2.yaml": "759a7c87d2afaec2f6ede7b6451154a287ea5527e00036e040beb26c32732b8e",
        "config/experiments/aegis_short_candidate_e3.yaml": "281e27f93f8be0f9c4fbe78673d55f1a4f3391aa421c3fcaf90823113dc81e62",
        "config/scientific_competition_v1.yaml": "6eb6e89b42ec518ddc7b277cec1ec7df22dd5f5b2fdbb78341d9885aaf27988c",
        "config/scientific_competition_v2.yaml": "70c889223b1466ed3e0817a63e7cfafb5b0966bf7c4d3eb3d06544c70c097f79",
    }
    assert {path: sha256_file(ROOT / path) for path in expected} == expected
    authority = json.loads((ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    assert authority["status"] == "NOT_CONSUMED" and authority["consumed_queries"] == []
