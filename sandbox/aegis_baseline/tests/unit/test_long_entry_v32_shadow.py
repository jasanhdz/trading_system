from __future__ import annotations

from pathlib import Path

import yaml

from training.train_long_entry_v32_shadow import FAMILY_FEATURE_NAMES, _augment


def test_v32_family_encoding_is_explicit_and_deterministic() -> None:
    rows = [
        {
            "candidate_family": "BREAKOUT_RETEST",
            "features": (1.0, 2.0),
            "opportunity_features": (3.0, 4.0),
        }
    ]
    result = _augment(rows)[0]
    assert len(result["features"]) == 2 + len(FAMILY_FEATURE_NAMES)
    assert sum(result["features"][-4:]) == 1.0
    assert result["features"][-4:] == result["opportunity_features"][-4:]


def test_v32_changes_structure_without_relaxing_v31_gates() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v32_shadow.yaml").read_text()
    )
    change = payload["structural_change"]
    assert change["thresholds_relaxed"] is False
    assert change["context_relaxed"] is False
    assert change["confirmation_relaxed"] is False
    assert change["horizons_changed"] is False
    assert payload["validation"]["pooled_policy_may_rescue_failed_family"] is False
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0
