from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.dual_side_shadow import (
    DualSideShadowError,
    build_composite_research_observer,
    load_dual_side_shadow_config,
)


ROOT = Path(__file__).parents[2]
PRIMARY = ROOT / "config/entry_quality_v2.yaml"
DUAL = ROOT / "config/entry_quality_v3_dual_shadow.yaml"
COMMITTEE = ROOT / "config/committee_v2_shadow.yaml"


def test_dual_side_repository_config_is_shadow_only_and_hash_pinned() -> None:
    config = load_dual_side_shadow_config(DUAL, repo_root=ROOT)
    assert config.mode.value == "SHADOW"
    assert config.artifact_path.is_file()
    assert config.signal_journal.parent.name == "entry_quality_v3_dual_shadow"


def test_failed_long_artifact_cannot_be_switched_to_live(tmp_path: Path) -> None:
    payload = yaml.safe_load(DUAL.read_text())
    payload["mode"] = "LIVE"
    payload["evidence"]["journal_root"] = str(
        ROOT / "data/entry_quality_v3_dual_shadow"
    )
    path = tmp_path / "dual.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(
        DualSideShadowError, match="LIVE_PROMOTION_PROHIBITED"
    ):
        load_dual_side_shadow_config(path, repo_root=ROOT)


def test_composite_observer_preserves_primary_and_adds_non_authoritative_long(
    tmp_path: Path,
) -> None:
    from test_entry_quality_v2_shadow_runtime import _batch

    primary_payload = yaml.safe_load(PRIMARY.read_text())
    primary_payload["evidence"]["journal_root"] = "../data/primary"
    primary_payload["opportunity"] = {
        "source": "CURRENT_EQM_CLEAN_PROBABILITY_PROXY",
        "artifact_path": None,
        "artifact_sha256": None,
    }
    primary_path = tmp_path / "config/entry_quality_v2.yaml"
    primary_path.parent.mkdir()
    primary_path.write_text(yaml.safe_dump(primary_payload, sort_keys=False))

    dual_payload = yaml.safe_load(DUAL.read_text())
    dual_payload["evidence"]["journal_root"] = "../data/dual"
    dual_payload["artifact"]["path"] = str(
        DUAL.parent.parent
        / dual_payload["artifact"]["path"]
    )
    dual_payload["artifact"]["readiness_path"] = str(
        DUAL.parent.parent
        / dual_payload["artifact"]["readiness_path"]
    )
    dual_path = tmp_path / "config/dual.yaml"
    dual_path.write_text(yaml.safe_dump(dual_payload, sort_keys=False))

    committee_payload = yaml.safe_load(COMMITTEE.read_text())
    committee_payload["evidence"]["journal_root"] = "../data/committee"
    committee_path = tmp_path / "config/committee.yaml"
    committee_path.write_text(
        yaml.safe_dump(committee_payload, sort_keys=False)
    )

    observer = build_composite_research_observer(
        primary_path,
        dual_path,
        committee_path,
        repo_root=tmp_path,
    )
    batch = _batch(0)
    for result in batch["results"].values():
        result["predictions"][0]["long_probability"] = 0.6
        result["predictions"][0]["short_probability"] = 0.3
        result["predictions"][0]["expected_return"] = 0.01
        result["research_features"]["market_direction_6"] = 0.004
    overlay = observer.observe_batch(batch)

    assert set(overlay) == set(CANONICAL_SYMBOLS)
    assert all("selected" in overlay[symbol] for symbol in CANONICAL_SYMBOLS)
    assert all(
        overlay[symbol]["dual_side_shadow"]["exchange_authority"] is False
        for symbol in CANONICAL_SYMBOLS
    )
    assert (
        sum(
            overlay[symbol]["dual_side_shadow"]["model_only_selected"]
            for symbol in CANONICAL_SYMBOLS
        )
        <= 1
    )
    assert all(
        overlay[symbol]["committee_v2_shadow"]["exchange_authority"] is False
        for symbol in CANONICAL_SYMBOLS
    )
    assert all(
        overlay[symbol]["committee_v2_shadow"]["control_selected"]
        == batch["results"][symbol]["selected"]
        for symbol in CANONICAL_SYMBOLS
    )
    assert observer.health()["dual_side_shadow"]["exchange_mutations"] == 0
    assert observer.health()["committee_v2_shadow"]["exchange_mutations"] == 0
