from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.live_decision import compatibility_response
from aegis.research.committee_v2_shadow import (
    CommitteeV2ShadowError,
    CommitteeV2ShadowRuntime,
    load_committee_v2_shadow_config,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "config/committee_v2_shadow.yaml"


def _config(tmp_path: Path):
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "committee_v2_shadow.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return load_committee_v2_shadow_config(path, repo_root=tmp_path)


def _overlays(batch: dict) -> tuple[dict, dict]:
    primary = {}
    dual = {}
    for symbol, result in batch["results"].items():
        primary[symbol] = {
            "mode": "SHADOW",
            "selected": result["selected"],
            "score": result["candidate"]["calibrated_score"],
            "regime": {
                "direction": "BEARISH",
                "volatility": "NORMAL",
                "structure": "TREND",
                "evidence_ready": True,
            },
            "exchange_authority": False,
        }
        dual[symbol] = {
            "status": "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY",
            "model_only_selected": False,
            "regime_confirmed_selected": False,
            "score": 0.1,
            "exchange_authority": False,
        }
    return primary, dual


def test_repository_contract_has_one_eligible_directional_member() -> None:
    config = load_committee_v2_shadow_config(CONFIG, repo_root=ROOT)
    eligible = {
        name
        for name, member in config.member_contracts.items()
        if member["directional_vote_eligible"]
    }

    assert eligible == {"short_opportunity"}
    assert config.signal_journal.parent == ROOT / "data/committee_v2_shadow"
    assert config.maximum_paper_entries_per_cycle == 1


def test_live_or_fabricated_vote_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["mode"] = "LIVE"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        CommitteeV2ShadowError,
        match="SHADOW_AUTHORITY_INVALID",
    ):
        load_committee_v2_shadow_config(path, repo_root=tmp_path)

    payload["mode"] = "SHADOW"
    payload["members"]["qmae"]["directional_vote_eligible"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        CommitteeV2ShadowError,
        match="DIRECTIONAL_EVIDENCE_INVALID",
    ):
        load_committee_v2_shadow_config(path, repo_root=tmp_path)


def test_shadow_records_specialists_without_mutating_canonical_batch(
    tmp_path: Path,
) -> None:
    from test_entry_quality_v2_shadow_runtime import _batch

    runtime = CommitteeV2ShadowRuntime(_config(tmp_path))
    batch = _batch(0)
    original = copy.deepcopy(batch)
    primary, dual = _overlays(batch)

    overlay = runtime.observe_batch(
        batch,
        primary_overlay=primary,
        dual_overlay=dual,
    )

    assert batch == original
    assert set(overlay) == set(CANONICAL_SYMBOLS)
    assert all(
        value["directional_consensus"]
        == "NOT_APPLICABLE_SINGLE_ELIGIBLE_DIRECTIONAL_MEMBER"
        for value in overlay.values()
    )
    assert all(
        value["eligible_directional_member_count"] == 1 for value in overlay.values()
    )
    assert all(
        value["exchange_authority"] is False and value["exchange_mutations"] == 0
        for value in overlay.values()
    )
    assert sum(value["paper_action"] == "ENTER_NOW" for value in overlay.values()) == 1


def test_reversal_observation_waits_without_blocking_control(
    tmp_path: Path,
) -> None:
    from test_entry_quality_v2_shadow_runtime import _batch

    runtime = CommitteeV2ShadowRuntime(_config(tmp_path))
    batch = _batch(0)
    selected = CANONICAL_SYMBOLS[0]
    batch["results"][selected]["research_features"]["rebound_risk_proxy"] = 1.0
    primary, dual = _overlays(batch)

    overlay = runtime.observe_batch(
        batch,
        primary_overlay=primary,
        dual_overlay=dual,
    )

    assert batch["results"][selected]["selected"] is True
    assert overlay[selected]["paper_action"] == "WAIT_CONFIRMATION"
    assert overlay[selected]["would_change_control"] is True
    signal = next(
        json.loads(line)
        for line in runtime.config.signal_journal.read_text(
            encoding="utf-8"
        ).splitlines()
        if json.loads(line)["symbol"] == selected
    )
    assert signal["members"]["short_reversal_risk"]["flags"]["rebound_risk_proxy"]
    assert signal["meta_selector"]["exchange_authority"] is False


def test_duplicate_cycle_is_idempotent_and_outcomes_mature(
    tmp_path: Path,
) -> None:
    from test_entry_quality_v2_shadow_runtime import _batch

    runtime = CommitteeV2ShadowRuntime(_config(tmp_path))
    first = _batch(0)
    primary, dual = _overlays(first)
    initial = runtime.observe_batch(
        first,
        primary_overlay=primary,
        dual_overlay=dual,
    )
    duplicate = runtime.observe_batch(
        first,
        primary_overlay=primary,
        dual_overlay=dual,
    )
    assert duplicate == initial

    for cycle in range(1, runtime.config.horizon_bars + 1):
        batch = _batch(cycle)
        primary, dual = _overlays(batch)
        runtime.observe_batch(
            batch,
            primary_overlay=primary,
            dual_overlay=dual,
        )

    signals = runtime.config.signal_journal.read_text(encoding="utf-8").splitlines()
    outcomes = runtime.config.outcome_journal.read_text(encoding="utf-8").splitlines()
    assert len(signals) == (runtime.config.horizon_bars + 1) * len(CANONICAL_SYMBOLS)
    assert len(outcomes) == len(CANONICAL_SYMBOLS)
    assert all('"exchange_mutations":0' in row for row in outcomes)
    assert runtime.config.signal_journal.stat().st_mode & 0o777 == 0o600
    assert runtime.config.outcome_journal.stat().st_mode & 0o777 == 0o600


def test_single_estimator_semantics_are_explicit_in_http_contract() -> None:
    from test_entry_quality_v2_shadow_runtime import _batch

    batch = _batch(0)
    response = compatibility_response(
        batch,
        CANONICAL_SYMBOLS[0],
        "trace",
    )

    assert (
        response["metadata"]["directional_consensus"]
        == "NOT_APPLICABLE_SINGLE_ESTIMATOR"
    )
    assert response["metadata"]["direction_probability_semantics"] == (
        "SIDE_AUTHORITY_NOT_PROFITABILITY_CONFIDENCE"
    )
    assert response["metadata"]["candidate_confidence_semantics"] == (
        "NOT_APPLICABLE_SINGLE_ESTIMATOR"
    )


def test_committee_module_has_no_exchange_or_process_surface() -> None:
    from aegis.research import committee_v2_shadow

    source = inspect.getsource(committee_v2_shadow).lower()
    forbidden = (
        "create_order",
        "cancel_order",
        "change_leverage",
        "change_margin",
        "api_secret",
        "api_key",
        "subprocess",
        "pm2",
    )
    assert all(value not in source for value in forbidden)
