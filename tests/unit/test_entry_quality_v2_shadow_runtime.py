from __future__ import annotations

import inspect
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import FEATURE_NAMES
from aegis.live_decision import compatibility_response
from aegis.research.shadow_runtime import (
    EntryQualityV2Error,
    EntryQualityV2Mode,
    EntryQualityV2ShadowRuntime,
    load_entry_quality_v2_config,
)
from scripts.set_entry_quality_v2_mode import (
    _live_authority,
    candidate_config,
    validate_candidate,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "config/entry_quality_v2.yaml"


def _runtime_config(tmp_path: Path):
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["opportunity"] = {
        "source": "CURRENT_EQM_CLEAN_PROBABILITY_PROXY",
        "artifact_path": None,
        "artifact_sha256": None,
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "entry_quality_v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_entry_quality_v2_config(path, repo_root=tmp_path)


def _batch(cycle: int) -> dict:
    timestamp = datetime(2026, 7, 25, tzinfo=timezone.utc) + timedelta(
        minutes=5 * cycle
    )
    results = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        raw = {name: 0.0 for name in FEATURE_NAMES}
        raw.update(
            {
                "market_direction_6": -0.004,
                "range_mean_24": 0.01 + cycle * 0.00001,
                "range_expansion": 0.1,
                "chop_12": 0.3,
                "trend_strength_12": 0.8,
            }
        )
        close = 100.0 + index - cycle * 0.1
        results[symbol] = {
            "symbol": symbol,
            "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "feature_schema": "aegis-features-v2",
            "feature_schema_hash": "f" * 64,
            "feature_vector_hash": f"{index:064x}",
            "feature_count": len(FEATURE_NAMES),
            "feature_quality": {
                "missing_values": 0,
                "clipped_values": 0,
                "finite": True,
                "history_rows": 96,
            },
            "research_features": raw,
            "research_normalized_features": raw,
            "market_bar": {
                "open": close + 0.02,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
            },
            "predictions": [
                {
                    "side": "SHORT",
                    "long_probability": 0.0,
                    "short_probability": 1.0,
                    "neutral_probability": 0.0,
                    "expected_return": -0.01,
                    "tail_risk_probability": 0.2,
                    "qmae_mean": 0.005,
                    "quality_probability": 0.8,
                    "calibration_valid": True,
                    "qmae_valid": True,
                    "qmae_q50": 0.005,
                    "qmae_q90": 0.01,
                    "qmae_coverage": 0.9,
                }
            ],
            "layer": {
                "side": "SHORT",
                "regime": "BEAR_TREND",
                "regime_confidence": 0.8,
                "rv2_tail_risk": 0.2,
                "trrm_compatibility": 0.8,
                "qmae_q90": 0.01,
                "qmae_quality": 0.67,
                "eqm_score": 0.008,
                "model_disagreement": 0.0,
                "calibrated_score": 0.008,
                "eligible": True,
                "reason_codes": ["ELIGIBLE"],
                "diagnostics": [],
            },
            "candidate": {
                "symbol": symbol,
                "side": "SHORT",
                "raw_score": 0.008,
                "calibrated_score": 0.008,
                "reason_codes": ["ELIGIBLE"],
            },
            "selected": symbol == CANONICAL_SYMBOLS[0],
            "selection_status": "SELECTED",
        }
    return {
        "schema_id": "aegis-current-brain-canonical-batch-v1",
        "decision_cycle_id": f"cycle-{cycle:04d}",
        "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "model_identifier": "test-current-brain",
        "model_sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
        "configuration_sha256": "c" * 64,
        "feature_schema": "aegis-features-v2",
        "feature_count": len(FEATURE_NAMES),
        "results": results,
    }


def test_repository_config_is_shadow_and_private() -> None:
    config = load_entry_quality_v2_config(CONFIG, repo_root=ROOT)
    assert config.mode is EntryQualityV2Mode.SHADOW
    assert config.journal_root == ROOT / "data/entry_quality_v2_shadow"
    assert config.opportunity_source == "SHORT_OPPORTUNITY_RF_SHADOW_CANDIDATE"
    assert config.opportunity_artifact_path is not None
    assert config.opportunity_artifact_path.is_file()


def test_live_mode_requires_promoted_model_and_record(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["mode"] = "LIVE"
    payload["opportunity"] = {
        "source": "CURRENT_EQM_CLEAN_PROBABILITY_PROXY",
        "artifact_path": None,
        "artifact_sha256": None,
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "entry_quality_v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        EntryQualityV2Error,
        match="AEGIS_ENTRY_QUALITY_V2_LIVE_MODEL_NOT_PROMOTED",
    ):
        load_entry_quality_v2_config(path, repo_root=tmp_path)


def test_live_mode_accepts_only_hash_bound_owner_approved_record(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifact = tmp_path / "opportunity.json"
    artifact.write_text("{}\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_id": "aegis-entry-quality-v2-live-promotion-v1",
                "state": "OWNER_APPROVED_FOR_LIVE_SWITCH",
                "artifact_sha256": artifact_sha256,
                "opportunity_source": "PROMOTED_SHORT_OPPORTUNITY_MODEL",
                "automatic_activation": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["mode"] = "LIVE"
    payload["opportunity"] = {
        "source": "PROMOTED_SHORT_OPPORTUNITY_MODEL",
        "artifact_path": "opportunity.json",
        "artifact_sha256": artifact_sha256,
    }
    payload["live_promotion"]["promotion_record_path"] = "promotion.json"
    payload["live_promotion"]["promotion_record_sha256"] = hashlib.sha256(
        promotion.read_bytes()
    ).hexdigest()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "entry_quality_v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_entry_quality_v2_config(path, repo_root=tmp_path)

    assert config.mode is EntryQualityV2Mode.LIVE


def test_mode_switch_validates_shadow_and_rejects_unapproved_live() -> None:
    shadow = candidate_config(CONFIG, "SHADOW")
    assert shadow == CONFIG.read_bytes()
    validate_candidate(CONFIG, shadow, ROOT)
    live = candidate_config(CONFIG, "LIVE")
    with pytest.raises(
        EntryQualityV2Error,
        match="AEGIS_ENTRY_QUALITY_V2_LIVE_AUTHORITY_INCOMPLETE",
    ):
        validate_candidate(CONFIG, live, ROOT)


def test_future_live_switch_is_valid_after_frozen_shadow_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "opportunity.json"
    artifact.write_text("{}\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "state": "LIVE_READY_NOT_ACTIVE",
                "artifact_sha256": artifact_sha256,
                "automatic_live_activation": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    data = tmp_path / "data" / "entry_quality_v2_shadow"
    data.mkdir(parents=True)
    signals = []
    outcomes = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(300):
        event_id = f"event-{index:04d}"
        timestamp = start + timedelta(hours=index)
        symbol = CANONICAL_SYMBOLS[index % len(CANONICAL_SYMBOLS)]
        signals.append(
            json.dumps(
                {
                    "event_id": event_id,
                    "decision_cycle_id": f"cycle-{index:04d}",
                    "market_timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "control": {"selected": False},
                    "v2": {"selected": True, "score": 0.01},
                },
                sort_keys=True,
            )
        )
        outcomes.append(
            json.dumps(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "horizon_bars": 12,
                    "net_return_fraction": 0.001,
                    "mae_fraction": 0.001,
                    "mfe_fraction": 0.002,
                },
                sort_keys=True,
            )
        )
    (data / "signals.jsonl").write_text("\n".join(signals) + "\n")
    (data / "outcomes.jsonl").write_text("\n".join(outcomes) + "\n")
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["evidence"]["journal_root"] = "../data/entry_quality_v2_shadow"
    payload["opportunity"] = {
        "source": "SHORT_OPPORTUNITY_RF_SHADOW_CANDIDATE",
        "artifact_path": "opportunity.json",
        "artifact_sha256": artifact_sha256,
    }
    payload["live_promotion"].update(
        {
            "technical_readiness_record_path": "readiness.json",
            "technical_readiness_record_sha256": hashlib.sha256(
                readiness.read_bytes()
            ).hexdigest(),
            "minimum_non_overlapping_episodes": 300,
            "promotion_record_path": "data/entry_quality_v2_shadow/live_promotion.json",
            "promotion_record_sha256": None,
        }
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "entry_quality_v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    promotion_path = tmp_path / "promotion-check.json"

    candidate, _, record = _live_authority(
        path,
        tmp_path,
        promotion_path_override=promotion_path,
    )
    promotion_path.write_bytes(record)
    validate_candidate(path, candidate, tmp_path)

    switched = yaml.safe_load(candidate)
    assert switched["mode"] == "LIVE"
    assert (
        switched["opportunity"]["source"]
        == "PROMOTED_SHORT_OPPORTUNITY_MODEL"
    )


def test_shadow_records_once_and_never_gets_exchange_authority(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    runtime = EntryQualityV2ShadowRuntime(config)
    batch = _batch(0)
    first = runtime.observe_batch(batch)
    second = runtime.observe_batch(batch)

    assert first == second
    assert len(first) == len(CANONICAL_SYMBOLS)
    assert sum(bool(value["selected"]) for value in first.values()) == 1
    assert all(value["mode"] == "SHADOW" for value in first.values())
    assert all(value["exchange_authority"] is False for value in first.values())
    assert len(config.signal_journal.read_text(encoding="utf-8").splitlines()) == 11
    assert config.signal_journal.stat().st_mode & 0o777 == 0o600


def test_shadow_reuses_same_market_timestamp_without_duplicate_evidence(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    runtime = EntryQualityV2ShadowRuntime(config)
    first_batch = _batch(0)
    second_batch = {
        **first_batch,
        "decision_cycle_id": "same-market-new-request",
    }

    first = runtime.observe_batch(first_batch)
    second = runtime.observe_batch(second_batch)

    assert second == first
    assert runtime.observation_errors == 0
    assert len(config.signal_journal.read_text().splitlines()) == len(
        CANONICAL_SYMBOLS
    )


def test_shadow_overlay_does_not_change_operational_response(
    tmp_path: Path,
) -> None:
    runtime = EntryQualityV2ShadowRuntime(_runtime_config(tmp_path))
    batch = _batch(0)
    symbol = CANONICAL_SYMBOLS[0]
    control = compatibility_response(batch, symbol, "trace")
    observed = compatibility_response(
        {**batch, "_entry_quality_v2": runtime.observe_batch(batch)},
        symbol,
        "trace",
    )

    assert observed["long_prob"] == control["long_prob"]
    assert observed["short_prob"] == control["short_prob"]
    assert observed["aegis"]["prod"] == control["aegis"]["prod"]
    assert observed["aegis"]["turbo"] == control["aegis"]["turbo"]
    assert (
        observed["aegis"]["decision_brain"]
        == control["aegis"]["decision_brain"]
    )
    assert observed["aegis"]["entry_quality_v2"]["mode"] == "SHADOW"
    assert observed["aegis"]["entry_quality_v2"]["exchange_authority"] is False


def test_paper_outcome_matures_after_exact_horizon(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    runtime = EntryQualityV2ShadowRuntime(config)
    for cycle in range(config.horizon_bars + 1):
        runtime.observe_batch(_batch(cycle))

    outcomes = config.outcome_journal.read_text(encoding="utf-8").splitlines()
    assert len(outcomes) == len(CANONICAL_SYMBOLS)
    assert all('"exchange_mutations":0' in row for row in outcomes)
    assert any('"execution":"COUNTERFACTUAL_OBSERVATION"' in row for row in outcomes)


def test_runtime_has_no_exchange_mutation_surface() -> None:
    from aegis.research import shadow_runtime

    source = inspect.getsource(shadow_runtime).lower()
    forbidden = (
        "create_order",
        "cancel_order",
        "cancel_all",
        "change_leverage",
        "change_margin",
        "/fapi/v1/order",
        "api_secret",
        "api_key",
    )
    assert all(value not in source for value in forbidden)
