from __future__ import annotations

import copy
import inspect
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import FEATURE_NAMES
from aegis.research.committee_v21_fit import (
    fit_committee_v21,
    load_committee_v21_fit_config,
)
from aegis.research.committee_v21_shadow import (
    CommitteeV21ShadowError,
    CommitteeV21ShadowRuntime,
    basis_term_names,
    committee_v21_calibrated_risk,
    load_committee_v21_artifact,
    load_committee_v21_contract,
    load_committee_v21_shadow_config,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file

ROOT = Path(__file__).parents[2]
PREREGISTRATION = ROOT / "config/experiments/aegis_committee_v21_preregistered_v1.yaml"


def _observation(index: int) -> dict[str, float | str]:
    contract = load_committee_v21_contract(PREREGISTRATION)
    adverse = index % 3 == 0
    values: dict[str, float | str] = {name: 0.0 for name in contract.numeric_terms}
    values.update(
        {
            "control_calibrated_score": 0.03 - 0.02 * adverse,
            "qmae_q90": 0.018 if adverse else 0.006,
            "tail_risk_probability": 0.72 if adverse else 0.28,
            "trrm_compatibility": 0.28 if adverse else 0.72,
            "extension_down_proxy": 0.02 if adverse else 0.002,
            "market_direction_6": 0.01 if adverse else -0.01,
            "ret_12": -0.03 if adverse else -0.005,
            "symbol": CANONICAL_SYMBOLS[index % len(CANONICAL_SYMBOLS)],
            "direction_regime": "BULLISH" if adverse else "BEARISH",
            "volatility_regime": "HIGH" if adverse else "NORMAL",
            "structure_regime": "RANGE" if adverse else "TREND",
        }
    )
    return values


def _fit_rows() -> tuple[dict, ...]:
    rows = []
    starts = {
        "TRAINING": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "CALIBRATION": datetime(2026, 6, 21, tzinfo=timezone.utc),
        "DIAGNOSTIC_ONLY": datetime(2026, 7, 5, tzinfo=timezone.utc),
    }
    for split in starts:
        for index in range(60):
            adverse = index % 3 == 0
            rows.append(
                {
                    "split": split,
                    "symbol": CANONICAL_SYMBOLS[index % len(CANONICAL_SYMBOLS)],
                    "signal_timestamp": (
                        starts[split] + timedelta(hours=index)
                    ).isoformat(),
                    "observation": _observation(index),
                    "net_return_fraction": -0.01 if adverse else 0.008,
                    "mae_fraction": 0.02 if adverse else 0.003,
                    "mfe_fraction": 0.002 if adverse else 0.015,
                    "adverse_label": int(adverse),
                    "feature_vector_hash": f"fixture-{split}-{index}",
                }
            )
    return tuple(rows)


def _artifact(tmp_path: Path):
    fit_config = replace(
        load_committee_v21_fit_config(
            PREREGISTRATION,
            repo_root=ROOT,
        ),
        artifact_path=tmp_path / "artifact.json",
        private_fit_root=tmp_path / "private",
    )
    payload, report = fit_committee_v21(
        fit_config,
        _fit_rows(),
        generated_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
    )
    atomic_write_json(fit_config.artifact_path, payload)
    artifact = load_committee_v21_artifact(
        fit_config.artifact_path,
        contract=fit_config.contract,
    )
    return fit_config, artifact, report


def _runtime_config(tmp_path: Path):
    fit_config, artifact, _ = _artifact(tmp_path)
    signal_root = tmp_path / "data/committee_v21_shadow"
    preregistration = tmp_path / "preregistration.yaml"
    preregistration.write_bytes(PREREGISTRATION.read_bytes())
    payload = {
        "schema_version": "aegis-specialized-committee-v21-shadow-runtime-v1",
        "experiment_id": fit_config.contract.experiment_id,
        "enabled": True,
        "mode": "SHADOW",
        "runtime_authority": "OBSERVATIONAL_ONLY",
        "authority": {
            "preregistration_path": str(preregistration),
            "preregistration_sha256": sha256_file(preregistration),
            "exchange_authority": False,
            "runtime_training": False,
            "automatic_promotion": False,
        },
        "artifact": {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
        },
        "evidence": {
            "journal_root": str(signal_root),
            "signal_journal": "signals.jsonl",
            "outcome_journal": "outcomes.jsonl",
            "evidence_start_utc": "2026-07-05T00:00:00Z",
        },
        "counterfactual": {
            "control_authority": "CURRENT_CANONICAL_SELECTION",
            "maximum_paper_entries_per_cycle": 1,
            "fabricated_votes_prohibited": True,
        },
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_committee_v21_shadow_config(path, repo_root=tmp_path)


def _primary_overlay(batch: dict) -> dict:
    return {
        symbol: {
            "regime": {
                "direction": "BEARISH",
                "volatility": "NORMAL",
                "structure": "TREND",
            }
        }
        for symbol in batch["results"]
    }


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
                "range_mean_24": 0.01,
                "range_expansion": 0.1,
                "chop_12": 0.3,
                "trend_strength_12": 0.8,
            }
        )
        close = 100.0 + index
        results[symbol] = {
            "symbol": symbol,
            "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "feature_schema": "aegis-features-v2",
            "feature_vector_hash": f"{index:064x}",
            "research_features": raw,
            "market_bar": {
                "open": close + 0.02,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
            },
            "layer": {
                "rv2_tail_risk": 0.2,
                "trrm_compatibility": 0.8,
                "qmae_q90": 0.01,
            },
            "candidate": {
                "symbol": symbol,
                "side": "SHORT",
                "raw_score": 0.008,
                "calibrated_score": 0.008,
            },
            "selected": symbol == CANONICAL_SYMBOLS[0],
        }
    return {
        "schema_id": "aegis-current-brain-canonical-batch-v1",
        "decision_cycle_id": f"cycle-{cycle:04d}",
        "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "feature_schema": "aegis-features-v2",
        "feature_count": len(FEATURE_NAMES),
        "results": results,
    }


def test_preregistration_freezes_interactions_and_shadow_authority() -> None:
    contract = load_committee_v21_contract(PREREGISTRATION)

    assert contract.experiment_id == "aegis-specialized-committee-v21-shadow-01"
    assert len(contract.numeric_terms) == 31
    assert len(contract.interaction_terms) == 13
    assert len(basis_term_names(contract)) > len(contract.numeric_terms)
    assert contract.horizon_bars == 12


def test_fit_uses_separate_calibration_and_exports_transparent_model(
    tmp_path: Path,
) -> None:
    fit_config, artifact, report = _artifact(tmp_path)

    assert artifact.contract_sha256 == fit_config.contract.sha256
    assert artifact.calibration_slope > 0.0
    assert 0.0 < artifact.calibrated_risk_threshold < 1.0
    assert set(artifact.coefficients) == set(basis_term_names(fit_config.contract))
    assert report["dataset"]["training_episodes"] == 60
    assert report["dataset"]["calibration_episodes"] == 60
    assert report["dataset"]["diagnostic_only_episodes"] == 60
    assert report["diagnostic_promotion_use"] == "PROHIBITED"


def test_calibrated_risk_is_finite_and_orders_fixture_risk(
    tmp_path: Path,
) -> None:
    fit_config, artifact, _ = _artifact(tmp_path)
    low = committee_v21_calibrated_risk(
        fit_config.contract,
        artifact,
        _observation(1),
    )
    high = committee_v21_calibrated_risk(
        fit_config.contract,
        artifact,
        _observation(3),
    )

    assert math.isfinite(float(low["calibrated_risk_probability"]))
    assert float(low["calibrated_risk_probability"]) < float(
        high["calibrated_risk_probability"]
    )


def test_runtime_records_counterfactual_without_mutating_control(
    tmp_path: Path,
) -> None:
    runtime = CommitteeV21ShadowRuntime(_runtime_config(tmp_path))
    batch = _batch(0)
    original = copy.deepcopy(batch)

    overlay = runtime.observe_batch(
        batch,
        primary_overlay=_primary_overlay(batch),
    )

    assert batch == original
    assert set(overlay) == set(CANONICAL_SYMBOLS)
    assert all(value["mode"] == "SHADOW" for value in overlay.values())
    assert all(value["exchange_authority"] is False for value in overlay.values())
    assert sum(value["paper_action"] == "ENTER_NOW" for value in overlay.values()) <= 1
    assert runtime.health()["runtime_training"] is False
    assert runtime.health()["automatic_promotion"] is False


def test_live_mode_or_wrong_artifact_hash_is_rejected(
    tmp_path: Path,
) -> None:
    runtime = _runtime_config(tmp_path)
    payload = yaml.safe_load(runtime.path.read_text(encoding="utf-8"))
    payload["mode"] = "LIVE"
    runtime.path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(
        CommitteeV21ShadowError,
        match="RUNTIME_AUTHORITY_INVALID",
    ):
        load_committee_v21_shadow_config(runtime.path, repo_root=tmp_path)

    payload["mode"] = "SHADOW"
    payload["artifact"]["sha256"] = "0" * 64
    runtime.path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(
        CommitteeV21ShadowError,
        match="RUNTIME_AUTHORITY_INVALID",
    ):
        load_committee_v21_shadow_config(runtime.path, repo_root=tmp_path)


def test_v21_modules_have_no_exchange_or_process_mutation_surface() -> None:
    from aegis.research import committee_v21_fit, committee_v21_shadow

    source = (
        inspect.getsource(committee_v21_shadow) + inspect.getsource(committee_v21_fit)
    ).lower()
    forbidden = (
        "create_order",
        "cancel_order",
        "change_leverage",
        "change_margin",
        "api_secret",
        "api_key",
        "subprocess",
        "pm2",
        "requests.",
    )

    assert all(value not in source for value in forbidden)


def test_v21_live_integration_is_observational_only() -> None:
    live_api = (ROOT / "src/aegis/live_api.py").read_text(encoding="utf-8")
    composite = (ROOT / "src/aegis/research/dual_side_shadow.py").read_text(
        encoding="utf-8"
    )
    runtime = yaml.safe_load(
        (ROOT / "config/committee_v21_shadow.yaml").read_text(encoding="utf-8")
    )

    assert "committee_v21_shadow.yaml" in live_api
    assert "committee_v21_shadow" in composite
    assert runtime["mode"] == "SHADOW"
    assert runtime["runtime_authority"] == "OBSERVATIONAL_ONLY"
    assert runtime["authority"]["exchange_authority"] is False
    assert runtime["authority"]["automatic_promotion"] is False


def test_runtime_observation_failure_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CommitteeV21ShadowRuntime(_runtime_config(tmp_path))
    batch = _batch(0)

    def fail(*args, **kwargs):
        raise CommitteeV21ShadowError("fixture failure")

    monkeypatch.setattr(runtime, "_build_rows", fail)

    assert (
        runtime.observe_batch(
            batch,
            primary_overlay=_primary_overlay(batch),
        )
        == {}
    )
    assert runtime.health()["observation_errors"] == 1


def test_malformed_or_nonfinite_observation_fails_closed(
    tmp_path: Path,
) -> None:
    fit_config, artifact, _ = _artifact(tmp_path)
    malformed = _observation(1)
    malformed["qmae_q90"] = float("nan")

    with pytest.raises(CommitteeV21ShadowError, match="NONFINITE"):
        committee_v21_calibrated_risk(
            fit_config.contract,
            artifact,
            malformed,
        )
