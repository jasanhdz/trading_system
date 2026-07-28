from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.entry_condition_study import (
    EntryConditionStudyError,
    evaluate_entry_condition_study,
    load_entry_condition_study_config,
    write_entry_condition_study_report,
)


START = datetime(2026, 7, 27, 21, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config_payload(tmp_path: Path) -> dict:
    return {
        "schema_version": "aegis-entry-condition-shadow-study-v1",
        "study_id": "test-study",
        "mode": "SHADOW",
        "runtime_authority": "OBSERVATIONAL_ONLY",
        "evidence_start_utc": _iso(START),
        "inputs": {
            "signal_journal": str(tmp_path / "signals.jsonl"),
            "outcome_journal": str(tmp_path / "outcomes.jsonl"),
            "v2_runtime_config": str(tmp_path / "v2.yaml"),
            "long_runtime_config": str(tmp_path / "long.yaml"),
        },
        "evidence": {
            "horizon_bars": 12,
            "minimum_observation_days": 14,
            "maximum_observation_days": 30,
            "minimum_non_overlapping_selected_episodes": 11,
            "minimum_per_symbol_non_overlapping_selected_episodes": 1,
            "minimum_embargo_minutes": 120,
            "minimum_temporal_blocks": 1,
            "maximum_symbol_concentration": 0.30,
            "maximum_mean_mae_fraction": 0.00275,
            "minimum_profit_factor": 1.0,
            "bootstrap_resamples": 50,
            "bootstrap_seed": 7,
            "require_expectancy_ci95_low_above_zero": True,
            "require_positive_first_and_second_half": True,
        },
        "regime_validation": {
            "minimum_observations_per_symbol": 2,
            "maximum_dominant_axis_fraction": 0.90,
            "minimum_distinct_direction_labels": 2,
            "minimum_distinct_structure_labels": 2,
            "minimum_axis_transitions": 1,
        },
        "hypotheses": [
            {
                "id": "CURRENT_CONTROL",
                "description": "control",
                "require_control_selected": True,
                "allowed_contexts": ["ANY"],
                "excluded_symbols": [],
            },
            {
                "id": "EXCLUDE_AVAX",
                "description": "exclude",
                "require_control_selected": True,
                "allowed_contexts": ["ANY"],
                "excluded_symbols": ["AVAXUSDT"],
            },
            {
                "id": "RANGE_OR_BEAR_TREND",
                "description": "context",
                "require_control_selected": True,
                "allowed_contexts": ["RANGE", "BEAR_TREND"],
                "excluded_symbols": [],
            },
            {
                "id": "RANGE_OR_BEAR_TREND_EXCLUDE_AVAX",
                "description": "combined",
                "require_control_selected": True,
                "allowed_contexts": ["RANGE", "BEAR_TREND"],
                "excluded_symbols": ["AVAXUSDT"],
            },
        ],
        "ranking_recalibration": {
            "enabled_for_research": True,
            "training_allowed_now": False,
            "historical_replay_required": True,
            "purged_walk_forward_required": True,
            "calibration_only_threshold_derivation": True,
            "champion_challenger_required": True,
            "minimum_positive_folds": 3,
            "fold_count": 4,
        },
        "promotion": {
            "automatic_training": False,
            "automatic_promotion": False,
            "live_configuration_changes": False,
            "owner_authorization_required": True,
            "same_evidence_discovery_and_validation_prohibited": True,
        },
        "outputs": {"report_path": str(tmp_path / "report.json")},
    }


def _prepare(tmp_path: Path) -> Path:
    payload = _config_payload(tmp_path)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (tmp_path / "v2.yaml").write_text("mode: SHADOW\n", encoding="utf-8")
    (tmp_path / "long.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "SHADOW",
                "selection": {"production_authority": False},
                "promotion": {"automatic_live_activation": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _evidence(tmp_path: Path) -> None:
    signals = []
    outcomes = []
    timestamps = (START + timedelta(hours=1), START + timedelta(hours=3))
    for index, timestamp in enumerate(timestamps):
        for symbol in CANONICAL_SYMBOLS:
            event_id = f"{index}-{symbol}"
            range_context = index == 0
            signals.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "market_timestamp": _iso(timestamp),
                    "control": {"selected": True},
                    "v2": {
                        "score": 0.2 + index,
                        "opportunity_probability": 0.3 + index * 0.1,
                        "regime": {
                            "direction": (
                                "NEUTRAL" if range_context else "BEARISH"
                            ),
                            "volatility": "NORMAL",
                            "structure": (
                                "RANGE" if range_context else "TREND"
                            ),
                        },
                    },
                }
            )
            outcomes.append(
                {
                    "event_id": event_id,
                    "net_return_fraction": 0.002 + index * 0.001,
                    "mae_fraction": 0.001,
                    "mfe_fraction": 0.004,
                }
            )
    signals.insert(
        0,
        {
            "event_id": "discovery-only",
            "symbol": "AVAXUSDT",
            "market_timestamp": _iso(START - timedelta(hours=1)),
            "control": {"selected": True},
            "v2": {
                "score": 9.0,
                "opportunity_probability": 0.99,
                "regime": {
                    "direction": "BEARISH",
                    "volatility": "HIGH",
                    "structure": "TREND",
                },
            },
        },
    )
    outcomes.insert(
        0,
        {
            "event_id": "discovery-only",
            "net_return_fraction": -0.5,
            "mae_fraction": 0.5,
            "mfe_fraction": 0.0,
        },
    )
    _write_jsonl(tmp_path / "signals.jsonl", signals)
    _write_jsonl(tmp_path / "outcomes.jsonl", outcomes)


def test_study_separates_discovery_and_passes_only_prospective_rows(
    tmp_path: Path,
) -> None:
    path = _prepare(tmp_path)
    _evidence(tmp_path)
    config = load_entry_condition_study_config(path, repo_root=tmp_path)
    report = evaluate_entry_condition_study(
        config, now=START + timedelta(days=15)
    )

    assert report["source"]["discovery_rows"] == 1
    assert report["source"]["prospective_rows"] == 22
    assert report["discovery_only"]["promotion_use"] == "PROHIBITED"
    assert report["regime_health"]["state"] == "HEALTHY"
    assert report["prospective_validation"]["CURRENT_CONTROL"][
        "evidence_passed"
    ]
    assert report["readiness"]["state"] == "READY_FOR_OWNER_REVIEW"
    assert report["exchange_mutations"] == 0


def test_study_remains_collecting_before_time_gate(tmp_path: Path) -> None:
    path = _prepare(tmp_path)
    _evidence(tmp_path)
    config = load_entry_condition_study_config(path, repo_root=tmp_path)
    report = evaluate_entry_condition_study(
        config, now=START + timedelta(days=2)
    )

    assert not report["time_checks"]["minimum_observation_days"]
    assert not report["readiness"]["evidence_ready"]
    assert report["readiness"]["state"] == (
        "COLLECTING_PROSPECTIVE_SHADOW_EVIDENCE"
    )


def test_live_or_automatic_authority_is_rejected(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["mode"] = "LIVE"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        EntryConditionStudyError,
        match="AEGIS_ENTRY_CONDITION_CONFIG_INVALID",
    ):
        load_entry_condition_study_config(path, repo_root=tmp_path)

    payload["mode"] = "SHADOW"
    payload["promotion"]["automatic_promotion"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        EntryConditionStudyError,
        match="AEGIS_ENTRY_CONDITION_RUNTIME_AUTHORITY_INVALID",
    ):
        load_entry_condition_study_config(path, repo_root=tmp_path)


def test_regime_collapse_blocks_readiness(tmp_path: Path) -> None:
    path = _prepare(tmp_path)
    _evidence(tmp_path)
    signals = [
        json.loads(line)
        for line in (tmp_path / "signals.jsonl").read_text().splitlines()
    ]
    for signal in signals:
        if signal["event_id"] != "discovery-only":
            signal["v2"]["regime"]["direction"] = "NEUTRAL"
            signal["v2"]["regime"]["structure"] = "RANGE"
    _write_jsonl(tmp_path / "signals.jsonl", signals)
    config = load_entry_condition_study_config(path, repo_root=tmp_path)
    report = evaluate_entry_condition_study(
        config, now=START + timedelta(days=15)
    )

    assert report["regime_health"]["state"] == "REGIME_COLLAPSE_DETECTED"
    assert not report["readiness"]["evidence_ready"]


def test_report_is_private_and_runtime_module_has_no_exchange_surface(
    tmp_path: Path,
) -> None:
    path = _prepare(tmp_path)
    _evidence(tmp_path)
    config = load_entry_condition_study_config(path, repo_root=tmp_path)
    report = evaluate_entry_condition_study(config, now=START)
    digest = write_entry_condition_study_report(report, config.report_path)

    assert len(digest) == 64
    assert config.report_path.stat().st_mode & 0o777 == 0o600
    source = Path(
        "src/aegis/research/entry_condition_study.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("binance", "create_order", "cancel_order", "pm2"):
        assert forbidden not in source.lower()


def test_repository_contract_freezes_recommendation_minimums() -> None:
    payload = yaml.safe_load(
        Path(
            "config/experiments/aegis_entry_condition_shadow_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    assert payload["mode"] == "SHADOW"
    assert payload["evidence"]["minimum_observation_days"] == 14
    assert (
        payload["evidence"]["minimum_non_overlapping_selected_episodes"]
        == 300
    )
    assert (
        payload["evidence"][
            "minimum_per_symbol_non_overlapping_selected_episodes"
        ]
        == 50
    )
    assert payload["ranking_recalibration"]["training_allowed_now"] is False
    assert payload["promotion"]["automatic_training"] is False
    assert payload["promotion"]["automatic_promotion"] is False
    assert payload["promotion"]["live_configuration_changes"] is False
