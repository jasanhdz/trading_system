from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.directional_shadow_evidence import (
    build_directional_shadow_evidence,
    load_directional_shadow_evidence_config,
)


def _write_fixture(
    root: Path,
    *,
    offline_state: str,
    selected_symbols: tuple[str, ...],
) -> Path:
    signals = root / "signals.jsonl"
    outcomes = root / "outcomes.jsonl"
    signal_rows = []
    outcome_rows = []
    start = datetime(2026, 7, 26, tzinfo=timezone.utc)
    for cycle, selected_symbol in enumerate(selected_symbols):
        timestamp = start + timedelta(hours=2 * cycle)
        for symbol in CANONICAL_SYMBOLS:
            event_id = f"{cycle}-{symbol}"
            selected = symbol == selected_symbol
            signal_rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "long_shadow": {
                        "score": 0.9 if selected else 0.1,
                        "regime": {
                            "direction": "BULLISH",
                            "volatility": "NORMAL",
                            "structure": "TREND",
                        },
                    },
                }
            )
            outcome_rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "signal_timestamp": timestamp.isoformat(),
                    "net_return_fraction": 0.01 if selected else -0.001,
                    "mae_fraction": 0.001,
                    "model_only_selected": selected,
                    "regime_confirmed_selected": selected,
                }
            )
    signals.write_text(
        "\n".join(json.dumps(row) for row in signal_rows) + "\n",
        encoding="utf-8",
    )
    outcomes.write_text(
        "\n".join(json.dumps(row) for row in outcome_rows) + "\n",
        encoding="utf-8",
    )
    config = root / "dual.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": (
                    "aegis-entry-quality-v3-dual-shadow-runtime-v1"
                ),
                "mode": "SHADOW",
                "side": "LONG",
                "artifact": {
                    "offline_validation_state": offline_state,
                },
                "evidence": {
                    "journal_root": ".",
                    "signal_journal": signals.name,
                    "outcome_journal": outcomes.name,
                },
                "evaluation": {
                    "report_path": "report.json",
                    "minimum_selected_outcomes": 4,
                    "minimum_independent_blocks": 4,
                    "maximum_symbol_concentration": 0.30,
                    "bootstrap_resamples": 200,
                    "bootstrap_seed": 7,
                    "bootstrap_block_minutes": 120,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def test_long_shadow_evidence_rejects_symbol_concentration(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(
        tmp_path,
        offline_state="PASSED",
        selected_symbols=("ADAUSDT",) * 4,
    )
    config = load_directional_shadow_evidence_config(
        config_path,
        repo_root=tmp_path,
    )
    report = build_directional_shadow_evidence(config)

    assert report["regime_confirmed"]["signals"] == 4
    assert report["regime_confirmed"]["symbol_concentration"] == 1.0
    assert report["regime_confirmed"]["evidence_passed"] is False
    assert (
        report["readiness"]["state"]
        == "COLLECTING_INDEPENDENT_SHADOW_EVIDENCE"
    )
    assert report["exchange_mutations"] == 0


def test_long_shadow_evidence_requires_offline_and_diverse_evidence(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(
        tmp_path,
        offline_state="FAILED",
        selected_symbols=(
            "ETHUSDT",
            "BTCUSDT",
            "SOLUSDT",
            "LTCUSDT",
        ),
    )
    config = load_directional_shadow_evidence_config(
        config_path,
        repo_root=tmp_path,
    )
    report = build_directional_shadow_evidence(config)

    assert report["regime_confirmed"]["independent_blocks"] == 4
    assert report["regime_confirmed"]["symbol_concentration"] == 0.25
    assert report["regime_confirmed"]["evidence_passed"] is True
    assert report["readiness"]["offline_validation_passed"] is False
    assert report["readiness"]["state"] == "OFFLINE_VALIDATION_FAILED"
