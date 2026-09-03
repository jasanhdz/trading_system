from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from aegis.features import FEATURE_NAMES
from aegis.research.live_feedback import (
    build_live_feedback_evidence,
    load_live_feedback_config,
)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def test_feedback_builder_joins_shadow_outcomes_and_actual_trade(tmp_path: Path) -> None:
    signals = tmp_path / "signals.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    trades = tmp_path / "turbo_trades_2026-07-25.jsonl"
    start = datetime(2026, 7, 25, tzinfo=timezone.utc)
    signal_rows = []
    for index in range(13):
        timestamp = start + timedelta(minutes=5 * index)
        close = 100.0 + index
        signal_rows.append(
            {
                "event_id": f"event-{index}",
                "decision_cycle_id": f"cycle-{index}",
                "market_timestamp": _iso(timestamp),
                "symbol": "AVAXUSDT",
                "feature_schema": "aegis-features-v2",
                "feature_vector_hash": f"{index:064x}",
                "feature_values": {name: 0.0 for name in FEATURE_NAMES},
                "market_bar": {
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.25,
                    "close": close,
                },
                "control": {
                    "selected": index == 0,
                    "side": "TradeSide.SHORT",
                    "raw_score": 1.0,
                    "calibrated_score": 0.01,
                    "reason_codes": ["ELIGIBLE"],
                },
                "v2": {
                    "selected": False,
                    "score": -0.01,
                    "opportunity_probability": 0.02,
                    "qmae_q90": 0.01,
                    "tail_risk_probability": 0.6,
                    "regime": {
                        "direction": "BULLISH",
                        "volatility": "HIGH",
                        "structure": "TREND",
                    },
                },
            }
        )
    signals.write_text(
        "\n".join(json.dumps(row) for row in signal_rows) + "\n",
        encoding="utf-8",
    )
    outcomes.write_text(
        json.dumps(
            {
                "event_id": "event-0",
                "symbol": "AVAXUSDT",
                "maturity_timestamp": _iso(start + timedelta(minutes=60)),
                "net_return_fraction": -0.121,
                "mfe_fraction": 0.0,
                "mae_fraction": 0.125,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trades.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "trade_id": "trade-1",
                    "symbol": "AVAXUSDT",
                    "side": "SHORT",
                    "status": "OPEN",
                    "opened_at": _iso(start + timedelta(minutes=2)),
                    "entry_price": 99.5,
                    "position_fraction": 0.08,
                    "leverage": 15,
                },
                {
                    "trade_id": "trade-1",
                    "symbol": "AVAXUSDT",
                    "side": "SHORT",
                    "status": "CLOSED",
                    "closed_at": _iso(start + timedelta(minutes=60)),
                    "exit_price": 112.0,
                    "pnl_usdt": -2.0,
                    "roe": -0.4,
                    "mfe_roe": 0.0,
                    "mae_roe": -0.4,
                    "metadata": {"exit_type": "STOP_LOSS"},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "feedback.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "aegis-entry-quality-live-feedback-v1",
                "symbols": [
                    "ETHUSDT",
                    "BTCUSDT",
                    "SOLUSDT",
                    "BNBUSDT",
                    "XRPUSDT",
                    "DOGEUSDT",
                    "ADAUSDT",
                    "AVAXUSDT",
                    "LINKUSDT",
                    "SUIUSDT",
                    "LTCUSDT",
                ],
                "inputs": {
                    "signal_journal": str(signals),
                    "outcome_journal": str(outcomes),
                    "trade_logs_glob": str(trades),
                },
                "labels": {
                    "schema": "aegis-labels-short-v4",
                    "horizon_bars": 12,
                },
                "evidence": {
                    "minimum_non_overlapping_episodes": 300,
                    "minimum_embargo_minutes": 120,
                },
                "outputs": {
                    "dataset_path": str(tmp_path / "dataset.jsonl"),
                    "report_path": str(tmp_path / "report.json"),
                },
                "automation": {
                    "automatic_training": False,
                    "automatic_promotion": False,
                },
                "training_controls": {
                    "historical_replay_required": True,
                    "live_only_training_allowed": False,
                    "purged_walk_forward_required": True,
                    "champion_challenger_required": True,
                    "owner_promotion_required": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_live_feedback_config(config_path, repo_root=tmp_path)

    report = build_live_feedback_evidence(config)
    row = json.loads(config.dataset_path.read_text(encoding="utf-8"))

    assert row["label"]["bad_entry"] is True
    assert row["classification"] == "ACTUAL_BAD_ENTRY_CONFIRMED"
    assert row["actual_trade"]["trade_id"] == "trade-1"
    assert row["observed"]["path_metrics"]["12"]["time_underwater_bars"] == 12
    assert report["counts"]["rows"] == 1
    assert report["counts"]["non_overlapping"] == 1
    assert report["signal_quality"]["control_selection"]["selected_outcomes"] == 1
    assert (
        report["signal_quality"]["challenger_selection"]["selected_outcomes"]
        == 0
    )
    assert (
        report["signal_quality"]["challenger_behavior"]["all_entries_rejected"]
        is True
    )
    assert (
        report["training_readiness"]["positive_selection_evidence_ready"]
        is False
    )
    assert (
        report["training_readiness"]["selection_failure"]
        == "NO_POSITIVE_SELECTION_EVIDENCE"
    )
    assert report["execution_quality"]["opened_trade_records"] == 1
    assert report["execution_quality"]["closed_trade_records"] == 1
    assert report["execution_quality"]["fill_price_capture_rate"] == 1.0
    assert report["execution_quality"]["signal_quality_inference_prohibited"] is True
    assert report["training_readiness"]["automatic_training"] is False
    assert report["training_readiness"]["automatic_promotion"] is False
    assert report["exchange_mutations"] == 0


def test_feedback_builder_serializes_unbounded_mfe_mae_ratio(tmp_path: Path) -> None:
    signals = tmp_path / "signals.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    start = datetime(2026, 7, 25, tzinfo=timezone.utc)
    rows = []
    for index in range(13):
        timestamp = start + timedelta(minutes=5 * index)
        close = 100.0 - index
        rows.append(
            {
                "event_id": f"event-{index}",
                "decision_cycle_id": f"cycle-{index}",
                "market_timestamp": _iso(timestamp),
                "symbol": "AVAXUSDT",
                "feature_schema": "aegis-features-v2",
                "feature_vector_hash": f"{index:064x}",
                "feature_values": {name: 0.0 for name in FEATURE_NAMES},
                "market_bar": {
                    "open": close,
                    "high": min(100.0, close + 0.25),
                    "low": close - 0.5,
                    "close": close,
                },
                "control": {
                    "selected": False,
                    "side": "TradeSide.SHORT",
                    "raw_score": 1.0,
                    "calibrated_score": 0.01,
                    "reason_codes": ["ELIGIBLE"],
                },
                "v2": {
                    "selected": False,
                    "score": -0.01,
                    "opportunity_probability": 0.02,
                    "qmae_q90": 0.01,
                    "tail_risk_probability": 0.2,
                    "regime": {
                        "direction": "BEARISH",
                        "volatility": "NORMAL",
                        "structure": "TREND",
                    },
                },
            }
        )
    signals.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    outcomes.write_text(
        json.dumps(
            {
                "event_id": "event-0",
                "symbol": "AVAXUSDT",
                "maturity_timestamp": _iso(start + timedelta(minutes=60)),
                "net_return_fraction": 0.119,
                "mfe_fraction": 0.125,
                "mae_fraction": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "feedback.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "aegis-entry-quality-live-feedback-v1",
                "symbols": [
                    "ETHUSDT",
                    "BTCUSDT",
                    "SOLUSDT",
                    "BNBUSDT",
                    "XRPUSDT",
                    "DOGEUSDT",
                    "ADAUSDT",
                    "AVAXUSDT",
                    "LINKUSDT",
                    "SUIUSDT",
                    "LTCUSDT",
                ],
                "inputs": {
                    "signal_journal": str(signals),
                    "outcome_journal": str(outcomes),
                    "trade_logs_glob": None,
                },
                "labels": {
                    "schema": "aegis-labels-short-v4",
                    "horizon_bars": 12,
                },
                "evidence": {
                    "minimum_non_overlapping_episodes": 300,
                    "minimum_embargo_minutes": 120,
                },
                "outputs": {
                    "dataset_path": str(tmp_path / "dataset.jsonl"),
                    "report_path": str(tmp_path / "report.json"),
                },
                "automation": {
                    "automatic_training": False,
                    "automatic_promotion": False,
                },
                "training_controls": {
                    "historical_replay_required": True,
                    "live_only_training_allowed": False,
                    "purged_walk_forward_required": True,
                    "champion_challenger_required": True,
                    "owner_promotion_required": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_live_feedback_config(config_path, repo_root=tmp_path)
    build_live_feedback_evidence(config)
    row = json.loads(config.dataset_path.read_text(encoding="utf-8"))

    assert row["label"]["mfe_mae_ratio"] is None
    assert row["label"]["mfe_mae_ratio_unbounded"] is True
