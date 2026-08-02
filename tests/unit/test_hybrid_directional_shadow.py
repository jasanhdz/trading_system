from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import FEATURE_NAMES
from aegis.research.hybrid_directional_shadow import (
    HybridDirectionalShadowConfig,
    HybridDirectionalShadowError,
    HybridDirectionalShadowRuntime,
    load_hybrid_directional_shadow_config,
)
from aegis.training.hybrid_directional import load_hybrid_directional_artifact

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> HybridDirectionalShadowConfig:
    artifact = ROOT / "config/bundles/aegis-hybrid-directional-committee-v1.json"
    return HybridDirectionalShadowConfig(
        config_path=tmp_path / "config.yaml",
        config_sha256="a" * 64,
        artifact_path=artifact,
        artifact_sha256="f52dcaa12fe94b6cc9023c25cf95ea2d6fc16296c9b65c2c93d00e13e66ba0e8",
        signal_journal=tmp_path / "signals.jsonl",
        outcome_journal=tmp_path / "outcomes.jsonl",
        horizon_bars=2,
        round_trip_cost_fraction=0.001,
    )


def _batch(timestamp: datetime) -> dict:
    results = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["ret_6"] = (index - 5) * 0.001
        results[symbol] = {
            "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "feature_schema": "aegis-features-v2",
            "feature_vector_hash": f"features-{timestamp.timestamp()}-{symbol}",
            "research_features": features,
            "market_bar": {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.2,
            },
            "candidate": {"side": "SHORT"},
            "selected": index == 0,
        }
    return {
        "decision_cycle_id": f"cycle-{timestamp.timestamp()}",
        "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "results": results,
    }


def test_shadow_observer_records_both_sides_and_matures_outcomes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runtime = HybridDirectionalShadowRuntime(
        config, load_hybrid_directional_artifact(config.artifact_path)
    )
    start = datetime(2026, 8, 2, tzinfo=timezone.utc)
    first = runtime.observe_batch(_batch(start))
    runtime.observe_batch(_batch(start + timedelta(minutes=5)))
    runtime.observe_batch(_batch(start + timedelta(minutes=10)))

    assert set(first["ADAUSDT"]["predictions"]) == {"LONG", "SHORT"}
    assert first["ADAUSDT"]["selection_effect"] == "NONE"
    assert first["ADAUSDT"]["exchange_authority"] is False
    health = runtime.health()
    assert health["signal_records"] == 3 * len(CANONICAL_SYMBOLS)
    assert health["outcome_records"] == len(CANONICAL_SYMBOLS)
    assert health["exchange_mutations"] == 0
    assert set(runtime._outcomes.rows[0]["directional_outcomes"]) == {
        "LONG",
        "SHORT",
    }


def test_runtime_is_idempotent_per_market_timestamp(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime = HybridDirectionalShadowRuntime(
        config, load_hybrid_directional_artifact(config.artifact_path)
    )
    timestamp = datetime(2026, 8, 2, tzinfo=timezone.utc)
    runtime.observe_batch(_batch(timestamp))
    runtime.observe_batch(_batch(timestamp))

    assert runtime.health()["signal_records"] == len(CANONICAL_SYMBOLS)


def test_config_rejects_live_mode(tmp_path: Path) -> None:
    source = (ROOT / "config/hybrid_directional_shadow.yaml").read_text()
    path = tmp_path / "hybrid.yaml"
    path.write_text(source.replace("mode: SHADOW", "mode: LIVE", 1))

    with pytest.raises(HybridDirectionalShadowError, match="AUTHORITY_INVALID"):
        load_hybrid_directional_shadow_config(path, repo_root=ROOT)


def test_shadow_observer_has_no_mutating_exchange_surface() -> None:
    source = (ROOT / "src/aegis/research/hybrid_directional_shadow.py").read_text()
    for forbidden in (
        "create_order",
        "cancel_order",
        "modify_order",
        "close_position",
        "BinanceAdapter",
        "api_secret",
    ):
        assert forbidden not in source
