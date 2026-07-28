from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import FEATURE_NAMES
from aegis.research.short_probability_shadow import (
    ShortProbabilityShadowConfig,
    ShortProbabilityShadowRuntime,
)


class _FixedProbabilityModel:
    def __init__(self, probability: float, semantics: str) -> None:
        self.value = probability
        self.probability_semantics = semantics

    def probability(self, symbol: str, features: list[float]) -> float:
        assert symbol in CANONICAL_SYMBOLS
        assert len(features) == len(FEATURE_NAMES)
        return self.value


def _batch(cycle: int) -> dict:
    timestamp = datetime(2026, 7, 28, tzinfo=timezone.utc) + timedelta(
        minutes=5 * cycle
    )
    results = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        close = 100.0 + index - cycle * 0.1
        features = {name: 0.0 for name in FEATURE_NAMES}
        results[symbol] = {
            "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "feature_schema": "aegis-features-v2",
            "feature_vector_hash": f"{cycle * 100 + index:064x}",
            "research_features": features,
            "market_bar": {
                "open": close + 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
            },
            "predictions": [
                {
                    "short_probability": 0.99,
                }
            ],
            "candidate": {
                "side": "SHORT",
            },
            "selected": symbol == CANONICAL_SYMBOLS[0],
        }
    return {
        "decision_cycle_id": f"cycle-{cycle:04d}",
        "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "feature_schema": "aegis-features-v2",
        "feature_count": len(FEATURE_NAMES),
        "results": results,
    }


def _config(tmp_path: Path) -> ShortProbabilityShadowConfig:
    return ShortProbabilityShadowConfig(
        config_path=tmp_path / "config.yaml",
        config_sha256="a" * 64,
        profitability_artifact_path=tmp_path / "profitability.json",
        profitability_artifact_sha256="b" * 64,
        clean_entry_artifact_path=tmp_path / "clean.json",
        clean_entry_artifact_sha256="c" * 64,
        signal_journal=tmp_path / "signals.jsonl",
        outcome_journal=tmp_path / "outcomes.jsonl",
        horizon_bars=12,
        round_trip_cost_fraction=0.001,
        maximum_clean_mae_fraction=0.01,
    )


def test_probability_semantics_are_distinct_and_have_no_selection_effect(
    tmp_path: Path,
) -> None:
    runtime = ShortProbabilityShadowRuntime(
        _config(tmp_path),
        _FixedProbabilityModel(
            0.63,
            "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS",
        ),  # type: ignore[arg-type]
        _FixedProbabilityModel(
            0.17,
            "CLEAN_ENTRY_LOW_MAE_H12",
        ),  # type: ignore[arg-type]
    )
    batch = _batch(0)
    original_selected = {
        symbol: bool(batch["results"][symbol]["selected"])
        for symbol in CANONICAL_SYMBOLS
    }

    overlay = runtime.observe_batch(batch)

    assert set(overlay) == set(CANONICAL_SYMBOLS)
    assert batch["results"][CANONICAL_SYMBOLS[0]]["selected"] is True
    assert original_selected == {
        symbol: bool(batch["results"][symbol]["selected"])
        for symbol in CANONICAL_SYMBOLS
    }
    for value in overlay.values():
        probabilities = value["probabilities"]
        assert probabilities["short_side_authority"] == 0.99
        assert probabilities["terminal_net_positive_h12_after_costs"] == 0.63
        assert probabilities["clean_entry_low_mae_h12"] == 0.17
        assert value["selection_effect"] == "NONE"
        assert value["exchange_authority"] is False


def test_outcomes_mature_without_exchange_authority(tmp_path: Path) -> None:
    runtime = ShortProbabilityShadowRuntime(
        _config(tmp_path),
        _FixedProbabilityModel(
            0.63,
            "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS",
        ),  # type: ignore[arg-type]
        _FixedProbabilityModel(
            0.17,
            "CLEAN_ENTRY_LOW_MAE_H12",
        ),  # type: ignore[arg-type]
    )

    for cycle in range(13):
        runtime.observe_batch(_batch(cycle))

    health = runtime.health()
    assert health["signal_records"] == 13 * len(CANONICAL_SYMBOLS)
    assert health["matured_outcomes"] == len(CANONICAL_SYMBOLS)
    assert health["exchange_mutations"] == 0
    assert health["selection_effect"] == "NONE"


def test_repeated_http_cycle_does_not_count_as_a_new_market_bar(
    tmp_path: Path,
) -> None:
    runtime = ShortProbabilityShadowRuntime(
        _config(tmp_path),
        _FixedProbabilityModel(
            0.63,
            "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS",
        ),  # type: ignore[arg-type]
        _FixedProbabilityModel(
            0.17,
            "CLEAN_ENTRY_LOW_MAE_H12",
        ),  # type: ignore[arg-type]
    )
    first = _batch(0)
    repeated = _batch(0)
    repeated["decision_cycle_id"] = "different-http-cycle"

    first_overlay = runtime.observe_batch(first)
    repeated_overlay = runtime.observe_batch(repeated)

    assert repeated_overlay == first_overlay
    assert runtime.health()["signal_records"] == len(CANONICAL_SYMBOLS)
