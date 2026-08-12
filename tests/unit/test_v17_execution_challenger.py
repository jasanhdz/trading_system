from pathlib import Path

import pytest

from aegis.v17_execution_challenger import (
    V17CanonicalDecision,
    V17ChallengerError,
    load_v17_challenger_config,
)
from aegis.live_decision import CurrentBrainDecisionService


ROOT = Path(__file__).resolve().parents[2]


def decision(**changes: object) -> V17CanonicalDecision:
    values = {
        "symbol": "ETHUSDT",
        "side": "LONG",
        "selected": True,
        "clean_probability": 0.72,
        "danger_probability": 0.18,
        "mae_q90": 0.004,
        "rank_score": 0.81,
        "minimum_clean_probability": 0.65,
        "maximum_danger_probability": 0.25,
        "maximum_mae_q90": 0.006,
        "minimum_rank_score": 0.75,
        "expected_price": 3000.0,
        "market_timestamp": "2026-08-12T00:00:00Z",
        "feature_hash": "feature-hash",
        "model_identifier": "v17-test-fixture",
        "model_sha256": "a" * 64,
        "policy_identifier": "v17-policy-test-fixture",
    }
    values.update(changes)
    return V17CanonicalDecision(**values)  # type: ignore[arg-type]


def test_tracked_challenger_is_fail_closed_and_preserves_v15() -> None:
    config = load_v17_challenger_config(ROOT / "config/v17_execution_challenger.yaml")
    assert config.execution_authority is False
    assert config.model_available is False
    assert config.execution_ready is False
    assert config.health()["blocker"] == "V17_REPRODUCIBLE_MODEL_ARTIFACT_REQUIRED"


def test_service_health_exposes_inactive_challenger_without_selecting_it() -> None:
    config = load_v17_challenger_config(ROOT / "config/v17_execution_challenger.yaml")
    engine = type("Engine", (), {"ready": True})()
    service = CurrentBrainDecisionService(
        engine, object(), v17_challenger_config=config  # type: ignore[arg-type]
    )
    assert service.health()["v17_execution_challenger"] == config.health()


def test_canonical_decision_preserves_all_v17_values() -> None:
    payload = decision().telemetry()
    assert payload["selected"] is True
    assert payload["clean_probability"] == 0.72
    assert payload["danger_probability"] == 0.18
    assert payload["mae_q90"] == 0.004
    assert payload["rank_score"] == 0.81
    assert payload["expected_price"] == 3000.0


def test_selected_cannot_disagree_with_policy() -> None:
    with pytest.raises(V17ChallengerError, match="selected flag"):
        decision(selected=True, clean_probability=0.1)


@pytest.mark.parametrize("field", ["clean_probability", "danger_probability"])
def test_probabilities_must_be_valid(field: str) -> None:
    with pytest.raises(V17ChallengerError):
        decision(**{field: float("nan")})
