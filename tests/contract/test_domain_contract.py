from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from aegis.domain import BrainManifest, Candle, DecisionStatus, DomainValidationError, ScientificLayerName, TradeSide
from aegis.utils import Sha256HashProvider, canonical_json


def test_contract_enums_keep_no_trade_and_layer_order_explicit() -> None:
    assert TradeSide.NO_TRADE.value == DecisionStatus.NO_TRADE.value
    assert tuple(layer.value for layer in ScientificLayerName) == ("D3", "RV2", "TRRM", "QMAE", "EQM", "ECON1")


def test_domain_is_immutable_and_rejects_non_finite_values() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candle = Candle(now, now.replace(minute=5), 1, 2, 0.5, 1.5, 4, True, "fixture")
    with pytest.raises(FrozenInstanceError):
        candle.close = 2  # type: ignore[misc]
    with pytest.raises(DomainValidationError):
        Candle(now, now.replace(minute=5), 1, 2, 0.5, float("nan"), 4, True, "fixture")


def test_canonical_serialization_and_hash_are_stable(decision_request) -> None:
    serialized = canonical_json(decision_request)
    assert "2026-07-17T12:00:00Z" in serialized
    assert Sha256HashProvider().digest_value(decision_request) == Sha256HashProvider().digest_value(decision_request)


def test_shared_manifest_fixture_matches_python_contract() -> None:
    payload = json.loads((Path(__file__).parents[1] / "fixtures" / "brain_manifest.json").read_text())
    manifest = BrainManifest(**{**payload, "symbols": tuple(payload["symbols"]), "capabilities": tuple(payload["capabilities"])})
    assert manifest.ready and len(manifest.symbols) == 11
