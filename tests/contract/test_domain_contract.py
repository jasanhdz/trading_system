import json
from pathlib import Path

from aegis.domain import BrainManifest, DecisionStatus, ScientificLayerName, TradeSide


def test_contract_enums_keep_no_trade_and_layer_order_explicit() -> None:
    assert TradeSide.NO_TRADE.value == "NO_TRADE"
    assert DecisionStatus.NO_TRADE.value == "NO_TRADE"
    assert tuple(layer.value for layer in ScientificLayerName) == (
        "D3",
        "RV2",
        "TRRM",
        "QMAE",
        "EQM",
        "ECON1",
    )


def test_manifest_fixture_has_the_domain_shape() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "brain_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = BrainManifest(
        contract_version=payload["contract_version"],
        universe_id=payload["universe_id"],
        symbols=tuple(payload["symbols"]),
        symbol_set_hash=payload["symbol_set_hash"],
        timeframe=payload["timeframe"],
        config_version=payload["config_version"],
        model_bundle_id=payload["model_bundle_id"],
        feature_schema_version=payload["feature_schema_version"],
        capabilities=tuple(payload["capabilities"]),
        build_id=payload["build_id"],
        ready=payload["ready"],
    )

    assert manifest.contract_version == "aegis-clean-rebuild-v1"
    assert len(manifest.symbols) == 11
    assert manifest.ready is False
