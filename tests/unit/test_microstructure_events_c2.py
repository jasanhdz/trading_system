import json
import sqlite3
import zipfile

import pytest

from aegis.domain import TradeSide
from aegis.research.market_event_lab_m1 import MarketEventContractError
from aegis.research.microstructure_events_c2 import (
    EVENT_FEATURES,
    C2Archive,
    C2EventFamily,
    C2Thresholds,
    archive_coverage,
    build_event_vector,
    detect_event,
    parse_aggregate_trade,
    parse_depth,
    parse_liquidation,
    parse_open_interest,
)
from collect_market_microstructure_c2 import normalize_message, stream_names
from import_market_microstructure_c2_archives import import_archives


TIMESTAMP = 1_800_000_000_000


def test_public_payloads_normalize_to_canonical_rows():
    trade = parse_aggregate_trade({
        "e": "aggTrade", "E": TIMESTAMP, "s": "ADAUSDT", "a": 7,
        "p": "0.5", "q": "12", "T": TIMESTAMP - 1, "m": True,
    })
    assert trade.quote_notional == 6.0
    liquidation = parse_liquidation({
        "e": "forceOrder", "E": TIMESTAMP, "o": {
            "s": "ADAUSDT", "S": "SELL", "q": "10", "p": "0.5",
            "ap": "0.49", "X": "FILLED", "z": "8", "T": TIMESTAMP - 1,
        },
    })
    assert liquidation.quote_notional == pytest.approx(3.92)
    depth = parse_depth({
        "e": "depthUpdate", "E": TIMESTAMP, "T": TIMESTAMP - 1,
        "s": "ADAUSDT", "u": 9,
        "b": [["0.49", "10"], ["0.48", "5"]],
        "a": [["0.50", "8"], ["0.51", "4"]],
    })
    assert depth.spread_bps > 0
    oi = parse_open_interest({
        "symbol": "ADAUSDT", "timestamp": TIMESTAMP,
        "sumOpenInterest": "100", "sumOpenInterestValue": "50",
    }, "ADAUSDT")
    assert oi.open_interest == 100


def test_stream_allowlist_and_envelope_are_strict():
    stream = "adausdt@aggTrade"
    message = json.dumps({"stream": stream, "data": {
        "e": "aggTrade", "E": TIMESTAMP, "s": "ADAUSDT", "a": 1,
        "p": "1", "q": "2", "T": TIMESTAMP, "m": False,
    }})
    source, row = normalize_message(message)
    assert source == "agg_trade"
    assert row.symbol == "ADAUSDT"
    assert len(stream_names()) == 33
    with pytest.raises(ValueError, match="NOT_ALLOWLISTED"):
        normalize_message(json.dumps({"stream": "adausdt@markPrice", "data": {}}))


def test_archive_is_idempotent_and_manifest_is_chained(tmp_path):
    path = tmp_path / "c2.db"
    archive = C2Archive(path)
    trade = parse_aggregate_trade({
        "e": "aggTrade", "E": TIMESTAMP, "s": "ADAUSDT", "a": 7,
        "p": "0.5", "q": "12", "T": TIMESTAMP, "m": True,
    })
    assert archive.insert(trade)
    assert not archive.insert(trade)
    first = archive.append_manifest({
        "created_at_utc": "2026-08-13T00:00:00Z", "source": "TEST",
        "first_timestamp_ms": TIMESTAMP, "last_timestamp_ms": TIMESTAMP,
        "accepted_rows": 1, "duplicate_rows": 1, "rejected_rows": 0,
    })
    second = archive.append_manifest({
        "created_at_utc": "2026-08-13T00:01:00Z", "source": "TEST",
        "first_timestamp_ms": TIMESTAMP + 1, "last_timestamp_ms": TIMESTAMP + 1,
        "accepted_rows": 0, "duplicate_rows": 1, "rejected_rows": 0,
    })
    assert archive.validate_manifest_chain() == (first, second)
    archive.close()
    assert archive_coverage(path)["aggregate_trades"]["rows"] == 1
    connection = sqlite3.connect(path)
    connection.execute("UPDATE c2_collection_manifest SET accepted_rows=99 WHERE sequence=1")
    connection.commit(); connection.close()
    archive = C2Archive(path)
    with pytest.raises(MarketEventContractError, match="CHAIN"):
        archive.validate_manifest_chain()
    archive.close()


@pytest.mark.parametrize("family", list(C2EventFamily))
def test_each_family_has_a_strict_passing_and_rejecting_vector(family):
    values = {
        "side_return_z": 3.0, "side_flow_z": 2.0, "oi_delta_z": 2.0,
        "price_acceptance": 1.0, "opposing_liquidation_z": 3.0,
        "price_response_abs": 0.01, "side_reclaim": 1.0,
        "side_depth_flip": 1.0, "aligned_liquidation_z": 3.0,
        "side_depth_imbalance": 1.0, "opposing_flow_z": 3.0,
        "spread_bps": 1.0, "side_flow_z_30s": 2.0,
        "side_flow_z_60s": 2.0, "side_flow_z_300s": 2.0,
        "side_leader_return_z": 3.0, "side_alt_residual_z": 0.01,
        "leader_flow_z": 2.0, "beta_btc": 1.2,
    }
    features = {name: values[name] for name in EVENT_FEATURES[family]}
    vector = build_event_vector(
        family=family, symbol="ADAUSDT", event_timestamp_ms=TIMESTAMP,
        side=TradeSide.LONG, features=features,
        source_max_timestamps_ms={"test": TIMESTAMP},
    )
    thresholds = C2Thresholds(2.5, 1.5, .05, .5, 5.0)
    assert detect_event(vector, thresholds) is TradeSide.LONG
    rejected = build_event_vector(
        family=family, symbol="ADAUSDT", event_timestamp_ms=TIMESTAMP,
        side=TradeSide.LONG,
        features={name: 0.0 for name in EVENT_FEATURES[family]},
        source_max_timestamps_ms={"test": TIMESTAMP},
    )
    assert detect_event(rejected, thresholds) is TradeSide.NO_TRADE


def test_event_vector_rejects_future_source_and_wrong_feature_order():
    family = C2EventFamily.OI_CONFIRMED_BREAKOUT
    features = dict(zip(EVENT_FEATURES[family], (3.0, 2.0, 2.0, 1.0)))
    with pytest.raises(MarketEventContractError, match="CAUSALITY"):
        build_event_vector(
            family=family, symbol="ADAUSDT", event_timestamp_ms=TIMESTAMP,
            side=TradeSide.LONG, features=features,
            source_max_timestamps_ms={"open_interest": TIMESTAMP + 1},
        )
    with pytest.raises(MarketEventContractError, match="CONTRACT"):
        build_event_vector(
            family=family, symbol="ADAUSDT", event_timestamp_ms=TIMESTAMP,
            side=TradeSide.LONG, features=dict(reversed(tuple(features.items()))),
            source_max_timestamps_ms={"open_interest": TIMESTAMP},
        )


def test_depth_requires_active_bid_and_ask_levels():
    with pytest.raises(MarketEventContractError, match="ACTIVE_LEVELS"):
        parse_depth({
            "e": "depthUpdate", "E": TIMESTAMP, "T": TIMESTAMP,
            "s": "ADAUSDT", "u": 9,
            "b": [["0.49", "0"]], "a": [["0.50", "1"]],
        })


def test_historical_aggregate_trade_archive_import_is_idempotent(tmp_path):
    path = tmp_path / "ADAUSDT-aggTrades-2026-07.zip"
    member = "ADAUSDT-aggTrades-2026-07.csv"
    payload = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        f"7,0.5,12,10,11,{TIMESTAMP},true\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload)
    output = tmp_path / "c2.db"
    first = import_archives((path,), output)
    second = import_archives((path,), output)
    assert first["archives"][0]["accepted_rows"] == 1
    assert second["archives"][0]["accepted_rows"] == 0
    assert second["archives"][0]["duplicate_rows"] == 1
    assert archive_coverage(output)["aggregate_trades"]["rows"] == 1
