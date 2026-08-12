import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from aegis.data import CanonicalBar
from aegis.domain import TradeSide
from aegis.research.market_event_lab_m1 import (
    AppendOnlyTrialLedger,
    EVENT_FEATURE_NAMES,
    FAMILY_REQUIREMENTS,
    EventFamily,
    EventThresholds,
    MarketEventContractError,
    TrialRecord,
    assess_database_readiness,
    build_event_vector,
    detect_event,
    initialize_market_event_schema,
    replay_event_path,
)


def _oi_vector():
    family = EventFamily.OI_CONFIRMED_BREAKOUT
    timestamp = 1_800_000_000_000
    vector = build_event_vector(
        family=family,
        symbol="ADAUSDT",
        event_timestamp_ms=timestamp,
        features=dict(zip(EVENT_FEATURE_NAMES[family], (3.1, 2.0, 1.8, 0.7))),
        source_max_timestamps_ms={name: timestamp for name in FAMILY_REQUIREMENTS[family]},
    )
    return vector


def test_event_contract_rejects_missing_extra_order_dtype_hash_and_future_source() -> None:
    vector = _oi_vector()
    vector.validate()
    family = vector.family
    base = dict(zip(vector.names, vector.values))
    with pytest.raises(MarketEventContractError, match="MISSING"):
        build_event_vector(
            family=family,
            symbol=vector.symbol,
            event_timestamp_ms=vector.event_timestamp_ms,
            features={name: value for name, value in base.items() if name != vector.names[0]},
            source_max_timestamps_ms=vector.source_max_timestamps_ms,
        )
    with pytest.raises(MarketEventContractError, match="EXTRA"):
        build_event_vector(
            family=family,
            symbol=vector.symbol,
            event_timestamp_ms=vector.event_timestamp_ms,
            features={**base, "future_return": 1.0},
            source_max_timestamps_ms=vector.source_max_timestamps_ms,
        )
    with pytest.raises(MarketEventContractError, match="ORDER"):
        build_event_vector(
            family=family,
            symbol=vector.symbol,
            event_timestamp_ms=vector.event_timestamp_ms,
            features=dict(reversed(tuple(base.items()))),
            source_max_timestamps_ms=vector.source_max_timestamps_ms,
        )
    with pytest.raises(MarketEventContractError, match="DTYPE"):
        replace(vector, dtype="float32").validate()
    with pytest.raises(MarketEventContractError, match="HASH"):
        replace(vector, feature_hash="0" * 64).validate()
    future_sources = dict(vector.source_max_timestamps_ms)
    future_sources["open_interest"] += 1
    with pytest.raises(MarketEventContractError, match="CAUSALITY"):
        replace(vector, source_max_timestamps_ms=future_sources).validate()


def test_detector_uses_supplied_train_thresholds_and_has_no_permissive_default() -> None:
    vector = _oi_vector()
    passing = EventThresholds(3.0, 1.5, 0.1, 0.5)
    rejecting = EventThresholds(3.2, 2.1, 0.1, 0.8)
    assert detect_event(vector, passing) is TradeSide.LONG
    assert detect_event(vector, rejecting) is TradeSide.NO_TRADE


def test_schema_is_prospective_and_readiness_fails_closed_on_immature_sources(tmp_path) -> None:
    database = tmp_path / "events.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE kline_microstructure (
          symbol TEXT, open_time_ms INTEGER, quote_volume REAL,
          trade_count INTEGER, taker_buy_base REAL, taker_buy_quote REAL
        );
        CREATE TABLE funding_history (
          symbol TEXT, funding_time_ms INTEGER, funding_rate REAL, mark_price REAL
        );
        CREATE TABLE open_interest_recent (
          symbol TEXT, timestamp_ms INTEGER, open_interest REAL,
          open_interest_value REAL
        );
        CREATE TABLE depth_snapshots (
          symbol TEXT, transaction_time_ms INTEGER, bid_notional_20 REAL,
          ask_notional_20 REAL, imbalance_20 REAL
        );
        """
    )
    initialize_market_event_schema(connection)
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "aggregate_trade_buckets",
        "liquidation_events",
        "spot_reference_snapshots",
        "basis_snapshots",
        "book_ticker_snapshots",
    } <= tables
    connection.close()
    report = assess_database_readiness(database)
    assert not report.M1_READY_FOR_EXPERIMENTS
    assert report.ready_families == ()
    assert all(not item.ready for item in report.family_readiness.values())


def _path(direction: float) -> tuple[CanonicalBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(4):
        open_price = 100.0 + direction * index
        rows.append(
            CanonicalBar(
                start + timedelta(minutes=5 * index),
                open_price,
                open_price + 1.0,
                open_price - 1.0,
                open_price + direction * 0.5,
                1000.0,
            )
        )
    return tuple(rows)


def test_economic_path_is_side_correct_costed_and_conservative_on_same_bar() -> None:
    long = replay_event_path(
        side=TradeSide.LONG,
        future=_path(1.0),
        horizon_bars=4,
        target_fraction=0.02,
        stop_fraction=0.02,
    )
    short = replay_event_path(
        side=TradeSide.SHORT,
        future=_path(-1.0),
        horizon_bars=4,
        target_fraction=0.02,
        stop_fraction=0.02,
    )
    assert long.gross_return_fraction > 0.0
    assert short.gross_return_fraction > 0.0
    assert long.net_return_fraction < long.gross_return_fraction
    assert short.net_return_fraction < short.gross_return_fraction
    same_bar = (
        CanonicalBar(datetime(2026, 1, 1, tzinfo=timezone.utc), 100, 103, 97, 100, 1),
    )
    ambiguous = replay_event_path(
        side=TradeSide.LONG,
        future=same_bar,
        horizon_bars=1,
        target_fraction=0.02,
        stop_fraction=0.02,
    )
    assert ambiguous.target_before_stop is False
    with pytest.raises(MarketEventContractError, match="INCOMPLETE"):
        replay_event_path(
            side=TradeSide.LONG,
            future=same_bar,
            horizon_bars=2,
            target_fraction=0.02,
            stop_fraction=0.02,
        )


def _trial(trial_id: str) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        created_at_utc="2026-08-12T00:00:00Z",
        preregistration_sha256="a" * 64,
        configuration_sha256="b" * 64,
        code_commit="c" * 40,
        dataset_sha256={"fixture": "d" * 64},
        status="FAILED",
        result_summary={"reason": "NO_EDGE"},
    )


def test_trial_ledger_is_hash_chained_append_only_and_detects_tampering(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = AppendOnlyTrialLedger(path)
    first_hash = ledger.append(_trial("M1-001"))
    second_hash = ledger.append(_trial("M1-002"))
    rows = ledger.validate()
    assert rows[1]["previous_record_hash"] == first_hash
    assert rows[1]["record_hash"] == second_hash
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(MarketEventContractError, match="DUPLICATE"):
        ledger.append(_trial("M1-001"))

    raw = [json.loads(line) for line in path.read_text().splitlines()]
    raw[0]["status"] = "PASSED"
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(MarketEventContractError, match="HASH_INVALID"):
        ledger.validate()
