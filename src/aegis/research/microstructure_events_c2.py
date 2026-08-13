"""Point-in-time public microstructure archive and event contracts for C2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import TradeSide
from ..utils import canonical_json
from .market_event_lab_m1 import EXPECTED_SYMBOLS, MarketEventContractError


SCHEMA_VERSION = "aegis-market-microstructure-c2-v1"
MANIFEST_VERSION = "aegis-market-microstructure-c2-manifest-v1"


class C2EventFamily(str, Enum):
    OI_CONFIRMED_BREAKOUT = "OI_CONFIRMED_BREAKOUT"
    LIQUIDATION_ABSORPTION_REVERSAL = "LIQUIDATION_ABSORPTION_REVERSAL"
    LIQUIDATION_CONTINUATION = "LIQUIDATION_CONTINUATION"
    DEPTH_ABSORPTION_REVERSAL = "DEPTH_ABSORPTION_REVERSAL"
    FLOW_IMPULSE_CONTINUATION = "FLOW_IMPULSE_CONTINUATION"
    BTC_ALT_LEAD_LAG = "BTC_ALT_LEAD_LAG"


EVENT_FEATURES: Mapping[C2EventFamily, tuple[str, ...]] = {
    C2EventFamily.OI_CONFIRMED_BREAKOUT: (
        "side_return_z", "side_flow_z", "oi_delta_z", "price_acceptance",
    ),
    C2EventFamily.LIQUIDATION_ABSORPTION_REVERSAL: (
        "opposing_liquidation_z", "side_flow_z", "price_response_abs",
        "side_reclaim", "side_depth_flip",
    ),
    C2EventFamily.LIQUIDATION_CONTINUATION: (
        "aligned_liquidation_z", "side_flow_z", "side_return_z",
        "oi_delta_z", "side_depth_imbalance",
    ),
    C2EventFamily.DEPTH_ABSORPTION_REVERSAL: (
        "opposing_flow_z", "price_response_abs", "side_depth_flip", "spread_bps",
    ),
    C2EventFamily.FLOW_IMPULSE_CONTINUATION: (
        "side_flow_z_30s", "side_flow_z_60s", "side_flow_z_300s",
        "side_return_z", "price_acceptance",
    ),
    C2EventFamily.BTC_ALT_LEAD_LAG: (
        "side_leader_return_z", "side_alt_residual_z", "side_flow_z",
        "leader_flow_z", "beta_btc",
    ),
}


@dataclass(frozen=True)
class AggregateTrade:
    symbol: str
    aggregate_trade_id: int
    event_time_ms: int
    trade_time_ms: int
    price: float
    quantity: float
    quote_notional: float
    buyer_is_maker: bool


@dataclass(frozen=True)
class LiquidationEvent:
    symbol: str
    order_trade_time_ms: int
    side: str
    price: float
    original_quantity: float
    event_time_ms: int
    order_status: str
    executed_quantity: float
    average_price: float
    quote_notional: float


@dataclass(frozen=True)
class DepthSnapshot:
    symbol: str
    transaction_time_ms: int
    last_update_id: int
    bid_notional_20: float
    ask_notional_20: float
    imbalance_20: float
    spread_bps: float


@dataclass(frozen=True)
class OpenInterestSnapshot:
    symbol: str
    timestamp_ms: int
    open_interest: float
    open_interest_value: float


@dataclass(frozen=True)
class C2EventVector:
    family: C2EventFamily
    symbol: str
    event_timestamp_ms: int
    side: TradeSide
    names: tuple[str, ...]
    values: tuple[float, ...]
    source_max_timestamps_ms: Mapping[str, int]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MarketEventContractError("AEGIS_C2_SCHEMA_MISMATCH")
        if self.symbol not in EXPECTED_SYMBOLS or self.event_timestamp_ms <= 0:
            raise MarketEventContractError("AEGIS_C2_IDENTITY_INVALID")
        if self.side not in {TradeSide.LONG, TradeSide.SHORT}:
            raise MarketEventContractError("AEGIS_C2_SIDE_INVALID")
        if self.names != EVENT_FEATURES[self.family]:
            raise MarketEventContractError("AEGIS_C2_FEATURE_CONTRACT_MISMATCH")
        if len(self.values) != len(self.names) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in self.values
        ):
            raise MarketEventContractError("AEGIS_C2_FEATURE_VALUE_INVALID")
        if not self.source_max_timestamps_ms or any(
            timestamp <= 0 or timestamp > self.event_timestamp_ms
            for timestamp in self.source_max_timestamps_ms.values()
        ):
            raise MarketEventContractError("AEGIS_C2_CAUSALITY_VIOLATION")


@dataclass(frozen=True)
class C2Thresholds:
    extreme_z: float
    confirmation_z: float
    nonresponse_maximum: float
    reclaim_minimum: float
    maximum_spread_bps: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0
            for value in asdict(self).values()
        ):
            raise MarketEventContractError("AEGIS_C2_THRESHOLD_INVALID")


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MarketEventContractError(f"AEGIS_C2_{field.upper()}_INVALID") from error
    if not math.isfinite(result) or positive and result <= 0:
        raise MarketEventContractError(f"AEGIS_C2_{field.upper()}_INVALID")
    return result


def _symbol(value: Any) -> str:
    symbol = str(value)
    if symbol not in EXPECTED_SYMBOLS:
        raise MarketEventContractError("AEGIS_C2_SYMBOL_INVALID")
    return symbol


def parse_aggregate_trade(payload: Mapping[str, Any]) -> AggregateTrade:
    if payload.get("e") != "aggTrade":
        raise MarketEventContractError("AEGIS_C2_AGG_TRADE_EVENT_INVALID")
    price = _finite(payload.get("p"), "price", positive=True)
    quantity = _finite(payload.get("q"), "quantity", positive=True)
    row = AggregateTrade(
        _symbol(payload.get("s")), int(payload["a"]), int(payload["E"]),
        int(payload["T"]), price, quantity, price * quantity,
        bool(payload.get("m")),
    )
    if min(row.aggregate_trade_id, row.event_time_ms, row.trade_time_ms) <= 0:
        raise MarketEventContractError("AEGIS_C2_AGG_TRADE_IDENTITY_INVALID")
    return row


def parse_liquidation(payload: Mapping[str, Any]) -> LiquidationEvent:
    if payload.get("e") != "forceOrder" or not isinstance(payload.get("o"), Mapping):
        raise MarketEventContractError("AEGIS_C2_LIQUIDATION_EVENT_INVALID")
    order = payload["o"]
    side = str(order.get("S"))
    if side not in {"BUY", "SELL"}:
        raise MarketEventContractError("AEGIS_C2_LIQUIDATION_SIDE_INVALID")
    price = _finite(order.get("p"), "liquidation_price", positive=True)
    original = _finite(order.get("q"), "liquidation_quantity", positive=True)
    executed = _finite(order.get("z", 0.0), "liquidation_executed")
    average = _finite(order.get("ap", price), "liquidation_average", positive=True)
    row = LiquidationEvent(
        _symbol(order.get("s")), int(order["T"]), side, price, original,
        int(payload["E"]), str(order.get("X", "UNKNOWN")), executed, average,
        average * executed,
    )
    if min(row.order_trade_time_ms, row.event_time_ms) <= 0:
        raise MarketEventContractError("AEGIS_C2_LIQUIDATION_TIMESTAMP_INVALID")
    return row


def parse_depth(payload: Mapping[str, Any]) -> DepthSnapshot:
    if payload.get("e") not in {None, "depthUpdate"}:
        raise MarketEventContractError("AEGIS_C2_DEPTH_EVENT_INVALID")
    bids, asks = payload.get("bids", payload.get("b")), payload.get("asks", payload.get("a"))
    if not isinstance(bids, Sequence) or not isinstance(asks, Sequence) or not bids or not asks:
        raise MarketEventContractError("AEGIS_C2_DEPTH_LEVELS_INVALID")
    bid_rows = [(_finite(item[0], "bid_price", positive=True), _finite(item[1], "bid_quantity")) for item in bids[:20]]
    ask_rows = [(_finite(item[0], "ask_price", positive=True), _finite(item[1], "ask_quantity")) for item in asks[:20]]
    bid_notional = sum(price * quantity for price, quantity in bid_rows)
    ask_notional = sum(price * quantity for price, quantity in ask_rows)
    total = bid_notional + ask_notional
    if total <= 0:
        raise MarketEventContractError("AEGIS_C2_DEPTH_NOTIONAL_INVALID")
    active_bids = [price for price, quantity in bid_rows if quantity > 0]
    active_asks = [price for price, quantity in ask_rows if quantity > 0]
    if not active_bids or not active_asks:
        raise MarketEventContractError("AEGIS_C2_DEPTH_ACTIVE_LEVELS_INVALID")
    best_bid, best_ask = max(active_bids), min(active_asks)
    midpoint = (best_bid + best_ask) / 2.0
    timestamp = int(payload.get("T") or payload.get("E") or payload.get("timestamp_ms") or 0)
    update_id = int(payload.get("lastUpdateId") or payload.get("u") or 0)
    row = DepthSnapshot(
        _symbol(payload.get("s")), timestamp, update_id, bid_notional, ask_notional,
        (bid_notional - ask_notional) / total, (best_ask - best_bid) / midpoint * 10_000.0,
    )
    if timestamp <= 0 or update_id <= 0 or best_ask < best_bid:
        raise MarketEventContractError("AEGIS_C2_DEPTH_IDENTITY_INVALID")
    return row


def parse_open_interest(payload: Mapping[str, Any], expected_symbol: str) -> OpenInterestSnapshot:
    symbol = _symbol(payload.get("symbol"))
    if symbol != expected_symbol:
        raise MarketEventContractError("AEGIS_C2_OPEN_INTEREST_SYMBOL_MISMATCH")
    row = OpenInterestSnapshot(
        symbol, int(payload["timestamp"]),
        _finite(payload.get("sumOpenInterest"), "open_interest"),
        _finite(payload.get("sumOpenInterestValue"), "open_interest_value"),
    )
    if row.timestamp_ms <= 0:
        raise MarketEventContractError("AEGIS_C2_OPEN_INTEREST_TIMESTAMP_INVALID")
    return row


def initialize_archive(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS c2_aggregate_trades (
          symbol TEXT NOT NULL, aggregate_trade_id INTEGER NOT NULL,
          event_time_ms INTEGER NOT NULL, trade_time_ms INTEGER NOT NULL,
          price REAL NOT NULL, quantity REAL NOT NULL, quote_notional REAL NOT NULL,
          buyer_is_maker INTEGER NOT NULL CHECK(buyer_is_maker IN (0,1)),
          PRIMARY KEY(symbol, aggregate_trade_id)
        );
        CREATE TABLE IF NOT EXISTS c2_liquidation_events (
          symbol TEXT NOT NULL, order_trade_time_ms INTEGER NOT NULL, side TEXT NOT NULL,
          price REAL NOT NULL, original_quantity REAL NOT NULL, event_time_ms INTEGER NOT NULL,
          order_status TEXT NOT NULL, executed_quantity REAL NOT NULL,
          average_price REAL NOT NULL, quote_notional REAL NOT NULL,
          PRIMARY KEY(symbol, order_trade_time_ms, side, price, original_quantity)
        );
        CREATE TABLE IF NOT EXISTS c2_depth_snapshots (
          symbol TEXT NOT NULL, transaction_time_ms INTEGER NOT NULL,
          last_update_id INTEGER NOT NULL, bid_notional_20 REAL NOT NULL,
          ask_notional_20 REAL NOT NULL, imbalance_20 REAL NOT NULL,
          spread_bps REAL NOT NULL, PRIMARY KEY(symbol, transaction_time_ms)
        );
        CREATE TABLE IF NOT EXISTS c2_open_interest (
          symbol TEXT NOT NULL, timestamp_ms INTEGER NOT NULL,
          open_interest REAL NOT NULL, open_interest_value REAL NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS c2_collection_manifest (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, created_at_utc TEXT NOT NULL,
          source TEXT NOT NULL, first_timestamp_ms INTEGER, last_timestamp_ms INTEGER,
          accepted_rows INTEGER NOT NULL, duplicate_rows INTEGER NOT NULL,
          rejected_rows INTEGER NOT NULL, previous_hash TEXT, record_hash TEXT UNIQUE NOT NULL
        );
        """
    )


class C2Archive:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        initialize_archive(self.connection)
        self.connection.commit()
        os.chmod(path, 0o600)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def insert(self, row: AggregateTrade | LiquidationEvent | DepthSnapshot | OpenInterestSnapshot) -> bool:
        before = self.connection.total_changes
        if isinstance(row, AggregateTrade):
            self.connection.execute(
                "INSERT OR IGNORE INTO c2_aggregate_trades VALUES(?,?,?,?,?,?,?,?)",
                (*asdict(row).values(),),
            )
        elif isinstance(row, LiquidationEvent):
            self.connection.execute(
                "INSERT OR IGNORE INTO c2_liquidation_events VALUES(?,?,?,?,?,?,?,?,?,?)",
                (*asdict(row).values(),),
            )
        elif isinstance(row, DepthSnapshot):
            self.connection.execute(
                "INSERT OR IGNORE INTO c2_depth_snapshots VALUES(?,?,?,?,?,?,?)",
                (*asdict(row).values(),),
            )
        elif isinstance(row, OpenInterestSnapshot):
            self.connection.execute(
                "INSERT OR IGNORE INTO c2_open_interest VALUES(?,?,?,?)",
                (*asdict(row).values(),),
            )
        else:
            raise MarketEventContractError("AEGIS_C2_ROW_TYPE_INVALID")
        return self.connection.total_changes > before

    def append_manifest(self, payload: Mapping[str, Any]) -> str:
        names = ("created_at_utc", "source", "first_timestamp_ms", "last_timestamp_ms", "accepted_rows", "duplicate_rows", "rejected_rows")
        if set(payload) != set(names):
            raise MarketEventContractError("AEGIS_C2_MANIFEST_FIELDS_INVALID")
        ordered = {name: payload[name] for name in names}
        previous_row = self.connection.execute(
            "SELECT record_hash FROM c2_collection_manifest ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = previous_row[0] if previous_row else None
        record = {"schema_version": MANIFEST_VERSION, **ordered, "previous_hash": previous}
        digest = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT INTO c2_collection_manifest(created_at_utc,source,first_timestamp_ms,last_timestamp_ms,accepted_rows,duplicate_rows,rejected_rows,previous_hash,record_hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (*ordered.values(), previous, digest),
        )
        self.connection.commit()
        return digest

    def validate_manifest_chain(self) -> tuple[str, ...]:
        previous = None
        hashes = []
        rows = self.connection.execute(
            "SELECT created_at_utc,source,first_timestamp_ms,last_timestamp_ms,accepted_rows,duplicate_rows,rejected_rows,previous_hash,record_hash FROM c2_collection_manifest ORDER BY sequence"
        )
        names = ("created_at_utc", "source", "first_timestamp_ms", "last_timestamp_ms", "accepted_rows", "duplicate_rows", "rejected_rows")
        for values in rows:
            payload = dict(zip(names, values[:7]))
            claimed_previous, claimed = values[7], values[8]
            record = {"schema_version": MANIFEST_VERSION, **payload, "previous_hash": previous}
            calculated = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
            if claimed_previous != previous or claimed != calculated:
                raise MarketEventContractError("AEGIS_C2_MANIFEST_CHAIN_INVALID")
            hashes.append(claimed)
            previous = claimed
        return tuple(hashes)


def build_event_vector(
    *, family: C2EventFamily, symbol: str, event_timestamp_ms: int,
    side: TradeSide, features: Mapping[str, float],
    source_max_timestamps_ms: Mapping[str, int],
) -> C2EventVector:
    names = EVENT_FEATURES[family]
    if tuple(features) != names:
        raise MarketEventContractError("AEGIS_C2_FEATURE_CONTRACT_MISMATCH")
    vector = C2EventVector(
        family, symbol, event_timestamp_ms, side, names,
        tuple(float(features[name]) for name in names), dict(source_max_timestamps_ms),
    )
    vector.validate()
    return vector


def detect_event(vector: C2EventVector, thresholds: C2Thresholds) -> TradeSide:
    vector.validate()
    value = dict(zip(vector.names, vector.values))
    passed = False
    if vector.family is C2EventFamily.OI_CONFIRMED_BREAKOUT:
        passed = value["side_return_z"] >= thresholds.extreme_z and value["side_flow_z"] >= thresholds.confirmation_z and value["oi_delta_z"] >= thresholds.confirmation_z and value["price_acceptance"] >= thresholds.reclaim_minimum
    elif vector.family is C2EventFamily.LIQUIDATION_ABSORPTION_REVERSAL:
        passed = value["opposing_liquidation_z"] >= thresholds.extreme_z and value["side_flow_z"] >= thresholds.confirmation_z and value["price_response_abs"] <= thresholds.nonresponse_maximum and value["side_reclaim"] >= thresholds.reclaim_minimum and value["side_depth_flip"] >= thresholds.reclaim_minimum
    elif vector.family is C2EventFamily.LIQUIDATION_CONTINUATION:
        passed = value["aligned_liquidation_z"] >= thresholds.extreme_z and value["side_flow_z"] >= thresholds.confirmation_z and value["side_return_z"] >= thresholds.confirmation_z and value["oi_delta_z"] >= thresholds.confirmation_z and value["side_depth_imbalance"] >= thresholds.reclaim_minimum
    elif vector.family is C2EventFamily.DEPTH_ABSORPTION_REVERSAL:
        passed = value["opposing_flow_z"] >= thresholds.extreme_z and value["price_response_abs"] <= thresholds.nonresponse_maximum and value["side_depth_flip"] >= thresholds.reclaim_minimum and value["spread_bps"] <= thresholds.maximum_spread_bps
    elif vector.family is C2EventFamily.FLOW_IMPULSE_CONTINUATION:
        passed = min(value["side_flow_z_30s"], value["side_flow_z_60s"], value["side_flow_z_300s"]) >= thresholds.confirmation_z and value["side_return_z"] >= thresholds.extreme_z and value["price_acceptance"] >= thresholds.reclaim_minimum
    elif vector.family is C2EventFamily.BTC_ALT_LEAD_LAG:
        passed = value["side_leader_return_z"] >= thresholds.extreme_z and value["side_alt_residual_z"] <= thresholds.nonresponse_maximum and value["side_flow_z"] >= thresholds.confirmation_z and value["leader_flow_z"] >= thresholds.confirmation_z and math.isfinite(value["beta_btc"])
    return vector.side if passed else TradeSide.NO_TRADE


def archive_coverage(path: Path) -> Mapping[str, Mapping[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    tables = {
        "aggregate_trades": ("c2_aggregate_trades", "trade_time_ms"),
        "liquidation_events": ("c2_liquidation_events", "order_trade_time_ms"),
        "depth_snapshots": ("c2_depth_snapshots", "transaction_time_ms"),
        "open_interest": ("c2_open_interest", "timestamp_ms"),
    }
    result = {}
    try:
        present = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for source, (table, timestamp) in tables.items():
            if table not in present:
                result[source] = {"rows": 0, "symbols": 0, "span_days": 0.0}
                continue
            count, symbols, minimum, maximum = connection.execute(
                f"SELECT COUNT(*),COUNT(DISTINCT symbol),MIN({timestamp}),MAX({timestamp}) FROM {table}"
            ).fetchone()
            span = (maximum - minimum) / 86_400_000 if minimum is not None and maximum is not None else 0.0
            result[source] = {"rows": int(count), "symbols": int(symbols), "span_days": float(span)}
    finally:
        connection.close()
    return result
