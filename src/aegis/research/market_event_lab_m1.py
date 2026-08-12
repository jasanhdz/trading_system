"""Causal, research-only contracts for the Market Event Laboratory M1."""

from __future__ import annotations

import fcntl
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data import CanonicalBar
from ..domain import TradeSide
from ..utils import Sha256HashProvider, canonical_json


SCHEMA_VERSION = "aegis-market-event-contract-v1"
LEDGER_SCHEMA_VERSION = "aegis-research-trial-ledger-v1"
EXPECTED_SYMBOLS = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "XRPUSDT",
)


class MarketEventContractError(ValueError):
    """Raised when an event observation violates the frozen causal contract."""


class EventFamily(str, Enum):
    LIQUIDATION_CASCADE_REVERSAL = "LIQUIDATION_CASCADE_REVERSAL"
    OI_CONFIRMED_BREAKOUT = "OI_CONFIRMED_BREAKOUT"
    SPOT_FUTURES_DISLOCATION = "SPOT_FUTURES_DISLOCATION"
    DEPTH_ABSORPTION_REVERSAL = "DEPTH_ABSORPTION_REVERSAL"


FAMILY_REQUIREMENTS: Mapping[EventFamily, tuple[str, ...]] = {
    EventFamily.LIQUIDATION_CASCADE_REVERSAL: (
        "futures_klines",
        "aggregate_trade_buckets",
        "liquidation_events",
        "depth_snapshots",
    ),
    EventFamily.OI_CONFIRMED_BREAKOUT: (
        "futures_klines",
        "aggregate_trade_buckets",
        "open_interest",
    ),
    EventFamily.SPOT_FUTURES_DISLOCATION: (
        "futures_klines",
        "funding",
        "mark_price",
        "spot_reference",
        "basis",
    ),
    EventFamily.DEPTH_ABSORPTION_REVERSAL: (
        "futures_klines",
        "aggregate_trade_buckets",
        "depth_snapshots",
    ),
}


SOURCE_TABLES: Mapping[str, tuple[str, str, str | None]] = {
    "futures_klines": ("kline_microstructure", "open_time_ms", None),
    "aggregate_trade_buckets": ("aggregate_trade_buckets", "bucket_open_ms", None),
    "open_interest": ("open_interest_recent", "timestamp_ms", None),
    "funding": ("funding_history", "funding_time_ms", None),
    "mark_price": ("funding_history", "funding_time_ms", "mark_price IS NOT NULL"),
    "spot_reference": ("spot_reference_snapshots", "timestamp_ms", None),
    "basis": ("basis_snapshots", "timestamp_ms", None),
    "depth_snapshots": ("depth_snapshots", "transaction_time_ms", None),
    "liquidation_events": ("liquidation_events", "event_time_ms", None),
}


EVENT_FEATURE_NAMES: Mapping[EventFamily, tuple[str, ...]] = {
    EventFamily.LIQUIDATION_CASCADE_REVERSAL: (
        "liquidation_intensity",
        "aggressive_flow_z",
        "price_response_abs",
        "reclaim_fraction",
        "depth_imbalance_change",
    ),
    EventFamily.OI_CONFIRMED_BREAKOUT: (
        "return_z",
        "quote_volume_z",
        "open_interest_delta_z",
        "taker_imbalance",
    ),
    EventFamily.SPOT_FUTURES_DISLOCATION: (
        "basis_z",
        "funding_z",
        "basis_convergence",
        "mark_return",
        "spot_return",
    ),
    EventFamily.DEPTH_ABSORPTION_REVERSAL: (
        "aggressive_flow_z",
        "price_response_abs",
        "depth_imbalance_before",
        "depth_imbalance_after",
    ),
}


@dataclass(frozen=True)
class EventFeatureVector:
    family: EventFamily
    symbol: str
    event_timestamp_ms: int
    names: tuple[str, ...]
    values: tuple[float, ...]
    source_max_timestamps_ms: Mapping[str, int]
    schema_version: str = SCHEMA_VERSION
    dtype: str = "float64"
    feature_hash: str = ""

    def unsigned_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "symbol": self.symbol,
            "event_timestamp_ms": self.event_timestamp_ms,
            "names": self.names,
            "values": self.values,
            "source_max_timestamps_ms": self.source_max_timestamps_ms,
            "dtype": self.dtype,
        }

    def calculated_hash(self) -> str:
        return Sha256HashProvider().digest_value(self.unsigned_payload())

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MarketEventContractError("AEGIS_M1_SCHEMA_MISMATCH")
        if self.symbol not in EXPECTED_SYMBOLS:
            raise MarketEventContractError("AEGIS_M1_SYMBOL_INVALID")
        if self.event_timestamp_ms <= 0:
            raise MarketEventContractError("AEGIS_M1_EVENT_TIMESTAMP_INVALID")
        if self.dtype != "float64":
            raise MarketEventContractError("AEGIS_M1_DTYPE_INVALID")
        expected = EVENT_FEATURE_NAMES[self.family]
        if self.names != expected:
            missing = sorted(set(expected) - set(self.names))
            extra = sorted(set(self.names) - set(expected))
            if missing:
                raise MarketEventContractError("AEGIS_M1_FEATURE_MISSING")
            if extra:
                raise MarketEventContractError("AEGIS_M1_FEATURE_EXTRA")
            raise MarketEventContractError("AEGIS_M1_FEATURE_ORDER_INVALID")
        if len(self.values) != len(self.names) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.values
        ):
            raise MarketEventContractError("AEGIS_M1_FEATURE_DTYPE_INVALID")
        if not all(math.isfinite(float(value)) for value in self.values):
            raise MarketEventContractError("AEGIS_M1_FEATURE_NON_FINITE")
        required_sources = set(FAMILY_REQUIREMENTS[self.family])
        if set(self.source_max_timestamps_ms) != required_sources:
            raise MarketEventContractError("AEGIS_M1_SOURCE_SET_INVALID")
        if any(
            timestamp > self.event_timestamp_ms or timestamp <= 0
            for timestamp in self.source_max_timestamps_ms.values()
        ):
            raise MarketEventContractError("AEGIS_M1_CAUSALITY_VIOLATION")
        if self.feature_hash != self.calculated_hash():
            raise MarketEventContractError("AEGIS_M1_FEATURE_HASH_INVALID")


def build_event_vector(
    *,
    family: EventFamily,
    symbol: str,
    event_timestamp_ms: int,
    features: Mapping[str, float],
    source_max_timestamps_ms: Mapping[str, int],
) -> EventFeatureVector:
    expected = EVENT_FEATURE_NAMES[family]
    if tuple(features) != expected:
        missing = sorted(set(expected) - set(features))
        extra = sorted(set(features) - set(expected))
        if missing:
            raise MarketEventContractError("AEGIS_M1_FEATURE_MISSING")
        if extra:
            raise MarketEventContractError("AEGIS_M1_FEATURE_EXTRA")
        raise MarketEventContractError("AEGIS_M1_FEATURE_ORDER_INVALID")
    vector = EventFeatureVector(
        family=family,
        symbol=symbol,
        event_timestamp_ms=event_timestamp_ms,
        names=expected,
        values=tuple(features[name] for name in expected),
        source_max_timestamps_ms=dict(source_max_timestamps_ms),
    )
    vector = replace(vector, feature_hash=vector.calculated_hash())
    vector.validate()
    return vector


@dataclass(frozen=True)
class EventThresholds:
    """Thresholds fitted on TRAIN only; M1 intentionally supplies no defaults."""

    absolute_extreme: float
    confirmation: float
    nonresponse_maximum: float
    reversal_minimum: float

    def __post_init__(self) -> None:
        values = (
            self.absolute_extreme,
            self.confirmation,
            self.nonresponse_maximum,
            self.reversal_minimum,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise MarketEventContractError("AEGIS_M1_THRESHOLD_INVALID")


def detect_event(
    vector: EventFeatureVector, thresholds: EventThresholds
) -> TradeSide:
    """Apply a frozen family rule without selecting or fitting thresholds."""

    vector.validate()
    values = dict(zip(vector.names, vector.values))
    if vector.family is EventFamily.OI_CONFIRMED_BREAKOUT:
        direction = math.copysign(1.0, values["return_z"])
        aligned = (
            values["quote_volume_z"] >= thresholds.confirmation
            and values["open_interest_delta_z"] >= thresholds.confirmation
            and values["taker_imbalance"] * direction >= thresholds.reversal_minimum
        )
        if abs(values["return_z"]) >= thresholds.absolute_extreme and aligned:
            return TradeSide.LONG if direction > 0 else TradeSide.SHORT
    elif vector.family is EventFamily.LIQUIDATION_CASCADE_REVERSAL:
        if (
            values["liquidation_intensity"] >= thresholds.absolute_extreme
            and values["aggressive_flow_z"] >= thresholds.confirmation
            and values["price_response_abs"] <= thresholds.nonresponse_maximum
            and abs(values["reclaim_fraction"]) >= thresholds.reversal_minimum
            and abs(values["depth_imbalance_change"]) >= thresholds.reversal_minimum
        ):
            return TradeSide.LONG if values["reclaim_fraction"] > 0 else TradeSide.SHORT
    elif vector.family is EventFamily.SPOT_FUTURES_DISLOCATION:
        extreme = max(abs(values["basis_z"]), abs(values["funding_z"]))
        if (
            extreme >= thresholds.absolute_extreme
            and abs(values["basis_convergence"]) >= thresholds.confirmation
        ):
            return TradeSide.LONG if values["basis_convergence"] > 0 else TradeSide.SHORT
    elif vector.family is EventFamily.DEPTH_ABSORPTION_REVERSAL:
        reversal = values["depth_imbalance_after"] - values["depth_imbalance_before"]
        if (
            abs(values["aggressive_flow_z"]) >= thresholds.absolute_extreme
            and values["price_response_abs"] <= thresholds.nonresponse_maximum
            and abs(reversal) >= thresholds.reversal_minimum
        ):
            return TradeSide.LONG if reversal > 0 else TradeSide.SHORT
    return TradeSide.NO_TRADE


@dataclass(frozen=True)
class SourceCoverage:
    source: str
    table_present: bool
    row_count: int
    symbols: int
    minimum_timestamp_ms: int | None
    maximum_timestamp_ms: int | None
    span_days: float


@dataclass(frozen=True)
class FamilyReadiness:
    family: EventFamily
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ReadinessReport:
    schema_version: str
    database: str
    source_coverage: Mapping[str, SourceCoverage]
    family_readiness: Mapping[str, FamilyReadiness]
    ready_families: tuple[str, ...]
    M1_READY_FOR_EXPERIMENTS: bool
    report_hash: str


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def assess_database_readiness(
    database: Path,
    *,
    minimum_days: float = 60.0,
    minimum_symbols: int = len(EXPECTED_SYMBOLS),
) -> ReadinessReport:
    """Inspect an existing evidence database through SQLite read-only mode."""

    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    coverage: dict[str, SourceCoverage] = {}
    try:
        tables = _table_names(connection)
        for source, (table, timestamp_column, condition) in SOURCE_TABLES.items():
            if table not in tables:
                coverage[source] = SourceCoverage(source, False, 0, 0, None, None, 0.0)
                continue
            where = f" WHERE {condition}" if condition else ""
            row = connection.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT symbol), MIN({timestamp_column}), "
                f"MAX({timestamp_column}) FROM {table}{where}"
            ).fetchone()
            minimum = int(row[2]) if row[2] is not None else None
            maximum = int(row[3]) if row[3] is not None else None
            span = (maximum - minimum) / 86_400_000 if minimum is not None and maximum is not None else 0.0
            coverage[source] = SourceCoverage(
                source, True, int(row[0]), int(row[1]), minimum, maximum, span
            )
    finally:
        connection.close()

    family_readiness: dict[str, FamilyReadiness] = {}
    for family, required in FAMILY_REQUIREMENTS.items():
        blockers = []
        for source in required:
            item = coverage[source]
            if not item.table_present or item.row_count == 0:
                blockers.append(f"{source}:SOURCE_ABSENT")
            elif item.symbols < minimum_symbols:
                blockers.append(f"{source}:SYMBOL_COVERAGE_{item.symbols}_OF_{minimum_symbols}")
            elif item.span_days < minimum_days:
                blockers.append(f"{source}:SPAN_{item.span_days:.3f}_DAYS_LT_{minimum_days:g}")
        family_readiness[family.value] = FamilyReadiness(family, not blockers, tuple(blockers))
    ready = tuple(name for name, item in family_readiness.items() if item.ready)
    payload = {
        "schema_version": "aegis-market-event-readiness-v1",
        "database": str(database.resolve()),
        "source_coverage": coverage,
        "family_readiness": family_readiness,
        "ready_families": ready,
        "M1_READY_FOR_EXPERIMENTS": len(ready) == len(EventFamily),
    }
    return ReadinessReport(
        **payload, report_hash=Sha256HashProvider().digest_value(payload)
    )


def initialize_market_event_schema(connection: sqlite3.Connection) -> None:
    """Create prospective source tables without starting any collector."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS aggregate_trade_buckets (
          symbol TEXT NOT NULL,
          bucket_open_ms INTEGER NOT NULL,
          buy_quantity REAL NOT NULL,
          sell_quantity REAL NOT NULL,
          quote_notional REAL NOT NULL,
          trade_count INTEGER NOT NULL,
          PRIMARY KEY(symbol, bucket_open_ms)
        );
        CREATE TABLE IF NOT EXISTS liquidation_events (
          event_id TEXT PRIMARY KEY,
          symbol TEXT NOT NULL,
          event_time_ms INTEGER NOT NULL,
          liquidation_side TEXT NOT NULL,
          price REAL NOT NULL,
          quantity REAL NOT NULL,
          quote_notional REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS spot_reference_snapshots (
          symbol TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          price REAL NOT NULL,
          source TEXT NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS basis_snapshots (
          symbol TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          mark_price REAL NOT NULL,
          spot_price REAL NOT NULL,
          basis_fraction REAL NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS book_ticker_snapshots (
          symbol TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          bid_price REAL NOT NULL,
          bid_quantity REAL NOT NULL,
          ask_price REAL NOT NULL,
          ask_quantity REAL NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        """
    )


@dataclass(frozen=True)
class EventPathOutcome:
    side: TradeSide
    horizon_bars: int
    entry_price: float
    exit_price: float
    gross_return_fraction: float
    cost_fraction: float
    net_return_fraction: float
    mae_fraction: float
    mfe_fraction: float
    target_before_stop: bool | None
    time_to_target_bars: int | None
    time_to_stop_bars: int | None


@dataclass(frozen=True)
class IndependentEvent:
    event_id: str
    family: EventFamily
    symbol: str
    side: TradeSide
    timestamp_ms: int
    score: float


def collapse_correlated_events(
    events: Sequence[IndependentEvent], *, window_minutes: int = 15
) -> tuple[IndependentEvent, ...]:
    """Keep the first causal observation in each market-wide family/side cluster."""

    if window_minutes < 0:
        raise MarketEventContractError("AEGIS_M1_CLUSTER_WINDOW_INVALID")
    accepted: list[IndependentEvent] = []
    last_by_group: dict[tuple[EventFamily, TradeSide], int] = {}
    window_ms = window_minutes * 60_000
    for event in sorted(events, key=lambda item: (item.timestamp_ms, item.event_id)):
        if (
            event.side not in {TradeSide.LONG, TradeSide.SHORT}
            or event.symbol not in EXPECTED_SYMBOLS
            or event.timestamp_ms <= 0
            or not math.isfinite(event.score)
        ):
            raise MarketEventContractError("AEGIS_M1_CLUSTER_EVENT_INVALID")
        group = (event.family, event.side)
        previous = last_by_group.get(group)
        if previous is not None and event.timestamp_ms - previous <= window_ms:
            continue
        accepted.append(event)
        last_by_group[group] = event.timestamp_ms
    return tuple(accepted)


@dataclass(frozen=True)
class PortfolioEvent:
    event_id: str
    timestamp_ms: int
    symbol: str
    side: TradeSide
    holding_bars: int
    net_return_fraction: float


@dataclass(frozen=True)
class PortfolioReplayResult:
    starting_equity: float
    ending_equity: float
    net_pnl: float
    trades_executed: int
    skipped_same_symbol: int
    skipped_capacity: int
    maximum_concurrent_positions: int
    maximum_drawdown_fraction: float


def replay_portfolio(
    events: Sequence[PortfolioEvent],
    *,
    starting_equity: float = 1.0,
    position_fraction: float = 0.1,
    maximum_concurrent_positions: int | None = None,
) -> PortfolioReplayResult:
    """Replay event economics with causal capital and overlap accounting."""

    if (
        not math.isfinite(starting_equity)
        or starting_equity <= 0.0
        or not math.isfinite(position_fraction)
        or not 0.0 <= position_fraction <= 1.0
        or maximum_concurrent_positions is not None
        and maximum_concurrent_positions <= 0
    ):
        raise MarketEventContractError("AEGIS_M1_PORTFOLIO_CONFIG_INVALID")
    equity = starting_equity
    peak = starting_equity
    maximum_drawdown = 0.0
    active: list[tuple[int, str]] = []
    executed = 0
    skipped_symbol = 0
    skipped_capacity = 0
    maximum_concurrent = 0
    seen_ids: set[str] = set()
    for event in sorted(events, key=lambda item: (item.timestamp_ms, item.event_id)):
        if (
            event.event_id in seen_ids
            or event.timestamp_ms <= 0
            or event.symbol not in EXPECTED_SYMBOLS
            or event.side not in {TradeSide.LONG, TradeSide.SHORT}
            or event.holding_bars <= 0
            or not math.isfinite(event.net_return_fraction)
        ):
            raise MarketEventContractError("AEGIS_M1_PORTFOLIO_EVENT_INVALID")
        seen_ids.add(event.event_id)
        active = [item for item in active if item[0] > event.timestamp_ms]
        if event.symbol in {symbol for _, symbol in active}:
            skipped_symbol += 1
            continue
        if maximum_concurrent_positions is not None and len(active) >= maximum_concurrent_positions:
            skipped_capacity += 1
            continue
        allocated = equity * position_fraction
        equity += allocated * event.net_return_fraction
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
        end_timestamp = event.timestamp_ms + event.holding_bars * 5 * 60_000
        active.append((end_timestamp, event.symbol))
        executed += 1
        maximum_concurrent = max(maximum_concurrent, len(active))
    return PortfolioReplayResult(
        starting_equity=starting_equity,
        ending_equity=equity,
        net_pnl=equity - starting_equity,
        trades_executed=executed,
        skipped_same_symbol=skipped_symbol,
        skipped_capacity=skipped_capacity,
        maximum_concurrent_positions=maximum_concurrent,
        maximum_drawdown_fraction=maximum_drawdown,
    )


def replay_event_path(
    *,
    side: TradeSide,
    future: Sequence[CanonicalBar],
    horizon_bars: int,
    target_fraction: float,
    stop_fraction: float,
    fee_bps_per_side: float = 5.0,
    slippage_bps_per_side: float = 2.0,
    funding_bps_per_hour: float = 1.0,
) -> EventPathOutcome:
    """Price a causal next-bar-open path with conservative same-bar ordering."""

    parameters = (
        target_fraction,
        stop_fraction,
        fee_bps_per_side,
        slippage_bps_per_side,
        funding_bps_per_hour,
    )
    if side not in {TradeSide.LONG, TradeSide.SHORT}:
        raise MarketEventContractError("AEGIS_M1_PATH_SIDE_INVALID")
    if horizon_bars <= 0 or len(future) < horizon_bars:
        raise MarketEventContractError("AEGIS_M1_PATH_INCOMPLETE")
    if not all(math.isfinite(value) and value >= 0.0 for value in parameters):
        raise MarketEventContractError("AEGIS_M1_PATH_PARAMETER_INVALID")
    path = tuple(future[:horizon_bars])
    entry = path[0].open
    if not math.isfinite(entry) or entry <= 0.0:
        raise MarketEventContractError("AEGIS_M1_ENTRY_INVALID")
    sign = 1.0 if side is TradeSide.LONG else -1.0
    favorable = [max(0.0, sign * (bar.high - entry) / entry) if sign > 0 else max(0.0, (entry - bar.low) / entry) for bar in path]
    adverse = [max(0.0, (entry - bar.low) / entry) if sign > 0 else max(0.0, (bar.high - entry) / entry) for bar in path]
    target_bar = next((index for index, value in enumerate(favorable, 1) if value >= target_fraction), None)
    stop_bar = next((index for index, value in enumerate(adverse, 1) if value >= stop_fraction), None)
    if target_bar is None and stop_bar is None:
        target_before_stop = None
    elif target_bar is None:
        target_before_stop = False
    elif stop_bar is None:
        target_before_stop = True
    else:
        target_before_stop = target_bar < stop_bar
    exit_price = path[-1].close
    gross = sign * (exit_price - entry) / entry
    costs = (
        2.0 * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
        + funding_bps_per_hour / 10_000.0 * horizon_bars * 5.0 / 60.0
    )
    return EventPathOutcome(
        side=side,
        horizon_bars=horizon_bars,
        entry_price=entry,
        exit_price=exit_price,
        gross_return_fraction=gross,
        cost_fraction=costs,
        net_return_fraction=gross - costs,
        mae_fraction=max(adverse),
        mfe_fraction=max(favorable),
        target_before_stop=target_before_stop,
        time_to_target_bars=target_bar,
        time_to_stop_bars=stop_bar,
    )


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    created_at_utc: str
    preregistration_sha256: str
    configuration_sha256: str
    code_commit: str
    dataset_sha256: Mapping[str, str]
    status: str
    result_summary: Mapping[str, Any]


class AppendOnlyTrialLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _record_hash(payload: Mapping[str, Any]) -> str:
        return Sha256HashProvider().digest_value(payload)

    def _read_and_validate(self) -> list[Mapping[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        previous = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise MarketEventContractError("AEGIS_M1_LEDGER_BLANK_LINE")
                row = json.loads(line)
                claimed = row.pop("record_hash", None)
                if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
                    raise MarketEventContractError("AEGIS_M1_LEDGER_SCHEMA_INVALID")
                if row.get("previous_record_hash") != previous:
                    raise MarketEventContractError("AEGIS_M1_LEDGER_CHAIN_INVALID")
                calculated = self._record_hash(row)
                if claimed != calculated:
                    raise MarketEventContractError(
                        f"AEGIS_M1_LEDGER_HASH_INVALID_LINE_{line_number}"
                    )
                rows.append({**row, "record_hash": claimed})
                previous = claimed
        if len({row["trial_id"] for row in rows}) != len(rows):
            raise MarketEventContractError("AEGIS_M1_LEDGER_DUPLICATE_TRIAL")
        return rows

    def append(self, record: TrialRecord) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                rows = self._read_and_validate()
                if record.trial_id in {row["trial_id"] for row in rows}:
                    raise MarketEventContractError("AEGIS_M1_LEDGER_DUPLICATE_TRIAL")
                previous = rows[-1]["record_hash"] if rows else None
                payload = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    **asdict(record),
                    "previous_record_hash": previous,
                }
                record_hash = self._record_hash(payload)
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json({**payload, "record_hash": record_hash}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.chmod(self.path, 0o600)
                return record_hash
        finally:
            if not os.path.exists(self.path):
                os.close(descriptor)

    def validate(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._read_and_validate())


def utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
