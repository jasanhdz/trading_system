"""Fail-closed order-book reconstruction and data gates for W9 research."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class W9CoverageRequirements:
    minimum_train_episodes: int = 1_000
    minimum_validation_episodes: int = 500
    minimum_symbols_per_partition: int = 8
    minimum_months_per_partition: int = 3


@dataclass
class L2Book:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    ready: bool = False

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.ready = False

    def apply_message(self, rows: Sequence[Mapping[str, str]]) -> None:
        if not rows:
            return
        snapshot = _boolean(rows[0]["is_snapshot"])
        if any(_boolean(row["is_snapshot"]) != snapshot for row in rows):
            raise ValueError("AEGIS_W9_MIXED_SNAPSHOT_MESSAGE")
        if snapshot:
            self.reset()
        elif not self.ready:
            raise ValueError("AEGIS_W9_UPDATE_BEFORE_SNAPSHOT")

        for row in rows:
            side = row["side"]
            if side not in {"bid", "ask"}:
                raise ValueError("AEGIS_W9_SIDE_INVALID")
            price = float(row["price"])
            amount = float(row["amount"])
            if not math.isfinite(price) or price <= 0.0:
                raise ValueError("AEGIS_W9_PRICE_INVALID")
            if not math.isfinite(amount) or amount < 0.0:
                raise ValueError("AEGIS_W9_AMOUNT_INVALID")
            levels = self.bids if side == "bid" else self.asks
            if amount == 0.0:
                levels.pop(price, None)
            else:
                levels[price] = amount

        if snapshot:
            self.ready = True
        self.assert_valid()

    def assert_valid(self) -> None:
        if not self.ready:
            return
        if not self.bids or not self.asks:
            raise ValueError("AEGIS_W9_BOOK_SIDE_EMPTY")
        if max(self.bids) >= min(self.asks):
            raise ValueError("AEGIS_W9_BOOK_CROSSED")

    def top(self) -> tuple[float, float, float, float]:
        self.assert_valid()
        bid = max(self.bids)
        ask = min(self.asks)
        return bid, self.bids[bid], ask, self.asks[ask]


def _boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("AEGIS_W9_BOOLEAN_INVALID")


def stable_opportunity_episode_id(symbol: str, timestamp_us: int) -> str:
    if not symbol or timestamp_us <= 0:
        raise ValueError("AEGIS_W9_EPISODE_ID_INVALID")
    material = f"W9|{symbol.upper()}|{timestamp_us}".encode()
    return "W9-" + hashlib.sha256(material).hexdigest()


def order_book_imbalance(bid_amount: float, ask_amount: float) -> float:
    total = bid_amount + ask_amount
    if not all(math.isfinite(value) and value >= 0.0 for value in (bid_amount, ask_amount)):
        raise ValueError("AEGIS_W9_DEPTH_INVALID")
    if total <= 0.0:
        raise ValueError("AEGIS_W9_DEPTH_EMPTY")
    return (bid_amount - ask_amount) / total


def microprice(best_bid: float, bid_amount: float, best_ask: float, ask_amount: float) -> float:
    if best_bid <= 0.0 or best_ask <= best_bid:
        raise ValueError("AEGIS_W9_BBO_INVALID")
    total = bid_amount + ask_amount
    if total <= 0.0:
        raise ValueError("AEGIS_W9_DEPTH_EMPTY")
    return (best_ask * bid_amount + best_bid * ask_amount) / total


def assess_coverage(
    *,
    train_episodes: int,
    validation_episodes: int,
    train_symbols: int,
    validation_symbols: int,
    train_months: int,
    validation_months: int,
    requirements: W9CoverageRequirements = W9CoverageRequirements(),
) -> dict[str, Any]:
    observed = {
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "train_symbols": train_symbols,
        "validation_symbols": validation_symbols,
        "train_months": train_months,
        "validation_months": validation_months,
    }
    required = {
        "train_episodes": requirements.minimum_train_episodes,
        "validation_episodes": requirements.minimum_validation_episodes,
        "train_symbols": requirements.minimum_symbols_per_partition,
        "validation_symbols": requirements.minimum_symbols_per_partition,
        "train_months": requirements.minimum_months_per_partition,
        "validation_months": requirements.minimum_months_per_partition,
    }
    blockers = [
        f"INSUFFICIENT_{name.upper()}:{value}<{required[name]}"
        for name, value in observed.items()
        if value < required[name]
    ]
    return {"passes": not blockers, "observed": observed, "required": required, "blockers": blockers}


def audit_incremental_l2(path: Path, *, maximum_messages: int | None = None) -> dict[str, Any]:
    """Stream a normalized Tardis L2 file and validate reconstructability."""

    book = L2Book()
    rows_count = messages = snapshots = crossed = invalid = 0
    local_out_of_order = exchange_out_of_order = 0
    gaps_over_5s = 0
    maximum_gap_us = 0
    previous_local: int | None = None
    previous_exchange: int | None = None
    current_local: int | None = None
    group: list[dict[str, str]] = []

    def consume(message_rows: list[dict[str, str]]) -> bool:
        nonlocal messages, snapshots, crossed, invalid
        if not message_rows:
            return True
        try:
            book.apply_message(message_rows)
            snapshots += int(_boolean(message_rows[0]["is_snapshot"]))
        except ValueError as error:
            invalid += 1
            crossed += int(str(error) == "AEGIS_W9_BOOK_CROSSED")
            return False
        messages += 1
        return True

    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "exchange", "symbol", "timestamp", "local_timestamp",
            "is_snapshot", "side", "price", "amount",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("AEGIS_W9_L2_SCHEMA_INVALID")
        for row in reader:
            rows_count += 1
            local = int(row["local_timestamp"])
            exchange = int(row["timestamp"])
            if current_local is None:
                current_local = local
            if local != current_local:
                if not consume(group):
                    break
                if maximum_messages is not None and messages >= maximum_messages:
                    group = []
                    break
                if previous_local is not None:
                    local_out_of_order += int(current_local < previous_local)
                    gap = max(0, current_local - previous_local)
                    maximum_gap_us = max(maximum_gap_us, gap)
                    gaps_over_5s += int(gap > 5_000_000)
                previous_local = current_local
                group = []
                current_local = local
            if previous_exchange is not None:
                exchange_out_of_order += int(exchange < previous_exchange)
            previous_exchange = exchange
            group.append(row)
        else:
            consume(group)

    return {
        "rows": rows_count,
        "messages": messages,
        "snapshot_messages": snapshots,
        "book_ready": book.ready,
        "bid_levels_final": len(book.bids),
        "ask_levels_final": len(book.asks),
        "crossed_or_locked_messages": crossed,
        "invalid_messages": invalid,
        "local_timestamp_out_of_order": local_out_of_order,
        "exchange_timestamp_out_of_order_rows": exchange_out_of_order,
        "gaps_over_5s": gaps_over_5s,
        "maximum_local_gap_us": maximum_gap_us,
        "native_sequence_ids_present": False,
        "provider_sequence_validation_documented": True,
        "passes_normalized_reconstruction": (
            book.ready and snapshots > 0 and invalid == 0 and local_out_of_order == 0
        ),
    }


def audit_quotes(path: Path) -> dict[str, Any]:
    rows = crossed = local_out_of_order = exchange_out_of_order = 0
    previous_local: int | None = None
    previous_exchange: int | None = None
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "exchange", "symbol", "timestamp", "local_timestamp",
            "ask_amount", "ask_price", "bid_price", "bid_amount",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("AEGIS_W9_QUOTES_SCHEMA_INVALID")
        for row in reader:
            rows += 1
            local = int(row["local_timestamp"])
            exchange = int(row["timestamp"])
            bid = float(row["bid_price"])
            ask = float(row["ask_price"])
            crossed += int(bid <= 0.0 or ask <= bid)
            if previous_local is not None:
                local_out_of_order += int(local < previous_local)
            if previous_exchange is not None:
                exchange_out_of_order += int(exchange < previous_exchange)
            previous_local = local
            previous_exchange = exchange
    return {
        "rows": rows,
        "crossed_or_invalid_quotes": crossed,
        "local_timestamp_out_of_order": local_out_of_order,
        "exchange_timestamp_out_of_order": exchange_out_of_order,
        "passes": rows > 0 and crossed == 0 and local_out_of_order == 0,
    }


def audit_trades(path: Path) -> dict[str, Any]:
    rows = duplicate_ids = local_out_of_order = exchange_out_of_order = 0
    previous_id: str | None = None
    previous_local: int | None = None
    previous_exchange: int | None = None
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "exchange", "symbol", "timestamp", "local_timestamp",
            "id", "side", "price", "amount",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("AEGIS_W9_TRADES_SCHEMA_INVALID")
        for row in reader:
            rows += 1
            local = int(row["local_timestamp"])
            exchange = int(row["timestamp"])
            identity = row["id"]
            duplicate_ids += int(identity == previous_id)
            if previous_local is not None:
                local_out_of_order += int(local < previous_local)
            if previous_exchange is not None:
                exchange_out_of_order += int(exchange < previous_exchange)
            if float(row["price"]) <= 0.0 or float(row["amount"]) <= 0.0:
                raise ValueError("AEGIS_W9_TRADE_INVALID")
            if row["side"] not in {"buy", "sell"}:
                raise ValueError("AEGIS_W9_TRADE_SIDE_INVALID")
            previous_id = identity
            previous_local = local
            previous_exchange = exchange
    return {
        "rows": rows,
        "adjacent_duplicate_ids": duplicate_ids,
        "local_timestamp_out_of_order": local_out_of_order,
        "exchange_timestamp_out_of_order": exchange_out_of_order,
        "passes": rows > 0 and duplicate_ids == 0 and local_out_of_order == 0,
    }


def first_barrier_label(
    midprices: Iterable[tuple[int, float]],
    *,
    reference_price: float,
    barrier_bps: float,
) -> str:
    if reference_price <= 0.0 or barrier_bps <= 0.0:
        raise ValueError("AEGIS_W9_BARRIER_INVALID")
    upper = reference_price * (1.0 + barrier_bps / 10_000.0)
    lower = reference_price * (1.0 - barrier_bps / 10_000.0)
    for _, price in midprices:
        if price >= upper:
            return "UP_FIRST"
        if price <= lower:
            return "DOWN_FIRST"
    return "NEITHER"
