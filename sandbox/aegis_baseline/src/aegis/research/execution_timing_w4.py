"""Fail-closed execution-cost primitives for W4 research."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REQUIRED_EXECUTION_FIELDS = (
    "exchange_timestamp_ms",
    "best_bid",
    "best_ask",
    "best_bid_quantity",
    "best_ask_quantity",
)


@dataclass(frozen=True)
class ExecutionDataCapabilities:
    agg_trades: bool
    best_bid_ask: bool
    depth_l2: bool
    local_receive_timestamp: bool
    decision_timestamp: bool
    fee_schedule: bool


def stable_execution_intent_id(
    signal_id: str, symbol: str, side: str, decision_timestamp_ms: int
) -> str:
    if not signal_id or side not in {"LONG", "SHORT"} or decision_timestamp_ms <= 0:
        raise ValueError("AEGIS_W4_INTENT_IDENTITY_INVALID")
    material = f"W4|{signal_id}|{symbol}|{side}|{decision_timestamp_ms}".encode()
    return "W4-" + hashlib.sha256(material).hexdigest()


def midprice(best_bid: float, best_ask: float) -> float:
    if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
        raise ValueError("AEGIS_W4_BBO_NONFINITE")
    if best_bid <= 0.0 or best_ask <= best_bid:
        raise ValueError("AEGIS_W4_BBO_INVALID")
    return (best_bid + best_ask) / 2.0


def microprice(
    best_bid: float, best_ask: float, best_bid_quantity: float, best_ask_quantity: float
) -> float:
    midprice(best_bid, best_ask)
    if best_bid_quantity < 0.0 or best_ask_quantity < 0.0:
        raise ValueError("AEGIS_W4_BBO_QUANTITY_INVALID")
    total = best_bid_quantity + best_ask_quantity
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("AEGIS_W4_BBO_QUANTITY_EMPTY")
    return (best_ask * best_bid_quantity + best_bid * best_ask_quantity) / total


def direction(side: str) -> int:
    if side == "LONG":
        return 1
    if side == "SHORT":
        return -1
    raise ValueError("AEGIS_W4_SIDE_INVALID")


def implementation_shortfall_bps(
    side: str, execution_price: float, decision_reference_price: float
) -> float:
    """Positive values always mean a worse execution for the frozen intent."""

    if min(execution_price, decision_reference_price) <= 0.0:
        raise ValueError("AEGIS_W4_PRICE_INVALID")
    return direction(side) * (execution_price / decision_reference_price - 1.0) * 10_000.0


def adverse_selection_bps(side: str, fill_price: float, future_midprice: float) -> float:
    """Positive values mean the post-fill market moved against the position."""

    if min(fill_price, future_midprice) <= 0.0:
        raise ValueError("AEGIS_W4_PRICE_INVALID")
    return -direction(side) * (future_midprice / fill_price - 1.0) * 10_000.0


def total_cost_per_intent_bps(
    *,
    implementation_shortfall: float,
    fee: float,
    delay_cost: float = 0.0,
    missed_opportunity_cost: float = 0.0,
) -> float:
    values = (implementation_shortfall, fee, delay_cost, missed_opportunity_cost)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AEGIS_W4_COST_NONFINITE")
    return float(sum(values))


def timestamp_diagnostics(
    event_timestamps_ms: Sequence[int], identities: Sequence[Any]
) -> dict[str, int | bool]:
    if len(event_timestamps_ms) != len(identities):
        raise ValueError("AEGIS_W4_TIMESTAMP_IDENTITY_LENGTH_MISMATCH")
    out_of_order = sum(
        current < previous
        for previous, current in zip(event_timestamps_ms, event_timestamps_ms[1:])
    )
    duplicates = len(identities) - len(set(identities))
    invalid = sum(timestamp <= 0 for timestamp in event_timestamps_ms)
    return {
        "rows": len(event_timestamps_ms),
        "invalid_timestamps": invalid,
        "out_of_order_events": out_of_order,
        "duplicate_identities": duplicates,
        "passes": invalid == 0 and out_of_order == 0 and duplicates == 0,
    }


def assess_data_quality(capabilities: ExecutionDataCapabilities) -> dict[str, Any]:
    blockers: list[str] = []
    if not capabilities.best_bid_ask:
        blockers.append("HISTORICAL_SYNCHRONIZED_BBO_MISSING")
    if not capabilities.local_receive_timestamp:
        blockers.append("LOCAL_RECEIVE_TIMESTAMP_MISSING")
    if not capabilities.decision_timestamp:
        blockers.append("FROZEN_INTENT_DECISION_TIMESTAMP_MISSING")
    if not capabilities.fee_schedule:
        blockers.append("AUTHORITATIVE_FEE_SCHEDULE_MISSING")
    return {
        "sufficient_for_w4a": not blockers,
        "sufficient_for_market_now": capabilities.best_bid_ask and not blockers,
        "sufficient_for_passive_limit": (
            capabilities.best_bid_ask
            and capabilities.depth_l2
            and capabilities.local_receive_timestamp
            and not blockers
        ),
        "blockers": blockers,
    }


def assert_causal_snapshot(snapshot: Mapping[str, Any], decision_timestamp_ms: int) -> None:
    missing = [field for field in REQUIRED_EXECUTION_FIELDS if field not in snapshot]
    if missing:
        raise ValueError("AEGIS_W4_SNAPSHOT_FIELDS_MISSING:" + ",".join(missing))
    if int(snapshot["exchange_timestamp_ms"]) > decision_timestamp_ms:
        raise ValueError("AEGIS_W4_SNAPSHOT_LOOKAHEAD")
    midprice(float(snapshot["best_bid"]), float(snapshot["best_ask"]))
    microprice(
        float(snapshot["best_bid"]),
        float(snapshot["best_ask"]),
        float(snapshot["best_bid_quantity"]),
        float(snapshot["best_ask_quantity"]),
    )


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)
