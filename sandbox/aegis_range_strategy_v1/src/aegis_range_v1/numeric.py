from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

QUANTUM_12DP = Decimal("0.000000000001")


def canonical_decimal_12dp(value: Decimal | float | int | str) -> str:
    if isinstance(value, float):
        decimal = Decimal.from_float(value)
    elif isinstance(value, Decimal):
        decimal = value
    else:
        decimal = Decimal(value)
    if not decimal.is_finite():
        raise ValueError("canonical decimal must be finite")
    with localcontext() as context:
        context.prec = max(50, len(decimal.as_tuple().digits) + 20)
        quantized = decimal.quantize(QUANTUM_12DP, rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = abs(quantized)
    return format(quantized, ".12f")


def iso_utc_millis(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cluster_id(symbol: str, side: str, first_pivot_at: datetime, source_price: str) -> str:
    return sha256_text(f"{symbol}|{side}|{iso_utc_millis(first_pivot_at)}|{source_price}")


def range_episode_id(
    symbol: str,
    range_confirmed_at: datetime,
    support_cluster_id: str,
    resistance_cluster_id: str,
) -> str:
    value = "|".join(
        (
            symbol,
            "5m",
            iso_utc_millis(range_confirmed_at),
            support_cluster_id,
            resistance_cluster_id,
        )
    )
    return sha256_text(value)


def range_id(
    episode_id: str,
    decision_at: datetime,
    support: float,
    resistance: float,
    midpoint: float,
) -> str:
    value = "|".join(
        (
            episode_id,
            iso_utc_millis(decision_at),
            canonical_decimal_12dp(support),
            canonical_decimal_12dp(resistance),
            canonical_decimal_12dp(midpoint),
        )
    )
    return sha256_text(value)
