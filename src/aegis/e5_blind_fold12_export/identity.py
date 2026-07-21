"""Owner-authorized Amendment 06 historical trade identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from aegis.e5_phase0.constants import CANONICAL_SYMBOLS

from .errors import BlindExportError


AUTHORITY_CLASSIFICATION = "OWNER_AUTHORIZED_CANONICAL_DERIVED_IDENTITY"
IDENTITY_SCHEME = "e5-historical-trade-id-v1"
IDENTITY_FIELDS = (
    "symbol",
    "fold",
    "signal_timestamp_utc_ms",
    "entry_timestamp_utc_ms",
    "entry_price_decimal",
    "side",
)
_OUTCOME_FIELDS = frozenset({
    "exit_timestamp", "exit_price", "gross_return", "gross_return_fraction",
    "cost_fraction", "funding", "funding_return", "net_return",
    "net_return_fraction", "mfe", "mfe_fraction", "mae", "mae_fraction",
    "realized_pnl", "label", "labels", "control", "discovery", "confirmation",
})
_BRAIN_FIELDS = frozenset({"d3", "rv2", "trrm", "qmae", "eqm", "econ1", "aegis"})
_ENVIRONMENT_FIELDS = frozenset({
    "wall_clock", "generation_timestamp", "filesystem_path", "source_path",
    "output_path", "checkout_path", "repository_path", "source_run_id",
    "horizon", "row_order",
})
_PRICE_PATTERN = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_ASCII_WHITESPACE = " \t\r\n"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class CanonicalTradeIdentity:
    trade_id: str
    preimage: bytes
    symbol: str
    fold: str
    signal_timestamp_utc_ms: int
    entry_timestamp_utc_ms: int
    entry_price_decimal: str
    side: str


def _normalize_symbol(value: Any) -> str:
    if not isinstance(value, str):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=symbol")
    normalized = value.strip(_ASCII_WHITESPACE).upper()
    if normalized not in CANONICAL_SYMBOLS or not normalized.isascii():
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=symbol")
    return normalized


def _normalize_fold(value: Any) -> str:
    if isinstance(value, bool):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=fold")
    if isinstance(value, int):
        normalized = f"F{value}"
    elif isinstance(value, str):
        normalized = value.strip(_ASCII_WHITESPACE).upper()
    else:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=fold")
    if normalized not in {"F1", "F2", "F3", "F4"}:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=fold")
    return normalized


def _normalize_side(value: Any) -> str:
    if not isinstance(value, str):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=side")
    normalized = value.strip(_ASCII_WHITESPACE).upper()
    if normalized not in {"LONG", "SHORT"}:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", "field=side")
    return normalized


def canonical_utc_ms(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
        return value
    if not isinstance(value, str):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond % 1000:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    milliseconds = delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000
    if not -(2**63) <= milliseconds < 2**63:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    return milliseconds


def canonical_decimal(value: Any, field: str, *, positive: bool = False) -> str:
    if isinstance(value, bool):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, str) and _PRICE_PATTERN.fullmatch(value):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}") from exc
    else:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise BlindExportError("E5_CANONICAL_TRADE_ID_INPUT_INVALID", f"field={field}")
    if parsed == 0:
        return "0.0"
    fixed = format(parsed, "f")
    if "." not in fixed:
        fixed += ".0"
    integer, fraction = fixed.split(".", 1)
    fraction = fraction.rstrip("0") or "0"
    sign = "-" if integer.startswith("-") else ""
    digits = integer.lstrip("-").lstrip("0") or "0"
    return f"{sign}{digits}.{fraction}"


def _validate_field_registry(values: Mapping[str, Any]) -> None:
    missing = set(IDENTITY_FIELDS) - set(values)
    if missing or any(values.get(field) is None for field in IDENTITY_FIELDS):
        field = sorted(missing or {name for name in IDENTITY_FIELDS if values.get(name) is None})[0]
        raise BlindExportError("E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING", f"field={field}")
    extras = set(values) - set(IDENTITY_FIELDS)
    if extras & _OUTCOME_FIELDS:
        raise BlindExportError(
            "E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED",
            f"field={sorted(extras & _OUTCOME_FIELDS)[0]}",
        )
    if extras & _BRAIN_FIELDS:
        raise BlindExportError(
            "E5_CANONICAL_TRADE_ID_CURRENT_BRAIN_DEPENDENCY_PROHIBITED",
            f"field={sorted(extras & _BRAIN_FIELDS)[0]}",
        )
    if extras & _ENVIRONMENT_FIELDS or extras:
        field = sorted(extras)[0]
        raise BlindExportError("E5_CANONICAL_TRADE_ID_CANONICALIZATION_FAILED", f"field={field}")


def derive_canonical_trade_identity(
    values: Mapping[str, Any],
    *,
    version: str = IDENTITY_SCHEME,
) -> CanonicalTradeIdentity:
    if version != IDENTITY_SCHEME:
        raise BlindExportError("E5_CANONICAL_TRADE_ID_VERSION_UNSUPPORTED", "identity version")
    _validate_field_registry(values)
    symbol = _normalize_symbol(values["symbol"])
    fold = _normalize_fold(values["fold"])
    signal_ms = canonical_utc_ms(values["signal_timestamp_utc_ms"], "signal_timestamp_utc_ms")
    entry_ms = canonical_utc_ms(values["entry_timestamp_utc_ms"], "entry_timestamp_utc_ms")
    entry_price = canonical_decimal(values["entry_price_decimal"], "entry_price_decimal", positive=True)
    side = _normalize_side(values["side"])
    preimage = json.dumps(
        [IDENTITY_SCHEME, symbol, fold, signal_ms, entry_ms, entry_price, side],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(preimage).hexdigest()
    return CanonicalTradeIdentity(digest, preimage, symbol, fold, signal_ms, entry_ms, entry_price, side)
