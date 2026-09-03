"""Deterministic identity, time, numerical, ATR, and seed primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence

from .constants import BASE_SEED, EXPERIMENT_ID, PROTOCOL_ID, SPECIFICATION_VERSION
from .errors import Phase0Error


UTC = timezone.utc
_CANONICAL_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise Phase0Error("IDENTITY_INVALID", "symbol must be nonempty uppercase ASCII alphanumeric text")
    return normalized


def normalize_fold(value: int | str) -> str:
    text = str(value).strip().upper()
    if text.isdigit():
        text = f"F{text}"
    if text not in {"F1", "F2", "F3", "F4"}:
        raise Phase0Error("FOLD_INVALID", f"invalid fold {value!r}")
    return text


def normalize_horizon(value: str) -> str:
    text = value.strip().upper()
    if text not in {"H12", "H48", "H96"}:
        raise Phase0Error("HORIZON_NOT_COMPUTABLE", f"invalid horizon {value!r}")
    return text


def identity_hash(version: str, *parts: Any) -> str:
    if not version or any(part is None for part in parts):
        raise Phase0Error("IDENTITY_INVALID", "identity parts must be explicit")
    return sha256_bytes(compact_json_bytes([version, *parts]))


def observation_id(fold: int | str, signal_ms: int, symbol: str, trade_id: str) -> str:
    if not trade_id or isinstance(signal_ms, bool) or not isinstance(signal_ms, int):
        raise Phase0Error("IDENTITY_INVALID", "observation identity fields are malformed")
    return identity_hash("obs-v1", normalize_fold(fold), signal_ms, normalize_symbol(symbol), "SHORT", trade_id)


def symbol_id(symbol: str) -> str:
    return identity_hash("symbol-v1", normalize_symbol(symbol))


def cycle_id(fold: int | str, signal_ms: int) -> str:
    if isinstance(signal_ms, bool) or not isinstance(signal_ms, int):
        raise Phase0Error("IDENTITY_INVALID", "cycle timestamp must be an integer")
    return identity_hash("cycle-v1", normalize_fold(fold), signal_ms)


def c1_control_id(experimental_observation_id: str, control_symbol: str, horizon: str, replicate_index: int) -> str:
    return identity_hash("c1-control-v1", experimental_observation_id, normalize_symbol(control_symbol), normalize_horizon(horizon), _index(replicate_index))


def c2_symbol_cycle_id(symbol: str, control_cycle_id: str) -> str:
    if not control_cycle_id:
        raise Phase0Error("IDENTITY_INVALID", "C2 cycle identity is empty")
    return identity_hash("c2-unit-v1", normalize_symbol(symbol), control_cycle_id)


def match_replicate_id(control: str, horizon: str, fold: int | str, month: str | None, index: int) -> str:
    control_name = control.strip().upper()
    if control_name not in {"C1", "C2"}:
        raise Phase0Error("IDENTITY_INVALID", "unknown matching control")
    return identity_hash("match-v1", control_name, normalize_horizon(horizon), normalize_fold(fold), month, _index(index))


def bootstrap_id(statistic: str, horizon: str, scope: str, index: int) -> str:
    return identity_hash("bootstrap-v1", statistic, normalize_horizon(horizon), scope, _index(index))


def permutation_id(test: str, horizon: str, index: int) -> str:
    return identity_hash("permutation-v1", test, normalize_horizon(horizon), _index(index))


def artifact_id(relative_posix_path: str, dependency_hash: str) -> str:
    if relative_posix_path.startswith("/") or ".." in relative_posix_path.split("/"):
        raise Phase0Error("IDENTITY_INVALID", "artifact path must be relative and traversal-free")
    _validate_sha256(dependency_hash)
    return identity_hash("artifact-v1", relative_posix_path, dependency_hash)


def confirmation_run_id(discovery_freeze_hash: str, confirmation_input_hash: str) -> str:
    _validate_sha256(discovery_freeze_hash)
    _validate_sha256(confirmation_input_hash)
    return identity_hash("confirmation-v1", discovery_freeze_hash, confirmation_input_hash)


def parse_utc_ms(value: str | datetime | int) -> int:
    if isinstance(value, bool):
        raise Phase0Error("TIMESTAMP_INVALID", "boolean is not a timestamp")
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise Phase0Error("TIMESTAMP_INVALID", f"invalid timestamp {value!r}") from exc
    else:
        raise Phase0Error("TIMESTAMP_INVALID", "unsupported timestamp type")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase0Error("TIMESTAMP_INVALID", "timestamp must carry an explicit timezone")
    utc_value = parsed.astimezone(UTC)
    return int(utc_value.timestamp() * 1000)


def funding_event_in_interval(entry_ms: int, termination_ms: int, funding_ms: int) -> bool:
    if termination_ms < entry_ms:
        raise Phase0Error("TIME_ALIGNMENT_FAILURE", "termination precedes entry")
    return entry_ms < funding_ms <= termination_ms


def ensure_unique_timestamps(values: Iterable[int]) -> tuple[int, ...]:
    ordered = tuple(sorted(values))
    if len(ordered) != len(set(ordered)):
        raise Phase0Error("TIME_ALIGNMENT_FAILURE", "duplicate timestamps")
    return ordered


def complete_iso_week_bounds(value_ms: int) -> tuple[int, int]:
    value = datetime.fromtimestamp(value_ms / 1000, tz=UTC)
    monday = (value - timedelta(days=value.weekday(), hours=value.hour, minutes=value.minute, seconds=value.second, microseconds=value.microsecond))
    return parse_utc_ms(monday), parse_utc_ms(monday + timedelta(days=7))


def is_complete_week_in_fold(week_start_ms: int, fold_start_ms: int, fold_end_ms: int) -> bool:
    start = datetime.fromtimestamp(week_start_ms / 1000, tz=UTC)
    if start.weekday() != 0 or start.time() != datetime.min.time():
        return False
    week_end_ms = week_start_ms + 7 * 24 * 60 * 60 * 1000
    return fold_start_ms <= week_start_ms and week_end_ms - 1 <= fold_end_ms


def canonical_decimal(value: str | int | Decimal) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise Phase0Error("FUNDING_RATE_INVALID", f"invalid decimal {value!r}") from exc
    if not parsed.is_finite():
        raise Phase0Error("FUNDING_RATE_INVALID", "funding rate must be finite")
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


def validate_canonical_decimal(value: str) -> Decimal:
    if not _CANONICAL_DECIMAL.fullmatch(value) or canonical_decimal(value) != value:
        raise Phase0Error("FUNDING_RATE_INVALID", "funding decimal is not canonical")
    return Decimal(value)


def deterministic_sum(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        if not math.isfinite(value):
            raise Phase0Error("OUTCOME_NOT_COMPUTABLE", "non-finite aggregation input")
        total += value
    return total


def type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise Phase0Error("BOOTSTRAP_VALIDITY_FAILURE", "invalid Type 7 input")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise Phase0Error("BOOTSTRAP_VALIDITY_FAILURE", "Type 7 input must be finite")
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    fraction = h - lower
    upper = min(lower + 1, len(ordered) - 1)
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]


@dataclass(frozen=True)
class OhlcBar:
    open_ms: int
    high: float
    low: float
    close: float


def short_barrier_event(entry_price: float, high: float, low: float, barrier_fraction: float = 0.0028) -> str:
    if not all(math.isfinite(value) and value > 0.0 for value in (entry_price, high, low)) or high < low:
        raise Phase0Error("BARRIER_NOT_COMPUTABLE", "invalid barrier prices")
    favorable = (entry_price - low) / entry_price >= barrier_fraction
    adverse = (entry_price - high) / entry_price <= -barrier_fraction
    if adverse:
        return "ADVERSE_FIRST"
    if favorable:
        return "FAVORABLE_FIRST"
    return "NO_BARRIER"


def wilder_atr(bars: Sequence[OhlcBar], period: int = 14) -> tuple[float | None, ...]:
    if period != 14:
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "E5 ATR period is fixed at 14")
    result: list[float | None] = [None] * len(bars)
    tr_window: list[float] = []
    previous: OhlcBar | None = None
    atr: float | None = None
    for index, bar in enumerate(bars):
        if not all(math.isfinite(value) for value in (bar.high, bar.low, bar.close)) or bar.high < bar.low:
            raise Phase0Error("ATR_NOT_COMPUTABLE", "invalid OHLC")
        continuous = previous is not None and bar.open_ms - previous.open_ms == 300_000
        if previous is None or not continuous:
            tr_window = []
            atr = None
            previous = bar
            continue
        true_range = max(bar.high - bar.low, abs(bar.high - previous.close), abs(bar.low - previous.close))
        if atr is None:
            tr_window.append(true_range)
            if len(tr_window) == period:
                atr = deterministic_sum(tr_window) / period
        else:
            atr = ((period - 1) * atr + true_range) / period
        result[index] = atr
        previous = bar
    return tuple(result)


def c1_seed(replicate_index: int) -> int:
    preimage = f"{PROTOCOL_ID}||C1||{_index(replicate_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(preimage).digest(), "big") % (2**64)


def namespaced_seed(namespace: str, *context: Any) -> tuple[int, str]:
    payload = [EXPERIMENT_ID, SPECIFICATION_VERSION, BASE_SEED, namespace, *context]
    digest = hashlib.sha256(compact_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex()


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase0Error("IDENTITY_INVALID", "index must be a nonnegative integer")
    return value


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Phase0Error("IDENTITY_INVALID", "expected lowercase SHA-256")
