"""Structural JSON scanning that decodes only the fold before partitioning."""

from __future__ import annotations

import json
import mmap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .errors import BlindExportError


_WHITESPACE = b" \t\r\n"


@dataclass(frozen=True)
class SourceRecordSpan:
    ordinal: int
    start: int
    end: int
    fold: str


def _skip_ws(buffer: mmap.mmap, position: int, limit: int) -> int:
    while position < limit and buffer[position] in _WHITESPACE:
        position += 1
    return position


def _scan_string(buffer: mmap.mmap, position: int, limit: int) -> int:
    if position >= limit or buffer[position] != ord('"'):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON string expected")
    position += 1
    while position < limit:
        byte = buffer[position]
        if byte == ord('"'):
            return position + 1
        if byte == ord('\\'):
            position += 2
        else:
            position += 1
    raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "unterminated JSON string")


def _scan_compound(buffer: mmap.mmap, position: int, limit: int) -> int:
    opening = buffer[position]
    closing = ord('}') if opening == ord('{') else ord(']')
    depth: list[int] = [closing]
    position += 1
    while position < limit and depth:
        byte = buffer[position]
        if byte == ord('"'):
            position = _scan_string(buffer, position, limit)
            continue
        if byte == ord('{'):
            depth.append(ord('}'))
        elif byte == ord('['):
            depth.append(ord(']'))
        elif byte == depth[-1]:
            depth.pop()
        position += 1
    if depth:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "unterminated JSON compound")
    return position


def _skip_value(buffer: mmap.mmap, position: int, limit: int) -> int:
    position = _skip_ws(buffer, position, limit)
    if position >= limit:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON value missing")
    byte = buffer[position]
    if byte == ord('"'):
        return _scan_string(buffer, position, limit)
    if byte in (ord('{'), ord('[')):
        return _scan_compound(buffer, position, limit)
    end = position
    while end < limit and buffer[end] not in b",]} \t\r\n":
        end += 1
    if end == position:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON scalar invalid")
    return end


def _decode_key(buffer: mmap.mmap, start: int, end: int) -> str:
    try:
        value = json.loads(buffer[start:end])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON object key invalid") from exc
    if not isinstance(value, str):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON object key invalid")
    return value


def _object_member(buffer: mmap.mmap, start: int, end: int, target: str) -> tuple[int, int] | None:
    position = _skip_ws(buffer, start, end)
    if position >= end or buffer[position] != ord('{'):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON object expected")
    position += 1
    while True:
        position = _skip_ws(buffer, position, end)
        if position >= end:
            raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON object truncated")
        if buffer[position] == ord('}'):
            return None
        key_start = position
        key_end = _scan_string(buffer, key_start, end)
        key = _decode_key(buffer, key_start, key_end)
        position = _skip_ws(buffer, key_end, end)
        if position >= end or buffer[position] != ord(':'):
            raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON member separator missing")
        value_start = _skip_ws(buffer, position + 1, end)
        value_end = _skip_value(buffer, value_start, end)
        if key == target:
            return value_start, value_end
        position = _skip_ws(buffer, value_end, end)
        if position < end and buffer[position] == ord(','):
            position += 1
            continue
        if position < end and buffer[position] == ord('}'):
            return None
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "JSON object delimiter invalid")


def _array_values(buffer: mmap.mmap, start: int, end: int) -> Iterator[tuple[int, int, int]]:
    position = _skip_ws(buffer, start, end)
    if position >= end or buffer[position] != ord('['):
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "report.trades must be an array")
    position += 1
    ordinal = 0
    while True:
        position = _skip_ws(buffer, position, end)
        if position >= end:
            raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "trades array truncated")
        if buffer[position] == ord(']'):
            return
        value_start = position
        value_end = _skip_value(buffer, value_start, end)
        yield ordinal, value_start, value_end
        ordinal += 1
        position = _skip_ws(buffer, value_end, end)
        if position < end and buffer[position] == ord(','):
            position += 1
            continue
        if position < end and buffer[position] == ord(']'):
            return
        raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "trades array delimiter invalid")


def _fold_from_record(buffer: mmap.mmap, start: int, end: int, ordinal: int) -> str:
    signal_span = _object_member(buffer, start, end, "signal")
    if signal_span is None:
        raise BlindExportError("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", f"ordinal={ordinal} field=signal.fold")
    fold_span = _object_member(buffer, *signal_span, "fold")
    if fold_span is None:
        raise BlindExportError("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", f"ordinal={ordinal} field=signal.fold")
    try:
        value = json.loads(buffer[fold_span[0]:fold_span[1]])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BlindExportError(
            "E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE",
            f"ordinal={ordinal} field=signal.fold",
        ) from exc
    if isinstance(value, bool):
        raise BlindExportError("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", f"ordinal={ordinal} field=signal.fold")
    if isinstance(value, int):
        fold = f"F{value}"
    elif isinstance(value, str):
        fold = value.strip(" \t\r\n").upper()
    else:
        raise BlindExportError("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", f"ordinal={ordinal} field=signal.fold")
    if fold not in {"F1", "F2", "F3", "F4"}:
        raise BlindExportError("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", f"ordinal={ordinal} field=signal.fold")
    return fold


@contextmanager
def blind_trade_stream(path: Path) -> Iterator[tuple[mmap.mmap, Iterator[SourceRecordSpan]]]:
    """Yield spans after validating the structural ``report.trades`` path."""

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            limit = len(buffer)
            root_start = _skip_ws(buffer, 0, limit)
            root_end = _skip_value(buffer, root_start, limit)
            if _skip_ws(buffer, root_end, limit) != limit:
                raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "trailing source content")
            report_span = _object_member(buffer, root_start, root_end, "report")
            if report_span is None:
                raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "field=report")
            trades_span = _object_member(buffer, *report_span, "trades")
            if trades_span is None:
                raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "field=report.trades")

            def records() -> Iterator[SourceRecordSpan]:
                for ordinal, start, end in _array_values(buffer, *trades_span):
                    yield SourceRecordSpan(ordinal, start, end, _fold_from_record(buffer, start, end, ordinal))

            yield buffer, records()
