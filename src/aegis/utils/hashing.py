"""Canonical serialization and SHA-256 helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


JsonPrimitive = None | bool | int | float | str | list["JsonPrimitive"] | dict[str, "JsonPrimitive"]


def to_primitive(value: Any) -> JsonPrimitive:
    """Convert contracts to a stable JSON-compatible representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime cannot be serialized")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


class HashProvider(Protocol):
    def digest_bytes(self, payload: bytes) -> str: ...

    def digest_value(self, value: Any) -> str: ...


class Sha256HashProvider:
    def digest_bytes(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def digest_value(self, value: Any) -> str:
        return self.digest_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_name_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
