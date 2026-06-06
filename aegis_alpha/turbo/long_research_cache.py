#!/usr/bin/env python3
"""Bounded research-only cache for LONG alpha experiments.

This module is intentionally not imported by live inference. It keeps cache
state instance-local, bounded, and rejects active/live paths.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


BAD_PATH_PARTS = ("/active/", "active_manifest.json", "phase_o_short_manifest.json")


def assert_research_cache_path(path: str | Path) -> None:
    text = str(path)
    if any(part in text for part in BAD_PATH_PARTS) or ("/models/turbo/" in text and "/active" in text):
        raise ValueError(f"refusing live/active path for LONG research cache: {text}")


def db_mtime_key(db_path: str | Path) -> tuple[str, int | None]:
    path = Path(db_path)
    assert_research_cache_path(path)
    try:
        return str(path), int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return str(path), None


def _rough_size_bytes(value: Any) -> int:
    try:
        if pd is not None and isinstance(value, pd.DataFrame):
            return int(value.memory_usage(deep=True).sum())
        if np is not None and isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(_rough_size_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(_rough_size_bytes(v) for v in value)
    except Exception:
        pass
    return int(sys.getsizeof(value))


class LongResearchCache:
    def __init__(self, max_items: int = 64):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = int(max_items)
        self._store: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _key(self, namespace: str, *parts: Any) -> tuple[Any, ...]:
        return (namespace, *parts)

    def get(self, namespace: str, *parts: Any) -> Any | None:
        key = self._key(namespace, *parts)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, namespace: str, *parts_and_value: Any) -> Any:
        if len(parts_and_value) < 1:
            raise ValueError("cache set requires value")
        *parts, value = parts_and_value
        key = self._key(namespace, *parts)
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)
            self.evictions += 1
        return value

    def get_or_set(self, namespace: str, parts: tuple[Any, ...], factory):
        cached = self.get(namespace, *parts)
        if cached is not None:
            return cached
        value = factory()
        return self.set(namespace, *parts, value)

    def clear(self) -> None:
        self._store.clear()

    def estimated_memory_mb(self) -> float:
        return sum(_rough_size_bytes(v) for v in self._store.values()) / (1024.0 * 1024.0)

    def summary(self) -> dict[str, Any]:
        return {
            "max_items": self.max_items,
            "items": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "estimated_memory_mb": self.estimated_memory_mb(),
            "namespaces": sorted({str(k[0]) for k in self._store}),
        }

    def ohlcv_key(self, symbol: str, lookback_days: int, db_path: str | Path) -> tuple[Any, ...]:
        return (symbol.upper(), int(lookback_days), db_mtime_key(db_path))

    def feature_key(self, symbol: str, lookback_days: int, family_or_mode: str, db_path: str | Path) -> tuple[Any, ...]:
        return (symbol.upper(), int(lookback_days), family_or_mode, db_mtime_key(db_path))

    def labels_key(self, symbol: str, target: str, horizon: int, lookback_days: int, db_path: str | Path) -> tuple[Any, ...]:
        return (symbol.upper(), target, int(horizon), int(lookback_days), db_mtime_key(db_path))

    def folds_key(self, sample_count: int, fold_count: int, min_train_samples: int, min_test_samples: int) -> tuple[Any, ...]:
        return (int(sample_count), int(fold_count), int(min_train_samples), int(min_test_samples))

    def x_key(self, symbol: str, feature_schema_hash: str, valid_idx_hash: str) -> tuple[Any, ...]:
        return (symbol.upper(), feature_schema_hash, valid_idx_hash)
