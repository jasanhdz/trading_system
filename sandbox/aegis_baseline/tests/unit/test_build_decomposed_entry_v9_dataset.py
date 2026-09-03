from __future__ import annotations

import io
import json

import pytest

from build_decomposed_entry_v9_dataset import _pairs


def encoded(rows: list[dict[str, str]]) -> io.StringIO:
    return io.StringIO("".join(json.dumps(row) + "\n" for row in rows))


def test_pairs_require_adjacent_long_and_short_for_same_identity() -> None:
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "side": "LONG"},
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "symbol": "BTCUSDT",
            "side": "SHORT",
        },
    ]
    pairs = list(_pairs(encoded(rows)))
    assert len(pairs) == 1
    assert pairs[0][0]["side"] == "LONG"
    assert pairs[0][1]["side"] == "SHORT"


def test_pairs_fail_closed_on_mismatched_symbol() -> None:
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "side": "LONG"},
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "symbol": "ETHUSDT",
            "side": "SHORT",
        },
    ]
    with pytest.raises(ValueError, match="complete adjacent"):
        list(_pairs(encoded(rows)))
