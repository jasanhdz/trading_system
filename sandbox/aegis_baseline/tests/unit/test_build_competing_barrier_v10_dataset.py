from __future__ import annotations

import io
import json

import pytest

from build_competing_barrier_v10_dataset import _pairs


def encoded(rows: list[dict[str, str]]) -> io.StringIO:
    return io.StringIO("".join(json.dumps(row) + "\n" for row in rows))


def test_v10_reuses_strict_complete_side_pairs() -> None:
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "side": "SHORT"},
        {"timestamp": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "side": "LONG"},
    ]
    pairs = list(_pairs(encoded(rows)))
    assert [row["side"] for row in pairs[0]] == ["LONG", "SHORT"]


def test_v10_fails_on_incomplete_pair() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        list(
            _pairs(
                encoded(
                    [
                        {
                            "timestamp": "2026-01-01T00:00:00+00:00",
                            "symbol": "BTCUSDT",
                            "side": "LONG",
                        }
                    ]
                )
            )
        )
