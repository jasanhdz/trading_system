from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(SANDBOX / "src"))
sys.path.insert(0, str(REPOSITORY / "src"))


def make_one_minute(rows: int, *, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="1min", tz="UTC")
    sequence = pd.Series(range(rows), dtype=float)
    base = 100.0 + sequence * 0.001
    close = base + ((sequence % 7) - 3.0) * 0.002
    return pd.DataFrame({
        "open_time_ms": index.as_unit("ns").asi8 // 1_000_000,
        "open": base,
        "high": pd.concat([base, close], axis=1).max(axis=1) + 0.05,
        "low": pd.concat([base, close], axis=1).min(axis=1) - 0.05,
        "close": close,
        "volume": 100.0 + sequence % 11,
        "taker_buy_volume": 48.0 + sequence % 9,
    })


@pytest.fixture
def one_minute() -> pd.DataFrame:
    return make_one_minute(900)
