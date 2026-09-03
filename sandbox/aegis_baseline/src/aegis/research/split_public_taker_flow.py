"""Causal V14 taker flow reconstructed from split public kline sources."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..features import CANONICAL_SYMBOLS
from .feature_information_v14 import (
    TAKER_FLOW_FEATURE_NAMES,
    local_taker_flow,
    market_taker_flow,
    taker_imbalance,
)


class SplitPublicTakerFlowError(ValueError):
    """Raised when split public flow sources violate their causal contract."""


def split_public_flow_lookup(
    *,
    candle_delta: Path,
    microstructure_database: Path,
    required_entry_timestamps_ms: Sequence[int],
    history_bars: int = 24,
    bar_spacing_ms: int = 300_000,
) -> tuple[Mapping[tuple[int, str], tuple[float, ...]], Mapping[str, Any]]:
    required = sorted(set(int(value) for value in required_entry_timestamps_ms))
    if not required or history_bars != 24 or bar_spacing_ms != 300_000:
        raise SplitPublicTakerFlowError("invalid split public flow request")
    start_bar = required[0] - history_bars * bar_spacing_ms
    end_bar = required[-1] - bar_spacing_ms
    volumes: dict[tuple[str, int], float] = {}
    with candle_delta.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                symbol = str(row["symbol"])
                timestamp = int(row["open_time_ms"])
                volume = float(row["volume"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SplitPublicTakerFlowError(
                    f"invalid public candle delta row: {line_number}"
                ) from exc
            if (
                symbol in CANONICAL_SYMBOLS
                and start_bar <= timestamp <= end_bar
                and math.isfinite(volume)
                and volume >= 0.0
            ):
                key = (symbol, timestamp)
                if key in volumes and not math.isclose(volumes[key], volume, abs_tol=1e-12):
                    raise SplitPublicTakerFlowError("conflicting public candle volume")
                volumes[key] = volume

    taker_buy: dict[tuple[str, int], float] = {}
    connection = sqlite3.connect(
        f"file:{microstructure_database.resolve()}?mode=ro", uri=True
    )
    try:
        for symbol, timestamp, value in connection.execute(
            "SELECT symbol,open_time_ms,taker_buy_base FROM kline_microstructure "
            "WHERE open_time_ms BETWEEN ? AND ? ORDER BY symbol,open_time_ms",
            (start_bar, end_bar),
        ):
            if str(symbol) in CANONICAL_SYMBOLS:
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    raise SplitPublicTakerFlowError("invalid public taker-buy volume")
                taker_buy[(str(symbol), int(timestamp))] = number
    finally:
        connection.close()

    lookup: dict[tuple[int, str], tuple[float, ...]] = {}
    invalid_physical_bars = 0
    incomplete_symbol_histories = 0
    complete_timestamps = 0
    for entry_timestamp in required:
        local: dict[str, Mapping[str, float]] = {}
        expected = [
            entry_timestamp - offset * bar_spacing_ms
            for offset in range(history_bars, 0, -1)
        ]
        for symbol in CANONICAL_SYMBOLS:
            imbalances = []
            for bar_timestamp in expected:
                key = (symbol, bar_timestamp)
                volume = volumes.get(key)
                bought = taker_buy.get(key)
                if volume is None or bought is None:
                    imbalances = []
                    break
                if bought > volume + 1e-9:
                    invalid_physical_bars += 1
                    imbalances = []
                    break
                imbalances.append(taker_imbalance(volume, bought))
            if len(imbalances) == history_bars:
                local[symbol] = local_taker_flow(imbalances)
            else:
                incomplete_symbol_histories += 1
        if set(local) != set(CANONICAL_SYMBOLS):
            continue
        complete_timestamps += 1
        for symbol in CANONICAL_SYMBOLS:
            values = {**local[symbol], **market_taker_flow(local, symbol=symbol)}
            lookup[(entry_timestamp, symbol)] = tuple(
                float(values[name]) for name in TAKER_FLOW_FEATURE_NAMES
            )
    return lookup, {
        "required_entry_timestamps": len(required),
        "complete_eleven_symbol_timestamps": complete_timestamps,
        "flow_keys": len(lookup),
        "candle_volume_rows_in_window": len(volumes),
        "taker_buy_rows_in_window": len(taker_buy),
        "invalid_physical_bars": invalid_physical_bars,
        "incomplete_symbol_histories": incomplete_symbol_histories,
        "history_bars": history_bars,
        "latest_bar_offset_minutes": 5,
        "strict_pre_entry_only": True,
        "zero_or_synthetic_fill": False,
    }

