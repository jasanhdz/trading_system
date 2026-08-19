from __future__ import annotations

import pandas as pd

from aegis_strategy_router.replay.fresh_pipeline import causal_candle_source_hash
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder
from aegis_strategy_router.replay.snapshot_builder import DeterministicSnapshotBuilder
from conftest import make_one_minute


def test_precomputed_builder_is_byte_equivalent_at_multiple_boundaries() -> None:
    frame = make_one_minute(130 * 1_440, start="2023-09-01T00:00:00Z")
    baseline = DeterministicSnapshotBuilder()
    precomputed = PrecomputedSnapshotBuilder()
    for offset in (100 * 1_440, 110 * 1_440 + 15, 129 * 1_440):
        boundary = pd.to_datetime(frame.iloc[offset - 1].open_time_ms, unit="ms", utc=True)
        boundary += pd.Timedelta(minutes=1)
        price = float(frame.iloc[offset - 1].close)
        source_hash = causal_candle_source_hash(frame, boundary.to_pydatetime())
        versions = {"fresh_candle_source_hash": source_hash}
        expected = baseline.build(
            symbol="TESTUSDT", decision_at=boundary.to_pydatetime(), built_at=boundary.to_pydatetime(),
            reference_price=price, one_minute=frame, source_versions=versions,
        )
        actual_hash = precomputed.causal_source_hash(
            "TESTUSDT", frame, boundary.to_pydatetime()
        )
        assert actual_hash == source_hash
        actual = precomputed.build(
            symbol="TESTUSDT", decision_at=boundary.to_pydatetime(), built_at=boundary.to_pydatetime(),
            reference_price=price, one_minute=frame,
            source_versions={"fresh_candle_source_hash": actual_hash},
        )
        assert actual.canonical_bytes() == expected.canonical_bytes()
