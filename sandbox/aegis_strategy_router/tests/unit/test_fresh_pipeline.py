from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from aegis_strategy_router.replay.fresh_pipeline import (
    FreshSignal,
    FreshPipelineDataError,
    FreshSnapshotCandidatePipeline,
    ParquetMinuteCandleSource,
    persist_pipeline_result,
)
from aegis_strategy_router.domain.types import Side
from conftest import make_one_minute


def _write(root: Path, symbol: str, frame: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / f"{symbol}_1m.parquet", index=False)


def test_candle_source_merges_contiguous_partitions(tmp_path: Path) -> None:
    frame = make_one_minute(20)
    first, second = tmp_path / "first", tmp_path / "second"
    _write(first, "TESTUSDT", frame.iloc[:10])
    _write(second, "TESTUSDT", frame.iloc[10:])
    merged, coverage = ParquetMinuteCandleSource((first, second)).load("TESTUSDT")
    assert len(merged) == 20
    assert coverage.gaps == 0


def test_boundary_revision_uses_later_complete_partition(tmp_path: Path) -> None:
    frame = make_one_minute(20)
    first, second = tmp_path / "first", tmp_path / "second"
    stale = frame.iloc[:11].copy()
    stale.loc[stale.index[-1], "close"] += 0.001
    _write(first, "TESTUSDT", stale)
    _write(second, "TESTUSDT", frame.iloc[10:])
    merged, coverage = ParquetMinuteCandleSource((first, second)).load("TESTUSDT")
    assert merged.loc[merged.open_time_ms.eq(frame.iloc[10].open_time_ms), "close"].item() == frame.iloc[10].close
    assert coverage.duplicate_rows_removed == 1


def test_interior_conflicting_duplicate_fails_closed(tmp_path: Path) -> None:
    frame = make_one_minute(20)
    first, second = tmp_path / "first", tmp_path / "second"
    stale = frame.iloc[:12].copy()
    stale.loc[stale.index[-2], "close"] += 0.001
    _write(first, "TESTUSDT", stale)
    _write(second, "TESTUSDT", frame.iloc[10:])
    with pytest.raises(FreshPipelineDataError, match="CONFLICTING_DUPLICATE"):
        ParquetMinuteCandleSource((first, second)).load("TESTUSDT")


def test_candle_gap_fails_closed(tmp_path: Path) -> None:
    frame = make_one_minute(20).drop(index=10)
    root = tmp_path / "only"
    _write(root, "TESTUSDT", frame)
    with pytest.raises(FreshPipelineDataError, match="CANDLE_GAPS"):
        ParquetMinuteCandleSource((root,)).load("TESTUSDT")


def test_full_fresh_snapshot_to_candidate_pipeline_is_deterministic(tmp_path: Path) -> None:
    frame = make_one_minute(100 * 1_440, start="2026-05-01T00:00:00Z")
    root = tmp_path / "candles"
    _write(root, "TESTUSDT", frame)
    decision = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    signal = FreshSignal("signal", decision.to_pydatetime(), "TESTUSDT", Side.SHORT, 100.0)
    pipeline = FreshSnapshotCandidatePipeline()
    first = pipeline.run((signal,), ParquetMinuteCandleSource((root,)))
    second = pipeline.run((signal,), ParquetMinuteCandleSource((root,)))
    assert len(first.snapshots) == 1
    assert len(first.candidates) == 5
    assert first.snapshots[0].snapshot_id == second.snapshots[0].snapshot_id
    assert first.candidates == second.candidates
    output = tmp_path / "output"
    persist_pipeline_result(first, output)
    bytes_before = {path.name: path.read_bytes() for path in output.iterdir()}
    persist_pipeline_result(second, output)
    assert bytes_before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_stale_candle_source_rejects_signal(tmp_path: Path) -> None:
    frame = make_one_minute(100 * 1_440, start="2026-05-01T00:00:00Z")
    root = tmp_path / "candles"
    _write(root, "TESTUSDT", frame)
    last_close = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    signal = FreshSignal(
        "stale", (last_close + pd.Timedelta(minutes=2)).to_pydatetime(),
        "TESTUSDT", Side.SHORT, 100.0,
    )
    result = FreshSnapshotCandidatePipeline().run((signal,), ParquetMinuteCandleSource((root,)))
    assert result.snapshots == ()
    assert result.rejected_signals[0][1].startswith("STALE_CANDLE_SOURCE")
