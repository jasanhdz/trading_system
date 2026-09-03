from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aegis_strategy_router.adapters.shared_market_data import (
    DecisionDerivedMarketDataError,
    SharedNeutralMinuteCandleSource,
)
from aegis_strategy_router.replay.fresh_pipeline import FreshPipelineDataError
from aegis_strategy_router.replay.general_market_pipeline import GeneralMarketCandidatePipeline
from conftest import make_one_minute


def _write(root: Path, symbol: str, frame: pd.DataFrame) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol}_1m.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_shared_market_source_runs_general_pipeline_without_aegis(tmp_path: Path) -> None:
    frame = make_one_minute(100 * 1_440, start="2026-05-01T00:00:00Z")
    root = tmp_path / "shared"
    path = _write(root, "TESTUSDT", frame)
    before = (path.stat().st_mtime_ns, path.read_bytes())
    end = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    result = GeneralMarketCandidatePipeline().run(
        symbols=("TESTUSDT",),
        start_at=(end - pd.Timedelta(minutes=15)).to_pydatetime(),
        end_at=end.to_pydatetime(),
        candle_source=SharedNeutralMinuteCandleSource((root,)),
    )
    assert result.snapshots
    assert all(item.signal_id is None and item.proposed_side is None for item in result.snapshots)
    assert {item.side.value for item in result.candidates} == {"LONG", "SHORT"}
    assert before == (path.stat().st_mtime_ns, path.read_bytes())


def test_shared_market_source_rejects_aegis_decision_columns(tmp_path: Path) -> None:
    frame = make_one_minute(10)
    frame["confidence"] = 0.9
    root = tmp_path / "unsafe"
    _write(root, "TESTUSDT", frame)
    with pytest.raises(DecisionDerivedMarketDataError, match="confidence"):
        SharedNeutralMinuteCandleSource((root,)).load("TESTUSDT")


def test_shared_market_source_deduplicates_exact_partitions_idempotently(tmp_path: Path) -> None:
    frame = make_one_minute(100)
    first, second = tmp_path / "first", tmp_path / "second"
    _write(first, "TESTUSDT", frame)
    _write(second, "TESTUSDT", frame)
    source = SharedNeutralMinuteCandleSource((first, second))
    one, audit_one = source.load_with_audit("TESTUSDT")
    two, audit_two = source.load_with_audit("TESTUSDT")
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == len(frame)
    assert audit_one == audit_two
    assert audit_one.coverage.duplicate_rows_removed == len(frame)


def test_shared_market_source_fails_closed_on_gap_and_staleness(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    frame = make_one_minute(120)
    _write(root, "GAPUSDT", frame.drop(index=60).reset_index(drop=True))
    with pytest.raises(FreshPipelineDataError, match="CANDLE_GAPS"):
        SharedNeutralMinuteCandleSource((root,)).load("GAPUSDT")

    _write(root, "STALEUSDT", frame)
    last_close = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    with pytest.raises(FreshPipelineDataError, match="STALE_SHARED_MARKET_DATA"):
        SharedNeutralMinuteCandleSource((root,)).assert_fresh_for(
            "STALEUSDT", (last_close + pd.Timedelta(minutes=2)).to_pydatetime()
        )


def test_future_candles_do_not_change_historical_snapshot(tmp_path: Path) -> None:
    full = make_one_minute(100 * 1_440 + 30, start="2026-05-01T00:00:00Z")
    historical = full.iloc[:-30].reset_index(drop=True)
    root = tmp_path / "shared"
    path = _write(root, "TESTUSDT", historical)
    boundary = pd.to_datetime(historical.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    start = boundary - pd.Timedelta(minutes=15)
    first = GeneralMarketCandidatePipeline().run(
        symbols=("TESTUSDT",), start_at=start.to_pydatetime(), end_at=boundary.to_pydatetime(),
        candle_source=SharedNeutralMinuteCandleSource((root,)),
    )
    full.to_parquet(path, index=False)
    second = GeneralMarketCandidatePipeline().run(
        symbols=("TESTUSDT",), start_at=start.to_pydatetime(), end_at=boundary.to_pydatetime(),
        candle_source=SharedNeutralMinuteCandleSource((root,)),
    )
    assert [item.canonical_bytes() for item in first.snapshots] == [
        item.canonical_bytes() for item in second.snapshots
    ]
    assert [item.to_primitive() for item in first.candidates] == [
        item.to_primitive() for item in second.candidates
    ]
