from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from aegis_strategy_router.candidates.base import GENERATOR_VERSION
from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateStatus,
    CandidateSubstate,
    Strategy,
)
from aegis_strategy_router.domain.types import Side
from aegis_strategy_router.replay.fresh_pipeline import ParquetMinuteCandleSource
from aegis_strategy_router.replay.general_market_pipeline import (
    GeneralMarketCandidatePipeline,
    merge_general_market_results,
    persist_general_market_result,
    select_independent_episodes,
)
from conftest import make_one_minute


def _write(root: Path, symbol: str, frame: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / f"{symbol}_1m.parquet", index=False)


def _candidate(at: datetime, *, setup_id: str | None = None) -> CandidateEvaluation:
    snapshot_id = f"snapshot:{at.isoformat()}"
    metadata = {"symbol": "TESTUSDT"}
    if setup_id:
        metadata["setup_episode_id"] = setup_id
    return CandidateEvaluation.create(
        snapshot_id=snapshot_id,
        signal_episode_id=None,
        strategy=Strategy.TREND_CONTINUATION,
        side=Side.LONG,
        decision_at=at,
        status=CandidateStatus.ELIGIBLE,
        substate=CandidateSubstate.TREND_CONFIRMED,
        reason_codes=("FIXTURE",),
        rules=(),
        frozen_gaps=(),
        generator_version=GENERATOR_VERSION,
        metadata=metadata,
    )


def test_independent_episode_selection_deduplicates_setup_and_60m_overlap() -> None:
    start = datetime(2026, 8, 18, tzinfo=timezone.utc)
    values = (
        _candidate(start, setup_id="setup-a"),
        _candidate(start + timedelta(minutes=15), setup_id="setup-a"),
        _candidate(start + timedelta(minutes=30), setup_id="setup-b"),
        _candidate(start + timedelta(minutes=60), setup_id="setup-c"),
    )
    selected, suppressed = select_independent_episodes(values)
    assert [item.decision_at for item in selected] == [start, start + timedelta(minutes=60)]
    assert {reason for _, reason in suppressed} == {
        "DUPLICATE_SETUP_EPISODE", "TEMPORAL_OVERLAP_60M",
    }


def test_general_market_replay_is_side_neutral_symmetric_and_deterministic(tmp_path: Path) -> None:
    frame = make_one_minute(100 * 1_440, start="2026-05-01T00:00:00Z")
    root = tmp_path / "candles"
    _write(root, "TESTUSDT", frame)
    final_close = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    start = final_close.floor("15min") - pd.Timedelta(minutes=15)
    source = ParquetMinuteCandleSource((root,))
    pipeline = GeneralMarketCandidatePipeline()
    first = pipeline.run(
        symbols=("TESTUSDT",), start_at=start.to_pydatetime(),
        end_at=final_close.to_pydatetime(), candle_source=source,
    )
    second = pipeline.run(
        symbols=("TESTUSDT",), start_at=start.to_pydatetime(),
        end_at=final_close.to_pydatetime(), candle_source=source,
    )
    assert first == second
    assert len(first.snapshots) == 2
    assert all(snapshot.proposed_side is None and snapshot.signal_id is None for snapshot in first.snapshots)
    assert len(first.candidates) == 20
    assert {candidate.side for candidate in first.candidates} == {Side.LONG, Side.SHORT}
    assert all(candidate.signal_episode_id is None for candidate in first.candidates)
    assert all(not candidate.frozen_gaps for candidate in first.candidates)
    manifest = first.manifest()
    assert manifest["initial_experiment_mode"] == "INDEPENDENT_STRATEGY_DISCOVERY"
    assert manifest["aegis_signals_loaded"] is False
    assert manifest["outcomes_loaded"] is False
    assert manifest["edge_validation_performed"] is False

    output = tmp_path / "output"
    persist_general_market_result(first, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    persist_general_market_result(second, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_symbol_partition_merge_matches_serial_replay(tmp_path: Path) -> None:
    frame = make_one_minute(100 * 1_440, start="2026-05-01T00:00:00Z")
    root = tmp_path / "candles"
    _write(root, "AAAUSDT", frame)
    _write(root, "BBBUSDT", frame)
    final_close = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
    start = final_close.floor("15min") - pd.Timedelta(minutes=15)
    source = ParquetMinuteCandleSource((root,))
    serial = GeneralMarketCandidatePipeline().run(
        symbols=("AAAUSDT", "BBBUSDT"), start_at=start.to_pydatetime(),
        end_at=final_close.to_pydatetime(), candle_source=source,
    )
    partitioned = merge_general_market_results(
        GeneralMarketCandidatePipeline().run(
            symbols=(symbol,), start_at=start.to_pydatetime(),
            end_at=final_close.to_pydatetime(), candle_source=source,
        )
        for symbol in ("AAAUSDT", "BBBUSDT")
    )
    assert [item.canonical_bytes() for item in partitioned.snapshots] == [
        item.canonical_bytes() for item in serial.snapshots
    ]
    assert [item.to_primitive() for item in partitioned.candidates] == [
        item.to_primitive() for item in serial.candidates
    ]
    assert partitioned.manifest() == serial.manifest()
