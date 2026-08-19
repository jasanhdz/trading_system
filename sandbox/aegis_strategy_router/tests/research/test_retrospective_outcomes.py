from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aegis_strategy_router.domain.types import Side
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder
from aegis_strategy_router.research.retrospective_falsification import reconstruct_outcome
from conftest import make_one_minute


SANDBOX = Path(__file__).resolve().parents[2]


def _snapshot_and_frame() -> tuple[object, pd.DataFrame]:
    frame = make_one_minute(10_000, start="2023-12-01T00:00:00Z")
    decision_index = 9_000
    decision_at = pd.to_datetime(
        frame.iloc[decision_index].open_time_ms, unit="ms", utc=True
    )
    reference = float(frame.iloc[decision_index - 1].close)
    builder = PrecomputedSnapshotBuilder()
    snapshot = builder.build(
        symbol="TESTUSDT",
        decision_at=decision_at.to_pydatetime(),
        built_at=decision_at.to_pydatetime(),
        reference_price=reference,
        one_minute=frame,
    )
    return snapshot, frame


def test_same_bar_barrier_ambiguity_is_adverse_first_for_both_sides() -> None:
    snapshot, frame = _snapshot_and_frame()
    start_ms = int(snapshot.decision_at.timestamp() * 1_000)
    row = frame.index[frame.open_time_ms.eq(start_ms)][0]
    frame.loc[row, "high"] = snapshot.reference_price + 10.0
    frame.loc[row, "low"] = snapshot.reference_price - 10.0
    for side in (Side.LONG, Side.SHORT):
        result = reconstruct_outcome(frame, snapshot, side)
        assert result["label"] == "ADVERSE_FIRST"
        assert result["gross_common_payoff_bps"] == -result["barrier_bps"]
        assert result["net_common_payoff_bps"] == result["gross_common_payoff_bps"] - 20.0


def test_symmetric_favorable_path_produces_symmetric_payoff() -> None:
    snapshot, long_frame = _snapshot_and_frame()
    short_frame = long_frame.copy()
    mask = long_frame.open_time_ms.ge(int(snapshot.decision_at.timestamp() * 1_000))
    reference = snapshot.reference_price
    long_frame.loc[mask, ["open", "high", "low", "close"]] = reference
    short_frame.loc[mask, ["open", "high", "low", "close"]] = reference
    first = long_frame.index[mask][0]
    long_frame.loc[first, ["high", "close"]] = reference + 10.0
    short_frame.loc[first, ["low", "close"]] = reference - 10.0
    long_result = reconstruct_outcome(long_frame, snapshot, Side.LONG)
    short_result = reconstruct_outcome(short_frame, snapshot, Side.SHORT)
    assert long_result["label"] == short_result["label"] == "FAVORABLE_FIRST"
    assert long_result["gross_common_payoff_bps"] == short_result["gross_common_payoff_bps"]


def test_retrospective_window_ends_before_first_excluded_holdout() -> None:
    config = json.loads(
        (SANDBOX / "config/retrospective_falsification_v1.json").read_text(encoding="utf-8")
    )
    assert pd.Timestamp(config["candidate_last_inclusive"]) < pd.Timestamp("2024-10-01T00:00:00Z")
    assert pd.Timestamp(config["source_end_exclusive"]) <= pd.Timestamp("2024-10-01T00:00:00Z")
