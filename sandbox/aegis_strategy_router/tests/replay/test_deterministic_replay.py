import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from aegis_strategy_router.domain.types import Side
from aegis_strategy_router.replay.snapshot_builder import DeterministicSnapshotBuilder, ReplayManifest
from conftest import make_one_minute


def test_same_input_produces_byte_equivalent_snapshot(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    kwargs = dict(
        symbol="SOLUSDT",
        decision_at=decision.to_pydatetime(),
        built_at=decision.to_pydatetime(),
        reference_price=145.25,
        one_minute=one_minute,
        proposed_side=Side.LONG,
        signal_id="signal-42",
        source_versions={"fixture": "v1"},
    )
    first = builder.build(**kwargs)
    second = builder.build(**kwargs)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert ReplayManifest.from_snapshot(first) == ReplayManifest.from_snapshot(second)


def test_input_row_order_does_not_change_replay(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    ordered = builder.build(
        symbol="ADAUSDT", decision_at=decision.to_pydatetime(), reference_price=0.55,
        one_minute=one_minute,
    )
    shuffled = builder.build(
        symbol="ADAUSDT", decision_at=decision.to_pydatetime(), reference_price=0.55,
        one_minute=one_minute.sample(frac=1.0, random_state=7),
    )
    assert ordered.canonical_bytes() == shuffled.canonical_bytes()


def test_golden_replay_fixture_is_stable() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "replay_case_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    source = make_one_minute(fixture["rows"], start=fixture["start"])
    decision = pd.Timestamp(fixture["decision_at"]).to_pydatetime()
    snapshot = DeterministicSnapshotBuilder().build(
        symbol=fixture["symbol"],
        decision_at=decision,
        built_at=decision,
        reference_price=fixture["reference_price"],
        one_minute=source,
        proposed_side=Side(fixture["side"]),
        signal_id=fixture["signal_id"],
        source_versions={"fixture": fixture["generator"]},
    )
    manifest = ReplayManifest.from_snapshot(snapshot)
    assert snapshot.snapshot_id == fixture["expected_snapshot_id"]
    assert manifest.sha256 == fixture["expected_canonical_sha256"]


def test_signal_conditioned_side_is_immutable(one_minute: pd.DataFrame) -> None:
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    snapshot = DeterministicSnapshotBuilder().build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=one_minute, proposed_side=Side.SHORT, signal_id="frozen-side",
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.proposed_side = Side.LONG  # type: ignore[misc]
