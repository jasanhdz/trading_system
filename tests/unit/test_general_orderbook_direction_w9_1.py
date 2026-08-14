from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.research.general_orderbook_direction_w9_1 import (
    CausalBook,
    _first_barrier,
    anchor_grid,
    requested_book_times,
    stable_episode_id,
)
from scripts.evaluate_general_orderbook_direction_w9_1 import (
    benjamini_hochberg,
    decisions,
    utilities,
    validate_dataset,
)


def _row(snapshot: bool, side: str, price: float, amount: float) -> dict[str, str]:
    return {
        "exchange": "binance-futures",
        "symbol": "ADAUSDT",
        "timestamp": "100",
        "local_timestamp": "101",
        "is_snapshot": str(snapshot).lower(),
        "side": side,
        "price": str(price),
        "amount": str(amount),
    }


def test_causal_book_tracks_replenishment_and_depletion() -> None:
    book = CausalBook()
    book.apply([_row(True, "bid", 99, 2), _row(True, "ask", 101, 3)])
    book.apply([_row(False, "bid", 99, 5), _row(False, "ask", 101, 1)])
    snapshot = book.snapshot()
    assert snapshot["snapshot_generation"] == 1
    assert snapshot["bid_replenishment"] == pytest.approx(3)
    assert snapshot["ask_depletion"] == pytest.approx(2)
    assert snapshot["obi_l1"] == pytest.approx((5 - 1) / 6)


def test_causal_book_rejects_update_before_snapshot() -> None:
    with pytest.raises(ValueError, match="UPDATE_BEFORE_SNAPSHOT"):
        CausalBook().apply([_row(False, "bid", 99, 2)])


def test_anchor_grid_is_non_overlapping_for_sixty_second_target() -> None:
    anchors = anchor_grid(0, 86_400_000_000)
    assert len(anchors) == 720
    assert np.diff(anchors).min() == 120_000_000


def test_requested_times_are_causal() -> None:
    anchors = np.array([10_000_000], dtype=np.int64)
    times = requested_book_times(anchors)
    assert times.max() == anchors[0]
    assert times.min() == anchors[0] - 5_000_000


def test_first_barrier_uses_capture_order_and_latency() -> None:
    times = np.array([0, 100_000, 200_000, 300_000], dtype=np.int64)
    mids = np.array([100.0, 100.3, 99.7, 100.0])
    label, terminal, mfe, mae = _first_barrier(
        times, mids, 0, latency_ms=0, barrier_bps=25, horizon_seconds=1
    )
    assert label == "UP_FIRST"
    assert terminal == pytest.approx(0)
    assert mfe == pytest.approx(30)
    assert mae == pytest.approx(-30)


def test_episode_identity_is_stable_and_symbol_specific() -> None:
    assert stable_episode_id("ADAUSDT", 1) == stable_episode_id("ADAUSDT", 1)
    assert stable_episode_id("ADAUSDT", 1) != stable_episode_id("BTCUSDT", 1)


def test_decision_requires_direction_to_beat_neither_and_threshold() -> None:
    probabilities = np.array([[0.70, 0.10, 0.20], [0.40, 0.10, 0.50], [0.20, 0.65, 0.15]])
    assert decisions(probabilities, 0.60).tolist() == ["LONG", "SKIP", "SHORT"]


def test_economic_utility_is_symmetric_and_charges_cost() -> None:
    frame = pd.DataFrame({
        "target__b25_h60_l0_label": ["UP_FIRST", "DOWN_FIRST", "NEITHER"],
        "target__b25_h60_l0_terminal_bps": [25.0, -25.0, 5.0],
    })
    result = utilities(frame, np.array(["LONG", "LONG", "SHORT"]), latency_ms=0, cost_bps=14.0)
    assert result.tolist() == pytest.approx([11.0, -39.0, -19.0])


def test_benjamini_hochberg_controls_family() -> None:
    adjusted = benjamini_hochberg([0.001, 0.02, 0.2])
    assert adjusted[0] == pytest.approx(0.003)
    assert adjusted[1] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.2)


def _integrity_fixture() -> tuple[pd.DataFrame, dict, dict]:
    frame = pd.DataFrame({
        "orderbook_episode_id": [f"episode-{index}" for index in range(720)],
        "symbol": "ADAUSDT",
        "date": "2025-01-01",
        "anchor_timestamp_us": np.arange(720, dtype=np.int64) * 120_000_000,
        "static__spread_bps": 1.0,
    })
    manifest = {
        "all_partitions_pass": True,
        "completed_parts": 1,
        "expected_parts": 1,
        "parts": [{"l2_quote_mid_difference_bps_p99": 0.0}],
    }
    config = {
        "partitions": {
            "train_months": ["2025-01"],
            "validation_months": [],
            "final_holdout": {"month": "2025-02"},
        },
        "anchors": {"interval_seconds": 120},
    }
    return frame, manifest, config


def test_dataset_integrity_accepts_preregistered_nonoverlapping_episodes() -> None:
    frame, manifest, config = _integrity_fixture()
    audit = validate_dataset(frame, manifest, config)
    assert all(audit["checks"].values())
    assert audit["unique_episode_ids"] == 720


def test_dataset_integrity_fails_closed_on_duplicate_episode() -> None:
    frame, manifest, config = _integrity_fixture()
    frame.loc[1, "orderbook_episode_id"] = frame.loc[0, "orderbook_episode_id"]
    with pytest.raises(RuntimeError, match="DATASET_INTEGRITY_FAILURE"):
        validate_dataset(frame, manifest, config)
