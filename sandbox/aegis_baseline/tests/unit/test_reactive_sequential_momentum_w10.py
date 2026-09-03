from __future__ import annotations

import numpy as np
import pandas as pd

from aegis.research.reactive_sequential_momentum_w10 import (
    decision_grid,
    episode_anchors,
    required_book_times,
    stable_momentum_episode_id,
)
from scripts.evaluate_reactive_sequential_momentum_w10 import Policy, simulate_policy


def test_episode_grid_is_non_overlapping() -> None:
    anchors = episode_anchors(0, 86_400_000_000)
    assert len(anchors) == 287
    assert np.diff(anchors).min() == 300_000_000


def test_decision_grid_has_twenty_five_causal_states_per_episode() -> None:
    decisions = decision_grid(np.array([300_000_000], dtype=np.int64))
    assert len(decisions) == 25
    assert decisions[0] == 300_000_000
    assert decisions[-1] == 420_000_000


def test_required_book_times_never_exceed_decision() -> None:
    decisions = np.array([30_000_000], dtype=np.int64)
    requested = required_book_times(decisions)
    assert requested.tolist() == [10_000_000, 20_000_000, 25_000_000, 30_000_000]


def test_episode_identity_is_stable_and_symbol_specific() -> None:
    assert stable_momentum_episode_id("BTCUSDT", 1) == stable_momentum_episode_id("BTCUSDT", 1)
    assert stable_momentum_episode_id("BTCUSDT", 1) != stable_momentum_episode_id("ETHUSDT", 1)


def test_state_machine_requires_exit_and_cooldown_before_opposite_entry() -> None:
    rows = []
    for step in range(25):
        price = 100.0 + step * 0.01
        rows.append({
            "momentum_episode_id": "episode",
            "symbol": "BTCUSDT",
            "date": "2025-01-01",
            "decision_timestamp_us": step * 5_000_000,
            "execution_mid_l0": price,
            "book__spread_bps": 1.0,
            "path__interval_high": price + 0.01,
            "path__interval_low": price - 0.01,
        })
    frame = pd.DataFrame(rows)
    probabilities = np.tile(np.array([0.10, 0.10, 0.80]), (25, 1))
    probabilities[1:4] = [0.80, 0.10, 0.10]
    probabilities[4:8] = [0.10, 0.80, 0.10]
    probabilities[12:16] = [0.10, 0.80, 0.10]
    trades, transitions = simulate_policy(
        frame,
        probabilities,
        Policy(0.75, 2, 0.45, 2, 20, 2),
        latency_ms=0,
        cost_bps=14.0,
    )
    assert trades["side"].tolist() == ["LONG", "SHORT"]
    assert trades.iloc[1]["entry_timestamp_us"] >= trades.iloc[0]["exit_timestamp_us"] + 20_000_000
    assert transitions["flip_attempts"] > 0
