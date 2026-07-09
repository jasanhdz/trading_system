#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.simulate_loss_cap_autopause_a import run_simulation, simulate_rules, synthetic_fixture_trades


def test_uses_fixture_without_trades() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_simulation(argparse.Namespace(out_dir=td, trades_csv="", grid=True))
        assert result["status"] == "USING_SYNTHETIC_FIXTURE_FOR_MECHANICS"
        assert Path(result["outputs"]["json"]).exists()


def test_loss_cap_reduces_max_loss() -> None:
    trades = synthetic_fixture_trades()
    out = simulate_rules(trades, {"loss_cap_multiplier": 2.0})
    assert out["simulated_max_loss"] > out["original_max_loss"]


def test_consecutive_losses_pause_blocks() -> None:
    trades = synthetic_fixture_trades()
    out = simulate_rules(trades, {"consecutive_losses": 2})
    assert out["trades_blocked_after_pause"] > 0


def test_daily_pause_blocks() -> None:
    trades = synthetic_fixture_trades()
    out = simulate_rules(trades, {"daily_loss_cap": 20.0})
    assert out["trades_blocked_after_pause"] > 0


if __name__ == "__main__":
    test_uses_fixture_without_trades()
    test_loss_cap_reduces_max_loss()
    test_consecutive_losses_pause_blocks()
    test_daily_pause_blocks()
    print("test_simulate_loss_cap_autopause_a: OK")
