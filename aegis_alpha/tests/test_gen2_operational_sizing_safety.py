#!/usr/bin/env python3
"""Safety gauntlet for the UNIFIED operational contract sizing (supersedes the
rejected Capital Contract V2 tests): liquidation buffer, stop<liq ordering,
fees inside margin, real-equity floor kill, and mode-bound arm token."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_operational_contract as oc

CID = "gen2-test"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)


def test_sizing_carries_liquidation_and_fee_safety_in_both_modes() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        for mode, lev in (("safe", 3), ("experimental", 10)):
            contract = oc.write_contract(CID, mode, 200.0, force=True)
            s = oc.compute_sizing(contract, price=0.70, available_balance=200.0, step_size=1.0, min_notional=5.0)
            assert s["decision"] == "SIZED" and s["leverage"] == lev
            assert s["stop_price"] < s["liquidation_price"], "stop must sit inside liquidation"
            assert s["liquidation_buffer_pct"] >= oc.MINIMUM_LIQUIDATION_BUFFER_PCT
            assert s["estimated_fees"] > 0
            assert s["per_stop_loss"] >= s["notional"] * oc.STOP_DISTANCE_PCT  # fees included


def test_insufficient_liquidation_buffer_blocks_trade() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        contract = oc.write_contract(CID, "experimental", 200.0)
        s = oc.compute_sizing(contract, price=0.70, available_balance=200.0, step_size=1.0, min_notional=5.0,
                              minimum_liquidation_buffer_pct=0.50)
        assert s["decision"] == "NO_TRADE" and s["reason"] == "LIQUIDATION_BUFFER_INSUFFICIENT"


def test_fees_count_against_available_balance() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        contract = oc.write_contract(CID, "safe", 200.0)
        # margin alone fits exactly, margin+fees does not
        margin_needed = 25.0 / 3
        s = oc.compute_sizing(contract, price=25.0, available_balance=margin_needed, step_size=1.0, min_notional=5.0)
        assert s["decision"] == "NO_TRADE" and s["reason"] == "INSUFFICIENT_BALANCE"


def test_equity_floor_breach_engages_kill_switch() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        oc.write_contract(CID, "experimental", 100.0)  # floor = 75
        ok, reason, meta = oc.check_equity_floor(CID, 80.0)
        assert ok is True and core.kill_switch_engaged(CID) is False
        ok2, reason2, meta2 = oc.check_equity_floor(CID, 74.9)
        assert (ok2, reason2) == (False, "EQUITY_FLOOR_REACHED")
        assert meta2["equity_floor"] == 75.0
        assert core.kill_switch_engaged(CID) is True


def test_incoherent_manual_contract_is_refused() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        # V2-style aggression (50% fraction @20x) is incoherent vs caps -> refused
        old = dict(oc.MODES["EXPERIMENTAL"])
        try:
            oc.MODES["EXPERIMENTAL"].update({"balance_fraction": 0.50, "default_leverage": 20, "max_leverage": 20})
            try:
                oc.write_contract(CID, "experimental", 16.24, force=True)
                raise AssertionError("V2-style contract must be refused as incoherent")
            except ValueError as err:
                assert "CONTRACT_INCOHERENT" in str(err)
        finally:
            oc.MODES["EXPERIMENTAL"] = old


def test_token_is_mode_bound() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        oc.write_contract(CID, "experimental", 200.0)
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        assert oc.verify_arm_token(CID)[1] == "TOKEN_VALID"
        oc.write_contract(CID, "safe", 200.0, force=True)
        assert oc.verify_arm_token(CID)[1] in {"TOKEN_MODE_MISMATCH", "TOKEN_CONTRACT_HASH_MISMATCH"}


if __name__ == "__main__":
    test_sizing_carries_liquidation_and_fee_safety_in_both_modes()
    test_insufficient_liquidation_buffer_blocks_trade()
    test_fees_count_against_available_balance()
    test_equity_floor_breach_engages_kill_switch()
    test_incoherent_manual_contract_is_refused()
    test_token_is_mode_bound()
    print("test_gen2_operational_sizing_safety: OK")
