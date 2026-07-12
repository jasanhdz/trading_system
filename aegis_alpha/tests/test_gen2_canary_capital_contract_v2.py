#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_canary_exec as ex

CID = "gen2-20260711T202935Z"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "GEN2_SYSTEM_FREEZE.json"
    ex.core.CANARY_ROOT = core.CANARY_ROOT
    ex.core.FREEZE_PATH = core.FREEZE_PATH
    core.FREEZE_PATH.write_text(json.dumps({
        "candidate_id": CID,
        "trrm_v2_sha256": "trrm",
        "eqm1_sha256": "eqm",
        "d3_dataset_sha256": "d3",
        "feature_hash": "features",
        "veto": {"threshold_full_dev_informational": 0.1},
    }))
    core.init_canary(CID)
    ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")


def filt(max_leverage: int = 20, min_notional: float = 5.0) -> ex.SymbolFilters:
    return ex.SymbolFilters("ADAUSDT", min_notional=min_notional, step_size=1.0, tick_size=0.0001, min_qty=1.0, max_leverage=max_leverage)


def opp(**kwargs):
    row = {"candidate_id": CID, "signal_id": "sig", "symbol": "ADAUSDT", "side": "SHORT", "primary_horizon": 12,
           "final_candle": True, "leverage": 20, "margin_type": "ISOLATED"}
    row.update(kwargs)
    return row


def test_margin_fraction_examples_and_rounding() -> None:
    for balance, expected_margin in [(16.0, 8.0), (8.0, 4.0), (4.0, 2.0)]:
        s = ex.compute_sizing_v2("ADAUSDT", filt(), 0.16, balance, 20)
        assert s.allocated_margin == expected_margin
        assert s.required_isolated_margin <= expected_margin
        assert s.actual_notional <= s.target_notional + 1e-9


def test_leverage_policy_and_liquidation_retry() -> None:
    assert ex.compute_sizing_v2("ADAUSDT", filt(), 0.16, 16.0, 15).leverage == 15
    assert ex.compute_sizing_v2("ADAUSDT", filt(), 0.16, 16.0, 20).leverage == 20
    assert ex.compute_sizing_v2("ADAUSDT", filt(), 0.16, 16.0, 10).reason == "LEVERAGE_NOT_ALLOWED"
    blocked = ex.compute_sizing_v2("ADAUSDT", filt(), 0.16, 16.0, 20, minimum_liquidation_buffer_pct=0.06)
    assert blocked.reason == "LIQUIDATION_BUFFER_INSUFFICIENT"
    retry = ex.select_sizing_v2("ADAUSDT", filt(), 0.16, 16.0)
    assert retry.leverage in {15, 20}
    assert retry.stop_price == ex.stop_price_for_short(0.16)
    assert retry.stop_price < retry.liquidation_price


def test_equity_floor_persists_and_engages_kill_switch() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        manifest = ex.init_operational_manifest_v2(CID, {"account_equity": 16.0, "available_balance": 16.0})
        assert manifest["initial_canary_equity"] == 16.0
        assert manifest["equity_floor"] == 8.0
        assert ex.init_operational_manifest_v2(CID, {"account_equity": 30.0})["initial_canary_equity"] == 16.0
        ok, reason, meta = ex.check_equity_floor(CID, 8.0)
        assert ok is False
        assert reason == "CANARY_EQUITY_FLOOR_REACHED"
        assert meta["equity_floor"] == 8.0
        assert core.kill_switch_engaged(CID) is True


def test_token_v2_and_adapter_gates() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.init_operational_manifest_v2(CID, {"account_equity": 16.0, "available_balance": 16.0})
        adapter = ex.CanaryExecutionAdapter(CID)
        assert adapter.validate(opp(), filt(), 0.16, 16.0, 16.0)[1] == "CANARY_UNARMED"
        core.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 5)
        assert ex.verify_arm_token_v2(CID)[1] == "CANARY_UNARMED"
        token = ex.create_arm_token_v2(CID, 24, 16.0)
        (core.canary_dir(CID) / "ARM_TOKEN_V2.json").write_text(json.dumps(token))
        assert ex.verify_arm_token_v2(CID)[1] == "TOKEN_V2_VALID"
        assert adapter.validate(opp(margin_type="CROSS"), filt(), 0.16, 16.0, 16.0)[1] == "ISOLATED_MARGIN_NOT_CONFIRMED"
        assert adapter.validate(opp(leverage=10), filt(), 0.16, 16.0, 16.0)[1] == "LEVERAGE_NOT_ALLOWED"
        ok, reason, sizing = adapter.validate(opp(), filt(), 0.16, 16.0, 16.0)
        assert (ok, reason) == (True, "READY")
        assert sizing["allocated_margin"] == 8.0
        assert sizing["leverage"] == 20


if __name__ == "__main__":
    test_margin_fraction_examples_and_rounding()
    test_leverage_policy_and_liquidation_retry()
    test_equity_floor_persists_and_engages_kill_switch()
    test_token_v2_and_adapter_gates()
    print("test_gen2_canary_capital_contract_v2: OK")
