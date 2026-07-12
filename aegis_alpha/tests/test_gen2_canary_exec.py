#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_canary_exec as ex
import aegis_alpha.tools.gen2_canary_reconciliation as rec

CID = "gen2-20260711T202935Z"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "GEN2_SYSTEM_FREEZE.json"
    ex.core.CANARY_ROOT = core.CANARY_ROOT
    ex.core.FREEZE_PATH = core.FREEZE_PATH
    ex.load_account_snapshot_from_logs = lambda ts_repo=None: {"available_balance": 16.24, "source": "test"}
    core.FREEZE_PATH.write_text(json.dumps({
        "candidate_id": CID,
        "trrm_v2_sha256": "trrm",
        "eqm1_sha256": "eqm",
        "d3_dataset_sha256": "d3",
        "feature_hash": "features",
        "veto": {"threshold_full_dev_informational": 0.1},
    }))
    core.init_canary(CID)


def opportunity(**overrides):
    base = {
        "candidate_id": CID,
        "signal_id": "sig-1",
        "symbol": "ADAUSDT",
        "side": "SHORT",
        "primary_horizon": 12,
        "final_candle": True,
        "leverage": 5,
    }
    base.update(overrides)
    return base


def filters(symbol: str = "ADAUSDT", min_notional: float = 5.0) -> ex.SymbolFilters:
    return ex.SymbolFilters(symbol, min_notional=min_notional, step_size=1.0, tick_size=0.0001)


def test_order_id_is_deterministic() -> None:
    a = ex.deterministic_client_order_id(CID, "s", "ADAUSDT", "SHORT")
    b = ex.deterministic_client_order_id(CID, "s", "ADAUSDT", "SHORT")
    c = ex.deterministic_client_order_id(CID, "s2", "ADAUSDT", "SHORT")
    assert a == b
    assert a != c
    assert a.startswith("GEN2-")


def test_unarmed_and_phase_o_gates_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (False, "PHASE_O_NEW_ENTRIES_NOT_PAUSED")
        adapter = ex.CanaryExecutionAdapter(CID)
        ok, reason, _ = adapter.validate(opportunity(), filters(), 0.7, 16.24)
        assert (ok, reason) == (False, "PHASE_O_NEW_ENTRIES_NOT_PAUSED")
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        ok, reason, _ = adapter.validate(opportunity(), filters(), 0.7, 16.24)
        assert (ok, reason) == (False, "CANARY_UNARMED")


def test_valid_token_still_blocks_invalid_inputs_and_min_notional() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        core.create_arm_token(CID, 15.0, 72, ["ADAUSDT", "DOGEUSDT"], 5)
        adapter = ex.CanaryExecutionAdapter(CID)
        assert adapter.validate(opportunity(symbol="BTCUSDT"), filters(), 0.7, 16.24)[1] == "SYMBOL_NOT_ALLOWED"
        assert adapter.validate(opportunity(side="LONG"), filters(), 0.7, 16.24)[1] == "ONLY_SHORT_ALLOWED"
        assert adapter.validate(opportunity(primary_horizon=6), filters(), 0.7, 16.24)[1] == "PRIMARY_H12_REQUIRED"
        assert adapter.validate(opportunity(final_candle=False), filters(), 0.7, 16.24)[1] == "PARTIAL_CANDLE"
        ok, reason, sizing = adapter.validate(opportunity(), filters(min_notional=5000), 0.7, 16.24)
        assert ok is False
        assert reason == "MIN_NOTIONAL_NOT_MET"
        assert sizing["notional"] < 5000


def test_dry_run_never_submits_and_logs_attempt() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        core.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 5)
        adapter = ex.CanaryExecutionAdapter(CID)
        record = adapter.submit(opportunity(), filters(), 0.7, 16.24, dry_run=True)
        assert record["order_action"] == "NO_ORDER"
        assert record["enforcement_action"] == "NONE"
        assert adapter.orders_submitted == 0
        assert (core.canary_dir(CID) / "live_orders.jsonl").exists()


def test_bracket_failure_and_reconciliation_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        bm = ex.BracketManager(CID)
        ok = bm.confirm({"client_order_id": "a", "filled_qty": 1.0}, True, True, 2.0)
        assert ok["ok"] is True
        fail = bm.confirm({"client_order_id": "b", "filled_qty": 1.0}, True, False, 61.0)
        assert fail["ok"] is False
        assert core.kill_switch_engaged(CID) is True
        rec = ex.Reconciler(CID).reconcile({}, {"orphan_position": True, "exposure": 0})
        assert rec["status"] == "RECONCILIATION_FAIL_CLOSED"
        assert "ORPHAN_POSITION" in rec["incidents"]


def test_capital_feasibility_and_yaml_phase_o_parser() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ts = tmp / "binance-futures-bot-ts"
        ts.mkdir()
        (ts / "regime_config.live.yaml").write_text("aegis:\n  phase_o_short_live:\n    enabled: true\n    allow_orders: false\n")
        assert ex.phase_o_new_entries_paused(ts) == (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        feasible = ex.capital_feasibility("ADAUSDT", filters(), 0.7)
        assert feasible.min_executable_notional >= 5.0
        assert feasible.required_isolated_margin <= feasible.min_executable_notional
        assert feasible.decision in {"CANARY_CAPITAL_SUFFICIENT", "CANARY_CAPITAL_INSUFFICIENT"}


def test_dry_run_report_has_zero_orders_and_no_outcomes() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        report = ex.dry_run(CID, use_public=False)
        assert report["orders_submitted"] == 0
        assert report["FORWARD_OUTCOMES_NOT_EVALUATED"] is True
        assert report["armed"] is False
        daily = rec.build_daily_report(CID)
        assert daily["orders_submitted"] == 0
        assert daily["FORWARD_OUTCOMES_NOT_EVALUATED"] is True
        assert daily["brackets"]["rows"] >= 2


if __name__ == "__main__":
    test_order_id_is_deterministic()
    test_unarmed_and_phase_o_gates_fail_closed()
    test_valid_token_still_blocks_invalid_inputs_and_min_notional()
    test_dry_run_never_submits_and_logs_attempt()
    test_bracket_failure_and_reconciliation_fail_closed()
    test_capital_feasibility_and_yaml_phase_o_parser()
    test_dry_run_report_has_zero_orders_and_no_outcomes()
    print("test_gen2_canary_exec: OK")
