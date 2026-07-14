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
import aegis_alpha.tools.gen2_operational_contract as oc

CID = "gen2-20260711T202935Z"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "GEN2_SYSTEM_FREEZE.json"
    ex.core.CANARY_ROOT = core.CANARY_ROOT
    ex.core.FREEZE_PATH = core.FREEZE_PATH
    ex.load_account_snapshot_from_logs = lambda ts_repo=None: {"available_balance": 116.24, "account_equity": 116.24, "source": "test"}
    core.FREEZE_PATH.write_text(json.dumps({
        "candidate_id": CID,
        "trrm_v2_sha256": "trrm",
        "eqm1_sha256": "eqm",
        "d3_dataset_sha256": "d3",
        "feature_hash": "features",
        "veto": {"threshold_full_dev_informational": 0.1},
    }))
    core.init_canary(CID)


def filters(symbol: str = "ADAUSDT", min_notional: float = 5.0) -> ex.SymbolFilters:
    return ex.SymbolFilters(symbol, min_notional=min_notional, step_size=1.0, tick_size=0.0001, min_qty=1.0)


def test_order_id_is_deterministic() -> None:
    a = ex.deterministic_client_order_id(CID, "s", "ADAUSDT", "SHORT")
    b = ex.deterministic_client_order_id(CID, "s", "ADAUSDT", "SHORT")
    c = ex.deterministic_client_order_id(CID, "s2", "ADAUSDT", "SHORT")
    assert a == b
    assert a != c
    assert a.startswith("GEN2-")


def test_module_has_no_order_submission_surface() -> None:
    assert ex.PYTHON_SUBMITS_ORDERS is False
    src = Path(ex.__file__).read_text(encoding="utf-8")
    for endpoint in ("/fapi/v1/order", "futuresOrder", "newOrder"):
        assert endpoint not in src, f"order endpoint {endpoint} must not exist in the python side"


def test_bracket_failure_and_reconciliation_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        bm = ex.BracketManager(CID)
        ok = bm.confirm({"client_order_id": "a", "filled_qty": 1.0}, True, True, 2.0)
        assert ok["ok"] is True
        fail = bm.confirm({"client_order_id": "b", "filled_qty": 1.0}, True, False, 61.0)
        assert fail["ok"] is False
        assert core.kill_switch_engaged(CID) is True
        r = ex.Reconciler(CID).reconcile({}, {"orphan_position": True, "exposure": 0})
        assert r["status"] == "RECONCILIATION_FAIL_CLOSED"
        assert "ORPHAN_POSITION" in r["incidents"]


class FakePrivateAdapter:
    def __init__(self, open_orders: dict | None = None, positions: dict | None = None, ready: bool = True,
                 balance: float = 100.0) -> None:
        self._open_orders = open_orders or {}
        self._positions = positions or {}
        self.available = ready
        self._balance = balance

    def private_snapshot(self, symbols=("ADAUSDT",)):
        return {"private_read_ready": True, "open_orders": self._open_orders, "positions": self._positions,
                "available_balance": self._balance, "account_equity": self._balance}


def test_second_opinion_flags_orphans_and_kills_on_exposure() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        # clean exchange -> reconciled
        r = ex.second_opinion_reconciliation(CID, FakePrivateAdapter())
        assert r["status"] == "RECONCILED"
        # orphan GEN2 order on exchange not present locally, plus exposure -> kill
        adapter = FakePrivateAdapter(
            open_orders={"ADAUSDT": [{"clientOrderId": "GEN2-deadbeef"}]},
            positions={"ADAUSDT": [{"positionAmt": "-12"}]},
        )
        r2 = ex.second_opinion_reconciliation(CID, adapter)
        assert r2["status"] == "RECONCILIATION_FAIL_CLOSED"
        assert "ORPHAN_ORDER_ON_EXCHANGE" in r2["incidents"]
        assert core.kill_switch_engaged(CID) is True


def test_second_opinion_without_credentials_is_recorded_not_silent() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        r = ex.second_opinion_reconciliation(CID, FakePrivateAdapter(ready=False))
        assert r["status"] == "PRIVATE_READ_UNAVAILABLE"
        rows = (core.canary_dir(CID) / "reconciliations.jsonl").read_text().splitlines()
        assert len(rows) == 1


def test_second_opinion_detects_duplicated_fills_and_missing_brackets() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        cdir = core.canary_dir(CID)
        with (cdir / "fills.jsonl").open("a") as f:
            f.write(json.dumps({"type": "FILL", "client_order_id": "GEN2-dup", "ts_sequence": 1}) + "\n")
            f.write(json.dumps({"type": "FILL", "client_order_id": "GEN2-dup", "ts_sequence": 2}) + "\n")
            f.write(json.dumps({"type": "FILL", "client_order_id": "GEN2-nobracket", "ts_sequence": 3}) + "\n")
        with (cdir / "brackets.jsonl").open("a") as f:
            f.write(json.dumps({"ok": True, "client_order_id": "GEN2-dup"}) + "\n")
        r = ex.second_opinion_reconciliation(CID, FakePrivateAdapter())
        assert "DUPLICATED_FILLS" in r["incidents"]
        assert "FILL_WITHOUT_CONFIRMED_BRACKET" in r["incidents"]
        assert r["duplicated_fills"] == ["GEN2-dup"]
        assert r["missing_brackets"] == ["GEN2-nobracket"]
        assert core.kill_switch_engaged(CID) is False  # incidents but no exposure -> no kill


def test_second_opinion_detects_leverage_margin_and_balance_drift() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        oc.write_contract(CID, "safe", 116.24)  # max_leverage 5
        adapter = FakePrivateAdapter(
            positions={"ADAUSDT": [{"symbol": "ADAUSDT", "positionAmt": "-12", "leverage": "20", "marginType": "cross"}]},
            balance=100.0,
        )
        r = ex.second_opinion_reconciliation(CID, adapter, bridge_status={"available_balance": 80.0})
        assert "LEVERAGE_ABOVE_CONTRACT" in r["incidents"]
        assert "MARGIN_MODE_NOT_ISOLATED" in r["incidents"]
        assert "BALANCE_MISMATCH_BRIDGE_VS_EXCHANGE" in r["incidents"]
        assert "POSITION_WITHOUT_LOCAL_ORDER" in r["incidents"]
        assert core.kill_switch_engaged(CID) is True  # incidents with exposure -> kill


def test_dry_run_uses_unified_contract_and_submits_nothing() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        ex.phase_o_new_entries_paused = lambda ts_repo=None: (True, "PHASE_O_NEW_ENTRIES_PAUSED")
        # without contract: fail-closed, reported, zero orders
        report = ex.dry_run(CID, use_public=False)
        assert report["orders_submitted"] == 0
        assert report["armed"] is False
        assert report["operational_mode"] is None
        assert report["risk_gate"] == "OPERATIONAL_CONTRACT_MISSING"
        # with SAFE contract: sizing preview comes from THE contract
        oc.write_contract(CID, "safe", 116.24)
        report2 = ex.dry_run(CID, use_public=False)
        assert report2["operational_mode"] == "SAFE"
        assert report2["orders_submitted"] == 0
        assert report2["FORWARD_OUTCOMES_NOT_EVALUATED"] is True
        ada = report2["sizing_preview"]["ADAUSDT"]
        assert ada["leverage"] == 3 and abs(ada["notional"] - 25.0) <= 0.8
        daily = rec.build_daily_report(CID)
        assert daily["orders_submitted"] == 0
        assert daily["FORWARD_OUTCOMES_NOT_EVALUATED"] is True
        assert daily["brackets"]["rows"] >= 4


def test_phase_o_yaml_parser_single_source() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ts = tmp / "binance-futures-bot-ts"
        ts.mkdir()
        (ts / "regime_config.live.yaml").write_text("aegis:\n  phase_o_short_live:\n    enabled: true\n    allow_orders: false\n")
        assert core.phase_o_new_entries_paused(ts) == (True, "PHASE_O_NEW_ENTRIES_PAUSED")


if __name__ == "__main__":
    test_order_id_is_deterministic()
    test_module_has_no_order_submission_surface()
    test_bracket_failure_and_reconciliation_fail_closed()
    test_second_opinion_flags_orphans_and_kills_on_exposure()
    test_second_opinion_without_credentials_is_recorded_not_silent()
    test_second_opinion_detects_duplicated_fills_and_missing_brackets()
    test_second_opinion_detects_leverage_margin_and_balance_drift()
    test_dry_run_uses_unified_contract_and_submits_nothing()
    test_phase_o_yaml_parser_single_source()
    print("test_gen2_canary_exec: OK")
