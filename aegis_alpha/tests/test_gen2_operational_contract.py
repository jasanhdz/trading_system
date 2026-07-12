#!/usr/bin/env python3
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


def test_modes_coherent_and_switch_requires_new_token() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup(tmp)
        safe = oc.write_contract(CID, "safe", 200.0)
        assert safe["mode"] == "SAFE" and safe["compounding"] is False
        token = oc.create_arm_token(CID, 72, ["ADAUSDT"])
        assert oc.verify_arm_token(CID)[0] is True
        assert token["mode"] == "SAFE"
        # switch mode -> old token invalid (mode + contract hash mismatch)
        exp = oc.write_contract(CID, "experimental", 200.0, force=True)
        assert exp["mode"] == "EXPERIMENTAL" and exp["balance_fraction"] == 0.25 and exp["max_leverage"] == 10
        ok, reason, _ = oc.verify_arm_token(CID)
        assert ok is False and reason in {"TOKEN_MODE_MISMATCH", "TOKEN_CONTRACT_HASH_MISMATCH"}
        # legacy schemas rejected
        (core.canary_dir(CID) / "ARM_TOKEN.json").write_text(json.dumps({"schema": "gen2_canary_arm_token_v2"}))
        assert oc.verify_arm_token(CID)[1] == "TOKEN_LEGACY_SCHEMA_REJECTED"


def test_incoherent_contract_refused() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        # tiny equity: SAFE fixed $25 notional stop-loss (0.375) > daily cap (2% of 10 = 0.2)
        try:
            oc.write_contract(CID, "safe", 10.0)
            raise AssertionError("must refuse incoherent contract")
        except ValueError as exc:
            assert "CONTRACT_INCOHERENT" in str(exc)


def test_risk_gate_uses_single_contract() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        assert oc.risk_gate(CID) == (True, "RISK_OK")
        for _ in range(2):
            core.record_trade_result(CID, -2.5)  # daily cap = 4.0
        assert oc.risk_gate(CID)[1] == "DAILY_LOSS_CAP"
        d = core.canary_dir(CID)
        state = json.loads((d / "risk_state.json").read_text())
        state.update({"daily_loss": 0.0, "total_loss": 11.0, "consecutive_losses": 0})
        (d / "risk_state.json").write_text(json.dumps(state))
        assert oc.risk_gate(CID)[1] == "TOTAL_LOSS_CAP"
        # equity floor engages kill switch (total_loss below the 5% cap so the floor is what fires)
        state.update({"total_loss": 9.0})
        contract = json.loads(oc.contract_path(CID).read_text())
        contract["equity_floor"] = 195.0
        core.atomic_write(oc.contract_path(CID), json.dumps(contract))
        (d / "risk_state.json").write_text(json.dumps(state))
        assert oc.risk_gate(CID)[1] == "EQUITY_FLOOR_REACHED"
        assert core.kill_switch_engaged(CID) is True


def test_sizing_by_mode_and_token_consumption() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        safe = oc.write_contract(CID, "safe", 200.0)
        s = oc.compute_sizing(safe, price=0.70, available_balance=200.0, step_size=1.0, min_notional=5.0)
        assert s["decision"] == "SIZED" and abs(s["notional"] - 25.0) < 0.8 and s["leverage"] == 3
        exp = oc.write_contract(CID, "experimental", 200.0, force=True)
        s2 = oc.compute_sizing(exp, price=0.70, available_balance=200.0, step_size=1.0, min_notional=5.0)
        assert s2["decision"] == "SIZED" and s2["leverage"] == 10 and s2["notional"] <= 200.0 * 0.25 * 10
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        oc.consume_order(CID)
        assert oc.verify_arm_token(CID)[1] == "TOKEN_MAX_ORDERS_EXHAUSTED"  # first_arm_max_orders=1


if __name__ == "__main__":
    test_modes_coherent_and_switch_requires_new_token()
    test_incoherent_contract_refused()
    test_risk_gate_uses_single_contract()
    test_sizing_by_mode_and_token_consumption()
    print("test_gen2_operational_contract: OK")
