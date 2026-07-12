#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as cc

CID = "gen2-test"


def setup(tmp: Path, freeze_ok: bool = True) -> None:
    cc.CANARY_ROOT = tmp / "live_canary"
    cc.FREEZE_PATH = tmp / "freeze.json"
    cc.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID if freeze_ok else "other", "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))


def good_decision() -> dict:
    return {"decision": "OK", "vetoed_by_trrm": False, "hypothetical_action": "CANDIDATE_SHORT", "eqm_score": 0.1}


def test_gauntlet_fail_closed_chain() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup(tmp)
        cc.init_canary(CID)
        no_conflict = {"conflict": False}
        original = cc.phase_o_conflict_audit
        try:
            cc.phase_o_conflict_audit = lambda: no_conflict
            # 1) unarmed by default
            ok, r = cc.canary_eligibility(CID, good_decision())
            assert (ok, r) == (False, "CANARY_UNARMED_NO_TOKEN")
            # 2) armed -> eligible
            cc.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 5)
            assert cc.canary_eligibility(CID, good_decision()) == (True, "ELIGIBLE")
            # 3) wrong candidate token
            tok = json.loads((cc.canary_dir(CID) / "ARM_TOKEN.json").read_text())
            tok["candidate_id"] = "other"
            (cc.canary_dir(CID) / "ARM_TOKEN.json").write_text(json.dumps(tok))
            assert cc.verify_arm_token(CID)[1] == "TOKEN_CHECKSUM_INVALID"
            cc.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 5)
            # 4) expired token
            tok = json.loads((cc.canary_dir(CID) / "ARM_TOKEN.json").read_text())
            # NOT_A trivially-forgeable expiry: checksum covers expires_at, so tampering fails checksum
            tok["expires_at"] = "2020-01-01T00:00:00+00:00"
            (cc.canary_dir(CID) / "ARM_TOKEN.json").write_text(json.dumps(tok))
            assert cc.verify_arm_token(CID)[1] == "TOKEN_CHECKSUM_INVALID"
            assert cc.create_arm_token(CID, 15.0, -1, ["ADAUSDT"], 5) and cc.verify_arm_token(CID)[1] == "TOKEN_EXPIRED"
            cc.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 1)
            # 5) consecutive loss cap pauses entries
            for _ in range(3):
                cc.record_trade_result(CID, -0.01)
            assert cc.risk_gate(CID)[1] == "CONSECUTIVE_LOSS_CAP"
            cc.record_trade_result(CID, +0.05)  # win resets streak
            assert cc.risk_gate(CID)[0] is True
            # 6) total loss cap
            cc.record_trade_result(CID, -10.0)
            assert cc.risk_gate(CID)[1] in {"DAILY_LOSS_CAP", "TOTAL_LOSS_CAP"}
            # 7) kill switch dominates and persists across restart (file-based)
            cc.engage_kill_switch(CID, "test")
            assert cc.kill_switch_engaged(CID) is True
            assert cc.canary_eligibility(CID, good_decision())[0] is False
            # 8) veto / NO_DECISION / non-finite are never eligible
            state = cc.canary_dir(CID) / "risk_state.json"
            state.write_text(json.dumps({"paused": False, "daily_loss": 0, "total_loss": 0, "consecutive_losses": 0, "day": "2026-07-12", "orders_sent": 0}))
            (cc.canary_dir(CID) / "KILL_SWITCH").unlink()
            for bad in ({"decision": "NO_DECISION"}, {**good_decision(), "vetoed_by_trrm": True}, {**good_decision(), "eqm_score": float("nan")}):
                assert cc.canary_eligibility(CID, {**good_decision(), **bad})[0] is False
            # 9) token max orders exhaustion
            state.write_text(json.dumps({"paused": False, "daily_loss": 0, "total_loss": 0, "consecutive_losses": 0, "day": "2026-07-12", "orders_sent": 1}))
            assert cc.verify_arm_token(CID)[1] == "TOKEN_MAX_ORDERS_EXHAUSTED"
        finally:
            cc.phase_o_conflict_audit = original


def test_phase_o_conflict_blocks_everything() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup(tmp)
        cc.init_canary(CID)
        cc.create_arm_token(CID, 15.0, 72, ["ADAUSDT"], 5)
        original = cc.phase_o_conflict_audit
        try:
            cc.phase_o_conflict_audit = lambda: {"conflict": True}
            ok, r = cc.canary_eligibility(CID, good_decision())
            assert (ok, r) == (False, "LIVE_CANARY_CONFLICT_WITH_EXISTING_STRATEGY")
        finally:
            cc.phase_o_conflict_audit = original


def test_candidate_mismatch_refused() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), freeze_ok=False)
        try:
            cc.init_canary(CID)
            raise AssertionError("must refuse candidate mismatch")
        except ValueError:
            pass


def test_phase_o_paused_without_open_position_clears_conflict() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        old_repo = cc.REPO
        try:
            cc.REPO = tmp
            ts = tmp / "binance-futures-bot-ts"
            logs = ts / "logs" / "aegis"
            logs.mkdir(parents=True)
            (ts / "regime_config.live.yaml").write_text("aegis:\n  phase_o_short_live:\n    enabled: true\n    allow_orders: false\n")
            (logs / "turbo_trades_2026-07-12.jsonl").write_text('{"event":"old"}\n')
            (logs / "account_snapshots_2026-07-12.jsonl").write_text('{"availableBalance":16.24,"positionOpen":false}\n')
            audit = cc.phase_o_conflict_audit()
            assert audit["phase_o_new_entries_paused"] is True
            assert audit["open_position"] is False
            assert audit["conflict"] is False
        finally:
            cc.REPO = old_repo


if __name__ == "__main__":
    test_gauntlet_fail_closed_chain()
    test_phase_o_conflict_blocks_everything()
    test_candidate_mismatch_refused()
    test_phase_o_paused_without_open_position_clears_conflict()
    print("test_gen2_canary_core: OK")
