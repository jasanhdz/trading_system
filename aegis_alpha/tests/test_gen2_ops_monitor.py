#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_operational_contract as oc
import aegis_alpha.tools.gen2_ops_monitor as mon

CID = "gen2-test"


def setup(tmp: Path) -> Path:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)
    # Phase O paused fixture
    ts = tmp / "binance-futures-bot-ts"
    (ts / "logs" / "aegis").mkdir(parents=True)
    (ts / "regime_config.live.yaml").write_text("aegis:\n  phase_o_short_live:\n    enabled: true\n    allow_orders: false\n")
    core.REPO = tmp
    decisions = tmp / "decisions.jsonl"
    decisions.write_text("")
    return decisions


def healthy_bridge():
    return {"schema": "gen2_status_v1", "gen2_enabled": True, "kill_switch": False,
            "phase_o_allow_orders": False, "available_balance": 116.0, "open_positions": []}


def report(decisions: Path, **kw):
    defaults = dict(status_fn=healthy_bridge, ping_fn=lambda: True, decisions_path=decisions)
    defaults.update(kw)
    return mon.build_report(CID, **defaults)


def test_healthy_system_no_alerts() -> None:
    with tempfile.TemporaryDirectory() as t:
        decisions = setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        r = report(decisions)
        assert r["healthy"] is True and r["alerts"] == []
        assert r["bridge"]["reachable"] is True and r["bridge"]["latency_ms"] is not None
        assert (core.canary_dir(CID) / "monitor_report.json").exists()
        assert not (core.canary_dir(CID) / "alerts.jsonl").exists()


def test_alerts_fire_and_are_append_only() -> None:
    with tempfile.TemporaryDirectory() as t:
        decisions = setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        core.engage_kill_switch(CID, "drill")

        def dead_bridge():
            raise RuntimeError("connection refused")

        r = report(decisions, status_fn=dead_bridge, ping_fn=lambda: (_ for _ in ()).throw(RuntimeError("dns")))
        assert r["healthy"] is False
        assert "LOCAL_KILL_SWITCH_ENGAGED" in r["alerts"]
        assert "BRIDGE_UNREACHABLE" in r["alerts"]
        assert "BINANCE_PUBLIC_UNREACHABLE" in r["alerts"]
        # incident growth alert on next run (kill switch wrote an incident before first run,
        # so seed state then add another incident)
        core.append_jsonl(core.canary_dir(CID) / "incidents" / "incidents.jsonl", {"type": "TEST_INCIDENT"})
        r2 = report(decisions, status_fn=dead_bridge, ping_fn=lambda: True)
        assert "NEW_INCIDENTS_SINCE_LAST_CHECK" in r2["alerts"]
        alerts_rows = (core.canary_dir(CID) / "alerts.jsonl").read_text().splitlines()
        assert len(alerts_rows) == 2  # one per unhealthy run, append-only


def test_bridge_kill_and_unpaused_phase_o_reported_by_bridge() -> None:
    with tempfile.TemporaryDirectory() as t:
        decisions = setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)

        def bad_status():
            return {**healthy_bridge(), "kill_switch": True, "phase_o_allow_orders": True}

        r = report(decisions, status_fn=bad_status)
        assert "BRIDGE_KILL_SWITCH_ENGAGED" in r["alerts"]
        assert "BRIDGE_REPORTS_PHASE_O_UNPAUSED" in r["alerts"]


def test_heartbeat_staleness_and_evidence_stall() -> None:
    with tempfile.TemporaryDirectory() as t:
        decisions = setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        now = time.time()
        # no heartbeat but watch expected -> alert
        r = report(decisions, expect_watch_running=True, now_epoch=now)
        assert "HEARTBEAT_MISSING" in r["alerts"]
        # stale heartbeat -> alert
        core.atomic_write(core.canary_dir(CID) / "heartbeat.json", json.dumps({
            "schema": "gen2_watch_heartbeat_v1", "cycle_finished_utc": "2026-07-13T00:00:00+00:00",
            "cycles": 10, "cycle_errors": 0, "decisions": 30}))
        r2 = report(decisions, expect_watch_running=True, now_epoch=now)
        assert "HEARTBEAT_STALE" in r2["alerts"]
        # stalled evidence: no decision growth across checks separated by > threshold
        r3 = report(decisions, expect_watch_running=True,
                    now_epoch=now + mon.EVIDENCE_STALE_SECONDS + 10)
        assert "FORWARD_EVIDENCE_STALLED" in r3["alerts"]
        # growth clears the stall
        decisions.write_text('{"candidate_id":"gen2-test"}\n')
        r4 = report(decisions, expect_watch_running=True,
                    now_epoch=now + 2 * (mon.EVIDENCE_STALE_SECONDS + 10))
        assert "FORWARD_EVIDENCE_STALLED" not in r4["alerts"]


def test_cycle_error_rate_alert() -> None:
    with tempfile.TemporaryDirectory() as t:
        decisions = setup(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        import pandas as pd

        core.atomic_write(core.canary_dir(CID) / "heartbeat.json", json.dumps({
            "schema": "gen2_watch_heartbeat_v1",
            "cycle_finished_utc": str(pd.Timestamp.utcnow()),
            "cycles": 4, "cycle_errors": 2, "decisions": 6}))
        r = report(decisions)
        assert "CYCLE_ERROR_RATE_HIGH" in r["alerts"]


if __name__ == "__main__":
    test_healthy_system_no_alerts()
    test_alerts_fire_and_are_append_only()
    test_bridge_kill_and_unpaused_phase_o_reported_by_bridge()
    test_heartbeat_staleness_and_evidence_stall()
    test_cycle_error_rate_alert()
    print("test_gen2_ops_monitor: OK")
