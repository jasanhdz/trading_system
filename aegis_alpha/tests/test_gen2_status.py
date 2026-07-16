#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_status as st


def test_health_and_exit_code_mapping() -> None:
    assert st.health_from_alerts([])[2:] == (0, "HEALTHY")
    assert st.health_from_alerts(["HEARTBEAT_STALE"])[2:] == (1, "WARNING")  # non-critical
    assert st.health_from_alerts(["LOCAL_KILL_SWITCH_ENGAGED"])[2:] == (2, "CRITICAL")
    # critical dominates a mix
    crit, warn, code, health = st.health_from_alerts(["HEARTBEAT_STALE", "LOCAL_KILL_SWITCH_ENGAGED"])
    assert code == 2 and health == "CRITICAL" and "LOCAL_KILL_SWITCH_ENGAGED" in crit and "HEARTBEAT_STALE" in warn


def _status_fixture(execution: bool) -> dict:
    return {
        "health": "HEALTHY", "exit_code": 0, "critical_alerts": [], "warnings": [],
        "timestamp_utc": "2026-07-16T00:00:00+00:00", "candidate_id": "gen2-x",
        "science": {"freeze_valid": True, "environment": "valid", "trrm": "hash-verified", "eqm": "hash-verified",
                    "selection_policy_valid": True, "frozen_threshold": 0.0143},
        "config": {"path": "/x/gen2_config.yaml", "checksum": "abc", "mode": "EXPERIMENTAL_CONTINUOUS",
                   "execution_config_enabled": execution, "emergency_deny_override": False,
                   "effective_execution": execution, "leverage": 10, "max_leverage": 10},
        "gates": {"phase_o_paused": True, "kill_switch": False, "bridge_kill": False,
                  "risk_gate": "RISK_OK", "armed": execution, "arm_reason": "ARMED_VIA_CONFIG"},
        "services": {"bridge": True, "bridge_latency_ms": 1.0, "binance_public": True, "heartbeat": {"age_seconds": 60}},
        "symbols": {"analyzed": ["ADAUSDT"], "executable": ["ADAUSDT"] if execution else []},
        "last_cycle": {"best_symbol": "ADAUSDT", "best_score": 0.001, "frozen_threshold": 0.0143,
                       "no_decision_reason": "BELOW_FROZEN_EQM_THRESHOLD", "orders_submitted": 0},
        "evidence": {"paper_decisions": 100, "outcomes": 90, "dryrun_requests": 3,
                     "real_order_submissions": 1, "real_fills": 1, "incidents": 2},
        "account": {"available": False, "reason": "SKIPPED"},
        "artifacts": {"total_mb": 5.0, "largest": []},
        "disk": {"free_gb": 700.0, "used_pct": 20.0},
    }


def test_render_counts_paper_dryrun_live_separately_and_no_secrets() -> None:
    os.environ["GEN2_BRIDGE_SECRET"] = "supersecret-hmac-value-abcdef"
    try:
        s = _status_fixture(execution=False)
        # inject the secret into a fully-rendered field to prove redaction scrubs it
        s["gates"]["arm_reason"] = "reason supersecret-hmac-value-abcdef"
        out = st.render_human(s)
        assert "supersecret-hmac-value-abcdef" not in out
        assert "<REDACTED>" in out
        # counts rendered separately
        assert "paper: 100" in out and "real orders: 1" in out and "real fills: 1" in out
        # effective execution shown unambiguously
        assert "EFFECTIVE: DISABLED" in out
    finally:
        os.environ.pop("GEN2_BRIDGE_SECRET", None)


def test_render_effective_enabled_when_config_on() -> None:
    out = st.render_human(_status_fixture(execution=True))
    assert "EFFECTIVE: ENABLED" in out
    assert "executable: ADAUSDT" in out


if __name__ == "__main__":
    test_health_and_exit_code_mapping()
    test_render_counts_paper_dryrun_live_separately_and_no_secrets()
    test_render_effective_enabled_when_config_on()
    print("test_gen2_status: OK")
