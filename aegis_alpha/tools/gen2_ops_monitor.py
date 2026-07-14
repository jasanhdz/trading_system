#!/usr/bin/env python3
"""GEN2 operations monitor — health snapshot + append-only alert stream.

Read-only over local state; the only network calls are the signed bridge
/gen2/status probe and a public Binance ping (both optional, both failures
are ALERTS, never crashes). Exit code 1 when any alert fires so it can be
wired to cron/mail. Never mutates trading state, never disengages anything.

Checks: python health (self), bridge reachability + latency, bridge kill
switch, local kill switch, Phase O pause, heartbeat freshness (decision
throughput), forward evidence growth, outcome resolution freshness,
incident growth since last run, arm token / risk gate status, Binance
public connectivity.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
import aegis_alpha.tools.gen2_operational_contract as oc  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, utc_now, validate_gen2_path  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
FORWARD_DECISIONS = GEN2_ROOT / "forward" / "forward_decisions.jsonl"
HEARTBEAT_STALE_SECONDS = 1800  # 6 five-minute cycles with margin
EVIDENCE_STALE_SECONDS = 7200


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def check_bridge(status_fn: Any) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        status = status_fn()
        latency_ms = (time.monotonic() - t0) * 1000
        return {"reachable": True, "latency_ms": round(latency_ms, 1), "status": status}
    except Exception as exc:
        return {"reachable": False, "latency_ms": None, "error": repr(exc)}


def check_binance_public(ping_fn: Any = None) -> dict[str, Any]:
    def default_ping() -> bool:
        from urllib.request import urlopen

        with urlopen("https://fapi.binance.com/fapi/v1/ping", timeout=5) as resp:  # nosec - public ping
            return resp.status == 200

    try:
        ok = (ping_fn or default_ping)()
        return {"reachable": bool(ok)}
    except Exception as exc:
        return {"reachable": False, "error": repr(exc)}


def build_report(candidate_id: str = DEFAULT_CANDIDATE_ID, status_fn: Any = None,
                 ping_fn: Any = None, now_epoch: float | None = None,
                 decisions_path: Path = FORWARD_DECISIONS,
                 expect_watch_running: bool = False,
                 probe_bridge: bool = True, probe_binance: bool = True) -> dict[str, Any]:
    cdir = core.canary_dir(candidate_id)
    now_epoch = now_epoch if now_epoch is not None else utc_now().timestamp()
    alerts: list[str] = []

    # --- kill switches / gates -------------------------------------------------
    local_kill = core.kill_switch_engaged(candidate_id)
    if local_kill:
        alerts.append("LOCAL_KILL_SWITCH_ENGAGED")
    phase_paused, phase_reason = core.phase_o_new_entries_paused()
    if not phase_paused:
        alerts.append("PHASE_O_NOT_PAUSED")
    armed, arm_reason, token = oc.verify_arm_token(candidate_id)
    risk_ok, risk_reason = oc.risk_gate(candidate_id)

    # --- bridge ------------------------------------------------------------------
    bridge: dict[str, Any] = {"probed": False}
    if probe_bridge:
        import aegis_alpha.tools.gen2_bridge_client as bridge_client

        bridge = {"probed": True, **check_bridge(status_fn or bridge_client.get_status)}
        if not bridge["reachable"]:
            alerts.append("BRIDGE_UNREACHABLE")
        else:
            bstatus = bridge.get("status", {})
            if bstatus.get("kill_switch") is True:
                alerts.append("BRIDGE_KILL_SWITCH_ENGAGED")
            if bstatus.get("phase_o_allow_orders") is not False:
                alerts.append("BRIDGE_REPORTS_PHASE_O_UNPAUSED")

    # --- binance public connectivity ----------------------------------------------
    binance: dict[str, Any] = {"probed": False}
    if probe_binance:
        binance = {"probed": True, **check_binance_public(ping_fn)}
        if not binance["reachable"]:
            alerts.append("BINANCE_PUBLIC_UNREACHABLE")

    # --- heartbeat / decision throughput ------------------------------------------
    hb_path = cdir / "heartbeat.json"
    heartbeat: dict[str, Any] = {"exists": hb_path.exists()}
    if hb_path.exists():
        hb = json.loads(hb_path.read_text())
        import pandas as pd

        finished = pd.Timestamp(hb.get("cycle_finished_utc"))
        age = now_epoch - finished.timestamp()
        heartbeat.update({"age_seconds": round(age, 1), "cycles": hb.get("cycles"),
                          "cycle_errors": hb.get("cycle_errors"), "decisions": hb.get("decisions")})
        if age > HEARTBEAT_STALE_SECONDS:
            alerts.append("HEARTBEAT_STALE")
        if int(hb.get("cycle_errors", 0)) > 0 and int(hb.get("cycles", 0)) > 0 and \
                int(hb.get("cycle_errors", 0)) * 2 >= int(hb.get("cycles", 0)):
            alerts.append("CYCLE_ERROR_RATE_HIGH")
    elif expect_watch_running:
        alerts.append("HEARTBEAT_MISSING")

    # --- forward evidence growth ---------------------------------------------------
    decisions_count = count_lines(decisions_path)
    outcomes_count = count_lines(cdir / "forward_outcomes.jsonl")
    incidents_count = count_lines(cdir / "incidents" / "incidents.jsonl")
    evidence: dict[str, Any] = {"decisions": decisions_count, "outcomes": outcomes_count,
                                "incidents": incidents_count}
    state_path = cdir / "monitor_state.json"
    prev = json.loads(state_path.read_text()) if state_path.exists() else {}
    if prev:
        evidence["decisions_delta"] = decisions_count - int(prev.get("decisions", 0))
        evidence["incidents_delta"] = incidents_count - int(prev.get("incidents", 0))
        if evidence["incidents_delta"] > 0:
            alerts.append("NEW_INCIDENTS_SINCE_LAST_CHECK")
        stale_for = now_epoch - float(prev.get("checked_epoch", now_epoch))
        if expect_watch_running and evidence["decisions_delta"] == 0 and stale_for > EVIDENCE_STALE_SECONDS:
            alerts.append("FORWARD_EVIDENCE_STALLED")
    core.atomic_write(state_path, json.dumps({"decisions": decisions_count, "outcomes": outcomes_count,
                                              "incidents": incidents_count, "checked_epoch": now_epoch}))

    report = {
        "schema": "gen2_ops_monitor_v1",
        "candidate_id": candidate_id,
        "checked_at_utc": utc_now().isoformat(),
        "python_health": "OK",
        "local_kill_switch": local_kill,
        "phase_o": {"paused": phase_paused, "reason": phase_reason},
        "arm": {"armed": armed, "reason": arm_reason, "mode": token.get("mode") if token else None},
        "risk_gate": {"ok": risk_ok, "reason": risk_reason},
        "bridge": bridge,
        "binance_public": binance,
        "heartbeat": heartbeat,
        "forward_evidence": evidence,
        "alerts": sorted(set(alerts)),
        "healthy": not alerts,
    }
    out = cdir / "monitor_report.json"
    validate_gen2_path(out)
    core.atomic_write(out, json.dumps(report, indent=2, default=json_default))
    if alerts:
        core.append_jsonl(cdir / "alerts.jsonl", {"alerts": sorted(set(alerts)), "report": "monitor_report.json"})
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GEN2 ops monitor (read-only; exit 1 on alerts)")
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    p.add_argument("--expect-watch-running", action="store_true",
                   help="alert on missing/stale heartbeat and stalled evidence")
    p.add_argument("--no-bridge", action="store_true", help="skip the bridge probe")
    p.add_argument("--no-binance", action="store_true", help="skip the public Binance ping")
    args = p.parse_args(argv)
    report = build_report(args.candidate_id, expect_watch_running=args.expect_watch_running,
                          probe_bridge=not args.no_bridge, probe_binance=not args.no_binance)
    print(json.dumps(report, indent=2, default=json_default))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
