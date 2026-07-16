#!/usr/bin/env python3
"""GEN2 unified status — one command, whole-system view.

Composes the existing pieces (startup state, ops monitor, maintenance audit,
optional private read) into a single human-readable or --json report with
health exit codes: 0 healthy, 1 warning, 2 critical. Never prints secrets.
No PM2 needed; run manually or with --watch.
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
import aegis_alpha.tools.gen2_config as gcfg  # noqa: E402
import aegis_alpha.tools.gen2_decision_loop as loop  # noqa: E402
import aegis_alpha.tools.gen2_maintenance as maint  # noqa: E402
import aegis_alpha.tools.gen2_ops_monitor as monitor  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, utc_now  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
SENSITIVE = ("GEN2_BRIDGE_SECRET", "TELEGRAM_BOT_TOKEN", "BINANCE_API_KEY", "BINANCE_API_SECRET",
             "BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET")


def _private_snapshot(candidate_id: str) -> dict[str, Any]:
    """Optional signed read-only account view (wallet/positions). Returns a
    'not available' marker if credentials are absent. Never returns secrets."""
    core.load_shared_env(("BINANCE_FUTURES_API_KEY", "BINANCE_API_KEY",
                          "BINANCE_FUTURES_API_SECRET", "BINANCE_API_SECRET"))
    try:
        from aegis_alpha.tools.gen2_canary_exec import BinancePrivateReadOnlyAdapter

        ad = BinancePrivateReadOnlyAdapter()
        if not ad.available:
            return {"available": False, "reason": "PRIVATE_CREDENTIALS_NOT_AVAILABLE"}
        acct = ad.account()
        symbols = tuple(gcfg.load_config(candidate_id).get("symbols", ("ADAUSDT", "DOGEUSDT"))) or ("ADAUSDT", "DOGEUSDT")
        positions, open_orders = [], 0
        for s in symbols:
            for p in ad.position_risk(s):
                if abs(float(p.get("positionAmt", 0) or 0)) > 0:
                    positions.append({"symbol": s, "amt": p.get("positionAmt"), "entry": p.get("entryPrice"),
                                      "mark": p.get("markPrice"), "uPnL": p.get("unRealizedProfit"),
                                      "leverage": p.get("leverage"), "owner": _owner_for_symbol(s)})
            open_orders += len(ad.open_orders(s))
        return {"available": True, "wallet": acct.get("totalWalletBalance"),
                "equity": acct.get("totalMarginBalance"), "available_balance": acct.get("availableBalance"),
                "open_positions": positions, "open_orders": open_orders}
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:120]}


def health_from_alerts(alerts: list[str]) -> tuple[list[str], list[str], int, str]:
    """Map monitor alerts to (critical, warning, exit_code, health).
    exit 0 healthy / 1 warning / 2 critical."""
    crit = sorted(set(alerts) & monitor.CRITICAL_ALERTS)
    warn = sorted(set(alerts) - monitor.CRITICAL_ALERTS)
    exit_code = 2 if crit else (1 if warn else 0)
    return crit, warn, exit_code, ("CRITICAL" if exit_code == 2 else "WARNING" if exit_code == 1 else "HEALTHY")


def _owner_for_symbol(symbol: str) -> str:
    """Best-effort owner from the shared ownership registry (a GEN2 record for the
    symbol => GEN2; else UNKNOWN — resolved definitively by the reconciler)."""
    reg = REPO / "binance-futures-bot-ts" / "logs" / "execution" / "position_ownership.json"
    try:
        data = json.loads(reg.read_text())
        for rec in data.values():
            if rec.get("symbol") == symbol and rec.get("owner") == "GEN2":
                return "GEN2"
    except Exception:
        pass
    return "UNKNOWN"


def collect(candidate_id: str, probe_bridge: bool = True, probe_binance: bool = True,
            private: bool = True) -> dict[str, Any]:
    core.load_shared_env(("GEN2_BRIDGE_SECRET",))
    cdir = core.canary_dir(candidate_id)
    st = loop.collect_startup_state(candidate_id)
    report = monitor.build_report(candidate_id, probe_bridge=probe_bridge, probe_binance=probe_binance, notifier=None)
    audit = maint.audit(candidate_id)
    # last cycle / decision
    last_cycle = {}
    lc = cdir / "loop_last_cycle.json"
    if lc.exists():
        try:
            c = json.loads(lc.read_text())
            so = c.get("selection_outcome", {})
            last_cycle = {"snapshot": c.get("snapshot"), "decisions": c.get("decisions"),
                          "ranked_candidates": c.get("ranked_candidates"), "orders_submitted": c.get("orders_submitted"),
                          "best_symbol": so.get("best_symbol"), "best_score": so.get("best_score"),
                          "frozen_threshold": so.get("frozen_threshold"), "no_decision_reason": so.get("no_decision_reason")}
        except Exception:
            pass
    try:
        threshold = loop.load_threshold_verified() if False else json.loads((GEN2_ROOT / "GEN2_SELECTION_POLICY.json").read_text())["threshold_value"]
        policy_valid = True
    except Exception:
        threshold, policy_valid = None, False
    ev = st.get("evidence", {})
    contract = st.get("contract") or {}
    private_view = _private_snapshot(candidate_id) if private else {"available": False, "reason": "SKIPPED"}
    # disk
    import shutil
    du = shutil.disk_usage(str(GEN2_ROOT))
    disk = {"free_gb": round(du.free / 1e9, 1), "used_pct": round(100 * du.used / du.total, 1)}

    # health: monitor alerts drive the exit code (critical alerts -> 2)
    crit, warn, exit_code, health = health_from_alerts(report.get("alerts", []))

    return {
        "schema": "gen2_status_v1",
        "timestamp_utc": utc_now().isoformat(),
        "candidate_id": candidate_id,
        "health": health, "exit_code": exit_code, "critical_alerts": crit, "warnings": warn,
        "science": {"freeze_valid": st.get("models", {}).get("freeze_valid"),
                    "environment": st.get("models", {}).get("environment"),
                    "trrm": st.get("models", {}).get("trrm"), "eqm": st.get("models", {}).get("eqm"),
                    "selection_policy_valid": policy_valid, "frozen_threshold": threshold},
        "config": {"path": str(gcfg.CONFIG_PATH), "checksum": contract.get("config_checksum"),
                   "mode": contract.get("mode"),
                   "execution_config_enabled": st.get("config_execution_enabled"),
                   "emergency_deny_override": st.get("emergency_deny_override"),
                   "effective_execution": st.get("effective_execution"),
                   "leverage": contract.get("default_leverage"), "max_leverage": contract.get("max_leverage")},
        "gates": {"phase_o_paused": st.get("phase_o_paused"), "kill_switch": st.get("kill_switch"),
                  "bridge_kill": report.get("bridge", {}).get("status", {}).get("kill_switch"),
                  "risk_gate": st.get("risk_reason"), "armed": st.get("armed"), "arm_reason": st.get("arm_reason")},
        "services": {"bridge": report.get("bridge", {}).get("reachable"),
                     "bridge_latency_ms": report.get("bridge", {}).get("latency_ms"),
                     "binance_public": report.get("binance_public", {}).get("reachable"),
                     "heartbeat": report.get("heartbeat")},
        "symbols": {"analyzed": st.get("symbols_analyzed"), "executable": st.get("symbols_executable")},
        "last_cycle": last_cycle,
        "evidence": {"paper_decisions": ev.get("paper_decisions"), "outcomes": ev.get("outcomes"),
                     "dryrun_requests": ev.get("dryrun_requests"), "real_order_submissions": ev.get("real_order_submissions"),
                     "real_fills": ev.get("real_fills"), "incidents": ev.get("incidents")},
        "account": private_view,
        "artifacts": {"total_mb": audit["total_mb"],
                      "largest": sorted(audit["items"], key=lambda i: -i["bytes"])[:3]},
        "disk": disk,
    }


def _redact(text: str) -> str:
    import os
    for name in SENSITIVE:
        v = os.environ.get(name)
        if v and len(v) >= 8:
            text = text.replace(v, "<REDACTED>")
    return text


def render_human(s: dict[str, Any]) -> str:
    icon = {"HEALTHY": "🟢", "WARNING": "🟠", "CRITICAL": "🔴"}[s["health"]]
    L = [f"{icon} GEN2 STATUS — {s['health']}  ({s['timestamp_utc']})",
         f"candidate: {s['candidate_id']}"]
    if s["critical_alerts"]:
        L.append(f"  CRITICAL: {', '.join(s['critical_alerts'])}")
    if s["warnings"]:
        L.append(f"  WARNING: {', '.join(s['warnings'])}")
    sc = s["science"]
    L += ["", "🔬 Science", f"  freeze_valid: {sc['freeze_valid']} | env: {sc['environment']} | "
          f"selection_policy: {'VALID' if sc['selection_policy_valid'] else 'INVALID'} | threshold: {sc['frozen_threshold']}"]
    c = s["config"]
    L += ["", "⚙️  Execution config (single source of truth)",
          f"  config: {'ENABLED' if c['execution_config_enabled'] else 'DISABLED'} | "
          f"deny override: {'ON' if c['emergency_deny_override'] else 'OFF'} | "
          f"EFFECTIVE: {'ENABLED' if c['effective_execution'] else 'DISABLED'}",
          f"  mode: {c['mode']} | leverage: {c['leverage']}x (max {c['max_leverage']}x) | checksum: {str(c['checksum'])[:12]}"]
    g = s["gates"]
    L += ["", "🛡️  Gates", f"  Phase O paused: {g['phase_o_paused']} | kill(py): {g['kill_switch']} | "
          f"kill(bridge): {g['bridge_kill']} | risk: {g['risk_gate']} | armed: {g['armed']} ({g['arm_reason']})"]
    sv = s["services"]
    hb = sv.get("heartbeat") or {}
    L += ["", "🔌 Services", f"  bridge: {sv['bridge']} ({sv['bridge_latency_ms']}ms) | binance: {sv['binance_public']} | "
          f"heartbeat age: {hb.get('age_seconds')}s"]
    L += ["", "🎯 Symbols", f"  analyzed: {' '.join((s['symbols']['analyzed'] or []))}",
          f"  executable: {' '.join((s['symbols']['executable'] or [])) or 'none'}"]
    lc = s["last_cycle"]
    if lc:
        L += ["", "📊 Last cycle", f"  best: {lc.get('best_symbol')} {lc.get('best_score')} vs threshold {lc.get('frozen_threshold')} | "
              f"decision: {lc.get('no_decision_reason') or 'OPEN'} | orders: {lc.get('orders_submitted')}"]
    e = s["evidence"]
    L += ["", "📈 Evidence", f"  paper: {e['paper_decisions']} | outcomes: {e['outcomes']} | "
          f"real orders: {e['real_order_submissions']} | real fills: {e['real_fills']} | incidents: {e['incidents']}"]
    a = s["account"]
    L += ["", "💰 Account"]
    if a.get("available"):
        L.append(f"  wallet: {a['wallet']} | equity: {a['equity']} | avail: {a['available_balance']} | "
                 f"open positions: {len(a.get('open_positions', []))} | open orders: {a.get('open_orders')}")
        for p in a.get("open_positions", []):
            L.append(f"    [{p['owner']}] {p['symbol']} {p['amt']} @ {p['entry']} uPnL {p['uPnL']}")
    else:
        L.append(f"  private read: not available ({a.get('reason')})")
    L += ["", "🗄️  Artifacts", f"  total: {s['artifacts']['total_mb']} MB | disk free: {s['disk']['free_gb']} GB "
          f"({s['disk']['used_pct']}% used)"]
    return _redact("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GEN2 unified status (exit 0 healthy / 1 warning / 2 critical)")
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-private", action="store_true", help="skip the signed private account read")
    p.add_argument("--no-bridge", action="store_true")
    p.add_argument("--no-binance", action="store_true")
    p.add_argument("--watch", type=float, metavar="SECONDS", default=None, help="re-print every N seconds")
    args = p.parse_args(argv)

    def once() -> int:
        s = collect(args.candidate_id, probe_bridge=not args.no_bridge, probe_binance=not args.no_binance,
                    private=not args.no_private)
        print(json.dumps(s, indent=2, default=json_default) if args.json else render_human(s), flush=True)
        return s["exit_code"]

    if args.watch:
        while True:
            once()
            print("—" * 40, flush=True)
            time.sleep(args.watch)
    return once()


if __name__ == "__main__":
    raise SystemExit(main())
