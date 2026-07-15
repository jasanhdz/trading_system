#!/usr/bin/env python3
"""GEN2 live-canary safety core: dual streams, risk guard, kill switch, arm token.

Research/execution boundary module. DISARMED BY DEFAULT: without a valid
human-created arm token AND absence of the kill switch AND a clean conflict
audit, no execution path can be reached. Fail-closed everywhere. No secrets
are ever read or written here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, sha256_text, utc_now, validate_gen2_path  # noqa: E402

CANARY_ROOT = GEN2_ROOT / "live_canary"
FREEZE_PATH = GEN2_ROOT / "GEN2_SYSTEM_FREEZE.json"


def _oc():
    """Lazy import of the single-source-of-truth operational contract (avoids the
    core<->contract import cycle). All risk limits, sizing and arm tokens live THERE."""
    import aegis_alpha.tools.gen2_operational_contract as oc

    return oc


def load_shared_env(keys: tuple[str, ...]) -> dict[str, bool]:
    """Reuse the TS bot's existing credentials for Python callers under PM2.

    Populates os.environ with the named keys, reading .env.gen2 then the bot's
    .env, WITHOUT overriding anything already set and WITHOUT printing values.
    Mirrors the bridge's dotenv reuse so Binance/Telegram secrets live in a
    single source and are never duplicated into GEN2 config or git.
    """
    import os

    result = {k: bool(os.environ.get(k)) for k in keys}
    ts_repo = REPO / "binance-futures-bot-ts"
    for env_file in (ts_repo / ".env.gen2", ts_repo / ".env"):
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                for k in keys:
                    if not os.environ.get(k) and line.startswith(f"{k}="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ[k] = val
                            result[k] = True
        except Exception:
            continue
    return result


def canary_dir(candidate_id: str) -> Path:
    d = CANARY_ROOT / candidate_id
    validate_gen2_path(d)
    return d


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"recorded_at": utc_now().isoformat(), **record}, default=str) + "\n")


def init_canary(candidate_id: str) -> Path:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze["candidate_id"] != candidate_id:
        raise ValueError(f"candidate mismatch: freeze={freeze['candidate_id']} requested={candidate_id}")
    d = canary_dir(candidate_id)
    d.mkdir(parents=True, exist_ok=True)
    for sub in ("daily_reports", "incidents", "parity_reports", "cost_reports"):
        (d / sub).mkdir(exist_ok=True)
    manifest = {"schema": "gen2_canary_manifest_v2", "candidate_id": candidate_id,
                "created_at_utc": utc_now().isoformat(),
                "limits_source": "operational_contract.json (gen2_operational_contract.py, GEN2_OPERATIONAL_MODES_SPEC.md)",
                "armed_default": False, "spec": "GEN2_F1_LIVE_CANARY_SPEC.md v1.0 + GEN2_OPERATIONAL_MODES_SPEC.md v1.0"}
    atomic_write(d / "canary_manifest.json", json.dumps(manifest, indent=2))
    (d / "canary_manifest.sha256").write_text(sha256_text(json.dumps(manifest, indent=2)) + "\n")
    atomic_write(d / "system_freeze_reference.json", json.dumps({"freeze_sha256_fields": {k: freeze[k] for k in ("candidate_id", "trrm_v2_sha256", "eqm1_sha256", "d3_dataset_sha256", "feature_hash")}}, indent=2))
    if not (d / "risk_state.json").exists():
        atomic_write(d / "risk_state.json", json.dumps({"paused": False, "daily_loss": 0.0, "total_loss": 0.0, "consecutive_losses": 0, "day": str(utc_now().date()), "orders_sent": 0}, indent=2))
    return d


# ------------------------------------------------------------------ kill switch
def kill_switch_engaged(candidate_id: str) -> bool:
    return (canary_dir(candidate_id) / "KILL_SWITCH").exists()


def engage_kill_switch(candidate_id: str, reason: str) -> None:
    """Idempotent: engaging an already-engaged kill switch preserves the original
    file and does NOT re-log an incident. Without this, a periodic caller (the
    reconciler, every 30 min) that keeps detecting the same unresolved exposure
    would spam KILL_SWITCH_ENGAGED incidents and NEW_INCIDENTS Telegram alerts."""
    d = canary_dir(candidate_id)
    if (d / "KILL_SWITCH").exists():
        return  # already engaged; keep the original engaged_at/reason
    atomic_write(d / "KILL_SWITCH", json.dumps({"engaged_at": utc_now().isoformat(), "reason": reason}))
    append_jsonl(d / "incidents" / "incidents.jsonl", {"type": "KILL_SWITCH_ENGAGED", "reason": reason})


# ------------------------------------------------- arm token / risk (delegated)
# The single implementations live in gen2_operational_contract.py. These thin
# delegates keep the historical 2-tuple call sites working without a second
# token schema or a second set of limits ever existing again.
def verify_arm_token(candidate_id: str) -> tuple[bool, str]:
    ok, reason, _token = _oc().verify_arm_token(candidate_id)
    return ok, reason


def risk_gate(candidate_id: str) -> tuple[bool, str]:
    return _oc().risk_gate(candidate_id)


def record_trade_result(candidate_id: str, pnl: float) -> None:
    d = canary_dir(candidate_id)
    state = json.loads((d / "risk_state.json").read_text())
    if pnl < 0:
        state["daily_loss"] += -pnl
        state["total_loss"] += -pnl
        state["consecutive_losses"] += 1
    else:
        state["consecutive_losses"] = 0
    atomic_write(d / "risk_state.json", json.dumps(state, indent=2))


# ---------------------------------------------------------- Phase O coexistence
def phase_o_new_entries_paused(ts_repo: Path | None = None) -> tuple[bool, str]:
    yaml_path = (ts_repo or REPO / "binance-futures-bot-ts") / "regime_config.live.yaml"
    if not yaml_path.exists():
        return False, "PHASE_O_CONFIG_NOT_FOUND"
    text = yaml_path.read_text(encoding="utf-8")
    m = re.search(r"phase_o_short_live:\n(?P<body>(?:[ \t]+[^\n]*\n)+)", text)
    if not m:
        return False, "PHASE_O_CONFIG_NOT_FOUND"
    body = m.group("body")
    enabled = re.search(r"^\s*enabled:\s*true\s*$", body, re.MULTILINE) is not None
    paused = re.search(r"^\s*allow_orders:\s*false\s*$", body, re.MULTILINE) is not None
    return enabled and paused, "PHASE_O_NEW_ENTRIES_PAUSED" if enabled and paused else "PHASE_O_NEW_ENTRIES_NOT_PAUSED"


def phase_o_conflict_audit() -> dict[str, Any]:
    """Read-only: conflict only if Phase O can open entries or exposure remains."""
    log_dir = REPO / "binance-futures-bot-ts" / "logs" / "aegis"
    recent = sorted(log_dir.glob("turbo_trades_2026-*.jsonl"))[-3:]
    recent_trades = sum(1 for f in recent for _ in f.open())
    snaps = sorted(log_dir.glob("account_snapshots_2026-*.jsonl"))
    balance = None
    open_position = None
    if snaps:
        last = snaps[-1].read_text().strip().splitlines()
        if last:
            row = json.loads(last[-1])
            balance = row.get("available_balance", row.get("availableBalance"))
            open_position = bool(row.get("position_open", row.get("positionOpen", False)))
    phase_paused, phase_reason = phase_o_new_entries_paused()
    conflict = (not phase_paused) or bool(open_position)
    return {"phase_o_recent_trades": recent_trades, "shared_wallet": True,
            "phase_o_new_entries_paused": phase_paused, "phase_o_pause_reason": phase_reason,
            "open_position": open_position,
            "available_balance": balance, "isolated_limits_possible": not conflict,
            "conflict": conflict,
            "resolution": "pause Phase O new entries and wait for existing exposure to close normally; no automatic position closing"}


# ---------------------------------------------------------------- dual streams
def record_paper_decision(candidate_id: str, decision: dict[str, Any]) -> None:
    append_jsonl(canary_dir(candidate_id) / "paper_decisions.jsonl", {"stream": "PAPER_CONTROL", **decision})


def canary_eligibility(candidate_id: str, decision: dict[str, Any]) -> tuple[bool, str]:
    """Full pre-order gauntlet. Paper stream is recorded by the caller regardless."""
    conflict = phase_o_conflict_audit()
    if conflict["conflict"]:
        return False, "LIVE_CANARY_CONFLICT_WITH_EXISTING_STRATEGY"
    ok, reason = risk_gate(candidate_id)
    if not ok:
        return False, reason
    ok, reason = verify_arm_token(candidate_id)
    if not ok:
        return False, reason
    if decision.get("decision") == "NO_DECISION" or decision.get("vetoed_by_trrm") or decision.get("hypothetical_action") != "CANDIDATE_SHORT":
        return False, "NOT_AN_ELIGIBLE_SIGNAL"
    score = decision.get("eqm_score")
    if score is None or not (score == score) or score in (float("inf"), float("-inf")):
        return False, "SCORE_NOT_FINITE"
    return True, "ELIGIBLE"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["init", "create-arm-token", "status", "kill", "conflict-audit"], required=True)
    p.add_argument("--candidate-id", default="gen2-20260711T202935Z")
    p.add_argument("--expiry-hours", type=int, default=72)
    p.add_argument("--allowed-symbols", default="ADAUSDT,DOGEUSDT")
    p.add_argument("--reason", default="manual")
    args = p.parse_args(argv)
    if args.mode == "init":
        out = {"initialized": str(init_canary(args.candidate_id)), "armed": False}
    elif args.mode == "create-arm-token":
        token = _oc().create_arm_token(args.candidate_id, args.expiry_hours, [s.strip().upper() for s in args.allowed_symbols.split(",")])
        out = {"token": token, "warning": "HUMAN ACTION ONLY - do not let automation call this"}
    elif args.mode == "kill":
        engage_kill_switch(args.candidate_id, args.reason)
        out = {"kill_switch": "ENGAGED"}
    elif args.mode == "conflict-audit":
        out = phase_o_conflict_audit()
    else:
        ok_t, r_t = verify_arm_token(args.candidate_id)
        ok_r, r_r = risk_gate(args.candidate_id)
        out = {"armed": ok_t, "arm_status": r_t, "risk": r_r, "kill_switch": kill_switch_engaged(args.candidate_id),
               "phase_o_conflict": phase_o_conflict_audit()["conflict"]}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
