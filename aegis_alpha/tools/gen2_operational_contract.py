#!/usr/bin/env python3
"""GEN2 single-source-of-truth operational contract (spec: GEN2_OPERATIONAL_MODES_SPEC.md).

One contract file per candidate, one arm token schema, one mode switch. The
scientific freeze (models, hashes, thresholds) is never touched here. All
consumers (risk gate, sizing, decision loop, bridge client) read THIS file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import utc_now  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
STOP_DISTANCE_PCT = 0.015
MODES: dict[str, dict[str, Any]] = {
    "SAFE": {
        "mode": "SAFE",
        "sizing_kind": "FIXED_NOTIONAL_USD",
        "fixed_notional_usd": 25.0,
        "balance_fraction": None,
        "max_leverage": 5,
        "default_leverage": 3,
        "risk_per_trade_pct": 0.01,
        "daily_loss_pct": 0.02,
        "total_loss_pct": 0.05,
        "equity_floor_fraction": 0.95,
        "compounding": False,
    },
    "EXPERIMENTAL": {
        "mode": "EXPERIMENTAL",
        "sizing_kind": "BALANCE_FRACTION",
        "fixed_notional_usd": None,
        "balance_fraction": 0.25,
        "max_leverage": 10,
        "default_leverage": 10,
        "risk_per_trade_pct": 0.0375,
        "daily_loss_pct": 0.10,
        "total_loss_pct": 0.25,
        "equity_floor_fraction": 0.75,
        "compounding": True,
        "sunset": {"technically_validated_orders": 20, "max_days": 60},
    },
}
COMMON = {
    "margin_type": "ISOLATED",
    "max_concurrent_positions": 1,
    "max_new_positions_per_30min": 1,
    "first_arm_max_orders": 1,
    "stop_distance_pct": STOP_DISTANCE_PCT,
    "martingale": False,
    "averaging_down": False,
    "pyramiding": False,
    "increase_after_losses": False,
    "scientific_freeze_untouched": True,
}


def contract_path(candidate_id: str) -> Path:
    return core.canary_dir(candidate_id) / "operational_contract.json"


def validate_coherence(contract: dict[str, Any], initial_equity: float) -> list[str]:
    """The V1/V2 incoherence class: per-stop loss must fit inside the caps."""
    errors: list[str] = []
    lev = contract["default_leverage"]
    if contract["sizing_kind"] == "FIXED_NOTIONAL_USD":
        notional = float(contract["fixed_notional_usd"])
    else:
        notional = initial_equity * float(contract["balance_fraction"]) * lev
    per_stop_loss = notional * STOP_DISTANCE_PCT
    daily_cap = contract["daily_loss_pct"] * initial_equity
    total_cap = contract["total_loss_pct"] * initial_equity
    if per_stop_loss > daily_cap + 1e-9:
        errors.append(f"per-stop loss {per_stop_loss:.4f} exceeds daily cap {daily_cap:.4f}")
    if daily_cap > total_cap + 1e-9:
        errors.append("daily cap exceeds total cap")
    if contract["risk_per_trade_pct"] * initial_equity + 1e-9 < per_stop_loss:
        errors.append(f"declared risk_per_trade {contract['risk_per_trade_pct']:.4f} understates per-stop loss {per_stop_loss / initial_equity:.4f} of equity")
    if contract["equity_floor_fraction"] >= 1.0 or contract["equity_floor_fraction"] < 0.5:
        errors.append("equity floor must be in [0.5, 1.0)")
    return errors


def write_contract(candidate_id: str, mode: str, initial_equity: float, force: bool = False) -> dict[str, Any]:
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    freeze = json.loads(core.FREEZE_PATH.read_text())
    if freeze["candidate_id"] != candidate_id:
        raise ValueError("candidate mismatch with scientific freeze")
    path = contract_path(candidate_id)
    if path.exists() and not force:
        current = json.loads(path.read_text())
        if current.get("open_position_at_switch"):
            raise ValueError("mode switch with open position is forbidden")
    contract = {
        "schema": "gen2_operational_contract_v1",
        "candidate_id": candidate_id,
        **MODES[mode],
        **COMMON,
        "initial_equity": float(initial_equity),
        "equity_floor": float(initial_equity) * MODES[mode]["equity_floor_fraction"],
        "created_at_utc": utc_now().isoformat(),
        "scientific_hashes": {k: freeze[k] for k in ("trrm_v2_sha256", "eqm1_sha256", "d3_dataset_sha256", "feature_hash")},
    }
    errors = validate_coherence(contract, float(initial_equity))
    if errors:
        raise ValueError(f"CONTRACT_INCOHERENT: {errors}")
    core.atomic_write(path, json.dumps(contract, indent=2))
    core.append_jsonl(core.canary_dir(candidate_id) / "contract_history.jsonl", {"event": "CONTRACT_WRITTEN", "mode": mode, "initial_equity": initial_equity})
    return contract


def load_contract(candidate_id: str) -> dict[str, Any]:
    path = contract_path(candidate_id)
    if not path.exists():
        raise FileNotFoundError("OPERATIONAL_CONTRACT_MISSING")
    contract = json.loads(path.read_text())
    if contract.get("schema") != "gen2_operational_contract_v1":
        raise ValueError("OPERATIONAL_CONTRACT_INVALID_SCHEMA")
    return contract


def create_arm_token(candidate_id: str, expiry_hours: int, allowed_symbols: list[str]) -> dict[str, Any]:
    """Single token schema. Bound to candidate + mode + contract hash."""
    contract = load_contract(candidate_id)
    body = {
        "schema": "gen2_arm_token_v3",
        "candidate_id": candidate_id,
        "mode": contract["mode"],
        "contract_sha256": hashlib.sha256(contract_path(candidate_id).read_bytes()).hexdigest(),
        "allowed_symbols": sorted(s.upper() for s in allowed_symbols),
        "max_orders": contract["first_arm_max_orders"],
        "expires_at_epoch": utc_now().timestamp() + expiry_hours * 3600,
        "consumed_orders": 0,
    }
    signed = {k: body[k] for k in sorted(body)}
    body["checksum"] = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    core.atomic_write(core.canary_dir(candidate_id) / "ARM_TOKEN.json", json.dumps(body, indent=2))
    return body


def verify_arm_token(candidate_id: str) -> tuple[bool, str, dict[str, Any]]:
    path = core.canary_dir(candidate_id) / "ARM_TOKEN.json"
    if not path.exists():
        return False, "CANARY_UNARMED_NO_TOKEN", {}
    try:
        t = json.loads(path.read_text())
    except Exception:
        return False, "TOKEN_CORRUPT", {}
    if t.get("schema") != "gen2_arm_token_v3":
        return False, "TOKEN_LEGACY_SCHEMA_REJECTED", {}
    signed = {k: t[k] for k in sorted(t) if k != "checksum"}
    if t.get("checksum") != hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest():
        return False, "TOKEN_CHECKSUM_INVALID", {}
    if t.get("candidate_id") != candidate_id:
        return False, "TOKEN_WRONG_CANDIDATE", {}
    try:
        contract = load_contract(candidate_id)
    except Exception:
        return False, "OPERATIONAL_CONTRACT_MISSING", {}
    if t.get("mode") != contract["mode"]:
        return False, "TOKEN_MODE_MISMATCH", {}
    if t.get("contract_sha256") != hashlib.sha256(contract_path(candidate_id).read_bytes()).hexdigest():
        return False, "TOKEN_CONTRACT_HASH_MISMATCH", {}
    if float(t.get("expires_at_epoch", 0)) < utc_now().timestamp():
        return False, "TOKEN_EXPIRED", {}
    if int(t.get("consumed_orders", 0)) >= int(t.get("max_orders", 0)):
        return False, "TOKEN_MAX_ORDERS_EXHAUSTED", {}
    return True, "TOKEN_VALID", t


def consume_order(candidate_id: str) -> None:
    path = core.canary_dir(candidate_id) / "ARM_TOKEN.json"
    t = json.loads(path.read_text())
    t["consumed_orders"] = int(t.get("consumed_orders", 0)) + 1
    signed = {k: t[k] for k in sorted(t) if k != "checksum"}
    t["checksum"] = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    core.atomic_write(path, json.dumps(t, indent=2))


def risk_gate(candidate_id: str) -> tuple[bool, str]:
    """THE risk gate: reads the single contract; replaces core.DEFAULT_LIMITS."""
    try:
        contract = load_contract(candidate_id)
    except Exception as exc:
        return False, str(exc)
    d = core.canary_dir(candidate_id)
    state = json.loads((d / "risk_state.json").read_text())
    today = str(utc_now().date())
    if state.get("day") != today:
        state["day"], state["daily_loss"] = today, 0.0
        core.atomic_write(d / "risk_state.json", json.dumps(state, indent=2))
    if state.get("paused"):
        return False, "CANARY_PAUSED"
    if core.kill_switch_engaged(candidate_id):
        return False, "KILL_SWITCH_ENGAGED"
    equity0 = contract["initial_equity"]
    if state["daily_loss"] >= contract["daily_loss_pct"] * equity0:
        return False, "DAILY_LOSS_CAP"
    if state["total_loss"] >= contract["total_loss_pct"] * equity0:
        return False, "TOTAL_LOSS_CAP"
    if state["consecutive_losses"] >= 3:
        return False, "CONSECUTIVE_LOSS_CAP"
    if equity0 - state["total_loss"] <= contract["equity_floor"]:
        core.engage_kill_switch(candidate_id, "EQUITY_FLOOR_REACHED")
        return False, "EQUITY_FLOOR_REACHED"
    return True, "RISK_OK"


def compute_sizing(contract: dict[str, Any], price: float, available_balance: float, step_size: float, min_notional: float, min_qty: float = 0.0) -> dict[str, Any]:
    lev = int(contract["default_leverage"])
    if contract["sizing_kind"] == "FIXED_NOTIONAL_USD":
        target_notional = float(contract["fixed_notional_usd"])
    else:
        target_notional = available_balance * float(contract["balance_fraction"]) * lev
    import math

    qty = math.floor((target_notional / price) / step_size) * step_size if step_size > 0 else target_notional / price
    qty = round(qty, 12)
    notional = qty * price
    margin = notional / lev
    result = {"quantity": qty, "leverage": lev, "notional": notional, "required_margin": margin,
              "stop_price": price * (1 + STOP_DISTANCE_PCT), "per_stop_loss": notional * STOP_DISTANCE_PCT}
    if qty <= 0 or qty < min_qty:
        result["decision"], result["reason"] = "NO_TRADE", "MIN_QTY_NOT_MET"
    elif notional < min_notional:
        result["decision"], result["reason"] = "NO_TRADE", "MIN_NOTIONAL_NOT_MET"
    elif margin > available_balance + 1e-9:
        result["decision"], result["reason"] = "NO_TRADE", "INSUFFICIENT_BALANCE"
    else:
        result["decision"], result["reason"] = "SIZED", "OK"
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode-op", choices=["write-contract", "create-arm-token", "status"], required=True)
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    p.add_argument("--mode", choices=["safe", "experimental"], default="safe")
    p.add_argument("--initial-equity", type=float, required=False)
    p.add_argument("--expiry-hours", type=int, default=72)
    p.add_argument("--allowed-symbols", default="ADAUSDT,DOGEUSDT")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.mode_op == "write-contract":
        if args.initial_equity is None:
            raise SystemExit("--initial-equity required")
        out = write_contract(args.candidate_id, args.mode, args.initial_equity, args.force)
    elif args.mode_op == "create-arm-token":
        out = create_arm_token(args.candidate_id, args.expiry_hours, args.allowed_symbols.split(","))
        out = {**out, "warning": "HUMAN ACTION ONLY"}
    else:
        ok, reason, token = verify_arm_token(args.candidate_id)
        gate_ok, gate_reason = risk_gate(args.candidate_id)
        out = {"armed": ok, "arm_reason": reason, "risk": gate_reason, "mode": token.get("mode") if token else None}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
