#!/usr/bin/env python3
"""GEN2 single operational config — the one source of truth (autonomous deploy).

Replaces the derived operational_contract.json + arm-token ceremony with a single
`gen2_config.yaml`. PM2 is the arm/disarm mechanism: a running watcher IS the
authorization. This file loads the config, resolves the mode preset (the safety
parameters stay versioned in code, not editable to unsafe values without failing
coherence), applies operator overrides, and VALIDATES COHERENCE (fail-closed).

Science is untouched: the frozen candidate, all hashes, and the frozen selection
threshold (GEN2_SELECTION_POLICY.json) are verified elsewhere and are NOT
editable from this config.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
import aegis_alpha.tools.gen2_operational_contract as oc  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, environment_info, git_commit, utc_now  # noqa: E402

CONFIG_PATH = GEN2_ROOT / "gen2_config.yaml"
DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
# Operator-facing keys the yaml may override on top of the mode preset. Anything
# else in the preset (leverage caps, loss caps, sunset, flags) is code-versioned;
# an override that produces an incoherent contract is rejected fail-closed.
OVERRIDABLE = ("default_leverage", "balance_fraction", "fixed_notional_usd",
               "daily_loss_pct", "total_loss_pct", "equity_floor_fraction",
               "max_concurrent_positions", "max_orders_per_cycle", "max_orders_per_day")


def config_checksum(path: Path | None = None) -> str:
    path = path or CONFIG_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("GEN2_CONFIG_MISSING")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("GEN2_CONFIG_INVALID")
    return data


def load_config(candidate_id: str = DEFAULT_CANDIDATE_ID, path: Path | None = None) -> dict[str, Any]:
    """Build the coherent, contract-shaped operational config from the yaml.
    Fail-closed: missing/invalid/incoherent config raises (the system will not
    trade). Same shape the risk gate and sizing already consume."""
    path = path or CONFIG_PATH
    data = _read_yaml(path)
    mode = str(data.get("mode", "")).upper()
    if mode not in oc.MODES:
        raise ValueError(f"GEN2_CONFIG_UNKNOWN_MODE:{mode}")
    capital = data.get("capital", {}) or {}
    initial_equity = capital.get("initial_equity")
    if initial_equity is None:
        raise ValueError("GEN2_CONFIG_MISSING_INITIAL_EQUITY")
    freeze = json.loads(core.FREEZE_PATH.read_text())
    if freeze["candidate_id"] != candidate_id:
        raise ValueError("GEN2_CONFIG_CANDIDATE_MISMATCH")

    contract: dict[str, Any] = {
        "schema": "gen2_operational_config_v1",
        "candidate_id": candidate_id,
        **oc.MODES[mode],
        **oc.COMMON,
    }
    # apply operator overrides (only the allowlisted keys)
    overrides = data.get("risk_overrides", {}) or {}
    applied: dict[str, Any] = {}
    for k, v in overrides.items():
        if k not in OVERRIDABLE:
            raise ValueError(f"GEN2_CONFIG_NON_OVERRIDABLE_KEY:{k}")
        contract[k] = v
        applied[k] = v
    contract["initial_equity"] = float(initial_equity)
    contract["equity_floor"] = float(initial_equity) * float(contract["equity_floor_fraction"])
    # Sunset anchor: the config file's mtime (when this operational config was
    # last written). Editing the config deliberately re-anchors the clock.
    from datetime import datetime, timezone

    contract["created_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    contract["scientific_hashes"] = {k: freeze[k] for k in ("trrm_v2_sha256", "eqm1_sha256", "d3_dataset_sha256", "feature_hash")}
    # operator config (not part of the risk contract shape, carried alongside)
    contract["symbols"] = [s.upper() for s in (data.get("symbols") or [])]
    contract["execution_enabled"] = bool(data.get("execution_enabled", False))
    contract["telegram_enabled"] = bool(data.get("telegram_enabled", True))
    contract["applied_overrides"] = applied
    contract["config_checksum"] = config_checksum(path)

    errors = oc.validate_coherence(contract, float(initial_equity))
    if errors:
        raise ValueError(f"GEN2_CONFIG_INCOHERENT:{errors}")
    return contract


def write_startup_audit(candidate_id: str = DEFAULT_CANDIDATE_ID, path: Path | None = None) -> dict[str, Any]:
    """Append-only startup record: every pm2 start is auditable without contracts."""
    path = path or CONFIG_PATH
    cfg = load_config(candidate_id, path)
    freeze = json.loads(core.FREEZE_PATH.read_text())
    record = {
        "event": "GEN2_STARTUP",
        "timestamp_utc": utc_now().isoformat(),
        "candidate_id": candidate_id,
        "code_commit": git_commit(),
        "config_checksum": cfg["config_checksum"],
        "mode": cfg["mode"],
        "execution_enabled": cfg["execution_enabled"],
        "symbols": cfg["symbols"],
        "applied_overrides": cfg["applied_overrides"],
        "scientific_hashes": cfg["scientific_hashes"],
        "environment": environment_info(),
    }
    core.append_jsonl(core.canary_dir(candidate_id) / "startup_audit.jsonl", record)
    return record


def detect_config_change(previous_checksum: str, candidate_id: str = DEFAULT_CANDIDATE_ID,
                         path: Path | None = None) -> dict[str, Any] | None:
    """Compare the current config against a prior checksum; if it changed, return
    a CONFIGURATION_CHANGED record with the field-level diff (and log it)."""
    path = path or CONFIG_PATH
    current = config_checksum(path)
    if current == previous_checksum:
        return None
    try:
        new_cfg = load_config(candidate_id, path)
        new_valid, new_reason = True, "OK"
    except Exception as exc:
        new_cfg, new_valid, new_reason = {}, False, str(exc)
    prior_path = core.canary_dir(candidate_id) / "config_last_seen.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    diff = {}
    watched = ("mode", "execution_enabled", "symbols", "initial_equity", "default_leverage",
               "balance_fraction", "daily_loss_pct", "total_loss_pct", "max_concurrent_positions",
               "max_orders_per_day", "telegram_enabled")
    for k in watched:
        if prior.get(k) != new_cfg.get(k):
            diff[k] = {"old": prior.get(k), "new": new_cfg.get(k)}
    record = {
        "event": "CONFIGURATION_CHANGED",
        "timestamp_utc": utc_now().isoformat(),
        "old_checksum": previous_checksum,
        "new_checksum": current,
        "new_config_valid": new_valid,
        "invalid_reason": None if new_valid else new_reason,
        "diff": diff,
        # sizing/leverage/caps/mode/symbols apply immediately (read each cycle);
        # nothing here needs a restart, but an INVALID new config is fail-closed
        # (the loop keeps the last valid contract and alerts).
        "requires_restart": False,
    }
    core.append_jsonl(core.canary_dir(candidate_id) / "config_changes.jsonl", record)
    if new_valid:
        core.atomic_write(prior_path, json.dumps({k: new_cfg.get(k) for k in watched}, default=str))
    return record


def snapshot_for_change_tracking(candidate_id: str = DEFAULT_CANDIDATE_ID, path: Path | None = None) -> None:
    path = path or CONFIG_PATH
    cfg = load_config(candidate_id, path)
    watched = ("mode", "execution_enabled", "symbols", "initial_equity", "default_leverage",
               "balance_fraction", "daily_loss_pct", "total_loss_pct", "max_concurrent_positions",
               "max_orders_per_day", "telegram_enabled")
    core.atomic_write(core.canary_dir(candidate_id) / "config_last_seen.json",
                      json.dumps({k: cfg.get(k) for k in watched}, default=str))


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="GEN2 operational config (single source of truth)")
    p.add_argument("--mode-op", choices=["validate", "show", "startup-audit"], default="validate")
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.candidate_id)
    except Exception as exc:
        print(json.dumps({"valid": False, "reason": str(exc)}, indent=2))
        return 1
    if args.mode_op == "startup-audit":
        print(json.dumps(write_startup_audit(args.candidate_id), indent=2, default=str))
    elif args.mode_op == "show":
        print(json.dumps(cfg, indent=2, default=str))
    else:
        print(json.dumps({"valid": True, "mode": cfg["mode"], "execution_enabled": cfg["execution_enabled"],
                          "symbols": cfg["symbols"], "checksum": cfg["config_checksum"][:16]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
