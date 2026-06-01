#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + ["LINKUSDT"]
LOOKBACK = {"LTCUSDT": 14, "AVAXUSDT": 14, "ETHUSDT": 30, "SUIUSDT": 7, "ADAUSDT": 30, "DOGEUSDT": 30, "BTCUSDT": 30, "BNBUSDT": 30, "XRPUSDT": 30, "SOLUSDT": 30}
ROOT = Path(__file__).resolve().parents[2]


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def nested(data: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_env_dirty(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    try:
        out = subprocess.check_output(["git", "status", "--short", "--", ".env", "binance-futures-bot-ts/.env"], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 128:
            return []
        return [f"git_env_status_failed:{exc!r}"]
    except Exception as exc:
        return [f"git_env_status_failed:{exc!r}"]
    return [line for line in out.splitlines() if line.strip()]


def validate(root: Path = ROOT, wallet_usdt: float = 20.0) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ts_yaml_path = root / "binance-futures-bot-ts/regime_config.live.yaml"
    py_yaml_path = root / "aegis_alpha/configs/turbo.yaml"
    ts = load_yaml(ts_yaml_path)
    py = load_yaml(py_yaml_path)
    phase = nested(ts, ["aegis", "phase_o_short_live"], {}) or {}
    turbo = nested(ts, ["aegis", "turbo"], {}) or {}
    entry_guards = nested(ts, ["aegis", "entry_policy", "guards"], {}) or {}
    py_sizing = nested(py, ["sizing"], {}) or {}

    if phase.get("enabled") is not True: errors.append("phase_o_short_live.enabled_not_true")
    if phase.get("allow_orders") is not True: errors.append("phase_o_short_live.allow_orders_not_true")
    if phase.get("require_brackets") is not True: errors.append("phase_o_short_live.require_brackets_not_true")
    if phase.get("allow_link_entry") is not False: errors.append("phase_o_short_live.allow_link_entry_not_false")
    if phase.get("link_avoid_only") is not True: errors.append("phase_o_short_live.link_avoid_only_not_true")
    if int(phase.get("max_open_phase_o_positions", 999)) > 1: errors.append("phase_o_short_live.max_open_phase_o_positions_gt_1")
    if int(phase.get("max_phase_o_trades_per_day", 999)) > 3: errors.append("phase_o_short_live.max_phase_o_trades_per_day_gt_3")

    lev = phase.get("leverage", {}) or {}
    expected_lev = {"conservative": 15, "normal": 20, "premium": 25, "max_allowed_leverage": 30}
    for key, expected in expected_lev.items():
        if float(lev.get(key, -1)) != float(expected): errors.append(f"phase_o_leverage_{key}_not_{expected}")
    if phase.get("align_short_leverage_with_long") is not True: errors.append("align_short_leverage_with_long_not_true")
    if phase.get("remove_legacy_short_leverage_downshift") is not True: errors.append("remove_legacy_short_leverage_downshift_not_true")

    for key, expected in expected_lev.items():
        if key == "max_allowed_leverage":
            value = py_sizing.get("max_allowed_leverage")
        else:
            value = (py_sizing.get(key, {}) or {}).get("leverage")
        if float(value or -1) != float(expected): errors.append(f"python_turbo_leverage_{key}_not_{expected}")
    if float(py_sizing.get("max_allowed_position_fraction", 999)) > 0.01: errors.append("python_turbo_position_fraction_gt_0.01")

    guard_modes = phase.get("guard_modes", {}) or {}
    desired = {
        "phase_o_ml": "ENFORCE", "momentum_ride": "ENFORCE", "clean_entry": "SHADOW", "event_risk": "SHADOW",
        "entry_quality": "SHADOW", "decision_brain": "SHADOW", "regime_engine": "SHADOW", "short_gate_legacy": "SHADOW", "risk_shadow_guards": "SHADOW",
    }
    for guard, mode in desired.items():
        if guard_modes.get(guard) != mode: errors.append(f"guard_modes.{guard}_not_{mode}")
    global_expected = {
        "momentum_ride": "ENFORCE",
        "clean_entry": "ENFORCE",
        "event_risk": "ENFORCE",
        "entry_quality": "ENFORCE",
        "decision_brain": "ENFORCE",
        "regime": "SHADOW",
        "short_gate": "ENFORCE",
        "long_risk_shadow": "ENFORCE_PROBE_LONG_CRITICAL",
    }
    for guard, expected in global_expected.items():
        actual = (entry_guards.get(guard, {}) or {}).get("mode")
        if actual != expected: errors.append(f"entry_policy_global.{guard}_not_{expected}")
    trading_service = root / "binance-futures-bot-ts/src/app/services/TradingService.ts"
    service_text = trading_service.read_text(encoding="utf-8") if trading_service.exists() else ""
    if "phase_o_short_guard_modes_applied" not in service_text or "withPhaseOShortGuardModes" not in service_text:
        errors.append("phase_o_short_scoped_guard_override_missing")

    hard = phase.get("hard_safety", {}) or {}
    for key in ["brackets", "max_open_positions", "max_trades_per_day", "daily_loss_stop", "exchange_min_notional", "exchange_order_errors", "link_no_entry"]:
        if hard.get(key) != "ENFORCE": errors.append(f"hard_safety.{key}_not_ENFORCE")
    if turbo.get("require_brackets") is not True: errors.append("aegis.turbo.require_brackets_not_true")
    if float(turbo.get("position_fraction_cap", 999)) > 0.01: errors.append("aegis.turbo.position_fraction_cap_gt_0.01")
    if int(turbo.get("max_trades_per_day", 999)) > 3: errors.append("aegis.turbo.max_trades_per_day_gt_3")
    if int(turbo.get("max_consecutive_losses", 999)) > 2: errors.append("aegis.turbo.max_consecutive_losses_gt_2")

    symbols = phase.get("symbols", {}) or {}
    for symbol in ENTRY_SYMBOLS:
        cfg = symbols.get(symbol, {}) or {}
        if cfg.get("enabled") is not True: errors.append(f"{symbol}:phase_symbol_not_enabled")
        if float(cfg.get("max_position_fraction", 999)) > 0.01: errors.append(f"{symbol}:max_position_fraction_gt_0.01")
        manifest = read_json(root / "aegis_alpha/models/turbo" / symbol / "active_manifest.json")
        mp = manifest.get("model_paths") if isinstance(manifest.get("model_paths"), dict) else {}
        pre = manifest.get("pre_phase_o_live_model_paths") if isinstance(manifest.get("pre_phase_o_live_model_paths"), dict) else {}
        short_key = f"short_{LOOKBACK[symbol]}d"
        if manifest.get("phase_o_live_enabled") is not True: errors.append(f"{symbol}:phase_o_live_enabled_not_true")
        if "/phase_o_" not in str(mp.get(short_key, "")): errors.append(f"{symbol}:{short_key}_not_phase_o")
        for key, value in mp.items():
            if key.startswith("long_") and pre and pre.get(key) != value:
                errors.append(f"{symbol}:long_model_path_changed:{key}")
    link = symbols.get("LINKUSDT", {}) or {}
    if link.get("entry_enabled") is not False or link.get("avoid_only") is not True:
        errors.append("LINKUSDT:phase_config_entry_not_disabled")
    link_manifest = read_json(root / "aegis_alpha/models/turbo/LINKUSDT/active_manifest.json")
    if link_manifest.get("phase_o_link_entry_enabled") is not False: errors.append("LINKUSDT:manifest_entry_not_disabled")
    if link_manifest.get("phase_o_avoid_only") is not True: errors.append("LINKUSDT:manifest_avoid_only_not_true")
    link_mp = link_manifest.get("model_paths") if isinstance(link_manifest.get("model_paths"), dict) else {}
    if any("/phase_o_" in str(v) for k, v in link_mp.items() if k.startswith("short_")):
        errors.append("LINKUSDT:entry_short_model_points_phase_o")

    overrides = nested(ts, ["SYMBOL_OVERRIDES"], {}) or {}
    for symbol in ENTRY_SYMBOLS:
        profile = nested(overrides, [symbol, "AEGIS_TURBO", "profile"])
        if profile in {"reduced_10x", "reduced_8x"}:
            errors.append(f"{symbol}:legacy_short_downshift_profile:{profile}")
        if profile == "reduced_15x":
            warnings.append(f"{symbol}:conservative_profile_15x")

    momentum = nested(ts, ["aegis", "momentum_ride"], {}) or {}
    caps = momentum.get("safety_caps", {}) or {}
    if float(caps.get("max_leverage", 999)) > 30: errors.append("momentum.max_leverage_gt_30")
    if float(caps.get("max_position_fraction", 999)) > 0.01: errors.append("momentum.max_position_fraction_gt_0.01")
    for name, cfg in (momentum.get("profiles", {}) or {}).items():
        if float((cfg or {}).get("leverage", 0) or 0) > 30: errors.append(f"momentum.profile.{name}.leverage_gt_30")
        if float((cfg or {}).get("position_fraction", 0) or 0) > 0.01: errors.append(f"momentum.profile.{name}.position_fraction_gt_0.01")

    for dirty in git_env_dirty(root):
        errors.append(f"env_dirty:{dirty}")

    status = "PASSED" if not errors else "FAILED"
    return {
        "schema_version": "aegis_phase_o3_live_controls_preflight_v1",
        "created_at": now_iso(),
        "wallet_usdt": wallet_usdt,
        "status": status,
        "block_pm2_restart": bool(errors),
        "errors": errors,
        "warnings": warnings,
        "manual_orders_sent": False,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    rs = stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"aegis_phase_o3_live_controls_preflight_{rs}.json"
    md_path = out_dir / f"aegis_phase_o3_live_controls_preflight_{rs}.md"
    write_json(json_path, report)
    lines = [f"# Phase O.3 Live Controls Preflight {rs}", "", f"- status: {report['status']}", f"- block_pm2_restart: {report['block_pm2_restart']}", "", "## Errors"]
    lines += [f"- {e}" for e in report["errors"]] or ["- none"]
    lines += ["", "## Warnings"]
    lines += [f"- {w}" for w in report["warnings"]] or ["- none"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet-usdt", type=float, default=20.0)
    ap.add_argument("--out-dir", default="/home/jasan/Develop")
    args = ap.parse_args()
    report = validate(ROOT, args.wallet_usdt)
    json_path, md_path = write_report(report, Path(args.out_dir))
    print(json.dumps({"status": report["status"], "block_pm2_restart": report["block_pm2_restart"], "errors": report["errors"][:30], "json": str(json_path), "md": str(md_path)}, indent=2))
    return 0 if report["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
