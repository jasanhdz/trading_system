#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

REPO = Path(__file__).resolve().parents[2]
TS_REPO = REPO / "binance-futures-bot-ts"
ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
PREDICT_SYMBOLS = ["AVAXUSDT", "ETHUSDT", "SUIUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT"]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if yaml is None:
        return {"_yaml_unavailable": True, "_raw": text}
    try:
        return yaml.safe_load(text) or {}
    except Exception as exc:
        return {"_parse_error": repr(exc)}


def nested(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def inspect_ts_source(text: str) -> dict[str, Any]:
    required_paths = [
        "signal.metadata.aegis.turbo.phase_o",
        "signal.metadata.aegis.turbo.raw.phase_o",
        "signal.metadata.turbo.phase_o",
        "signal.aegis.turbo.phase_o",
        "signal.metadata.rawPrediction.aegis.turbo.phase_o",
        "signal.phase_o",
        "signal.phaseO",
    ]
    present_paths = [path for path in required_paths if path in text]
    checks = {
        "extractor_present": "extractPhaseOTurboMetadata" in text,
        "is_phase_o_uses_extractor": "const metadata = this.extractPhaseOTurboMetadata(signal, side)" in text,
        "multiple_metadata_paths": len(present_paths) >= 5,
        "phase_o_guard_log_present": "phase_o_short_guard_modes_applied" in text,
        "source_path_logged": "phase_o_metadata_source_path" in text,
        "single_rigid_path_only": "turbo.phase_o ?? turbo.raw?.phase_o" in text and "extractPhaseOTurboMetadata" not in text,
        "present_paths": present_paths,
    }
    checks["ok"] = all([checks["extractor_present"], checks["is_phase_o_uses_extractor"], checks["multiple_metadata_paths"], checks["phase_o_guard_log_present"], checks["source_path_logged"]]) and not checks["single_rigid_path_only"]
    return checks


def audit_config(ts_cfg_path: Path = TS_REPO / "regime_config.live.yaml") -> dict[str, Any]:
    cfg = load_yaml(ts_cfg_path)
    phase = nested(cfg, "aegis", "phase_o_short_live", default={}) or {}
    turbo = nested(cfg, "aegis", "turbo", default={}) or {}
    hard = phase.get("hard_safety") or {}
    errors = []
    if phase.get("enabled") is not True:
        errors.append("PHASE_O_DISABLED")
    if phase.get("allow_orders") is not True:
        errors.append("ALLOW_ORDERS_DISABLED")
    if phase.get("require_brackets") is not True:
        errors.append("BRACKETS_NOT_REQUIRED")
    if phase.get("allow_link_entry") is not False or phase.get("link_avoid_only") is not True:
        errors.append("LINK_ENTRY_RISK")
    for key in ["brackets", "daily_loss_stop", "exchange_min_notional", "exchange_order_errors", "link_no_entry"]:
        if hard.get(key) != "ENFORCE":
            errors.append(f"HARD_SAFETY_NOT_ENFORCE:{key}")
    return {"phase_o_short_live": phase, "turbo": turbo, "hard_safety": hard, "errors": errors, "ok": not errors}


def audit_manifests(model_root: Path = REPO / "aegis_alpha" / "models" / "turbo") -> dict[str, Any]:
    rows = []
    errors = []
    for symbol in ENTRY_SYMBOLS + ["LINKUSDT"]:
        path = model_root / symbol / "active_manifest.json"
        row = {"symbol": symbol, "path": str(path), "exists": path.exists(), "status": "UNKNOWN"}
        if not path.exists():
            row["status"] = "MISSING"
            errors.append(f"{symbol}:manifest_missing")
            rows.append(row)
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            row["status"] = "PARSE_ERROR"
            row["error"] = repr(exc)
            errors.append(f"{symbol}:manifest_parse_error")
            rows.append(row)
            continue
        model_paths = manifest.get("model_paths") or {}
        short_paths = {k: str(v) for k, v in model_paths.items() if str(k).startswith("short_")}
        phase_paths = {k: v for k, v in short_paths.items() if "phase_o" in v.lower()}
        missing = [v for v in phase_paths.values() if not Path(v).exists()]
        row.update({
            "phase_o_live_enabled": manifest.get("phase_o_live_enabled"),
            "phase_o_overlay_persistence_enabled": manifest.get("phase_o_overlay_persistence_enabled"),
            "phase_o_avoid_only": manifest.get("phase_o_avoid_only"),
            "phase_o_link_entry_enabled": manifest.get("phase_o_link_entry_enabled"),
            "phase_short_path_count": len(phase_paths),
            "missing_phase_path_count": len(missing),
        })
        if symbol == "LINKUSDT":
            ok = manifest.get("phase_o_avoid_only") is True and manifest.get("phase_o_link_entry_enabled") is False and not phase_paths
            row["status"] = "LINK_AVOID_ONLY_OK" if ok else "LINK_BAD_ENTRY_ENABLED"
        else:
            ok = manifest.get("phase_o_live_enabled") is True and manifest.get("phase_o_overlay_persistence_enabled") is True and bool(phase_paths) and not missing
            row["status"] = "PHASE_O_MANIFEST_OK" if ok else "PHASE_O_MANIFEST_DRIFTED"
        if not row["status"].endswith("OK"):
            errors.append(f"{symbol}:{row['status']}")
        rows.append(row)
    return {"rows": rows, "errors": errors, "ok": not errors}


def predict_symbol(symbol: str, timeout: float) -> dict[str, Any]:
    req = Request("http://127.0.0.1:8001/ml-v2/predict", data=json.dumps({"symbol": symbol}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        turbo = ((payload.get("aegis") or {}).get("turbo") or {}) if isinstance(payload, dict) else {}
        phase = turbo.get("phase_o") or (turbo.get("raw") or {}).get("phase_o") or payload.get("phase_o") or {}
        action = turbo.get("action") or (turbo.get("gated") or {}).get("action") or (turbo.get("raw") or {}).get("action")
        return {"symbol": symbol, "http_status": resp.status, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "action": action, "reason": turbo.get("reason") or (turbo.get("gated") or {}).get("reason") or (turbo.get("raw") or {}).get("reason"), "phase_o_metadata_present": bool(phase), "link_avoid_only": phase.get("phase_o_link_avoid_only"), "link_entry_enabled": phase.get("phase_o_link_entry_enabled"), "error": None}
    except Exception as exc:
        return {"symbol": symbol, "http_status": None, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "phase_o_metadata_present": False, "error": repr(exc)}


def audit_predict(timeout: float, skip: bool = False) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "rows": [], "errors": [], "ok": True}
    rows = [predict_symbol(symbol, timeout) for symbol in PREDICT_SYMBOLS]
    errors = []
    for row in rows:
        symbol = row["symbol"]
        if row.get("http_status") != 200:
            errors.append(f"{symbol}:predict_error")
        if symbol != "LINKUSDT" and not row.get("phase_o_metadata_present"):
            errors.append(f"{symbol}:phase_o_metadata_missing")
        if symbol == "LINKUSDT" and not (row.get("link_avoid_only") is True and row.get("link_entry_enabled") is False):
            errors.append("LINKUSDT:not_avoid_only")
    return {"rows": rows, "errors": errors, "ok": not errors}


def git_status() -> list[str]:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, timeout=10, check=False).stdout
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_path = TS_REPO / "src" / "app" / "services" / "TradingService.ts"
    source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path.exists() else ""
    ts_source = inspect_ts_source(source)
    config = audit_config()
    manifests = audit_manifests()
    predict = audit_predict(args.predict_timeout, args.skip_predict)
    status_lines = git_status()
    active_manifest_changes = [line for line in status_lines if "active_manifest" in line]
    yaml_changes = [line for line in status_lines if line.endswith(".yaml") or line.endswith(".yml")]
    errors = []
    if not ts_source["ok"]:
        errors.append("TS_EXTRACTOR_NOT_ROBUST")
    errors.extend(config["errors"])
    errors.extend(manifests["errors"])
    errors.extend(predict["errors"])
    if active_manifest_changes:
        errors.append("ACTIVE_MANIFEST_CHANGED")
    if yaml_changes:
        errors.append("YAML_CHANGED")
    result = "PASSED" if not errors else "FAILED"
    return {
        "schema_version": "phase_o_short_metadata_fix_preflight_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "BLOCK_PM2_RESTART": result != "PASSED",
        "errors": errors,
        "ts_source": ts_source,
        "config": config,
        "manifests": manifests,
        "predict": predict,
        "git_status": status_lines,
        "confirmations": {"read_only": True, "no_active_manifest": not active_manifest_changes, "no_yaml": not yaml_changes, "link_no_entry": "LINKUSDT:not_avoid_only" not in errors},
    }


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    md = out_dir / f"aegis_phase_o_short_metadata_fix_preflight_{stamp}.md"
    js = out_dir / f"aegis_phase_o_short_metadata_fix_preflight_{stamp}.json"
    js.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Phase O SHORT Metadata Fix Preflight", "", f"- Result: `{payload['result']}`", f"- BLOCK_PM2_RESTART: `{payload['BLOCK_PM2_RESTART']}`", "", "## Errors"]
    lines += [f"- {err}" for err in payload["errors"]] or ["- none"]
    lines += ["", "## TS Source", f"- extractor_present: `{payload['ts_source']['extractor_present']}`", f"- multiple_metadata_paths: `{payload['ts_source']['multiple_metadata_paths']}`", f"- present_paths: `{payload['ts_source']['present_paths']}`", "", "## Predict"]
    for row in payload["predict"]["rows"]:
        lines.append(f"- {row['symbol']}: http={row.get('http_status')} action={row.get('action')} phase_o={row.get('phase_o_metadata_present')} link_avoid={row.get('link_avoid_only')} error={row.get('error')}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"md": str(md), "json": str(js)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/jasan/Develop")
    ap.add_argument("--predict-timeout", type=float, default=8.0)
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()
    payload = run_preflight(args)
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({"result": payload["result"], "BLOCK_PM2_RESTART": payload["BLOCK_PM2_RESTART"], "errors": payload["errors"], "reports": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
