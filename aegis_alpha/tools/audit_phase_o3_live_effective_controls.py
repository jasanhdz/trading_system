#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + ["LINKUSDT"]
ROOT = Path(__file__).resolve().parents[2]
TS_DIR = ROOT / "binance-futures-bot-ts"
LOOKBACK = {"LTCUSDT": 14, "AVAXUSDT": 14, "ETHUSDT": 30, "SUIUSDT": 7, "ADAUSDT": 30, "DOGEUSDT": 30, "BTCUSDT": 30, "BNBUSDT": 30, "XRPUSDT": 30, "SOLUSDT": 30}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def nested(data: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def code_refs(root: Path) -> list[dict[str, Any]]:
    files = [
        root / "aegis_alpha/configs/turbo.yaml",
        root / "aegis_alpha/turbo/turbo_signal.py",
        root / "aegis_alpha/inference/server.py",
        root / "binance-futures-bot-ts/regime_config.live.yaml",
        root / "binance-futures-bot-ts/src/app/services/TradingService.ts",
        root / "binance-futures-bot-ts/src/domain/services/aegis-entry/AegisEntryGuardOrchestrator.ts",
        root / "binance-futures-bot-ts/src/domain/services/aegis-entry/guards/ShortGateGuardAdapter.ts",
        root / "binance-futures-bot-ts/src/domain/services/AegisShortGate.ts",
        root / "binance-futures-bot-ts/src/domain/services/AegisMicroLiveGate.ts",
        root / "binance-futures-bot-ts/src/infra/config/ConfigLoader.ts",
    ]
    patterns = ["leverage", "position_fraction", "positionFraction", "phase_o_short_live", "short_gate", "entry_quality", "event_risk", "decision_brain", "clean_entry", "regime", "momentum", "bracket", "setLeverage", "wouldBlock"]
    rows: list[dict[str, Any]] = []
    for path in files:
        text = read_text(path)
        for line_no, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            matched = [p for p in patterns if p.lower() in low]
            if matched:
                rows.append({"file": str(path.relative_to(root)), "line": line_no, "patterns": ",".join(matched[:4]), "text": line.strip()[:220]})
    return rows


def active_manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for symbol in ALL_SYMBOLS:
        path = root / "aegis_alpha/models/turbo" / symbol / "active_manifest.json"
        data = read_json(path)
        model_paths = data.get("model_paths") if isinstance(data.get("model_paths"), dict) else {}
        pre = data.get("pre_phase_o_live_model_paths") if isinstance(data.get("pre_phase_o_live_model_paths"), dict) else {}
        short_key = f"short_{LOOKBACK.get(symbol, 30)}d" if symbol in LOOKBACK else ""
        rows.append({
            "symbol": symbol,
            "manifest_exists": path.exists(),
            "phase_o_live_enabled": data.get("phase_o_live_enabled"),
            "link_entry_enabled": data.get("phase_o_link_entry_enabled"),
            "link_avoid_only": data.get("phase_o_avoid_only"),
            "short_key": short_key,
            "short_path": model_paths.get(short_key, ""),
            "short_points_phase_o": "/phase_o_" in str(model_paths.get(short_key, "")),
            "long_paths_intact": all(pre.get(k) == v for k, v in model_paths.items() if k.startswith("long_") and pre),
        })
    return rows


def guard_rows(ts_yaml: dict[str, Any]) -> list[dict[str, Any]]:
    phase_modes = nested(ts_yaml, ["aegis", "phase_o_short_live", "guard_modes"], {}) or {}
    entry_guards = nested(ts_yaml, ["aegis", "entry_policy", "guards"], {}) or {}
    wanted = {
        "phase_o_ml": "ENFORCE", "momentum_ride": "ENFORCE", "clean_entry": "SHADOW", "event_risk": "SHADOW",
        "entry_quality": "SHADOW", "decision_brain": "SHADOW", "regime_engine": "SHADOW", "short_gate_legacy": "SHADOW", "risk_shadow_guards": "SHADOW",
    }
    rows = []
    for name, desired in wanted.items():
        effective_key = {"regime_engine": "regime", "short_gate_legacy": "short_gate", "risk_shadow_guards": "long_risk_shadow", "phase_o_ml": "aegis_turbo"}.get(name, name)
        effective = "ENFORCE" if name == "phase_o_ml" else (entry_guards.get(effective_key, {}) or {}).get("mode")
        rows.append({"guard": name, "desired_mode": desired, "phase_o_mode": phase_modes.get(name), "effective_guard": effective_key, "effective_mode": effective})
    return rows


def sizing_rows(ts_yaml: dict[str, Any], wallet_usdt: float) -> list[dict[str, Any]]:
    phase = nested(ts_yaml, ["aegis", "phase_o_short_live"], {}) or {}
    lev = phase.get("leverage", {}) or {}
    symbols = phase.get("symbols", {}) or {}
    normal_leverage = float(lev.get("normal", 20))
    rows = []
    for symbol in ENTRY_SYMBOLS:
        cfg = symbols.get(symbol, {}) or {}
        frac = float(cfg.get("max_position_fraction", phase.get("max_position_fraction_default", 0.01)))
        margin = wallet_usdt * frac
        rows.append({
            "symbol": symbol,
            "max_position_fraction": frac,
            "margin_usdt_at_wallet": round(margin, 6),
            "normal_leverage": normal_leverage,
            "notional_usdt_at_normal": round(margin * normal_leverage, 6),
            "caution_level": cfg.get("caution_level", ""),
        })
    return rows


def run_audit(root: Path = ROOT, out_dir: Path = Path("/home/jasan/Develop"), wallet_usdt: float = 20.0) -> dict[str, Any]:
    stamp = utc_stamp()
    ts_yaml_path = root / "binance-futures-bot-ts/regime_config.live.yaml"
    py_yaml_path = root / "aegis_alpha/configs/turbo.yaml"
    ts_yaml = load_yaml(ts_yaml_path)
    py_yaml = load_yaml(py_yaml_path)
    manifests = active_manifest_rows(root)
    guards = guard_rows(ts_yaml)
    sizing = sizing_rows(ts_yaml, wallet_usdt)
    refs = code_refs(root)
    phase = nested(ts_yaml, ["aegis", "phase_o_short_live"], {}) or {}
    report = {
        "schema_version": "aegis_phase_o3_controls_audit_v1",
        "created_at": now_iso(),
        "wallet_usdt": wallet_usdt,
        "python_turbo_sizing": nested(py_yaml, ["sizing"], {}),
        "ts_phase_o_short_live": phase,
        "effective_summary": {
            "long_leverage_scheme": nested(py_yaml, ["sizing"], {}),
            "short_leverage_policy": phase.get("leverage", {}),
            "short_downshift_symbols": [],
            "secondary_guards_shadow_requested": phase.get("guard_modes", {}),
            "hard_safety": phase.get("hard_safety", {}),
        },
        "symbols": manifests,
        "guards": guards,
        "sizing_20usdt": sizing,
        "code_refs_count": len(refs),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"aegis_phase_o3_controls_audit_{stamp}.json", report)
    with (out_dir / f"aegis_phase_o3_controls_audit_symbols_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifests[0].keys()))
        w.writeheader(); w.writerows(manifests)
    with (out_dir / f"aegis_phase_o3_controls_audit_guards_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(guards[0].keys()))
        w.writeheader(); w.writerows(guards)
    with (out_dir / f"aegis_phase_o3_effective_sizing_20usdt_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sizing[0].keys()))
        w.writeheader(); w.writerows(sizing)
    with (out_dir / f"aegis_phase_o3_controls_audit_code_refs_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "patterns", "text"])
        w.writeheader(); w.writerows(refs)
    md = [
        f"# Phase O.3 Controls Audit {stamp}", "",
        "## Summary",
        f"- wallet_usdt: {wallet_usdt}",
        f"- Phase O enabled: {phase.get('enabled')}",
        f"- leverage policy: {phase.get('leverage')}",
        f"- guard modes: {phase.get('guard_modes')}", "",
        "## Effective Sizing 20 USDT",
        "|symbol|fraction|margin_usdt|normal_leverage|notional_usdt|caution|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in sizing:
        md.append(f"|{row['symbol']}|{row['max_position_fraction']}|{row['margin_usdt_at_wallet']}|{row['normal_leverage']}|{row['notional_usdt_at_normal']}|{row['caution_level']}|")
    md += ["", "## Guards", "|guard|desired|configured|effective|", "|---|---|---|---|"]
    for row in guards:
        md.append(f"|{row['guard']}|{row['desired_mode']}|{row['phase_o_mode']}|{row['effective_mode']}|")
    (out_dir / f"aegis_phase_o3_controls_audit_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"stamp": stamp, "report": str(out_dir / f'aegis_phase_o3_controls_audit_{stamp}.md')}, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet-usdt", type=float, default=20.0)
    ap.add_argument("--out-dir", default="/home/jasan/Develop")
    args = ap.parse_args()
    run_audit(ROOT, Path(args.out_dir), args.wallet_usdt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
