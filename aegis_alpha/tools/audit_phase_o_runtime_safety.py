#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol, turbo_symbol_model_dir  # noqa: E402

ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
AVOID_ONLY_SYMBOLS = ["LINKUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + AVOID_ONLY_SYMBOLS
SHORT_KEYS = ("short_7d", "short_14d", "short_30d")
CODE_FILES = [
    "aegis_alpha/inference/server.py",
    "aegis_alpha/turbo/turbo_signal.py",
    "aegis_alpha/turbo/snapshot_utils.py",
    "aegis_alpha/turbo/train_recent_edge.py",
    "aegis_alpha/tools/generate_short_shadow_artifacts_o.py",
    "aegis_alpha/tools/validate_short_shadow_artifacts_o.py",
]
PATTERNS = ("active_manifest.json", "model_paths", "active/", "phase_o", "short_7d", "short_14d", "short_30d", "evaluate_turbo_shadow")
REPORT_SCHEMA_VERSION = "aegis_phase_o_runtime_safety_v1"


@dataclass(frozen=True)
class CodeFindings:
    rows: list[dict[str, Any]]
    turbo_signal_loads_active_manifest: bool
    turbo_signal_uses_model_paths_directly: bool
    turbo_signal_phase_o_aware: bool
    server_calls_evaluate_turbo_shadow: bool
    yaml_guard_detected: bool
    shadow_only_flag_detected: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__read_error__": repr(exc)}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_phase_o_path(path: Any) -> bool:
    return "/active/phase_o_" in str(path).replace("\\", "/")


def parse_short_model_paths(manifest: Mapping[str, Any]) -> dict[str, str]:
    model_paths = manifest.get("model_paths") if isinstance(manifest, Mapping) else None
    if not isinstance(model_paths, Mapping):
        return {}
    return {str(key): str(value) for key, value in model_paths.items() if str(key).startswith("short_")}


def classify_phase_o_runtime_risk(row: Mapping[str, Any]) -> str:
    symbol = str(row.get("symbol", ""))
    if bool(row.get("manifest_read_error")) or bool(row.get("loader_unknown")):
        return "LIVE_IMPACT_UNKNOWN"
    if symbol == "LINKUSDT" and bool(row.get("entry_enabled")):
        return "DANGEROUS"
    if bool(row.get("affects_orders")) or bool(row.get("affects_gating")) or bool(row.get("affects_sizing")):
        return "DANGEROUS"
    if bool(row.get("link_runtime_confusion_risk")):
        return "DANGEROUS"
    if bool(row.get("model_paths_point_to_phase_o")) and bool(row.get("runtime_uses_model_paths_directly")) and not bool(row.get("yaml_guard_detected")):
        return "LIVE_PATH_REPLACED"
    if bool(row.get("global_artifact_path_missing")):
        return "LIVE_IMPACT_UNKNOWN"
    if bool(row.get("active_manifest_has_phase_o_fields")):
        return "PASSIVE_BUT_AMBIGUOUS"
    return "SAFE_PASSIVE"


def classify_global_risk(symbol_rows: Sequence[Mapping[str, Any]]) -> str:
    classes = {str(row.get("risk_class")) for row in symbol_rows}
    if "DANGEROUS" in classes:
        return "NEEDS_ROLLBACK_OR_MANIFEST_PATCH"
    if "LIVE_PATH_REPLACED" in classes:
        return "NEEDS_FIX_BEFORE_YAML"
    if "LIVE_IMPACT_UNKNOWN" in classes:
        return "UNKNOWN_NEEDS_MANUAL_REVIEW"
    return "SAFE_TO_ENABLE_YAML_SHADOW_LATER"


def inspect_code(root: Path) -> CodeFindings:
    rows: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    for rel in CODE_FILES:
        path = root / rel
        if not path.exists():
            rows.append({"file": rel, "line": "", "pattern": "missing_file", "text": ""})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        contents[rel] = text
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in PATTERNS:
                if pattern in line:
                    rows.append({"file": rel, "line": lineno, "pattern": pattern, "text": line.strip()[:240]})
    turbo = contents.get("aegis_alpha/turbo/turbo_signal.py", "")
    server = contents.get("aegis_alpha/inference/server.py", "")
    all_text = "\n".join(contents.values())
    runtime_text = turbo + "\n" + server
    return CodeFindings(
        rows=rows,
        turbo_signal_loads_active_manifest='manifest_path = symbol_dir / "active_manifest.json"' in turbo,
        turbo_signal_uses_model_paths_directly="model_paths.get(manifest_key)" in turbo or "model_paths" in turbo and "candidate.exists()" in turbo,
        turbo_signal_phase_o_aware="phase_o" in turbo,
        server_calls_evaluate_turbo_shadow="evaluate_turbo_shadow" in server,
        yaml_guard_detected="yaml_shadow_expected" in runtime_text or ("phase_o" in runtime_text and "YAML" in runtime_text),
        shadow_only_flag_detected="SHADOW_ONLY" in server or "shadow" in server.lower(),
    )


def load_phase_o_global(root: Path) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    pointer_path = root / "aegis_alpha/models/turbo/phase_o_short_manifest.json"
    pointer = read_json(pointer_path)
    latest_text = pointer.get("latest_manifest") if isinstance(pointer, dict) else None
    latest_path = Path(str(latest_text)) if latest_text else None
    if latest_path is not None and not latest_path.is_absolute():
        latest_path = root / latest_path
    global_manifest = read_json(latest_path) if latest_path else {}
    return pointer, latest_path, global_manifest


def find_symbol_shadow_manifest(symbol: str, active: Mapping[str, Any], global_manifest: Mapping[str, Any]) -> Path | None:
    artifact_paths = global_manifest.get("artifact_paths") if isinstance(global_manifest, Mapping) else None
    if isinstance(artifact_paths, Mapping) and artifact_paths.get(symbol):
        return Path(str(artifact_paths[symbol]))
    phase = active.get("phase_o_symbols") if isinstance(active, Mapping) else None
    if isinstance(phase, Mapping):
        symbol_phase = phase.get(symbol)
        if isinstance(symbol_phase, Mapping) and symbol_phase.get("symbol_manifest"):
            return Path(str(symbol_phase["symbol_manifest"]))
    candidates = sorted((turbo_symbol_model_dir(symbol) / "active").glob("phase_o_*/symbol_shadow_manifest.json"))
    return candidates[-1] if candidates else None


def simulate_runtime_model_resolution(symbol: str, root: Optional[Path] = None) -> dict[str, Any]:
    symbol = normalize_turbo_symbol(symbol)
    symbol_dir = turbo_symbol_model_dir(symbol)
    manifest_path = symbol_dir / "active_manifest.json"
    active = read_json(manifest_path)
    resolved: dict[str, str] = {}
    phase_paths: list[str] = []
    for lookback in DEFAULT_TURBO_CONFIG.lookback_days:
        key = f"short_{int(lookback)}d"
        raw = (active.get("model_paths") or {}).get(key) if isinstance(active.get("model_paths"), Mapping) else None
        if raw:
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                candidate = (root or Path.cwd()) / candidate
            if candidate.exists():
                resolved[key] = str(candidate)
                if is_phase_o_path(candidate):
                    phase_paths.append(str(candidate))
                continue
        fallback = symbol_dir / "active" / f"turbo_short_edge_{int(lookback)}d_v010.joblib"
        if fallback.exists():
            resolved[key] = str(fallback)
            if is_phase_o_path(fallback):
                phase_paths.append(str(fallback))
        else:
            legacy = symbol_dir / f"turbo_short_edge_{int(lookback)}d_v010.joblib"
            resolved[key] = str(legacy)
            if is_phase_o_path(legacy):
                phase_paths.append(str(legacy))
    return {
        "symbol": symbol,
        "resolved_manifest_path": str(manifest_path),
        "resolved_model_paths": resolved,
        "resolved_short_windows": sorted(resolved),
        "phase_o_paths_detected": phase_paths,
        "would_load_phase_o_without_yaml": bool(phase_paths),
        "reason": "turbo_signal._model_path reads active_manifest.model_paths directly when file exists" if phase_paths else "no resolved short path points to phase_o",
    }


def inspect_symbol(symbol: str, code: CodeFindings, global_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol = normalize_turbo_symbol(symbol)
    manifest_path = turbo_symbol_model_dir(symbol) / "active_manifest.json"
    active = read_json(manifest_path)
    manifest_read_error = bool(active.get("__read_error__"))
    short_paths = parse_short_model_paths(active)
    phase_o_short_paths = {key: value for key, value in short_paths.items() if is_phase_o_path(value)}
    artifact_paths = global_manifest.get("artifact_paths") if isinstance(global_manifest, Mapping) else None
    expected_symbol_manifest_path = Path(str(artifact_paths[symbol])) if isinstance(artifact_paths, Mapping) and artifact_paths.get(symbol) else None
    expected_symbol_manifest_exists = bool(expected_symbol_manifest_path and expected_symbol_manifest_path.exists())
    symbol_manifest_path = find_symbol_shadow_manifest(symbol, active, global_manifest)
    symbol_manifest = read_json(symbol_manifest_path) if symbol_manifest_path and symbol_manifest_path.exists() else {}
    active_phase_fields = any(key in active for key in ("phase_o_prod_ready", "yaml_shadow_expected", "phase_o_artifact_stamp", "phase_o_symbols", "phase_o_avoid_only"))
    replaced_keys = sorted(phase_o_short_paths)
    sim = simulate_runtime_model_resolution(symbol)
    entry_enabled = bool(symbol_manifest.get("entry_enabled")) if symbol_manifest else None
    avoid_only = bool(symbol_manifest.get("avoid_only")) if symbol_manifest else None
    row: dict[str, Any] = {
        "symbol": symbol,
        "active_manifest_exists": manifest_path.exists(),
        "active_manifest_path": str(manifest_path),
        "manifest_read_error": manifest_read_error,
        "phase_o_prod_ready": bool(active.get("phase_o_prod_ready")),
        "yaml_shadow_expected": bool(active.get("yaml_shadow_expected")),
        "phase_o_artifact_stamp": active.get("phase_o_artifact_stamp", ""),
        "has_phase_o_symbols": isinstance(active.get("phase_o_symbols"), Mapping) and symbol in active.get("phase_o_symbols", {}),
        "active_manifest_has_phase_o_fields": active_phase_fields,
        "short_model_path_keys": sorted(short_paths),
        "has_short_7d": "short_7d" in short_paths,
        "has_short_14d": "short_14d" in short_paths,
        "has_short_30d": "short_30d" in short_paths,
        "model_paths_point_to_phase_o": bool(phase_o_short_paths),
        "phase_o_model_path_keys": replaced_keys,
        "potentially_replaced_model_paths": phase_o_short_paths,
        "has_backup_previous_paths": any(key in active for key in ("previous_model_paths", "backup_model_paths", "pre_phase_o_model_paths", "phase_o_previous_model_paths")),
        "global_artifact_path": str(expected_symbol_manifest_path) if expected_symbol_manifest_path else "",
        "global_artifact_path_exists": expected_symbol_manifest_exists,
        "global_artifact_path_missing": bool(expected_symbol_manifest_path and not expected_symbol_manifest_exists),
        "symbol_shadow_manifest_exists": bool(symbol_manifest_path and symbol_manifest_path.exists()),
        "symbol_shadow_manifest_path": str(symbol_manifest_path) if symbol_manifest_path else "",
        "shadow_type": symbol_manifest.get("shadow_type", "") if symbol_manifest else "",
        "entry_enabled": entry_enabled,
        "avoid_only": avoid_only,
        "affects_orders": bool(symbol_manifest.get("affects_orders")) if symbol_manifest else None,
        "affects_gating": bool(symbol_manifest.get("affects_gating")) if symbol_manifest else None,
        "affects_sizing": bool(symbol_manifest.get("affects_sizing")) if symbol_manifest else None,
        "affects_decision": bool(symbol_manifest.get("affects_decision")) if symbol_manifest else None,
        "runtime_uses_model_paths_directly": code.turbo_signal_uses_model_paths_directly,
        "yaml_guard_detected": code.yaml_guard_detected,
        "runtime_can_read_phase_o_without_yaml": bool(sim["would_load_phase_o_without_yaml"]),
        "link_runtime_confusion_risk": symbol == "LINKUSDT" and bool(sim["would_load_phase_o_without_yaml"]) and bool(entry_enabled),
        "simulation_reason": sim["reason"],
        "resolved_short_windows": sim["resolved_short_windows"],
    }
    row["risk_class"] = classify_phase_o_runtime_risk(row)
    if row["risk_class"] == "LIVE_PATH_REPLACED":
        row["risk_reason"] = "active_manifest.model_paths short key points to /active/phase_o_ and turbo_signal uses model_paths directly without Phase O guard"
    elif row["risk_class"] == "DANGEROUS":
        row["risk_reason"] = "unsafe symbol manifest flags or LINK entry confusion risk"
    elif row["risk_class"] == "LIVE_IMPACT_UNKNOWN" and row.get("global_artifact_path_missing"):
        row["risk_reason"] = "Phase O global manifest points to a missing symbol artifact path; active runtime paths are not Phase O but shadow artifact pointer is stale"
    elif row["risk_class"] == "PASSIVE_BUT_AMBIGUOUS":
        row["risk_reason"] = "Phase O fields are present but active model paths do not resolve to Phase O"
    elif row["risk_class"] == "SAFE_PASSIVE":
        row["risk_reason"] = "No active model path points to Phase O"
    else:
        row["risk_reason"] = "runtime impact could not be determined"
    model_path_rows = []
    for key, value in short_paths.items():
        model_path_rows.append({
            "symbol": symbol,
            "source": "active_manifest.model_paths",
            "key": key,
            "path": value,
            "points_to_phase_o": is_phase_o_path(value),
            "exists": Path(value).exists(),
            "would_runtime_load": key in sim["resolved_model_paths"] and sim["resolved_model_paths"].get(key) == value,
        })
    for key, value in sim["resolved_model_paths"].items():
        if key not in short_paths:
            model_path_rows.append({
                "symbol": symbol,
                "source": "runtime_fallback_resolution",
                "key": key,
                "path": value,
                "points_to_phase_o": is_phase_o_path(value),
                "exists": Path(value).exists(),
                "would_runtime_load": True,
            })
    return row, model_path_rows


def build_global_summary(pointer: Mapping[str, Any], latest_path: Path | None, global_manifest: Mapping[str, Any], symbol_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    link_row = next((row for row in symbol_rows if row.get("symbol") == "LINKUSDT"), {})
    inconsistencies: list[str] = []
    if not pointer:
        inconsistencies.append("missing_phase_o_short_manifest")
    if pointer and not pointer.get("latest_manifest"):
        inconsistencies.append("pointer_missing_latest_manifest")
    if latest_path and not latest_path.exists():
        inconsistencies.append("latest_manifest_missing")
    if global_manifest.get("symbol_count") != 11:
        inconsistencies.append("global_symbol_count_not_11")
    if global_manifest.get("entry_shadow_count") != 10:
        inconsistencies.append("global_entry_count_not_10")
    if global_manifest.get("avoid_only_count") != 1:
        inconsistencies.append("global_avoid_count_not_1")
    if link_row and link_row.get("entry_enabled") is not False:
        inconsistencies.append("link_entry_enabled_not_false")
    if link_row and link_row.get("avoid_only") is not True:
        inconsistencies.append("link_avoid_only_not_true")
    for row in symbol_rows:
        if row.get("global_artifact_path_missing"):
            inconsistencies.append(f"global_artifact_path_missing:{row.get('symbol')}")
    return {
        "phase_o_short_manifest_exists": bool(pointer),
        "phase_o_short_manifest_latest_manifest": pointer.get("latest_manifest", "") if pointer else "",
        "latest_manifest_path": str(latest_path) if latest_path else "",
        "latest_manifest_exists": bool(latest_path and latest_path.exists()),
        "global_symbol_count": global_manifest.get("symbol_count"),
        "global_entry_count": global_manifest.get("entry_shadow_count"),
        "global_avoid_only_count": global_manifest.get("avoid_only_count"),
        "link_entry_enabled_false": link_row.get("entry_enabled") is False if link_row else False,
        "missing_global_artifact_path_count": sum(1 for row in symbol_rows if row.get("global_artifact_path_missing")),
        "inconsistencies": inconsistencies,
    }


def write_reports(out_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    stamp = str(payload["created_stamp"])
    base = out_dir / f"aegis_phase_o_runtime_safety_{stamp}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    symbols_csv = out_dir / f"aegis_phase_o_runtime_safety_symbols_{stamp}.csv"
    code_csv = out_dir / f"aegis_phase_o_runtime_safety_code_refs_{stamp}.csv"
    paths_csv = out_dir / f"aegis_phase_o_runtime_safety_model_paths_{stamp}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_rows = list(payload["symbol_rows"])
    code_findings = payload["code_findings"]
    link = next((row for row in symbol_rows if row.get("symbol") == "LINKUSDT"), {})
    md_lines = [
        f"# Phase O Runtime Safety Audit {stamp}",
        "",
        "## Safety",
        "- READ_ONLY: true",
        "- manifest_changes: false",
        "- yaml_changes: false",
        "- pm2_restart: false",
        "- orders: false",
        "",
        "## Executive Summary",
        f"- global_risk: {payload['global_risk']}",
        f"- biggest_risk: {payload['biggest_risk']}",
        f"- phase_o_already_affects_runtime: {payload['phase_o_already_affects_runtime']}",
        "",
        "## Runtime Code Findings",
        f"- turbo_signal_loads_active_manifest: {code_findings['turbo_signal_loads_active_manifest']}",
        f"- turbo_signal_uses_model_paths_directly: {code_findings['turbo_signal_uses_model_paths_directly']}",
        f"- turbo_signal_phase_o_aware: {code_findings['turbo_signal_phase_o_aware']}",
        f"- server_calls_evaluate_turbo_shadow: {code_findings['server_calls_evaluate_turbo_shadow']}",
        f"- yaml_guard_detected: {code_findings['yaml_guard_detected']}",
        "",
        "## Per-Symbol Risk",
        "| symbol | phase_o_fields | model_paths_phase_o | shadow_type | risk | reason |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in symbol_rows:
        md_lines.append(f"| {row['symbol']} | {row['active_manifest_has_phase_o_fields']} | {row['model_paths_point_to_phase_o']} | {row['shadow_type']} | {row['risk_class']} | {row['risk_reason']} |")
    md_lines += [
        "",
        "## LINK Safety",
        f"- entry_enabled: {link.get('entry_enabled')}",
        f"- avoid_only: {link.get('avoid_only')}",
        f"- model_paths_point_to_phase_o: {link.get('model_paths_point_to_phase_o')}",
        f"- runtime_confusion_risk: {link.get('link_runtime_confusion_risk')}",
        "",
        "## Recommendations",
    ]
    for rec in payload["recommendations"]:
        md_lines.append(f"- {rec}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_json(json_path, payload)
    symbol_fields = ["symbol", "risk_class", "risk_reason", "phase_o_prod_ready", "yaml_shadow_expected", "phase_o_artifact_stamp", "has_phase_o_symbols", "model_paths_point_to_phase_o", "phase_o_model_path_keys", "has_backup_previous_paths", "global_artifact_path", "global_artifact_path_exists", "global_artifact_path_missing", "shadow_type", "entry_enabled", "avoid_only", "affects_orders", "affects_gating", "affects_sizing", "affects_decision", "runtime_can_read_phase_o_without_yaml"]
    write_csv(symbols_csv, symbol_rows, symbol_fields)
    write_csv(code_csv, payload["code_refs"], ["file", "line", "pattern", "text"])
    write_csv(paths_csv, payload["model_path_rows"], ["symbol", "source", "key", "path", "points_to_phase_o", "exists", "would_runtime_load"])
    return {"md": str(md_path), "json": str(json_path), "symbols_csv": str(symbols_csv), "code_refs_csv": str(code_csv), "model_paths_csv": str(paths_csv)}


def run_audit(out_dir: Path, verbose: bool = False) -> dict[str, Any]:
    root = repo_root()
    code = inspect_code(root)
    pointer, latest_path, global_manifest = load_phase_o_global(root)
    symbol_rows: list[dict[str, Any]] = []
    model_path_rows: list[dict[str, Any]] = []
    for symbol in ALL_SYMBOLS:
        row, path_rows = inspect_symbol(symbol, code, global_manifest)
        symbol_rows.append(row)
        model_path_rows.extend(path_rows)
    global_summary = build_global_summary(pointer, latest_path, global_manifest, symbol_rows)
    global_risk = classify_global_risk(symbol_rows)
    phase_o_already_affects = any(bool(row.get("runtime_can_read_phase_o_without_yaml")) for row in symbol_rows)
    if global_risk == "NEEDS_FIX_BEFORE_YAML":
        recommendations = [
            "Do not activate YAML shadow or restart PM2 yet.",
            "Prepare Phase O.2 manifest patch: move phase_o short paths out of active model_paths into phase_o_model_paths.",
            "Restore previous live model_paths from backup or current non-phase_o active files before service restart.",
            "Add explicit loader guard if Phase O artifacts should be read only by a separate shadow path.",
        ]
    elif global_risk == "SAFE_TO_ENABLE_YAML_SHADOW_LATER":
        recommendations = ["Proceed to YAML shadow integration after review."]
    elif global_risk == "NEEDS_ROLLBACK_OR_MANIFEST_PATCH":
        recommendations = ["Patch or rollback manifests before any YAML/PM2 action."]
    else:
        recommendations = ["Manual review required before YAML/PM2 action."]
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": now_iso(),
        "created_stamp": now_stamp(),
        "read_only": True,
        "manifest_changes": False,
        "yaml_changes": False,
        "pm2_restart": False,
        "orders_sent": False,
        "runtime_files_inspected": CODE_FILES,
        "code_findings": {
            "turbo_signal_loads_active_manifest": code.turbo_signal_loads_active_manifest,
            "turbo_signal_uses_model_paths_directly": code.turbo_signal_uses_model_paths_directly,
            "turbo_signal_phase_o_aware": code.turbo_signal_phase_o_aware,
            "server_calls_evaluate_turbo_shadow": code.server_calls_evaluate_turbo_shadow,
            "yaml_guard_detected": code.yaml_guard_detected,
            "shadow_only_flag_detected": code.shadow_only_flag_detected,
        },
        "global_manifest_summary": global_summary,
        "global_risk": global_risk,
        "biggest_risk": "active_manifest.model_paths short keys point to Phase O and turbo_signal loads them directly" if phase_o_already_affects else "no Phase O runtime path detected",
        "phase_o_already_affects_runtime": phase_o_already_affects,
        "symbol_rows": symbol_rows,
        "model_path_rows": model_path_rows,
        "code_refs": code.rows,
        "recommendations": recommendations,
    }
    payload["report_paths"] = write_reports(out_dir, payload)
    if verbose:
        payload["verbose_note"] = "Detailed code references and model path rows are included in CSV reports."
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Phase O runtime safety auditor.")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_audit(Path(args.out_dir), verbose=args.verbose)
    print(json.dumps({
        "global_risk": payload["global_risk"],
        "phase_o_already_affects_runtime": payload["phase_o_already_affects_runtime"],
        "risk_counts": {risk: sum(1 for row in payload["symbol_rows"] if row["risk_class"] == risk) for risk in sorted({row["risk_class"] for row in payload["symbol_rows"]})},
        "report_paths": payload["report_paths"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
