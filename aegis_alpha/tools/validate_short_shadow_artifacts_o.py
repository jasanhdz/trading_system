#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.snapshot_utils import turbo_symbol_model_dir  # noqa: E402

ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
AVOID_ONLY_SYMBOLS = ["LINKUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + AVOID_ONLY_SYMBOLS
REPORT_SCHEMA_VERSION = "aegis_short_phase_o_validation_v1"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _path_ok(path: str) -> bool:
    text = path.replace("\\", "/")
    return "/models/turbo/" in text and ("/active/phase_o_" in text or text.endswith("active_manifest.json") or text.endswith("phase_o_short_manifest.json") or "phase_o_global_short_manifest_" in text)


def validate(base_dir: Path, out_dir: Path, require_model_files: Optional[bool] = None) -> dict[str, Any]:
    pointer_path = base_dir / "phase_o_short_manifest.json"
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if not pointer_path.exists():
        errors.append(f"missing pointer manifest: {pointer_path}")
        latest_manifest = None
        pointer = {}
        global_manifest = {}
    else:
        pointer = load_json(pointer_path)
        latest_manifest = Path(str(pointer.get("latest_manifest", "")))
        if not latest_manifest.exists():
            errors.append(f"missing global manifest: {latest_manifest}")
            global_manifest = {}
        else:
            global_manifest = load_json(latest_manifest)

    artifact_paths = global_manifest.get("artifact_paths", {}) if isinstance(global_manifest, dict) else {}
    if set(artifact_paths) != set(ALL_SYMBOLS):
        errors.append(f"expected 11 symbols {ALL_SYMBOLS}, got {sorted(artifact_paths)}")

    entry_count = 0
    avoid_count = 0
    for symbol in ALL_SYMBOLS:
        path_text = artifact_paths.get(symbol)
        if not path_text:
            errors.append(f"missing symbol manifest for {symbol}")
            continue
        manifest_path = Path(path_text)
        if not manifest_path.exists():
            errors.append(f"missing symbol manifest file for {symbol}: {manifest_path}")
            continue
        if not _path_ok(str(manifest_path)):
            errors.append(f"unexpected artifact path for {symbol}: {manifest_path}")
        manifest = load_json(manifest_path)
        entry_enabled = bool(manifest.get("entry_enabled"))
        avoid_only = bool(manifest.get("avoid_only"))
        if symbol == "LINKUSDT":
            if entry_enabled or not avoid_only:
                errors.append("LINKUSDT must be entry_enabled=false and avoid_only=true")
        else:
            if not entry_enabled or avoid_only:
                errors.append(f"{symbol} must be entry_enabled=true and avoid_only=false")
        if bool(manifest.get("affects_orders")):
            errors.append(f"{symbol} affects_orders must be false")
        if not manifest.get("feature_schema_hash"):
            errors.append(f"{symbol} missing feature_schema_hash")
        metadata_only = bool(manifest.get("metadata_only"))
        model_files = list(manifest.get("model_files") or [])
        if require_model_files is None:
            must_have_models = not metadata_only
        else:
            must_have_models = require_model_files
        if must_have_models and not model_files:
            errors.append(f"{symbol} has no model_files")
        for model_file in model_files:
            model_path = Path(model_file)
            if not model_path.exists():
                errors.append(f"{symbol} missing model file: {model_file}")
            if not _path_ok(model_file):
                errors.append(f"{symbol} model file not in phase_o active path: {model_file}")
        active_manifest_path = turbo_symbol_model_dir(symbol) / "active_manifest.json"
        if not active_manifest_path.exists():
            errors.append(f"{symbol} missing active_manifest.json")
        else:
            active = load_json(active_manifest_path)
            phase = active.get("phase_o_symbols", {}).get(symbol, {}) if isinstance(active.get("phase_o_symbols"), dict) else {}
            if not phase:
                errors.append(f"{symbol} active manifest missing phase_o_symbols entry")
            if symbol != "LINKUSDT":
                if not metadata_only or require_model_files:
                    key = f"short_{int(manifest.get('lookback_days', 0))}d"
                    path_for_key = (active.get("model_paths") or {}).get(key)
                    if not path_for_key:
                        errors.append(f"{symbol} active manifest missing model_paths.{key}")
            else:
                if not active.get("phase_o_avoid_only"):
                    errors.append("LINKUSDT active manifest missing phase_o_avoid_only=true")
        if entry_enabled:
            entry_count += 1
        if avoid_only:
            avoid_count += 1
        rows.append({"symbol": symbol, "entry_enabled": entry_enabled, "avoid_only": avoid_only, "metadata_only": metadata_only, "model_file_count": len(model_files), "manifest": str(manifest_path)})

    if entry_count != 10:
        errors.append(f"expected 10 entry symbols, got {entry_count}")
    if avoid_count != 1:
        errors.append(f"expected 1 avoid-only symbol, got {avoid_count}")
    if pointer and pointer.get("active_manifest_touched") is not True:
        errors.append("pointer active_manifest_touched must be true for prod-ready Phase O")
    if global_manifest and global_manifest.get("entry_shadow_count") != 10:
        errors.append("global manifest entry_shadow_count must be 10")
    if global_manifest and global_manifest.get("avoid_only_count") != 1:
        errors.append("global manifest avoid_only_count must be 1")

    status = "passed" if not errors else "failed"
    stamp = now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"aegis_short_shadow_artifacts_validation_{stamp}.md"
    json_path = out_dir / f"aegis_short_shadow_artifacts_validation_{stamp}.json"
    csv_path = out_dir / f"aegis_short_shadow_artifacts_validation_summary_{stamp}.csv"
    payload = {"schema_version": REPORT_SCHEMA_VERSION, "created_at": now_iso(), "status": status, "errors": errors, "pointer_manifest": str(pointer_path), "global_manifest": str(latest_manifest) if latest_manifest else "", "symbol_count": len(rows), "entry_count": entry_count, "avoid_only_count": avoid_count, "rows": rows}
    md = [f"# Phase O Validation {stamp}", "", f"- status: {status}", f"- pointer: {pointer_path}", f"- global: {latest_manifest}", f"- symbol_count: {len(rows)}", f"- entry_count: {entry_count}", f"- avoid_only_count: {avoid_count}", "", "## Errors"] + [f"- {err}" for err in errors]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(json_path, payload)
    write_csv(csv_path, rows, ["symbol", "entry_enabled", "avoid_only", "metadata_only", "model_file_count", "manifest"])
    payload["report_paths"] = {"md": str(md_path), "json": str(json_path), "csv": str(csv_path)}
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase O SHORT prod-ready artifacts.")
    parser.add_argument("--shadow-model-dir", default=str(Path(__file__).resolve().parents[2] / "aegis_alpha/models/turbo"))
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--require-model-files", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(Path(args.shadow_model_dir), Path(args.out_dir), True if args.require_model_files else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2

if __name__ == "__main__":
    raise SystemExit(main())
