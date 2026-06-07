#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_SYMBOLS = "AVAXUSDT,ETHUSDT,SUIUSDT,BNBUSDT,XRPUSDT,LINKUSDT"
PREDICT_URL = "http://127.0.0.1:8001/ml-v2/predict"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def post_predict(symbol: str, timeout: float = 20.0) -> dict[str, Any]:
    started = time.perf_counter()
    req = Request(PREDICT_URL, data=json.dumps({"symbol": symbol}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
        return {"symbol": symbol, "http_status": resp.status, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "payload": payload, "error": None}
    except Exception as exc:
        return {"symbol": symbol, "http_status": None, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "payload": None, "error": repr(exc)}


def walk_paths(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, child
            yield from walk_paths(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{path}[{idx}]"
            yield child_path, child
            yield from walk_paths(child, child_path)


def find_paths(payload: Any) -> dict[str, Any]:
    phase_paths = []
    side_paths = []
    entry_paths = []
    avoid_paths = []
    model_paths = []
    score_paths = []
    known = {}
    for candidate in [
        "metadata", "metadata.aegis", "metadata.aegis.turbo", "aegis", "aegis.turbo", "turbo", "phase_o",
        "phase_o_short_live", "model_metadata", "prediction.metadata", "signal.metadata", "decision.metadata",
    ]:
        cur = payload
        ok = True
        for part in candidate.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            known[candidate] = cur
    for path, value in walk_paths(payload):
        leaf = path.split(".")[-1].lower()
        sval = str(value).lower()
        if "phase_o" in path.lower() or "phaseo" in path.lower() or (isinstance(value, bool) and "phase" in path.lower()):
            phase_paths.append({"path": path, "value": value})
        if leaf in {"side", "action", "decision"} and str(value).upper() in {"SHORT", "LONG", "HOLD", "BUY", "SELL"}:
            side_paths.append({"path": path, "value": value})
        if "entry" in path.lower() and "enabled" in path.lower():
            entry_paths.append({"path": path, "value": value})
        if "avoid" in path.lower():
            avoid_paths.append({"path": path, "value": value})
        if "model" in path.lower() and "path" in path.lower():
            model_paths.append({"path": path, "value": value})
        if any(token in path.lower() for token in ["score", "confidence", "prob"]):
            if isinstance(value, (int, float, str)):
                score_paths.append({"path": path, "value": value})
    return {
        "known_paths_present": known,
        "phase_o_paths": phase_paths[:100],
        "side_paths": side_paths[:100],
        "entry_enabled_paths": entry_paths[:100],
        "avoid_only_paths": avoid_paths[:100],
        "model_path_fields": model_paths[:100],
        "score_paths": score_paths[:100],
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload") or {}
    paths = find_paths(payload)
    turbo = payload.get("aegis", {}).get("turbo", {}) if isinstance(payload, dict) else {}
    raw = turbo.get("raw", {}) if isinstance(turbo, dict) else {}
    gated = turbo.get("gated", {}) if isinstance(turbo, dict) else {}
    phase = turbo.get("phase_o") or raw.get("phase_o") or payload.get("phase_o") if isinstance(payload, dict) else {}
    return {
        "symbol": result["symbol"],
        "http_status": result.get("http_status"),
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
        "decision": turbo.get("action") or gated.get("action") or raw.get("action") if isinstance(turbo, dict) else None,
        "side": turbo.get("action") or gated.get("action") or raw.get("action") if isinstance(turbo, dict) else None,
        "reason": turbo.get("reason") or gated.get("reason") or raw.get("reason") if isinstance(turbo, dict) else None,
        "phase_o_flags": phase if isinstance(phase, dict) else {},
        "phase_o_present": bool(phase),
        "phase_o_path_candidates": [p["path"] for p in paths["phase_o_paths"][:20]],
        "side_path_candidates": paths["side_paths"][:20],
        "entry_enabled_paths": paths["entry_enabled_paths"][:20],
        "avoid_only_paths": paths["avoid_only_paths"][:20],
        "model_path_fields": paths["model_path_fields"][:20],
        "score_paths": paths["score_paths"][:20],
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase O Predict Shape Dump", "", "## Safety", "- read-only", "- no orders", "", "## Summary", "| Symbol | HTTP | Decision | Reason | Phase O | Phase path examples |", "| --- | ---: | --- | --- | --- | --- |"]
    for row in payload["summaries"]:
        phase_paths = ", ".join(row.get("phase_o_path_candidates", [])[:5])
        lines.append(f"| {row['symbol']} | {row.get('http_status')} | {row.get('decision')} | {row.get('reason')} | {row.get('phase_o_present')} | {phase_paths} |")
    lines.append("")
    lines.append("## Metadata Paths")
    for row in payload["summaries"]:
        lines.append(f"### {row['symbol']}")
        lines.append(f"- side paths: `{row.get('side_path_candidates')}`")
        lines.append(f"- entry paths: `{row.get('entry_enabled_paths')}`")
        lines.append(f"- avoid paths: `{row.get('avoid_only_paths')}`")
        lines.append(f"- model paths: `{row.get('model_path_fields')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out-dir", default="/home/jasan/Develop")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    results = [post_predict(symbol, args.timeout) for symbol in symbols]
    summaries = [summarize(result) for result in results]
    stamp = utc_stamp()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "READ_ONLY", "url": PREDICT_URL, "symbols": symbols, "summaries": summaries, "responses": results}
    json_path = out / f"aegis_phase_o_predict_shape_{stamp}.json"
    md_path = out / f"aegis_phase_o_predict_shape_{stamp}.md"
    json_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(md_path, payload)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
