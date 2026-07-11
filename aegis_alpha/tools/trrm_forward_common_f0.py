#!/usr/bin/env python3
"""Shared research-only helpers for FASE-F0 TRRM forward collection."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import (
    ORIGINAL_THRESHOLD,
    build_internal_folds,
    feature_hash,
    load_json,
    load_pipeline,
    policy_predictions,
    score_frame,
    validate_research_path,
)
from aegis_alpha.tools.train_trrm_honest_e2 import (
    MedianImputer,
    StandardScalerLite,
    TARGET,
    target_values,
)

EXPECTED_FEATURE_HASH = "fbb1fee2cf0c42d21c591169b25452eb65e932bf5bd76109ca8447a4dfd7057e"
CHAMPION_POLICY = "ROLLING_GLOBAL_QUANTILE_PAST_ONLY"
CHAMPION_ENGINE = "E21_PER_ROW_CANONICAL"
CHAMPION_BUDGET = 0.30
CHAMPION_WINDOW_DAYS = 30
PRIMARY_HORIZON = 12
DIAGNOSTIC_HORIZONS = [6, 24]
SCHEMA_VERSION = "trrm_f0_forward_research_v1"
RESEARCH_OUTPUT_ROOT = Path("/home/jasan/Develop/aegis_forward_research/trrm_f0")
DEFAULT_SIGNAL_GLOB = "/home/jasan/Develop/trading_system/binance-futures-bot-ts/logs/aegis/turbo_signals_*.jsonl"
BLOCKED_OUTPUT_PARTS = {"active", "active_manifest"}
LABEL_TOKENS = (
    "target",
    "label",
    "future",
    "mae",
    "mfe",
    "pnl",
    "outcome",
    "win",
    "loss",
    "tail_event",
    "realized",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_dt(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        fast = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if fast.tzinfo is None:
            fast = fast.replace(tzinfo=timezone.utc)
        return pd.Timestamp(fast).tz_convert("UTC")
    except Exception:
        pass
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default).encode())


def safe_research_path(path: Path) -> None:
    validate_research_path(path)
    if any(part in BLOCKED_OUTPUT_PARTS for part in path.resolve().parts):
        raise ValueError(f"refusing active/live path: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(f"refusing YAML write: {path}")


def atomic_write_text(path: Path, text: str) -> None:
    safe_research_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=json_default) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    safe_research_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, default=json_default) + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_paths_from_args(
    source_path: str | Sequence[str] | None = None,
    source_glob: str | None = None,
    source_dir: str | None = None,
    source_pattern: str = "turbo_signals_*.jsonl",
) -> list[Path]:
    paths: list[Path] = []
    explicit = bool(source_path or source_glob or source_dir)
    if source_path:
        raw_paths = [source_path] if isinstance(source_path, str) else list(source_path)
        for raw in raw_paths:
            if not raw:
                continue
            matches = glob.glob(str(raw))
            if matches:
                paths.extend(Path(p) for p in matches)
            else:
                paths.append(Path(raw))
    if source_glob:
        paths.extend(Path(p) for p in glob.glob(str(source_glob)))
    if source_dir:
        paths.extend(Path(p) for p in Path(source_dir).glob(source_pattern or "turbo_signals_*.jsonl"))
    if not paths and not explicit:
        paths.extend(Path(p) for p in glob.glob(DEFAULT_SIGNAL_GLOB))
    resolved: dict[str, Path] = {}
    for path in paths:
        try:
            key = str(path.resolve())
        except FileNotFoundError:
            key = str(path.absolute())
        resolved[key] = Path(key)
    return [resolved[k] for k in sorted(resolved)]


def _event_sort_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    ts = parse_dt(str(row.get("timestamp")))
    ts_key = ts.isoformat() if ts is not None else ""
    return (ts_key, str(row.get("signal_id") or ""), str(row.get("_source_path") or ""), int(row.get("_source_line") or 0))


def turbo_signal_file_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"turbo_signals_(\d{4}-\d{2}-\d{2})\.jsonl$", path.name)
    if not match:
        return None
    return pd.Timestamp(match.group(1), tz="UTC")


def read_turbo_signal_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    incomplete = 0
    malformed = 0
    last_complete = 0
    try:
        stat = path.stat()
    except FileNotFoundError:
        return [], {
            "absolute_path": str(path),
            "readable": False,
            "exists": False,
            "empty": True,
            "events": 0,
            "incomplete_trailing_lines": 0,
            "malformed_lines": 0,
        }
    with path.open("rb") as f:
        line_no = 0
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            line_no += 1
            if not raw.strip():
                last_complete = f.tell()
                continue
            has_newline = raw.endswith(b"\n")
            try:
                text = raw.decode("utf-8")
                row = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                if not has_newline:
                    incomplete += 1
                    break
                malformed += 1
                continue
            row["_source_path"] = str(path)
            row["_source_line"] = line_no
            row["_source_offset"] = offset
            row["_source_event_hash"] = sha256_json(row)
            rows.append(row)
            last_complete = f.tell()
    try:
        stat_after = path.stat()
    except FileNotFoundError:
        stat_after = stat
    size_after = int(getattr(stat_after, "st_size", stat.st_size))
    last_complete = min(last_complete, size_after)
    timestamps = [parse_dt(str(r.get("timestamp"))) for r in rows]
    timestamps = [t for t in timestamps if t is not None]
    prefix_sha = None
    if last_complete:
        h = hashlib.sha256()
        with path.open("rb") as f:
            remaining = last_complete
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                h.update(chunk)
        prefix_sha = h.hexdigest()
    meta = {
        "absolute_path": str(path),
        "readable": True,
        "exists": True,
        "empty": size_after == 0,
        "inode": getattr(stat_after, "st_ino", getattr(stat, "st_ino", None)),
        "size_bytes": size_after,
        "last_read_offset": size_after,
        "last_complete_line_offset": last_complete,
        "last_event_timestamp": str(max(timestamps)) if timestamps else None,
        "first_event_timestamp": str(min(timestamps)) if timestamps else None,
        "sha256_prefix_processed": prefix_sha,
        "events": len(rows),
        "incomplete_trailing_lines": incomplete,
        "malformed_lines": malformed,
        "rotation_detected": False,
        "truncation_detected": False,
    }
    return rows, meta


def load_rotating_signal_events(
    source_path: str | Sequence[str] | None = None,
    source_glob: str | None = None,
    source_dir: str | None = None,
    source_pattern: str = "turbo_signals_*.jsonl",
    since: str | None = None,
    until: str | None = None,
    previous_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = source_paths_from_args(source_path, source_glob, source_dir, source_pattern)
    since_ts = parse_dt(since)
    until_ts = parse_dt(until) if until else None
    all_rows: list[dict[str, Any]] = []
    file_meta: list[dict[str, Any]] = []
    previous_files = {
        str(item.get("absolute_path")): item
        for item in (previous_state or {}).get("source_files", [])
        if item.get("absolute_path")
    }
    for path in paths:
        file_day = turbo_signal_file_date(path)
        if file_day is not None and since_ts is not None and file_day < since_ts.normalize() - pd.Timedelta(days=1):
            file_meta.append({
                "absolute_path": str(path),
                "readable": False,
                "exists": path.exists(),
                "empty": False,
                "events": 0,
                "skipped_by_date_filter": True,
                "rotation_detected": False,
                "truncation_detected": False,
            })
            continue
        if file_day is not None and until_ts is not None and file_day > until_ts.normalize() + pd.Timedelta(days=1):
            file_meta.append({
                "absolute_path": str(path),
                "readable": False,
                "exists": path.exists(),
                "empty": False,
                "events": 0,
                "skipped_by_date_filter": True,
                "rotation_detected": False,
                "truncation_detected": False,
            })
            continue
        rows, meta = read_turbo_signal_file(path)
        old = previous_files.get(meta.get("absolute_path"))
        if old:
            old_size = int(old.get("size_bytes") or 0)
            new_size = int(meta.get("size_bytes") or 0)
            meta["truncation_detected"] = new_size < old_size
            meta["rotation_detected"] = old.get("inode") is not None and meta.get("inode") is not None and old.get("inode") != meta.get("inode")
        file_meta.append(meta)
        for row in rows:
            ts = parse_dt(str(row.get("timestamp")))
            if ts is None:
                continue
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            all_rows.append(row)
    all_rows.sort(key=_event_sort_key)
    ids: dict[str, str] = {}
    duplicates = 0
    mutations: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    for row in all_rows:
        sid = str(row.get("signal_id") or deterministic_id([row.get("timestamp"), row.get("symbol"), row.get("_source_path"), row.get("_source_line")]))
        fp = source_event_fingerprint(row)
        if sid in ids:
            if ids[sid] == fp:
                duplicates += 1
                continue
            mutations.append({"source_event_id": sid, "source_reference": f"{row.get('_source_path')}:{row.get('_source_line')}"})
            continue
        ids[sid] = fp
        unique_rows.append(row)
    timestamps = [parse_dt(str(r.get("timestamp"))) for r in unique_rows]
    timestamps = [t for t in timestamps if t is not None]
    post_freeze_files = sorted({str(r.get("_source_path")) for r in unique_rows})
    report = {
        "files_matched": len(paths),
        "files_readable": sum(1 for m in file_meta if m.get("readable")),
        "files_empty": sum(1 for m in file_meta if m.get("empty")),
        "files": file_meta,
        "incomplete_trailing_lines": sum(int(m.get("incomplete_trailing_lines") or 0) for m in file_meta),
        "malformed_lines": sum(int(m.get("malformed_lines") or 0) for m in file_meta),
        "earliest_event": str(min(timestamps)) if timestamps else None,
        "latest_event": str(max(timestamps)) if timestamps else None,
        "total_events": len(unique_rows),
        "unique_source_event_ids": len(ids),
        "duplicates": duplicates,
        "mutations": mutations,
        "post_freeze_files": post_freeze_files,
        "rotation_detected": any(bool(m.get("rotation_detected")) for m in file_meta),
        "truncation_detected": any(bool(m.get("truncation_detected")) for m in file_meta),
        "readiness": "SOURCE_MUTATION_DETECTED" if mutations else "SOURCE_TRUNCATION_DETECTED" if any(bool(m.get("truncation_detected")) for m in file_meta) else "ROTATING_SOURCE_READY" if unique_rows or paths else "ROTATING_SOURCE_EMPTY",
    }
    return unique_rows, report


def existing_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in read_jsonl(path) if row.get(key)}


def deterministic_id(parts: Iterable[Any]) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def contains_label_columns(row: dict[str, Any]) -> bool:
    for key in row:
        if key == "labels_resolved":
            continue
        low = str(key).lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
        if any(token in tokens for token in LABEL_TOKENS):
            return True
        if any(pattern in low for pattern in ("future_mae", "future_mfe", "realized_pnl", "trade_outcome", "is_tail_event")):
            return True
    return False


def guard_no_enforcement_imports(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    blocked = ("import binance", "from binance", "import ccxt", "create_order", "close_position", "cancel_order", "subprocess.*pm2")
    bad = [token for token in blocked if token in text]
    if "active_manifest" in text and "No active_manifest" not in text:
        bad.append("active_manifest")
    if bad:
        raise ValueError(f"ENFORCEMENT_PATH_DETECTED: {bad}")


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    return {str(p.relative_to(model_dir)): sha256_file(p) for p in files}


def combined_model_hash(model_dir: Path) -> str:
    return sha256_json(model_file_hashes(model_dir))


def load_pipeline_checked(model_dir: Path) -> dict[str, Any]:
    return load_pipeline(model_dir)


def validate_champion_artifacts(e2: dict[str, Any], e21: dict[str, Any], fable: dict[str, Any], pipeline: dict[str, Any], args: Any) -> dict[str, Any]:
    features = list(pipeline["features"])
    checks = {
        "target_exact": args.target == TARGET and pipeline.get("target") == TARGET and e21.get("target") == TARGET,
        "feature_hash_exact": feature_hash(features) == args.feature_hash == EXPECTED_FEATURE_HASH,
        "model_class_exact": e2.get("selected_candidate", {}).get("model") == "random_forest",
        "policy_exact": (e21.get("selected_policy") or {}).get("method") == CHAMPION_POLICY,
        "budget_exact": abs(float(args.budget) - CHAMPION_BUDGET) < 1e-12 and abs(float((e21.get("selected_policy") or {}).get("budget")) - CHAMPION_BUDGET) < 1e-12,
        "rolling_window_exact": int(args.rolling_window_days) == CHAMPION_WINDOW_DAYS and int((e21.get("selected_policy") or {}).get("rolling_window_days")) == CHAMPION_WINDOW_DAYS,
        "fable_approved_e21": fable.get("status") == "METRICS_SCOPE_AMBIGUOUS" and any("F0 freeze must name exactly one engine" in str(x.get("detail", "")) for x in fable.get("findings", [])),
        "no_raw_close": "feature.close" not in features,
        "no_symbol_feature": "symbol" not in features and "id.symbol" not in features,
        "no_label_future_features": not any(c.startswith(("target.", "label.", "future_eval.", "reference.", "id.")) for c in features),
        "primary_horizon_h12": int(args.primary_horizon) == PRIMARY_HORIZON,
        "diagnostics_h6_h24": sorted(int(x) for x in args.diagnostic_horizons) == DIAGNOSTIC_HORIZONS,
    }
    return {
        "status": "OK" if all(checks.values()) else "ARTIFACT_INTEGRITY_ERROR",
        "checks": checks,
        "feature_hash": feature_hash(features),
        "feature_count": len(features),
        "feature_list": features,
    }


def replay_e21_engine(dense_csv: Path, internal_predictions: Path, pipeline: dict[str, Any], budget: float, window_days: int) -> dict[str, Any]:
    dense = pd.read_csv(dense_csv)
    known = pd.read_csv(internal_predictions)
    score = score_frame(dense, pipeline)
    folds = build_internal_folds(dense, 120, 500, 20)
    if not folds:
        return {"status": "ENGINE_REPLAY_MISMATCH", "reason": "no folds"}
    fold = folds[-1]
    s, thr, _ = policy_predictions(dense, fold, score, budget, CHAMPION_POLICY, 500, 20, window_days)
    cols = ["id.symbol", "id.timestamp", "id.timeframe", "id.horizon"]
    replay = dense.iloc[fold.evaluation_idx][cols].copy().reset_index(drop=True)
    replay["risk_score"] = s
    replay["policy_threshold"] = thr
    replay["reject"] = (s >= thr).astype(int)
    cmp_cols = cols + ["risk_score", "policy_threshold", "reject"]
    if len(replay) != len(known):
        return {"status": "ENGINE_REPLAY_MISMATCH", "rows_compared": min(len(replay), len(known)), "row_count_match": False}
    threshold_match = bool(np.allclose(replay["policy_threshold"], known["policy_threshold"], rtol=0, atol=1e-12))
    score_match = bool(np.allclose(replay["risk_score"], known["risk_score"], rtol=0, atol=1e-12))
    decision_match = bool((replay["reject"].astype(int).to_numpy() == known["reject"].astype(int).to_numpy()).all())
    order_match = bool((replay[cols].astype(str).to_numpy() == known[cols].astype(str).to_numpy()).all())
    return {
        "status": "OK" if threshold_match and score_match and decision_match and order_match else "ENGINE_REPLAY_MISMATCH",
        "rows_compared": int(len(replay)),
        "threshold_match": threshold_match,
        "score_match": score_match,
        "decision_match": decision_match,
        "ordering_match": order_match,
        "rejection_rate": float(replay["reject"].mean()) if len(replay) else 0.0,
        "engine": CHAMPION_ENGINE,
    }


def make_history_seed(internal_predictions: Path, freeze_time: str, out_path: Path) -> dict[str, Any]:
    freeze_ts = parse_dt(freeze_time)
    if freeze_ts is None:
        raise ValueError("invalid freeze_time")
    df = pd.read_csv(internal_predictions)
    rows = []
    for row in df.to_dict(orient="records"):
        ts = parse_dt(str(row.get("id.timestamp")))
        if ts is None or ts >= freeze_ts:
            continue
        out = {
            "schema_version": SCHEMA_VERSION,
            "stream": "HISTORY_SEED",
            "market_timestamp": str(ts),
            "symbol": row.get("id.symbol"),
            "timeframe": row.get("id.timeframe"),
            "horizon": int(row.get("id.horizon")),
            "score": float(row.get("risk_score")),
            "source_reference": str(internal_predictions),
        }
        rows.append(out)
    rows.sort(key=lambda r: (r["market_timestamp"], r["symbol"], r["horizon"]))
    append_jsonl(out_path, rows)
    checksum = sha256_file(out_path)
    times = [parse_dt(r["market_timestamp"]) for r in rows]
    return {
        "path": str(out_path),
        "sha256": checksum,
        "rows": len(rows),
        "history_start": str(min(times)) if times else None,
        "history_end": str(max(times)) if times else None,
        "labels_present": False,
    }


def threshold_from_history(history_rows: list[dict[str, Any]], now_ts: pd.Timestamp, budget: float, window_days: int) -> tuple[float | None, dict[str, Any]]:
    start = now_ts - pd.Timedelta(days=window_days)
    vals = [
        float(r["score"])
        for r in history_rows
        if parse_dt(str(r.get("market_timestamp"))) is not None
        and start <= parse_dt(str(r.get("market_timestamp"))) < now_ts
    ]
    if not vals:
        vals = [float(r["score"]) for r in history_rows if parse_dt(str(r.get("market_timestamp"))) is not None and parse_dt(str(r.get("market_timestamp"))) < now_ts]
    if not vals:
        return None, {"history_rows": 0, "history_start": None, "history_end": None}
    times = [parse_dt(str(r.get("market_timestamp"))) for r in history_rows if parse_dt(str(r.get("market_timestamp"))) is not None and parse_dt(str(r.get("market_timestamp"))) < now_ts]
    return float(np.quantile(np.asarray(vals, dtype=float), 1.0 - budget)), {
        "history_rows": len(vals),
        "history_start": str(min(times)) if times else None,
        "history_end": str(max(times)) if times else None,
    }


def source_event_fingerprint(event: dict[str, Any]) -> str:
    clean = {k: v for k, v in event.items() if not str(k).startswith("_")}
    return sha256_json(clean)


def inspect_turbo_signal_source(
    source_path: str | Sequence[str] | None = None,
    source_glob: str | None = None,
    source_dir: str | None = None,
    source_pattern: str = "turbo_signals_*.jsonl",
) -> dict[str, Any]:
    paths = source_paths_from_args(source_path, source_glob, source_dir, source_pattern)
    sample: dict[str, Any] | None = None
    for path in reversed(paths):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    break
        if sample:
            break
    ready = bool(sample and sample.get("signal_id") and sample.get("timestamp") and sample.get("symbol"))
    return {
        "decision": "OPPORTUNITY_SOURCE_READY" if ready else "OPPORTUNITY_SEMANTICS_NOT_READY",
        "source_kind": "turbo_signals_jsonl",
        "path_pattern": source_glob or (str(Path(source_dir) / source_pattern) if source_dir else str(source_path or DEFAULT_SIGNAL_GLOB)),
        "files": [str(p) for p in paths[-5:]],
        "schema_keys": sorted(sample.keys()) if sample else [],
        "id_field": "signal_id",
        "timestamp_field": "timestamp",
        "symbol_field": "symbol",
        "side_field": "raw_action/final_action",
        "strategy_field": "strategy",
        "timeframe": "5m inferred from Turbo snapshots",
        "read_only": True,
        "incremental_read": True,
        "opportunity_definition": "one Turbo signal/candidate event keyed by signal_id; horizons are risk evaluations of the same event",
        "feature_alignment": "requires frozen causal features at or before source timestamp; absent features produce NO_DECISION",
    }


def iter_turbo_signal_events(pattern: str, since: str | None, until: str | None, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows, _ = load_rotating_signal_events(source_glob=pattern, since=since, until=until)
    return rows[:max_rows] if max_rows else rows


def opportunity_id_from_event(event: dict[str, Any], source_name: str) -> str:
    return str(event.get("signal_id") or deterministic_id([
        source_name,
        event.get("symbol"),
        parse_dt(str(event.get("timestamp"))),
        event.get("raw_action") or event.get("final_action"),
        event.get("strategy"),
        "5m",
        "turbo_signal",
    ]))


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "streams": {
            "MODEL_MONITOR_STREAM": [
                "schema_version",
                "candidate_id",
                "stream",
                "recorded_at_utc",
                "market_timestamp",
                "symbol",
                "horizon",
                "score",
                "threshold",
                "hypothetical_row_decision",
                "policy_engine",
                "policy_budget",
                "rolling_window_days",
                "history_rows",
                "history_start",
                "history_end",
                "feature_hash",
                "model_hash",
                "source_reference",
                "enforcement_action",
            ],
            "OPPORTUNITY_STREAM": [
                "schema_version",
                "candidate_id",
                "stream",
                "opportunity_id",
                "source_event_id",
                "source_event_type",
                "source_timestamp",
                "recorded_at_utc",
                "symbol",
                "side",
                "strategy_name",
                "strategy_version",
                "timeframe",
                "primary_horizon",
                "diagnostic_horizons",
                "score_h6",
                "score_h12",
                "score_h24",
                "primary_threshold",
                "hypothetical_decision",
                "no_decision_reason",
                "policy_engine",
                "policy_budget",
                "rolling_window_days",
                "history_rows",
                "feature_hash",
                "model_hash",
                "source_reference",
                "enforcement_action",
                "labels_resolved",
            ],
        },
        "forbidden_columns": list(LABEL_TOKENS),
        "forward_outcomes": "FORWARD_OUTCOMES_NOT_EVALUATED",
    }
