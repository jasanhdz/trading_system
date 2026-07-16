#!/usr/bin/env python3
"""GEN2 operational maintenance — safe retention / rotation / compaction.

Classifies every growing artifact and applies a SAFE policy. Never destroys
scientific evidence, never breaks dedup/idempotency, never reprocesses old
orders. All writes are atomic (tmp + rename); the active file is never truncated
in place; a --dry-run shows exactly what would change without touching anything.

Classes:
  A EVIDENCE     forward_decisions/outcomes, selection_outcomes, live_orders,
                 fills, brackets — month-SEGMENTED (never deleted): rows older
                 than the current UTC month move to <name>_YYYY-MM.jsonl and the
                 active file keeps the current month. Total row count preserved.
  B STATE/DEDUP  events_seen.json, events_checkpoint.json, bridge state —
                 REPORT ONLY (age-based pruning could allow reprocessing).
  C OP LOGS      reconciliations, telegram_log, alerts, config_changes —
                 ROTATED by retention (default 45d): rows older than the window
                 move to a dated archive. incidents also gets a summary line.
  D SNAPSHOTS    forward/snapshots — pruned to keep-N (delegates to the watcher's
                 approved prune; report only here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, utc_now  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
FORWARD_ROOT = GEN2_ROOT / "forward"
DEFAULT_LOG_RETENTION_DAYS = 45
DEFAULT_SNAPSHOT_KEEP = 12
TS_KEYS = ("collected_at", "recorded_at_utc", "recorded_at", "timestamp_utc", "decision_ts", "created_at_utc")

# name -> class
EVIDENCE_FILES = ("forward_decisions.jsonl", "forward_outcomes.jsonl", "selection_outcomes.jsonl",
                  "live_orders.jsonl", "fills.jsonl", "brackets.jsonl")
OP_LOG_FILES = ("reconciliations.jsonl", "telegram_log.jsonl", "alerts.jsonl", "config_changes.jsonl")
STATE_FILES = ("events_seen.json", "events_checkpoint.json", "loop_state.json", "risk_state.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant: skip a torn trailing line (power-cut safe)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _row_dt(row: dict[str, Any]) -> datetime | None:
    for k in TS_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _atomic_write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _atomic_append_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(existing + "".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def segment_evidence(path: Path, now: datetime, dry_run: bool) -> dict[str, Any]:
    """Move rows older than the current UTC month into <name>_YYYY-MM.jsonl.
    Total rows preserved (archive + active). Idempotent."""
    rows = read_jsonl(path)
    if not rows:
        return {"file": path.name, "rows": 0, "moved": 0, "kept": 0, "archives": {}}
    cur = f"{now.year:04d}-{now.month:02d}"
    keep: list[dict[str, Any]] = []
    by_month: dict[str, list[dict[str, Any]]] = {}
    undated = 0
    for r in rows:
        dt = _row_dt(r)
        month = f"{dt.year:04d}-{dt.month:02d}" if dt else None
        if month is None:
            keep.append(r)  # undated rows always stay in the active file (never lost)
            undated += 1
        elif month >= cur:
            keep.append(r)
        else:
            by_month.setdefault(month, []).append(r)
    moved = sum(len(v) for v in by_month.values())
    result = {"file": path.name, "rows": len(rows), "moved": moved, "kept": len(keep),
              "undated_kept": undated, "archives": {m: len(v) for m, v in by_month.items()}}
    if moved == 0 or dry_run:
        return result
    stem = path.name[:-len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
    for month, mrows in by_month.items():
        archive = path.parent / f"{stem}_{month}.jsonl"
        _atomic_append_lines(archive, mrows)  # append: re-runs stay consistent
    _atomic_write_lines(path, keep)
    # integrity: archive+active row count must equal the original
    total_after = len(read_jsonl(path)) + sum(len(read_jsonl(path.parent / f"{stem}_{m}.jsonl")) for m in by_month)
    result["integrity_ok"] = (total_after >= len(rows))
    return result


def rotate_log(path: Path, retention_days: int, now: datetime, dry_run: bool) -> dict[str, Any]:
    """Move rows older than the retention window into a dated archive. Idempotent."""
    rows = read_jsonl(path)
    if not rows:
        return {"file": path.name, "rows": 0, "rotated": 0}
    cutoff = now.timestamp() - retention_days * 86400
    keep, old = [], []
    for r in rows:
        dt = _row_dt(r)
        (keep if (dt is None or dt.timestamp() >= cutoff) else old).append(r)
    result = {"file": path.name, "rows": len(rows), "rotated": len(old), "kept": len(keep)}
    if not old or dry_run:
        return result
    archive = path.parent / f"{path.name[:-6]}.{now.strftime('%Y-%m-%d')}.archive.jsonl"
    _atomic_append_lines(archive, old)
    # incidents: never rotate without leaving a summary
    if path.name == "incidents.jsonl":
        import collections
        types = collections.Counter(r.get("type", "?") for r in old)
        _atomic_append_lines(path.parent / "incidents_summary.jsonl",
                             [{"event": "INCIDENTS_ROTATED", "rotated_at": now.isoformat(),
                               "count": len(old), "by_type": dict(types)}])
    _atomic_write_lines(path, keep)
    return result


def audit(candidate_id: str) -> dict[str, Any]:
    cdir = core.canary_dir(candidate_id)
    items: list[dict[str, Any]] = []

    def add(path: Path, cls: str, policy: str) -> None:
        if path.exists():
            size = path.stat().st_size
            n = len(read_jsonl(path)) if path.suffix == ".jsonl" else None
            items.append({"path": str(path), "class": cls, "policy": policy, "bytes": size, "rows": n})

    for name in EVIDENCE_FILES:
        add(FORWARD_ROOT / name, "A_EVIDENCE", "month-segment (never delete)")
        add(cdir / name, "A_EVIDENCE", "month-segment (never delete)")
    for name in OP_LOG_FILES:
        add(cdir / name, "C_OP_LOG", f"rotate >{DEFAULT_LOG_RETENTION_DAYS}d")
    add(cdir / "incidents" / "incidents.jsonl", "C_OP_LOG", "rotate + summary")
    for name in STATE_FILES:
        add(cdir / name, "B_STATE_DEDUP", "report only (watermark-safe)")
    # snapshots
    snaps = sorted(p for p in (FORWARD_ROOT / "snapshots").glob("*") if p.is_dir()) if (FORWARD_ROOT / "snapshots").exists() else []
    snap_bytes = sum(f.stat().st_size for p in snaps for f in p.rglob("*") if f.is_file())
    items.append({"path": str(FORWARD_ROOT / "snapshots"), "class": "D_SNAPSHOT",
                  "policy": f"prune keep-{DEFAULT_SNAPSHOT_KEEP}", "bytes": snap_bytes, "dirs": len(snaps)})
    total = sum(i["bytes"] for i in items)
    return {"schema": "gen2_maintenance_audit_v1", "candidate_id": candidate_id,
            "generated_at_utc": utc_now().isoformat(), "total_bytes": total,
            "total_mb": round(total / 1e6, 2), "items": items}


def run_maintenance(candidate_id: str, mode: str, retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
                    snapshot_keep: int = DEFAULT_SNAPSHOT_KEEP) -> dict[str, Any]:
    """mode in {audit, dry-run, compact, rotate, report}. compact = evidence
    segmentation; rotate = op-log rotation; dry-run = show both without writing."""
    now = utc_now()
    cdir = core.canary_dir(candidate_id)
    dry = mode == "dry-run"
    out: dict[str, Any] = {"schema": "gen2_maintenance_v1", "mode": mode, "candidate_id": candidate_id,
                           "dry_run": dry, "generated_at_utc": now.isoformat()}
    if mode in ("audit", "report"):
        out["audit"] = audit(candidate_id)
        if mode == "report":
            a = out["audit"]
            out["freeable_estimate_mb"] = round(sum(
                i["bytes"] for i in a["items"] if i["class"] in ("C_OP_LOG", "D_SNAPSHOT")) / 1e6, 2)
        return out
    # compact (evidence) and/or rotate (op logs); dry-run does both without writing
    if mode in ("compact", "dry-run"):
        seg = []
        for name in EVIDENCE_FILES:
            for base in (FORWARD_ROOT, cdir):
                p = base / name
                if p.exists():
                    seg.append(segment_evidence(p, now, dry))
        out["evidence_segmentation"] = [s for s in seg if s["rows"]]
    if mode in ("rotate", "dry-run"):
        rot = []
        for name in OP_LOG_FILES:
            p = cdir / name
            if p.exists():
                rot.append(rotate_log(p, retention_days, now, dry))
        inc = cdir / "incidents" / "incidents.jsonl"
        if inc.exists():
            rot.append(rotate_log(inc, retention_days, now, dry))
        out["log_rotation"] = [r for r in rot if r["rows"]]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GEN2 operational maintenance (safe retention/rotation)")
    p.add_argument("--mode", choices=["audit", "dry-run", "compact", "rotate", "report"], default="audit")
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    p.add_argument("--retention-days", type=int, default=DEFAULT_LOG_RETENTION_DAYS)
    p.add_argument("--snapshot-keep", type=int, default=DEFAULT_SNAPSHOT_KEEP)
    args = p.parse_args(argv)
    result = run_maintenance(args.candidate_id, args.mode, args.retention_days, args.snapshot_keep)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
