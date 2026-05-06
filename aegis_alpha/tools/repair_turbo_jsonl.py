#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.jsonl_utils import parse_jsonl_line_safe  # noqa: E402


DEFAULT_LOG_GLOB = str(DEFAULT_TURBO_CONFIG.log_dir / "turbo_shadow_*.jsonl")
DEFAULT_BACKUP_DIR = DEFAULT_TURBO_CONFIG.log_dir / "quarantine"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _scan_file(path: Path, max_preview: int) -> tuple[dict[str, Any], list[str]]:
    valid_lines: list[str] = []
    corrupt_line_numbers: list[int] = []
    first_error: dict[str, Any] | None = None
    total_lines = 0
    recovered_lines = 0
    skipped_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            total_lines = line_no
            record, error = parse_jsonl_line_safe(line, line_no, str(path), max_preview=max_preview)
            if error is not None:
                corrupt_line_numbers.append(line_no)
                first_error = first_error or error
                if error.get("recovered"):
                    recovered_lines += 1
                else:
                    skipped_lines += 1
            if record is None:
                continue
            valid_lines.append(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
    summary = {
        "file": str(path),
        "total_lines": int(total_lines),
        "valid_lines": int(len(valid_lines)),
        "corrupt_lines": int(len(corrupt_line_numbers)),
        "recovered_lines": int(recovered_lines),
        "skipped_lines": int(skipped_lines),
        "corrupt_line_numbers": corrupt_line_numbers,
        "first_error_preview": first_error.get("preview") if first_error else None,
        "first_error": first_error,
    }
    return summary, valid_lines


def _targets(file_path: str | None) -> list[Path]:
    if file_path:
        return [Path(file_path)]
    return [Path(item) for item in sorted(glob.glob(DEFAULT_LOG_GLOB))]


def repair_turbo_jsonl(
    *,
    apply: bool,
    file_path: str | None,
    backup_dir: Path,
    max_preview: int,
) -> dict[str, Any]:
    stamp = _utc_stamp()
    backup_dir.mkdir(parents=True, exist_ok=True) if apply else None
    files: list[dict[str, Any]] = []
    modified_files: list[str] = []
    backup_files: list[str] = []
    for path in _targets(file_path):
        summary, valid_lines = _scan_file(path, max_preview=max_preview)
        files.append(summary)
        if not apply or summary["corrupt_lines"] == 0:
            continue
        backup_path = backup_dir / f"{path.name}.bak_{stamp}"
        shutil.copy2(path, backup_path)
        backup_files.append(str(backup_path))
        path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
        modified_files.append(str(path))
    report = {
        "schema_version": "aegis_turbo_jsonl_repair_v1",
        "created_at": stamp,
        "dry_run": not apply,
        "backup_dir": str(backup_dir),
        "files": files,
        "files_scanned": int(len(files)),
        "files_modified": modified_files,
        "backup_files": backup_files,
        "total_corrupt_lines": int(sum(file["corrupt_lines"] for file in files)),
        "total_recovered_lines": int(sum(file["recovered_lines"] for file in files)),
        "total_skipped_lines": int(sum(file["skipped_lines"] for file in files)),
    }
    if apply:
        report_path = DEFAULT_TURBO_CONFIG.log_dir / f"turbo_jsonl_repair_{stamp}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or repair Aegis Turbo JSONL logs safely.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Report corrupt lines without modifying files. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Create backups and rewrite repaired JSONL files.")
    parser.add_argument("--file", default=None, help="Optional specific JSONL file to scan or repair.")
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--max-preview", type=int, default=160)
    args = parser.parse_args()
    report = repair_turbo_jsonl(
        apply=bool(args.apply),
        file_path=args.file,
        backup_dir=Path(args.backup_dir),
        max_preview=int(args.max_preview),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
