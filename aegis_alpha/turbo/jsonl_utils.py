from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def parse_jsonl_line_safe(line: str, line_no: int, path: str, max_preview: int = 160) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    had_nul_bytes = "\x00" in line
    cleaned = line.replace("\x00", "").strip()
    preview = cleaned[:max_preview]
    if not cleaned:
        return None, {
            "path": str(path),
            "line_no": int(line_no),
            "error": "empty_or_nul_only_line",
            "preview": preview,
            "had_nul_bytes": bool(had_nul_bytes),
            "recovered": False,
        }
    try:
        record = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, {
            "path": str(path),
            "line_no": int(line_no),
            "error": str(exc),
            "preview": preview,
            "had_nul_bytes": bool(had_nul_bytes),
            "recovered": False,
        }
    if not isinstance(record, dict):
        return None, {
            "path": str(path),
            "line_no": int(line_no),
            "error": f"expected_json_object_got_{type(record).__name__}",
            "preview": preview,
            "had_nul_bytes": bool(had_nul_bytes),
            "recovered": False,
        }
    if had_nul_bytes:
        return record, {
            "path": str(path),
            "line_no": int(line_no),
            "error": "nul_bytes_removed",
            "preview": preview,
            "had_nul_bytes": True,
            "recovered": True,
        }
    return record, None


def iter_jsonl_safe(path: str | Path, max_preview: int = 160) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            yield parse_jsonl_line_safe(line, line_no, str(path_obj), max_preview=max_preview)


def load_jsonl_safe(path: str | Path, max_preview: int = 160) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record, error in iter_jsonl_safe(path, max_preview=max_preview):
        if record is not None:
            rows.append(record)
        if error is not None:
            errors.append(error)
    return rows, errors
