#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.validate_trrm_rotating_sources_f01 import parse_args, run_validate


def row(signal_id: str, ts: str, symbol: str = "SOLUSDT", action: str = "SHORT") -> dict[str, object]:
    return {
        "timestamp": ts,
        "signal_id": signal_id,
        "symbol": symbol,
        "strategy": "AEGIS_TURBO",
        "raw_action": action,
        "gated_action": action,
        "final_action": action,
        "gate_allowed": True,
        "executed": False,
    }


def write_rows(path: Path, rows: list[dict[str, object]], incomplete: bool = False) -> None:
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if incomplete:
        text += '{"timestamp":"2026-07-11T00:30:00Z",'
    path.write_text(text)


def test_rotating_source_glob_directory_duplicates_and_incomplete() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "turbo_signals_2026-07-10.jsonl"
        b = root / "turbo_signals_2026-07-11.jsonl"
        write_rows(a, [row("SIG-2", "2026-07-11T00:10:00Z"), row("SIG-1", "2026-07-11T00:00:00Z")])
        write_rows(b, [row("SIG-1", "2026-07-11T00:00:00Z"), row("SIG-3", "2026-07-11T00:20:00Z")], incomplete=True)
        payload = run_validate(
            parse_args(
                [
                    "--source-dir",
                    str(root),
                    "--source-pattern",
                    "turbo_signals_*.jsonl",
                    "--since",
                    "2026-07-11T00:00:00Z",
                    "--until",
                    "2026-07-11T01:00:00Z",
                    "--write-report",
                    "false",
                ]
            )
        )
        assert payload["decision"] == "ROTATING_SOURCE_READY"
        assert payload["files_matched"] == 2
        assert payload["files_readable"] == 2
        assert payload["total_events"] == 3
        assert payload["duplicates"] == 1
        assert payload["incomplete_trailing_lines"] == 1
        assert payload["earliest_event"].startswith("2026-07-11 00:00:00")
        assert payload["latest_event"].startswith("2026-07-11 00:20:00")
        payload2 = run_validate(
            parse_args(
                [
                    "--source-glob",
                    str(root / "turbo_signals_*.jsonl"),
                    "--since",
                    "2026-07-11T00:00:00Z",
                    "--until",
                    "2026-07-11T01:00:00Z",
                    "--write-report",
                    "false",
                ]
            )
        )
        assert payload2["decision"] == "ROTATING_SOURCE_READY"


def test_mutation_and_empty_source_decisions() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "turbo_signals_2026-07-10.jsonl"
        b = root / "turbo_signals_2026-07-11.jsonl"
        write_rows(a, [row("SIG-1", "2026-07-11T00:00:00Z", symbol="SOLUSDT")])
        write_rows(b, [row("SIG-1", "2026-07-11T00:00:00Z", symbol="BTCUSDT")])
        payload = run_validate(parse_args(["--source-glob", str(root / "turbo_signals_*.jsonl"), "--write-report", "false"]))
        assert payload["decision"] == "SOURCE_MUTATION_DETECTED"
        empty = run_validate(parse_args(["--source-glob", str(root / "none_*.jsonl"), "--write-report", "false"]))
        assert empty["decision"] == "ROTATING_SOURCE_EMPTY"


if __name__ == "__main__":
    test_rotating_source_glob_directory_duplicates_and_incomplete()
    test_mutation_and_empty_source_decisions()
    print("test_validate_trrm_rotating_sources_f01: OK")
