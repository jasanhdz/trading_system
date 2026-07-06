#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402


DEFAULT_GROUPS = (
    "ETHUSDT,BTCUSDT,SOLUSDT",
    "BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT",
    "AVAXUSDT,LINKUSDT,SUIUSDT,LTCUSDT",
)
REPORT_DIR = DEFAULT_TURBO_CONFIG.log_dir / "scheduler"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def write_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"turbo_refresh_scheduler_{utc_stamp()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_groups(values: list[str] | None) -> list[str]:
    groups = values or list(DEFAULT_GROUPS)
    return [group for group in (value.strip() for value in groups) if group]


def refresh_command(symbols: str, min_available_mem_gb: float, sleep_between_symbols_seconds: float) -> list[str]:
    return [
        sys.executable,
        "aegis_alpha/tools/refresh_turbo_snapshots.py",
        "--mode",
        "features-only",
        "--symbols",
        symbols,
        "--min-available-mem-gb",
        str(min_available_mem_gb),
        "--sleep-between-symbols-seconds",
        str(sleep_between_symbols_seconds),
    ]


def run_group(symbols: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    cmd = refresh_command(symbols, args.min_available_mem_gb, args.sleep_between_symbols_seconds)
    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=args.group_timeout_seconds,
        check=False,
    )
    return {
        "symbols": symbols,
        "command": cmd,
        "returncode": int(proc.returncode),
        "success": proc.returncode == 0,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_cycle(args: argparse.Namespace, groups: list[str]) -> dict[str, Any]:
    cycle_started = time.time()
    report: dict[str, Any] = {
        "schema_version": "aegis_turbo_refresh_scheduler_v1",
        "started_at": utc_now().isoformat(),
        "groups_requested": groups,
        "groups": [],
        "success": False,
        "partial_success": False,
    }
    successes = 0
    for index, symbols in enumerate(groups):
        try:
            group_report = run_group(symbols, args)
        except subprocess.TimeoutExpired as exc:
            group_report = {
                "symbols": symbols,
                "success": False,
                "returncode": None,
                "duration_seconds": args.group_timeout_seconds,
                "error": f"group_timeout:{exc!r}",
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        report["groups"].append(group_report)
        if group_report.get("success"):
            successes += 1
        if index < len(groups) - 1 and args.sleep_between_groups_seconds > 0:
            time.sleep(args.sleep_between_groups_seconds)
    report["success"] = successes == len(groups)
    report["partial_success"] = 0 < successes < len(groups)
    report["finished_at"] = utc_now().isoformat()
    report["duration_seconds"] = round(time.time() - cycle_started, 3)
    report["report_path"] = str(write_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", help="Comma-separated symbol group. Can be passed multiple times.")
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_INTERVAL_SECONDS", "900")))
    parser.add_argument("--min-sleep-seconds", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_MIN_SLEEP_SECONDS", "30")))
    parser.add_argument("--sleep-between-groups-seconds", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_SLEEP_BETWEEN_GROUPS_SECONDS", "5")))
    parser.add_argument("--sleep-between-symbols-seconds", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_SLEEP_BETWEEN_SYMBOLS_SECONDS", "2")))
    parser.add_argument("--min-available-mem-gb", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_MIN_AVAILABLE_MEM_GB", "8")))
    parser.add_argument("--group-timeout-seconds", type=float, default=float(os.getenv("AEGIS_TURBO_REFRESH_GROUP_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    groups = parse_groups(args.group)
    while True:
        started = time.time()
        report = run_cycle(args, groups)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.once:
            raise SystemExit(0 if report.get("success") else 1)
        elapsed = time.time() - started
        sleep_seconds = max(float(args.min_sleep_seconds), float(args.interval_seconds) - elapsed)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
