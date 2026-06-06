#!/usr/bin/env python3
"""Research-only system health audit for LONG training runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("/home/jasan/Develop/trading_system/data/binance_candles.db")
DEFAULT_OUT_DIR = Path("/home/jasan/Develop")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_cmd(args: list[str], timeout: float = 8.0) -> dict[str, Any]:
    if not args or shutil.which(args[0]) is None:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "command not found", "cmd": args}
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "available": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "cmd": args,
        }
    except Exception as exc:
        return {"available": True, "returncode": None, "stdout": "", "stderr": str(exc), "cmd": args}


def import_version(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "version": None, "error": str(exc)}


def python_env() -> dict[str, Any]:
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": import_version("numpy"),
        "pandas": import_version("pandas"),
        "sklearn": import_version("sklearn"),
    }
    torch_probe = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " print(json.dumps({'available': True, 'version': getattr(torch, '__version__', None), "
        "'cuda_is_available': bool(torch.cuda.is_available()), 'device_count': int(torch.cuda.device_count()), "
        "'hip': getattr(torch.version, 'hip', None), 'cuda': getattr(torch.version, 'cuda', None)}))\n"
        "except Exception as exc:\n"
        " print(json.dumps({'available': False, 'error': str(exc)}))\n"
    )
    probe = run_cmd([sys.executable, "-c", torch_probe], timeout=3.0)
    try:
        env["torch"] = json.loads(probe.get("stdout") or "{}")
    except Exception:
        env["torch"] = {"available": False, "error": probe.get("stderr") or "torch probe timeout/invalid output"}
    return env


def parse_ps_output(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines()[1:]:
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, pcpu, pmem, rss, comm, command = parts
        rows.append({
            "pid": pid,
            "ppid": ppid,
            "pcpu": pcpu,
            "pmem": pmem,
            "rss_kb": rss,
            "command_name": comm,
            "command": command,
        })
    return rows


def sqlite_snapshot(db_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": db_path.stat().st_size if db_path.exists() else None,
        "mode": "best_effort_timeout_limited",
    }
    if not db_path.exists():
        return info
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"]
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        deadline = time.monotonic() + 4.0
        def progress() -> int:
            return 1 if time.monotonic() > deadline else 0
        con.set_progress_handler(progress, 10_000)
        try:
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
            info["tables"] = tables
            info["indexes"] = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name LIMIT 50").fetchall()]
            table_rows: list[dict[str, Any]] = []
            if "ohlcv_data" in tables:
                for symbol in symbols:
                    row = {"symbol": symbol, "timeframe": "5m"}
                    try:
                        count = con.execute("SELECT COUNT(*) FROM ohlcv_data WHERE symbol = ? AND timeframe = '5m'", (symbol.replace('USDT', '/USDT'),)).fetchone()[0]
                        last_ts = con.execute("SELECT MAX(timestamp) FROM ohlcv_data WHERE symbol = ? AND timeframe = '5m'", (symbol.replace('USDT', '/USDT'),)).fetchone()[0]
                        row.update({"rows": count, "last_ts": last_ts})
                    except Exception as exc:
                        row.update({"rows": None, "last_ts": None, "error": str(exc)})
                    table_rows.append(row)
            info["ohlcv_rows"] = table_rows
        finally:
            con.close()
    except Exception as exc:
        info["error"] = str(exc)
    return info

def collect_health(db_path: Path) -> dict[str, Any]:
    commands = {
        "date_utc": ["date", "-u"],
        "hostname": ["hostname"],
        "uptime": ["uptime"],
        "whoami": ["whoami"],
        "df_h": ["df", "-h"],
        "df_i": ["df", "-i"],
        "free_h": ["free", "-h"],
        "nproc": ["nproc"],
        "lscpu": ["lscpu"],
        "pm2_status": ["pm2", "status"],
        "pm2_jlist": ["pm2", "jlist"],
        "top_cpu": ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,comm,args", "--sort=-pcpu"],
        "top_mem": ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,comm,args", "--sort=-rss"],
        "python_processes": ["pgrep", "-af", "python"],
        "node_processes": ["pgrep", "-af", "node"],
        "iostat": ["iostat", "-xz", "1", "1"],
        "rocm_smi": ["rocm-smi"],
        "radeontop": ["radeontop", "-d", "-", "-l", "1"],
        "nvidia_smi": ["nvidia-smi"],
    }
    results = {name: run_cmd(cmd, timeout=6.0 if name == "pm2_jlist" else 3.0) for name, cmd in commands.items()}
    processes = parse_ps_output(results["top_cpu"].get("stdout", ""))
    return {
        "schema_version": "aegis_long_perf_system_health_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "safety": {"no_live_changes": True, "no_pm2_restart": True, "no_orders": True, "no_yaml": True, "no_active_manifest": True},
        "commands": results,
        "processes_top_cpu": processes[:40],
        "processes_top_ram": parse_ps_output(results["top_mem"].get("stdout", ""))[:40],
        "python_env": python_env(),
        "sqlite": sqlite_snapshot(db_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    paths = {
        "md": str(out_dir / f"aegis_long_perf_system_health_{stamp}.md"),
        "json": str(out_dir / f"aegis_long_perf_system_health_{stamp}.json"),
        "processes": str(out_dir / f"aegis_long_perf_processes_{stamp}.csv"),
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    write_csv(Path(paths["processes"]), payload.get("processes_top_cpu", []))
    commands = payload["commands"]
    py = payload["python_env"]
    sqlite_info = payload["sqlite"]
    gpu_available = any(commands[name]["available"] and commands[name].get("stdout") for name in ("rocm_smi", "radeontop", "nvidia_smi"))
    lines = [
        "# Aegis LONG Training System Health",
        "",
        "## Safety",
        "- research-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2 restart",
        "- no orders",
        "",
        "## System",
        f"- Host: `{commands['hostname'].get('stdout', '')}`",
        f"- Uptime: `{commands['uptime'].get('stdout', '')}`",
        f"- Python: `{py.get('executable')}` `{py.get('python', '').splitlines()[0]}`",
        f"- GPU tool available: `{gpu_available}`",
        f"- Torch CUDA/ROCm available: `{py.get('torch', {}).get('cuda_is_available')}` hip=`{py.get('torch', {}).get('hip')}` devices=`{py.get('torch', {}).get('devices')}`",
        f"- SQLite DB: `{sqlite_info.get('path')}` size_bytes=`{sqlite_info.get('size_bytes')}`",
        "",
        "## PM2",
        "```text",
        commands["pm2_status"].get("stdout", "")[:4000],
        "```",
        "",
        "## Top CPU Processes",
        "| pid | cpu | mem | rss_kb | command |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("processes_top_cpu", [])[:15]:
        lines.append(f"| {row.get('pid')} | {row.get('pcpu')} | {row.get('pmem')} | {row.get('rss_kb')} | `{str(row.get('command', ''))[:120]}` |")
    lines += [
        "",
        "## Memory",
        "```text",
        commands["free_h"].get("stdout", ""),
        "```",
        "",
        "## Disk",
        "```text",
        commands["df_h"].get("stdout", "")[:4000],
        "```",
    ]
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = collect_health(Path(args.db_path))
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({"reports": paths, "gpu_tools_detected": any(payload["commands"][k]["available"] for k in ("rocm_smi", "radeontop", "nvidia_smi"))}, indent=2))


if __name__ == "__main__":
    main()
