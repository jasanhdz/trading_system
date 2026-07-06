#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import settings  # noqa: E402
from data.storage.database_manager import DatabaseManager  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import load_turbo_snapshot_status, normalize_turbo_symbol, turbo_snapshot_path  # noqa: E402
from aegis_alpha.turbo.evaluate_recent_models import evaluate_recent_models  # noqa: E402
from aegis_alpha.turbo.train_recent_edge import train_recent_edge_models  # noqa: E402


BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
LOCK_PATH = DEFAULT_TURBO_CONFIG.log_dir / "turbo_snapshot_refresh.lock"
DEFAULT_MIN_AVAILABLE_MEM_GB = float(os.getenv("AEGIS_TURBO_REFRESH_MIN_AVAILABLE_MEM_GB", "8"))
DEFAULT_SLEEP_BETWEEN_SYMBOLS_SECONDS = float(os.getenv("AEGIS_TURBO_REFRESH_SLEEP_BETWEEN_SYMBOLS_SECONDS", "2"))
LOCKED_REASON = "another_refresh_turbo_snapshots_instance_is_running"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _report_path() -> Path:
    path = DEFAULT_TURBO_CONFIG.log_dir / f"turbo_snapshot_refresh_{_utc_stamp()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(report: dict[str, Any]) -> Path:
    path = _report_path()
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return None
    return None


def _ensure_memory_available(min_available_gb: float) -> None:
    available = _mem_available_gb()
    if available is None:
        return
    if available < min_available_gb:
        raise RuntimeError(
            f"insufficient memory for turbo refresh: MemAvailable={available:.2f}GiB "
            f"< required={min_available_gb:.2f}GiB"
        )


@contextmanager
def turbo_refresh_lock() -> Any:
    DEFAULT_TURBO_CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(LOCKED_REASON) from exc
    try:
        yield
    finally:
        lock_file.close()


def _fetch_binance_klines(params: dict[str, Any]) -> list[list[Any]]:
    url = f"{BINANCE_KLINES_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Binance response: {data!r}")
    return data


def _refresh_market_history(symbol: str, timeframe: str) -> dict[str, Any]:
    db = DatabaseManager(settings.DATABASE_URL)
    db.create_tables()
    db_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")
    binance_symbol = db_symbol.replace("/", "")
    latest = db.get_latest_timestamp(db_symbol, timeframe)
    end_date = datetime.now(timezone.utc)
    interval_minutes = int(timeframe.rstrip("m"))
    if latest is not None:
        latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
        start_date = latest_utc + timedelta(minutes=interval_minutes)
    else:
        start_date = end_date - timedelta(days=30)

    if start_date >= end_date:
        return {
            "success": True,
            "symbol": db_symbol,
            "timeframe": timeframe,
            "inserted": 0,
            "last_timestamp": latest.isoformat() if latest is not None else None,
            "reason": "already_up_to_date",
        }

    rows: list[list[Any]] = []
    current_start = start_date
    while current_start < end_date:
        params = {
            "symbol": binance_symbol,
            "interval": timeframe,
            "limit": 1000,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
        }
        chunk = _fetch_binance_klines(params)
        if not chunk:
            break
        rows.extend(chunk)
        last_open_ms = int(chunk[-1][0])
        current_start = datetime.fromtimestamp((last_open_ms / 1000.0) + interval_minutes * 60, tz=timezone.utc)
        if len(chunk) < 1000:
            break

    if rows:
        df = pd.DataFrame(rows, columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "buy_volume"]:
            df[col] = df[col].astype(float)
    else:
        df = pd.DataFrame()

    if latest is not None and not df.empty:
        latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
        df = df[df["timestamp"] > latest_utc]

    if df.empty:
        return {
            "success": True,
            "symbol": db_symbol,
            "timeframe": timeframe,
            "inserted": 0,
            "last_timestamp": latest.isoformat() if latest is not None else None,
            "reason": "no_new_candles",
        }

    inserted = db.insert_ohlcv_data(df, db_symbol, timeframe)
    return {
        "success": True,
        "symbol": db_symbol,
        "timeframe": timeframe,
        "inserted": int(inserted),
        "last_timestamp": df["timestamp"].max().isoformat(),
        "reason": "refreshed_from_binance",
    }


def refresh_features_only(symbol: str) -> dict[str, Any]:
    symbol = normalize_turbo_symbol(symbol)
    market_refresh = _refresh_market_history(symbol, DEFAULT_TURBO_CONFIG.timeframe)
    files_written: list[str] = []
    sample_count_per_file: dict[str, int] = {}
    last_timestamp_per_file: dict[str, str | None] = {}
    reports: list[dict[str, Any]] = []
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        built = build_recent_dataset(symbol, int(lookback_days), save=True)
        report = built["report"]
        dataset_path = turbo_snapshot_path(int(lookback_days), symbol)
        files_written.append(str(dataset_path))
        sample_count_per_file[str(dataset_path)] = int(report.get("sample_count") or 0)
        last_timestamp_per_file[str(dataset_path)] = report.get("last_timestamp")
        reports.append(report)
    return {
        "mode": "features-only",
        "symbol": symbol,
        "market_refresh": market_refresh,
        "files_written": files_written,
        "sample_count_per_file": sample_count_per_file,
        "last_timestamp_per_file": last_timestamp_per_file,
        "dataset_reports": reports,
    }


def refresh_full(symbol: str) -> dict[str, Any]:
    symbol = normalize_turbo_symbol(symbol)
    train_report = train_recent_edge_models(symbol)
    eval_report = evaluate_recent_models(symbol)
    dataset_reports = train_report.get("dataset_reports", []) if isinstance(train_report, dict) else []
    files_written = []
    sample_count_per_file: dict[str, int] = {}
    last_timestamp_per_file: dict[str, str | None] = {}
    for report in dataset_reports:
        dataset_path = report.get("dataset_path")
        if not dataset_path:
            continue
        files_written.append(str(dataset_path))
        sample_count_per_file[str(dataset_path)] = int(report.get("sample_count") or 0)
        last_timestamp_per_file[str(dataset_path)] = report.get("last_timestamp")
    return {
        "mode": "full",
        "symbol": symbol,
        "files_written": files_written,
        "sample_count_per_file": sample_count_per_file,
        "last_timestamp_per_file": last_timestamp_per_file,
        "train_report": train_report,
        "eval_report": eval_report,
    }


def _symbol_summary(payload: dict[str, Any]) -> dict[str, Any]:
    statuses = {
        f"{lookback_days}d": load_turbo_snapshot_status(turbo_snapshot_path(int(lookback_days), payload["symbol"]), include_sample_count=True)
        for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days
    }
    selected = None
    for status in statuses.values():
        if not status.get("exists"):
            continue
        if selected is None or ((status.get("feature_timestamp") or ""), (status.get("snapshot_mtime") or "")) > ((selected.get("feature_timestamp") or ""), (selected.get("snapshot_mtime") or "")):
            selected = status
    return {
        "success": True,
        "last_feature_timestamp": (selected or {}).get("feature_timestamp") or (selected or {}).get("last_ts"),
        "is_fresh": bool((selected or {}).get("is_fresh", False)),
        "files_written": payload.get("files_written", []),
        "snapshot_statuses": statuses,
    }


def _parse_symbols(symbol: str, symbols: str | None) -> list[str]:
    raw_values = symbols.split(",") if symbols else [symbol]
    parsed = [normalize_turbo_symbol(value) for value in raw_values if value.strip()]
    return list(dict.fromkeys(parsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. ETHUSDT,BTCUSDT")
    parser.add_argument("--mode", choices=("features-only", "full"), default="features-only")
    parser.add_argument("--min-available-mem-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEM_GB)
    parser.add_argument("--sleep-between-symbols-seconds", type=float, default=DEFAULT_SLEEP_BETWEEN_SYMBOLS_SECONDS)
    args = parser.parse_args()

    try:
        lock_context = turbo_refresh_lock()
        lock_context.__enter__()
    except RuntimeError as exc:
        if str(exc) != LOCKED_REASON:
            raise
        report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "symbol": None,
            "symbols_requested": _parse_symbols(args.symbol, args.symbols),
            "mode": args.mode,
            "success": False,
            "partial_success": False,
            "skipped": True,
            "reason": LOCKED_REASON,
        }
        report_path = _write_report(report)
        report["report_path"] = str(report_path)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(75)

    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    symbols = _parse_symbols(args.symbol, args.symbols)
    report: dict[str, Any] = {
        "started_at": started_iso,
        "symbol": symbols[0] if len(symbols) == 1 else None,
        "symbols_requested": symbols,
        "mode": args.mode,
        "success": False,
        "partial_success": False,
        "files_written": [],
        "sample_count_per_file": {},
        "last_timestamp_per_file": {},
        "symbols": {},
    }
    try:
        success_count = 0
        for index, symbol in enumerate(symbols):
            try:
                _ensure_memory_available(float(args.min_available_mem_gb))
                payload = refresh_features_only(symbol) if args.mode == "features-only" else refresh_full(symbol)
                summary = _symbol_summary(payload)
                report["symbols"][symbol] = summary
                report["files_written"].extend(payload.get("files_written", []))
                report["sample_count_per_file"].update(payload.get("sample_count_per_file", {}))
                report["last_timestamp_per_file"].update(payload.get("last_timestamp_per_file", {}))
                success_count += 1
                if len(symbols) == 1:
                    report.update(payload)
            except Exception as exc:
                report["symbols"][symbol] = {
                    "success": False,
                    "last_feature_timestamp": None,
                    "is_fresh": False,
                    "files_written": [],
                    "error": repr(exc),
                }
            finally:
                gc.collect()
                if index < len(symbols) - 1 and args.sleep_between_symbols_seconds > 0:
                    time.sleep(float(args.sleep_between_symbols_seconds))
        report["success"] = success_count > 0
        report["partial_success"] = 0 < success_count < len(symbols)
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(time.time() - started_at, 3)
        report_path = _write_report(report)
        report["report_path"] = str(report_path)
        lock_context.__exit__(None, None, None)
        if report.get("success"):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
