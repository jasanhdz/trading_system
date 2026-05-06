#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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
from aegis_alpha.turbo.snapshot_utils import load_turbo_snapshot_status, turbo_snapshot_path  # noqa: E402
from aegis_alpha.turbo.evaluate_recent_models import evaluate_recent_models  # noqa: E402
from aegis_alpha.turbo.train_recent_edge import train_recent_edge_models  # noqa: E402


BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


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
    market_refresh = _refresh_market_history(symbol, DEFAULT_TURBO_CONFIG.timeframe)
    files_written: list[str] = []
    sample_count_per_file: dict[str, int] = {}
    last_timestamp_per_file: dict[str, str | None] = {}
    reports: list[dict[str, Any]] = []
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        built = build_recent_dataset(symbol, int(lookback_days), save=True)
        report = built["report"]
        dataset_path = turbo_snapshot_path(int(lookback_days))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--mode", choices=("features-only", "full"), default="features-only")
    args = parser.parse_args()

    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "started_at": started_iso,
        "symbol": args.symbol,
        "mode": args.mode,
        "success": False,
        "files_written": [],
        "sample_count_per_file": {},
        "last_timestamp_per_file": {},
    }
    try:
        payload = refresh_features_only(args.symbol) if args.mode == "features-only" else refresh_full(args.symbol)
        report.update(payload)
        report["success"] = True
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(time.time() - started_at, 3)
        report_path = _write_report(report)
        report["report_path"] = str(report_path)
        if report.get("success"):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
