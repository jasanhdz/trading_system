#!/usr/bin/env python3
"""FASE-F0.2 recent market source audit and read-only snapshot.

Research-only. The script inspects local OHLCV coverage first and can fall back
to Binance USD-M public klines without API keys. It writes snapshots outside the
repository and never writes to the operational SQLite database.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import db_symbol
from aegis_alpha.tools.trrm_forward_common_f0 import atomic_write_text, safe_research_path, sha256_file, write_json

DEFAULT_DB = REPO / "data" / "binance_candles.db"
DEFAULT_OUTPUT = Path("/home/jasan/Develop/aegis_forward_research/trrm_f02/market_snapshots")
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,AVAXUSDT,SOLUSDT,LINKUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SUIUSDT,LTCUSDT"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MINUTES = 5


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_dt(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def closed_candle_end(freeze: pd.Timestamp) -> pd.Timestamp:
    floored = freeze.floor(f"{INTERVAL_MINUTES}min")
    return floored - pd.Timedelta(minutes=INTERVAL_MINUTES)


def parse_symbols(text: str) -> list[str]:
    return [s.strip().upper().replace("/", "") for s in text.replace(" ", ",").split(",") if s.strip()]


def open_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)


def local_coverage(db_path: Path, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    rows = []
    with open_ro(db_path) as con:
        for symbol in symbols:
            row = con.execute(
                """
                select min(timestamp), max(timestamp), count(*), count(distinct timestamp)
                from ohlcv_data
                where symbol=? and timeframe=? and timestamp between ? and ?
                """,
                (db_symbol(symbol), timeframe, str(start.tz_convert(None)), str(end.tz_convert(None))),
            ).fetchone()
            first, last, count, distinct_count = row
            expected = int(((end - start).total_seconds() // 300) + 1)
            rows.append(
                {
                    "symbol": symbol,
                    "first_timestamp": str(first) if first else None,
                    "last_timestamp": str(last) if last else None,
                    "rows": int(count or 0),
                    "distinct_rows": int(distinct_count or 0),
                    "expected_rows": expected,
                    "gaps": max(0, expected - int(distinct_count or 0)),
                    "duplicates": max(0, int(count or 0) - int(distinct_count or 0)),
                    "complete": bool(last and pd.Timestamp(last).tz_localize("UTC") >= end),
                }
            )
    return rows


def fetch_binance_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp, retries: int = 3) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cursor = start
    end_ms = int(end.timestamp() * 1000)
    while cursor <= end:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": int(cursor.timestamp() * 1000),
            "endTime": end_ms,
            "limit": 1500,
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                req = Request(f"{BINANCE_KLINES_URL}?{urlencode(params)}", headers={"User-Agent": "aegis-f0.2-research"})
                with urlopen(req, timeout=15) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise RuntimeError(f"unexpected Binance response: {payload!r}")
                chunk = payload
                break
            except Exception as exc:  # pragma: no cover - network branch
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"BINANCE_PUBLIC_KLINES_FAILED:{symbol}:{last_error}")
        if not chunk:
            break
        rows.extend(chunk)
        last_open = pd.to_datetime(int(chunk[-1][0]), unit="ms", utc=True)
        next_cursor = last_open + pd.Timedelta(minutes=INTERVAL_MINUTES)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(chunk) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "buy_volume"])
    df = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def read_local_klines(db_path: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    with open_ro(db_path) as con:
        df = pd.read_sql_query(
            """
            select timestamp, open, high, low, close, volume, buy_volume
            from ohlcv_data
            where symbol=? and timeframe=? and timestamp between ? and ?
            order by timestamp asc
            """,
            con,
            params=(db_symbol(symbol), timeframe, str(start.tz_convert(None)), str(end.tz_convert(None))),
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def write_symbol_csv(path: Path, df: pd.DataFrame) -> None:
    safe_research_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def gap_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    diff = df["timestamp"].sort_values().diff().dropna()
    return int((diff != pd.Timedelta(minutes=INTERVAL_MINUTES)).sum())


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    freeze = parse_dt(args.freeze_time) if args.freeze_time else pd.Timestamp(datetime.now(timezone.utc))
    start = parse_dt(args.start)
    end = min(closed_candle_end(freeze), parse_dt(args.until) if args.until else closed_candle_end(freeze))
    symbols = parse_symbols(args.symbols)
    local = local_coverage(Path(args.db_path), symbols, start, end, args.timeframe)
    local_ready = bool(local) and all(item["complete"] and item["gaps"] == 0 for item in local)
    source = "local_sqlite_read_only" if local_ready else "binance_futures_public_klines"
    stamp = utc_stamp()
    snapshot_dir = Path(args.output_root) / stamp
    manifest: dict[str, Any] = {
        "phase": "F0.2",
        "created_at_utc": stamp,
        "freeze_time": str(freeze),
        "start": str(start),
        "end": str(end),
        "symbols": symbols,
        "timeframe": args.timeframe,
        "local_db": str(Path(args.db_path)),
        "local_coverage": local,
        "source_selected": source,
        "read_only": True,
        "uses_api_key": False,
        "private_endpoints": False,
        "rows_by_symbol": {},
        "gaps_by_symbol": {},
        "duplicates_by_symbol": {},
        "checksums": {},
        "files": {},
    }
    if args.write_artifacts.lower() not in {"0", "false", "no"}:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        if source == "local_sqlite_read_only":
            df = read_local_klines(Path(args.db_path), symbol, start, end, args.timeframe)
        else:
            df = fetch_binance_klines(symbol, start, end)
        if df.empty:
            manifest["rows_by_symbol"][symbol] = 0
            manifest["gaps_by_symbol"][symbol] = None
            manifest["duplicates_by_symbol"][symbol] = None
            continue
        out_path = snapshot_dir / f"{symbol}_5m.csv"
        if args.write_artifacts.lower() not in {"0", "false", "no"}:
            write_symbol_csv(out_path, df)
            manifest["checksums"][symbol] = sha256_file(out_path)
            manifest["files"][symbol] = str(out_path)
        manifest["rows_by_symbol"][symbol] = int(len(df))
        manifest["gaps_by_symbol"][symbol] = gap_count(df)
        manifest["duplicates_by_symbol"][symbol] = int(len(df) - df["timestamp"].nunique())
    manifest["latest_closed_candle"] = str(end)
    manifest["decision"] = "RECENT_MARKET_SOURCE_READY" if all(manifest["rows_by_symbol"].get(s, 0) > 0 for s in symbols) else "RECENT_MARKET_SOURCE_PARTIAL"
    if args.write_artifacts.lower() not in {"0", "false", "no"}:
        write_json(snapshot_dir / "snapshot_manifest.json", manifest)
        atomic_write_text(snapshot_dir / "snapshot_manifest.sha256", sha256_file(snapshot_dir / "snapshot_manifest.json") + "\n")
    print(json.dumps({**manifest, "snapshot_dir": str(snapshot_dir)}, indent=2, default=json_default))
    return {**manifest, "snapshot_dir": str(snapshot_dir)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit and snapshot recent F0.2 5m market data")
    p.add_argument("--db-path", default=str(DEFAULT_DB))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", default="2026-06-01T00:00:00Z")
    p.add_argument("--until", default="")
    p.add_argument("--freeze-time", default="")
    p.add_argument("--write-artifacts", default="true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_audit(parse_args(argv))
        return 0 if payload["decision"] == "RECENT_MARKET_SOURCE_READY" else 2
    except Exception as exc:
        print(json.dumps({"decision": "RECENT_MARKET_SOURCE_NOT_READY", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
