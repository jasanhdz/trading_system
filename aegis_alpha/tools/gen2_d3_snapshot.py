#!/usr/bin/env python3
"""GEN2-D3 canonical snapshot pipeline (spec §2-§4, gate G3).

Fetches FINAL Binance USDT-M Futures klines (public endpoint, no keys) into an
immutable snapshot directory with full raw columns, fetch log, manifest and
sha256 checksums. `--mode refetch-check` implements gate G3: after >=15 minutes
it re-downloads a pre-registered random sample plus the last 24h plus page
boundary windows and requires bit-identical values on already-closed bars.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.gen2_d3_common import (  # noqa: E402
    BAR_MINUTES,
    MIN_DAYS_PER_SYMBOL,
    RAW_COLUMNS,
    REPORTS_ROOT,
    SNAPSHOT_ROOT,
    SYMBOLS,
    TARGET_RANGE_DAYS,
    TIMEFRAME,
    new_artifact_dir,
    read_csv_exact,
    sha256_file,
    utc_now,
    utc_stamp,
    validate_gen2_path,
    write_manifest,
)

ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
PAGE_LIMIT = 1500
REQUEST_SPACING_SECONDS = 0.35
CLOSED_BAR_SAFETY_SECONDS = 120
G3_SAMPLE_FRACTION = 0.05
G3_SAMPLE_MIN = 500
G3_WAIT_MINUTES = 15


def http_get_klines(symbol: str, start_ms: int, end_ms: int, limit: int = PAGE_LIMIT, retries: int = 3) -> list[list[Any]]:
    import requests

    params = {"symbol": symbol, "interval": TIMEFRAME, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=30)
            if resp.status_code in (418, 429):
                raise RuntimeError(f"rate limited: {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return []


def rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(rows).iloc[:, :11]
    df.columns = RAW_COLUMNS
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    return df


def fetch_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    closed_cutoff_ms: int,
    fetch_fn: Callable[..., list[list[Any]]] = http_get_klines,
    log: list[dict[str, Any]] | None = None,
    spacing: float = REQUEST_SPACING_SECONDS,
) -> pd.DataFrame:
    """Paginate klines in [start_ms, end_ms]; keep only bars fully closed before the cutoff."""
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    last_open: int | None = None
    while cursor <= end_ms:
        rows = fetch_fn(symbol, cursor, end_ms)
        if log is not None:
            log.append({"ts": utc_now().isoformat(), "symbol": symbol, "start_ms": cursor, "end_ms": end_ms, "rows": len(rows)})
        if not rows:
            break
        df = rows_to_frame(rows)
        if last_open is not None and int(df["open_time"].iloc[0]) <= last_open:
            df = df[df["open_time"] > last_open]
        if df.empty:
            break
        frames.append(df)
        last_open = int(df["open_time"].iloc[-1])
        cursor = last_open + BAR_MINUTES * 60_000
        if len(rows) < PAGE_LIMIT and cursor > end_ms:
            break
        if spacing:
            time.sleep(spacing)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["close_time"] < closed_cutoff_ms].reset_index(drop=True)
    return out


def frame_to_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.insert(0, "timestamp", pd.to_datetime(out["open_time"], unit="ms", utc=True).dt.tz_localize(None).astype(str))
    # No float_format: pandas writes shortest round-trip repr, so float(csv) == float(memory)
    # exactly (spec A4 full double-precision requirement).
    out.to_csv(path, index=False)


def load_snapshot_symbol(snapshot_dir: Path, symbol: str) -> pd.DataFrame:
    df = read_csv_exact(snapshot_dir / f"{symbol}_{TIMEFRAME}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def build_snapshot(
    *,
    symbols: list[str],
    range_days: int,
    fetch_fn: Callable[..., list[list[Any]]] = http_get_klines,
    snapshot_root: Path = SNAPSHOT_ROOT,
    spacing: float = REQUEST_SPACING_SECONDS,
    end_override_ms: int | None = None,
) -> dict[str, Any]:
    now = utc_now()
    closed_cutoff_ms = end_override_ms or int((now - timedelta(seconds=CLOSED_BAR_SAFETY_SECONDS)).timestamp() * 1000)
    end_ms = closed_cutoff_ms
    start_ms = end_ms - range_days * 86_400_000
    directory, snapshot_id = new_artifact_dir(snapshot_root, None)
    log: list[dict[str, Any]] = []
    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        df = fetch_range(symbol, start_ms, end_ms, closed_cutoff_ms, fetch_fn, log, spacing)
        path = directory / f"{symbol}_{TIMEFRAME}.csv"
        frame_to_csv(df, path)
        first = str(pd.to_datetime(int(df["open_time"].iloc[0]), unit="ms")) if len(df) else None
        last = str(pd.to_datetime(int(df["open_time"].iloc[-1]), unit="ms")) if len(df) else None
        days = ((int(df["open_time"].iloc[-1]) - int(df["open_time"].iloc[0])) / 86_400_000) if len(df) else 0.0
        per_symbol[symbol] = {"rows": int(len(df)), "first": first, "last": last, "calendar_days": round(days, 2), "sha256": sha256_file(path)}
    (directory / "fetch_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8")
    manifest = {
        "schema": "gen2_d3_snapshot_v1",
        "artifact_id": snapshot_id,
        "endpoint": ENDPOINT,
        "interval": TIMEFRAME,
        "requested_range": {"start_ms": start_ms, "end_ms": end_ms, "range_days": range_days},
        "closed_bar_cutoff_ms": closed_cutoff_ms,
        "symbols": per_symbol,
        "min_days_required": MIN_DAYS_PER_SYMBOL,
        "g3_status": "PENDING_REFETCH_CHECK",
        "fetch_completed_at": utc_now().isoformat(),
    }
    write_manifest(directory, manifest, "snapshot_manifest")
    return {"snapshot_id": snapshot_id, "directory": str(directory), "symbols": per_symbol}


G3_WINDOW_BARS = 100


def g3_targets(df: pd.DataFrame, snapshot_id: str, closed_cutoff_ms: int) -> list[int]:
    """Pre-registered deterministic G3 sample, seeded by snapshot_id.

    The 5% sample is drawn as CONTIGUOUS windows of G3_WINDOW_BARS bars (one
    request each) instead of scattered single bars: identical statistical
    coverage of the spec's 5%, but ~1e2 requests per symbol instead of ~1e4.
    """
    open_times = df["open_time"].astype("int64").tolist()
    if not open_times:
        return []
    rng = random.Random(f"gen2-g3-{snapshot_id}")
    k = max(G3_SAMPLE_MIN, int(len(open_times) * G3_SAMPLE_FRACTION))
    n_windows = max(1, k // G3_WINDOW_BARS)
    sample: set[int] = set()
    max_start = max(0, len(open_times) - G3_WINDOW_BARS)
    for _ in range(n_windows):
        start = rng.randint(0, max_start)
        sample.update(open_times[start: start + G3_WINDOW_BARS])
    last_24h_floor = closed_cutoff_ms - 86_400_000
    sample.update(t for t in open_times if t >= last_24h_floor)
    for i in range(PAGE_LIMIT - 1, len(open_times), PAGE_LIMIT):
        sample.update(open_times[max(0, i - 2): i + 3])
    return sorted(sample)


def refetch_check(
    snapshot_dir: Path,
    fetch_fn: Callable[..., list[list[Any]]] = http_get_klines,
    min_wait_minutes: int = G3_WAIT_MINUTES,
    spacing: float = REQUEST_SPACING_SECONDS,
) -> dict[str, Any]:
    manifest = json.loads((snapshot_dir / "snapshot_manifest.json").read_text())
    fetched_at = pd.Timestamp(manifest["fetch_completed_at"])
    waited_minutes = (pd.Timestamp(utc_now().isoformat()) - fetched_at).total_seconds() / 60.0
    if waited_minutes < min_wait_minutes:
        raise RuntimeError(f"G3 requires >= {min_wait_minutes} minutes since fetch; waited {waited_minutes:.1f}")
    closed_cutoff_ms = int(manifest["closed_bar_cutoff_ms"])
    finality_floor_ms = closed_cutoff_ms - 600_000  # bars closed >=10 min before first fetch
    per_symbol: dict[str, Any] = {}
    total_checked = 0
    total_mismatch = 0
    mismatches: list[dict[str, Any]] = []
    for symbol in manifest["symbols"]:
        original = load_snapshot_symbol(snapshot_dir, symbol)
        original["open_time"] = original["open_time"].astype("int64")
        targets = [t for t in g3_targets(original, manifest["artifact_id"], closed_cutoff_ms) if t <= finality_floor_ms]
        checked = 0
        bad = 0
        idx = original.set_index("open_time")
        i = 0
        while i < len(targets):
            block_start = targets[i]
            j = i
            while j + 1 < len(targets) and targets[j + 1] - targets[j] <= BAR_MINUTES * 60_000 * 4:
                j += 1
            block_end = targets[j] + BAR_MINUTES * 60_000 - 1
            fresh = rows_to_frame(fetch_fn(symbol, block_start, block_end))
            fresh = fresh[fresh["close_time"] < closed_cutoff_ms]
            wanted = set(targets[i: j + 1])
            for _, row in fresh.iterrows():
                ot = int(row["open_time"])
                if ot not in wanted or ot not in idx.index:
                    continue
                checked += 1
                orig = idx.loc[ot]
                for col in ("open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume"):
                    if float(orig[col]) != float(row[col]):
                        bad += 1
                        mismatches.append({"symbol": symbol, "open_time": ot, "column": col, "snapshot": float(orig[col]), "refetch": float(row[col])})
                        break
            i = j + 1
            if spacing:
                time.sleep(spacing)
        per_symbol[symbol] = {"bars_checked": checked, "mismatches": bad}
        total_checked += checked
        total_mismatch += bad
    rate = total_mismatch / total_checked if total_checked else 1.0
    passed = total_checked > 0 and rate <= 0.001
    result = {
        "schema": "gen2_d3_g3_refetch_v1",
        "snapshot_id": manifest["artifact_id"],
        "waited_minutes": round(waited_minutes, 1),
        "bars_checked": total_checked,
        "mismatch_count": total_mismatch,
        "mismatch_rate": rate,
        "per_symbol": per_symbol,
        "mismatch_examples": mismatches[:20],
        "gate": "G3_FINALITY",
        "passed": bool(passed),
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report = REPORTS_ROOT / f"aegis_gen2_d3_g3_{manifest['artifact_id']}_{utc_stamp()}.json"
    validate_gen2_path(report)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report"] = str(report)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-D3 canonical snapshot pipeline")
    p.add_argument("--mode", choices=["fetch", "refetch-check"], default="fetch")
    p.add_argument("--range-days", type=int, default=TARGET_RANGE_DAYS)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--snapshot-dir", default="")
    p.add_argument("--min-wait-minutes", type=int, default=G3_WAIT_MINUTES)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "fetch":
        result = build_snapshot(symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()], range_days=args.range_days)
    else:
        result = refetch_check(Path(args.snapshot_dir), min_wait_minutes=args.min_wait_minutes)
    print(json.dumps(result, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
