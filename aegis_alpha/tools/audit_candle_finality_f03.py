#!/usr/bin/env python3
"""F0.3 diagnostic: detect non-final (partial) candles in the local SQLite OHLCV store.

Root cause of the F0.2 parity failure: the live refresher can capture a 5m bar
mid-bar and never re-fetch it after close, leaving a strict subset of the final
candle (open exact; high lower-or-equal, low higher-or-equal, volume smaller).
This audit compares the local store against a canonical Binance snapshot (the
immutable final klines already downloaded by F0.2), quantifies partial-bar rates
per symbol/day, and estimates how many tail labels flip when computed on final
candles. Read-only: opens SQLite in ro mode, no network, research artifacts only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SNAPSHOT_DIR = Path("/home/jasan/Develop/aegis_forward_research/trrm_f02/market_snapshots/20260711T080908Z")
DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "binance_candles.db"
DEFAULT_OUT = Path("/home/jasan/Develop")
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,SUIUSDT,LTCUSDT,LINKUSDT"
LEVERAGE = 20.0
TAIL_ROE = 0.30


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def db_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "")
    return s[:-4] + "/USDT" if s.endswith("USDT") else s


def load_local(db_path: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as con:
        df = pd.read_sql_query(
            "select timestamp, open, high, low, close, volume from ohlcv_data where symbol = ? and timeframe = ? order by timestamp",
            con,
            params=(db_symbol(symbol), timeframe),
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["timestamp"])


def load_snapshot(snapshot_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = snapshot_dir / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.dropna(subset=["timestamp"])


def future_max_high(high: np.ndarray, horizon: int) -> np.ndarray:
    """future_max[i] = max(high[i+1 .. i+horizon]); trailing rows padded with nan."""
    n = len(high)
    out = np.full(n, np.nan)
    if n <= horizon:
        return out
    stacked = np.column_stack([np.roll(high, -k) for k in range(1, horizon + 1)])
    out[: n - horizon] = stacked[: n - horizon].max(axis=1)
    return out


def tail_label_flip(merged: pd.DataFrame, horizon: int) -> dict[str, Any]:
    hi_db = merged["high_db"].to_numpy(float)
    hi_bn = merged["high_bn"].to_numpy(float)
    cl_db = merged["close_db"].to_numpy(float)
    cl_bn = merged["close_bn"].to_numpy(float)
    fut_db = future_max_high(hi_db, horizon)
    fut_bn = future_max_high(hi_bn, horizon)
    valid = ~np.isnan(fut_db)
    mae_db = np.maximum(0.0, (fut_db[valid] - cl_db[valid]) / cl_db[valid]) * LEVERAGE
    mae_bn = np.maximum(0.0, (fut_bn[valid] - cl_bn[valid]) / cl_bn[valid]) * LEVERAGE
    lab_db = mae_db >= TAIL_ROE
    lab_bn = mae_bn >= TAIL_ROE
    return {
        "horizon": horizon,
        "rows": int(valid.sum()),
        "prevalence_local": float(lab_db.mean()),
        "prevalence_final": float(lab_bn.mean()),
        "label_flip_rate": float((lab_db != lab_bn).mean()),
        "real_tails_missed_by_local": int(((~lab_db) & lab_bn).sum()),
        "fake_tails_in_local": int((lab_db & (~lab_bn)).sum()),
    }


def audit_symbol(db_path: Path, snapshot_dir: Path, symbol: str, timeframe: str, horizons: list[int]) -> dict[str, Any] | None:
    local = load_local(db_path, symbol, timeframe)
    snap = load_snapshot(snapshot_dir, symbol, timeframe)
    if local.empty or snap.empty:
        return None
    m = local.merge(snap, on="timestamp", suffixes=("_db", "_bn")).sort_values("timestamp").reset_index(drop=True)
    if m.empty:
        return None
    differs = ~np.isclose(m["close_db"].astype(float), m["close_bn"].astype(float), rtol=1e-9, atol=1e-12)
    for col in ("high", "low", "volume"):
        differs |= ~np.isclose(m[f"{col}_db"].astype(float), m[f"{col}_bn"].astype(float), rtol=1e-9, atol=1e-12)
    sub = m[differs]
    open_exact = bool(np.isclose(m["open_db"].astype(float), m["open_bn"].astype(float), rtol=1e-9).all())
    subset_signature = bool(
        len(sub) == 0
        or (
            (sub["volume_db"].astype(float) <= sub["volume_bn"].astype(float) + 1e-9).all()
            and (sub["high_db"].astype(float) <= sub["high_bn"].astype(float) + 1e-9).all()
            and (sub["low_db"].astype(float) >= sub["low_bn"].astype(float) - 1e-9).all()
        )
    )
    daily = (
        m.assign(bad=differs, date=m["timestamp"].dt.date)
        .groupby("date")["bad"]
        .mean()
        .round(4)
    )
    return {
        "symbol": symbol,
        "overlap_rows": int(len(m)),
        "overlap_start": str(m["timestamp"].min()),
        "overlap_end": str(m["timestamp"].max()),
        "partial_candle_count": int(differs.sum()),
        "partial_candle_rate": float(differs.mean()),
        "open_always_exact": open_exact,
        "partial_is_strict_subset_of_final": subset_signature,
        "worst_days": {str(k): float(v) for k, v in daily.sort_values(ascending=False).head(8).items()},
        "label_flip_by_horizon": [tail_label_flip(m, h) for h in horizons],
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    per_symbol = []
    for symbol in symbols:
        row = audit_symbol(Path(args.db_path), Path(args.snapshot_dir), symbol, args.timeframe, horizons)
        if row is not None:
            per_symbol.append(row)
    if not per_symbol:
        decision, reason = "NO_OVERLAP_DATA", "no symbol had both local and snapshot coverage"
        agg: dict[str, Any] = {}
    else:
        rates = [r["partial_candle_rate"] for r in per_symbol]
        flips = [f["label_flip_rate"] for r in per_symbol for f in r["label_flip_by_horizon"]]
        missed = sum(f["real_tails_missed_by_local"] for r in per_symbol for f in r["label_flip_by_horizon"])
        signature_ok = all(r["partial_is_strict_subset_of_final"] and r["open_always_exact"] for r in per_symbol)
        agg = {
            "symbols_audited": len(per_symbol),
            "mean_partial_candle_rate": float(np.mean(rates)),
            "max_partial_candle_rate": float(np.max(rates)),
            "mean_label_flip_rate": float(np.mean(flips)) if flips else None,
            "total_real_tails_missed_by_local": int(missed),
            "partial_signature_confirmed": signature_ok,
        }
        if agg["max_partial_candle_rate"] <= 0.001:
            decision, reason = "CANDLES_FINAL_OK", "local store matches final candles within tolerance"
        else:
            decision = "PARTIAL_CANDLES_DETECTED"
            reason = (
                "local SQLite bars were captured mid-bar by the refresher and never re-fetched after close; "
                "training/lockbox labels understate real tails and feature replay from canonical data cannot match"
            )
    payload = {
        "schema_version": "audit_candle_finality_f03_v1",
        "generated_at": stamp,
        "mode": "research-only read-only",
        "decision": decision,
        "reason": reason,
        "db_path": str(args.db_path),
        "snapshot_dir": str(args.snapshot_dir),
        "timeframe": args.timeframe,
        "tail_definition": f"future max-high MAE at {LEVERAGE:.0f}x >= {TAIL_ROE} ROE",
        "aggregate": agg,
        "per_symbol": per_symbol,
        "recommendation": [
            "Treat Binance final klines as the canonical source for research datasets and labels.",
            "Rebuild the D2 dataset and re-run the E2/E2.1 gauntlet on canonical candles before freezing any forward candidate.",
            "Operational (owner decision, touches live infra): make the refresher re-fetch the last N closed bars each cycle so stored bars finalize, and backfill-repair the historical DB.",
            "Add this audit to preflight for any future forward freeze; require partial rate <= 0.1%.",
        ],
    }
    json_path = out_dir / f"aegis_candle_finality_f03_{stamp}.json"
    md_path = out_dir / f"aegis_candle_finality_f03_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_lines = [
        "# F0.3 Candle Finality Audit",
        "",
        f"- decision: {decision}",
        f"- reason: {reason}",
        f"- aggregate: {json.dumps(agg, default=str)}",
        "",
        "| symbol | overlap | partial rate | subset sig | flips h12 | flips h24 | tails missed |",
        "|---|---:|---:|---|---:|---:|---:|",
        *[
            "| {s} | {n} | {r:.3f} | {sig} | {f12:.4f} | {f24:.4f} | {miss} |".format(
                s=r["symbol"], n=r["overlap_rows"], r=r["partial_candle_rate"], sig=r["partial_is_strict_subset_of_final"],
                f12=next((f["label_flip_rate"] for f in r["label_flip_by_horizon"] if f["horizon"] == 12), float("nan")),
                f24=next((f["label_flip_rate"] for f in r["label_flip_by_horizon"] if f["horizon"] == 24), float("nan")),
                miss=sum(f["real_tails_missed_by_local"] for f in r["label_flip_by_horizon"]),
            )
            for r in per_symbol
        ],
        "",
        "## Recommendation",
        *[f"- {x}" for x in payload["recommendation"]],
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    print(json.dumps({"decision": decision, "aggregate": agg, "md": str(md_path)}, indent=2, default=str))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F0.3 candle finality audit (research-only, read-only)")
    p.add_argument("--db-path", default=str(DEFAULT_DB))
    p.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--horizons", default="12,24")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
