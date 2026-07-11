#!/usr/bin/env python3
"""FASE-F0.2 refresh TRRM causal features from a market snapshot.

Research-only. Reuses the D/D2 causal feature functions and never writes labels
or target columns to the refreshed feature dataset.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import add_market_context, compute_causal_features, db_symbol, is_leakage_column
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import feature_hash, load_pipeline
from aegis_alpha.tools.train_trrm_honest_e2 import MedianImputer, StandardScalerLite
from aegis_alpha.tools.trrm_forward_common_f0 import EXPECTED_FEATURE_HASH, atomic_write_text, safe_research_path, sha256_file, write_json

DEFAULT_D2 = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_MODEL = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_OUTPUT = Path("/home/jasan/Develop/aegis_forward_research/trrm_f02/features")
DEFAULT_DB = REPO / "data" / "binance_candles.db"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "SOLUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SUIUSDT", "LTCUSDT"]
HORIZONS = [6, 12, 24]


def utc_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_dt(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value}")
    return ts


def load_local_candles(db_path: Path, symbols: list[str], start: pd.Timestamp | None, end: pd.Timestamp, d2_lookback_days: int = 365) -> dict[str, pd.DataFrame]:
    if not db_path.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as con:
        for symbol in symbols:
            symbol_start = start
            if symbol_start is None:
                max_row = con.execute("select max(timestamp) from ohlcv_data where symbol=? and timeframe='5m'", (db_symbol(symbol),)).fetchone()
                if not max_row or not max_row[0]:
                    continue
                symbol_start = pd.Timestamp(max_row[0], tz="UTC") - pd.Timedelta(days=d2_lookback_days)
            df = pd.read_sql_query(
                """
                select timestamp, open, high, low, close, volume, buy_volume
                from ohlcv_data
                where symbol=? and timeframe='5m' and timestamp between ? and ?
                order by timestamp asc
                """,
                con,
                params=(db_symbol(symbol), str(symbol_start.tz_convert(None)), str(end.tz_convert(None))),
            )
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
            for col in ("open", "high", "low", "close", "volume", "buy_volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            out[symbol] = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return out


def load_snapshot(snapshot_dir: Path, symbols: list[str], local_db: Path | None = None, warmup_start: pd.Timestamp | None = None, d2_warmup: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    snapshot_frames: dict[str, pd.DataFrame] = {}
    snapshot_end: pd.Timestamp | None = None
    for symbol in symbols:
        path = snapshot_dir / f"{symbol}_5m.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        for col in ("open", "high", "low", "close", "volume", "buy_volume"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]].reset_index(drop=True)
        snapshot_frames[symbol] = df
        if not df.empty:
            mx = pd.Timestamp(df["timestamp"].max(), tz="UTC")
            snapshot_end = mx if snapshot_end is None else max(snapshot_end, mx)
    local: dict[str, pd.DataFrame] = {}
    if local_db is not None and snapshot_end is not None and (warmup_start is not None or d2_warmup):
        local = load_local_candles(local_db, symbols, None if d2_warmup else warmup_start, snapshot_end)
    for symbol in symbols:
        frames = []
        if symbol in local:
            frames.append(local[symbol])
        if symbol in snapshot_frames:
            frames.append(snapshot_frames[symbol])
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        out[symbol] = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return out


def build_feature_tables(candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    base = {symbol: compute_causal_features(df) for symbol, df in candles.items() if not df.empty}
    context = {k: v for k, v in base.items() if k in {"BTCUSDT", "ETHUSDT"}}
    return {symbol: add_market_context(frame, context) for symbol, frame in base.items()}


def model_feature_columns(model_dir: Path) -> tuple[list[str], list[str]]:
    pipe = load_pipeline(model_dir)
    all_features = list(pipe["features"])
    causal = [c for c in all_features if c.startswith("feature.")]
    horizon = [c for c in all_features if c.startswith("horizon_")]
    return causal, horizon


def safe_feature_payload(row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for col in feature_cols:
        source = col.removeprefix("feature.")
        leak, reason = is_leakage_column(col)
        if leak:
            raise ValueError(f"LEAKAGE_FEATURE_DETECTED:{col}:{reason}")
        payload[col] = pd.to_numeric(row.get(source, np.nan), errors="coerce")
    return payload


def rows_for_targets(tables: dict[str, pd.DataFrame], targets: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol, group in targets.groupby("id.symbol"):
        table = tables.get(str(symbol))
        if table is None or table.empty:
            continue
        lookup = table.set_index("timestamp")
        for _, target in group.iterrows():
            ts = pd.to_datetime(target["id.timestamp"], utc=True).tz_convert(None)
            if ts not in lookup.index:
                continue
            feat = lookup.loc[ts]
            if isinstance(feat, pd.DataFrame):
                feat = feat.iloc[-1]
            for horizon in HORIZONS:
                if int(target["id.horizon"]) != horizon:
                    continue
                row = {
                    "id.symbol": symbol,
                    "id.timestamp": str(ts),
                    "id.timeframe": "5m",
                    "id.horizon": horizon,
                }
                row.update(safe_feature_payload(feat, feature_cols))
                frames.append(row)
    return pd.DataFrame(frames)


def rows_for_interval(tables: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_naive = start.tz_convert(None)
    end_naive = end.tz_convert(None)
    for symbol in SYMBOLS:
        table = tables.get(symbol)
        if table is None or table.empty:
            continue
        part = table[(table["timestamp"] >= start_naive) & (table["timestamp"] <= end_naive)].copy()
        for _, feat in part.iterrows():
            for horizon in HORIZONS:
                row = {
                    "id.symbol": symbol,
                    "id.timestamp": str(feat["timestamp"]),
                    "id.timeframe": "5m",
                    "id.horizon": horizon,
                }
                row.update(safe_feature_payload(feat, feature_cols))
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True)
    return out


def compare_overlap(rebuilt: pd.DataFrame, d2: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    keys = ["id.symbol", "id.timestamp", "id.timeframe", "id.horizon"]
    left = d2[keys + feature_cols].copy()
    right = rebuilt[keys + feature_cols].copy()
    for frame in (left, right):
        frame["id.timestamp"] = pd.to_datetime(frame["id.timestamp"], utc=True).dt.tz_convert(None).astype(str)
    merged = left.merge(right, on=keys, how="outer", suffixes=("_expected", "_rebuilt"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    total_expected = len(left)
    row_match_rate = len(matched) / total_expected if total_expected else 0.0
    total_values = 0
    matching_values = 0
    max_abs = 0.0
    abs_sum = 0.0
    dtype_mismatches = []
    for col in feature_cols:
        a = pd.to_numeric(matched[f"{col}_expected"], errors="coerce")
        b = pd.to_numeric(matched[f"{col}_rebuilt"], errors="coerce")
        both_nan = a.isna() & b.isna()
        diff = (a - b).abs()
        eq = both_nan | (diff <= 1e-12)
        total_values += len(eq)
        matching_values += int(eq.sum())
        finite = diff.replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            max_abs = max(max_abs, float(finite.max()))
            abs_sum += float(finite.sum())
        if str(left[col].dtype) != str(right[col].dtype):
            dtype_mismatches.append({"column": col, "expected": str(left[col].dtype), "rebuilt": str(right[col].dtype)})
    value_match_rate = matching_values / total_values if total_values else 0.0
    return {
        "rows_expected": int(total_expected),
        "rows_rebuilt": int(len(right)),
        "rows_matched": int(len(matched)),
        "row_match_rate": row_match_rate,
        "feature_value_match_rate": value_match_rate,
        "missing_mismatches": int((merged["_merge"] != "both").sum()),
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": abs_sum / total_values if total_values else 0.0,
        "dtype_mismatches": dtype_mismatches[:20],
        "timestamp_mismatches": int((merged["_merge"] != "both").sum()),
        "passes": row_match_rate >= 0.999 and value_match_rate >= 0.9999,
    }


def forbidden_columns(df: pd.DataFrame) -> list[str]:
    bad = []
    for col in df.columns:
        low = col.lower()
        if any(token in low for token in ("target", "label", "future", "future_mae", "pnl", "outcome")):
            bad.append(col)
    return bad


def run_refresh(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(args.snapshot_dir)
    if not snapshot_dir.exists():
        raise FileNotFoundError(snapshot_dir)
    feature_cols, horizon_cols = model_feature_columns(Path(args.model_dir))
    if len(feature_cols) != 111 or sorted(horizon_cols) != ["horizon_12", "horizon_24", "horizon_6"]:
        raise ValueError("ARTIFACT_INTEGRITY_ERROR: unexpected model feature layout")
    if feature_hash(feature_cols + horizon_cols) != args.feature_hash:
        raise ValueError("ARTIFACT_INTEGRITY_ERROR: feature hash mismatch")
    d2_warmup = str(args.warmup_start).lower() == "d2_365d_by_symbol"
    candles = load_snapshot(
        snapshot_dir,
        SYMBOLS,
        Path(args.local_db) if args.local_db else None,
        None if d2_warmup or not args.warmup_start else parse_dt(args.warmup_start),
        d2_warmup=d2_warmup,
    )
    tables = build_feature_tables(candles)
    start = parse_dt(args.start)
    end = parse_dt(args.end)
    output_dir = Path(args.output_dir)
    safe_research_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    payload: dict[str, Any] = {
        "phase": "F0.2",
        "mode": args.mode,
        "snapshot_dir": str(snapshot_dir),
        "local_warmup_db": str(args.local_db) if args.local_db else None,
        "warmup_start": args.warmup_start,
        "d2_warmup_by_symbol": d2_warmup,
        "feature_hash": args.feature_hash,
        "feature_count": len(feature_cols),
        "horizon_features": horizon_cols,
        "labels_present": False,
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    if args.mode == "overlap":
        d2 = pd.read_csv(args.d2_csv)
        ts = pd.to_datetime(d2["id.timestamp"], utc=True)
        d2_overlap = d2[(ts >= start) & (ts <= end)].copy()
        rebuilt = rows_for_targets(tables, d2_overlap, feature_cols)
        comp = compare_overlap(rebuilt, d2_overlap, feature_cols)
        out_path = output_dir / f"aegis_trrm_causal_features_overlap_replay_{stamp}.csv"
        rebuilt.to_csv(out_path, index=False)
        payload.update({"decision": "OVERLAP_REPLAY_OK" if comp["passes"] else "CAUSAL_FEATURE_REPLAY_MISMATCH", "overlap": comp, "output_csv": str(out_path), "output_sha256": sha256_file(out_path)})
    else:
        out = rows_for_interval(tables, start, end, feature_cols)
        bad = forbidden_columns(out)
        if bad:
            raise ValueError(f"FORWARD_LABEL_COLUMN_DETECTED:{bad}")
        out_path = output_dir / f"aegis_trrm_causal_features_incremental_{stamp}.csv"
        out.to_csv(out_path, index=False)
        payload.update(
            {
                "decision": "INCREMENTAL_FEATURES_READY" if not out.empty else "RECENT_FEATURE_REFRESH_FAILED",
                "output_csv": str(out_path),
                "output_sha256": sha256_file(out_path),
                "rows": int(len(out)),
                "symbols": sorted(out["id.symbol"].unique().tolist()) if not out.empty else [],
                "horizons": sorted(int(x) for x in out["id.horizon"].unique()) if not out.empty else [],
                "timestamp_min": str(out["id.timestamp"].min()) if not out.empty else None,
                "timestamp_max": str(out["id.timestamp"].max()) if not out.empty else None,
                "forbidden_columns": bad,
            }
        )
    js = output_dir / f"aegis_phase_f02_feature_refresh_{args.mode}_{stamp}.json"
    md = output_dir / f"aegis_phase_f02_feature_refresh_{args.mode}_{stamp}.md"
    if args.write_report.lower() not in {"0", "false", "no"}:
        write_json(js, payload)
        atomic_write_text(md, f"# FASE-F0.2 Feature Refresh\n\n- decision: {payload['decision']}\n- mode: {args.mode}\n- FORWARD_OUTCOMES_NOT_EVALUATED\n")
        payload["report_json"] = str(js)
        payload["report_md"] = str(md)
    print(json.dumps(payload, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh F0.2 TRRM causal features from market snapshot")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--mode", choices=["overlap", "incremental"], default="overlap")
    p.add_argument("--d2-csv", default=str(DEFAULT_D2))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p.add_argument("--local-db", default=str(DEFAULT_DB))
    p.add_argument("--warmup-start", default="d2_365d_by_symbol")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--start", default="2026-07-01T00:00:00Z")
    p.add_argument("--end", default="2026-07-08T22:45:00Z")
    p.add_argument("--feature-hash", default=EXPECTED_FEATURE_HASH)
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_refresh(parse_args(argv))
        return 0 if payload["decision"] in {"OVERLAP_REPLAY_OK", "INCREMENTAL_FEATURES_READY"} else 2
    except Exception as exc:
        print(json.dumps({"decision": "RECENT_FEATURE_REFRESH_FAILED", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
