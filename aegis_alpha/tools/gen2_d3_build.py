#!/usr/bin/env python3
"""GEN2-D3 canonical series and dataset builder (spec v1.1 §3, §6-§8).

Stages:
  --mode series        snapshot -> gates G1/G2/G4/G5 -> canonical_series/<version>/
  --mode dataset       canonical series -> D3 dataset (frozen Gen1 builder + labels)
  --mode double-check  compare two dataset builds for byte identity (spec §8)
  --mode replay        gate G7: rebuild rows from an independent snapshot and compare

The feature and label logic is imported UNCHANGED from the frozen Gen1 modules
(build_trrm_causal_feature_dataset_d, build_risk_v4_long_horizon_dataset_d2,
short_quality_v4_labels, audit_tail_risk_targets_d2). This file is I/O and gate
orchestration only (Amendment A1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import audit_targets, json_default, strided_view  # noqa: E402
from aegis_alpha.tools.build_risk_v4_long_horizon_dataset_d2 import build_rows_for_symbol  # noqa: E402
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import compute_causal_features  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import (  # noqa: E402
    BAR_MINUTES,
    DATASET_ROOT,
    MIN_DAYS_PER_SYMBOL,
    REPORTS_ROOT,
    SERIES_ROOT,
    SYMBOLS,
    TIMEFRAME,
    read_csv_exact,
    sha256_file,
    sha256_text,
    utc_stamp,
    validate_gen2_path,
    write_manifest,
)
from aegis_alpha.tools.gen2_d3_snapshot import load_snapshot_symbol  # noqa: E402

# Frozen Gen1 sampling contract (Amendment A1) — identical to FASE-D2 defaults.
FROZEN_HORIZONS = [6, 12, 24]
FROZEN_MAX_ROWS_PER_COMBO = 5000
FROZEN_SEED = 42
FROZEN_WARMUP_INDEX = 64
FROZEN_EMBARGO_MINUTES = 120
FROZEN_TARGET = "target.tail_risk_roe_030"
SERIES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]
GAP_RATIO_MAX = 0.001
GAP_RUN_MAX_BARS = 12
GEN1_REFERENCE_DATASET = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
REPLAY_WARMUP_DAYS = 14
REPLAY_RTOL = 1e-9
REPLAY_ATOL = 1e-12
REPLAY_PARITY_MIN = 0.999


# ---------------------------------------------------------------- series stage
def gate_g1_schema(df: pd.DataFrame, symbol: str) -> list[str]:
    errors: list[str] = []
    for col in ("timestamp", "open", "high", "low", "close", "volume", "taker_buy_base_volume"):
        if col not in df.columns:
            errors.append(f"{symbol}: missing column {col}")
    if errors:
        return errors
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        errors.append(f"{symbol}: unparseable timestamps")
    if not ts.is_monotonic_increasing:
        errors.append(f"{symbol}: timestamps not monotonic increasing")
    ohlcv = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if ohlcv.isna().any().any():
        errors.append(f"{symbol}: NaN in OHLCV")
        return errors
    body_high = ohlcv[["open", "close"]].max(axis=1)
    body_low = ohlcv[["open", "close"]].min(axis=1)
    if (ohlcv["high"] < body_high - 1e-12).any():
        errors.append(f"{symbol}: high < max(open, close) on {(ohlcv['high'] < body_high - 1e-12).sum()} bars")
    if (ohlcv["low"] > body_low + 1e-12).any():
        errors.append(f"{symbol}: low > min(open, close) on {(ohlcv['low'] > body_low + 1e-12).sum()} bars")
    if (ohlcv["volume"] < 0).any():
        errors.append(f"{symbol}: negative volume")
    tb = pd.to_numeric(df["taker_buy_base_volume"], errors="coerce")
    if (tb > ohlcv["volume"] + 1e-9).any():
        errors.append(f"{symbol}: taker_buy_base > volume")
    return errors


def gate_g2_duplicates(df: pd.DataFrame, symbol: str) -> list[str]:
    dup = df["open_time"].duplicated()
    return [f"{symbol}: {int(dup.sum())} duplicated open_time values"] if dup.any() else []


def gate_g4_gaps(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    ot = df["open_time"].astype("int64").to_numpy()
    step = BAR_MINUTES * 60_000
    diffs = np.diff(ot) // step
    gap_bars = int((diffs - 1).clip(min=0).sum())
    expected = int((ot[-1] - ot[0]) // step) + 1 if len(ot) else 0
    max_run = int(diffs.max() - 1) if len(diffs) else 0
    last_365_floor = ot[-1] - 365 * 86_400_000 if len(ot) else 0
    recent = diffs[ot[1:] >= last_365_floor]
    recent_gap_bars = int((recent - 1).clip(min=0).sum())
    recent_expected = int(max(1, (ot[-1] - max(ot[0], last_365_floor)) // step))
    recent_max_run = int(recent.max() - 1) if len(recent) else 0
    ratio = recent_gap_bars / recent_expected if recent_expected else 1.0
    return {
        "symbol": symbol,
        "expected_bars": expected,
        "actual_bars": int(len(ot)),
        "gap_bars_total": gap_bars,
        "gap_ratio_last_365d": ratio,
        "max_gap_run_last_365d": recent_max_run,
        "max_gap_run_total": max_run,
        "passes": bool(ratio <= GAP_RATIO_MAX and recent_max_run <= GAP_RUN_MAX_BARS),
    }


def gate_g5_coverage(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    ot = df["open_time"].astype("int64")
    days = (int(ot.iloc[-1]) - int(ot.iloc[0])) / 86_400_000 if len(ot) else 0.0
    return {"symbol": symbol, "calendar_days": round(days, 2), "passes": bool(days >= MIN_DAYS_PER_SYMBOL)}


def build_series(snapshot_dir: Path, series_version: str) -> dict[str, Any]:
    manifest_src = json.loads((snapshot_dir / "snapshot_manifest.json").read_text())
    out_dir = SERIES_ROOT / series_version
    validate_gen2_path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"series version exists (immutability): {out_dir}")
    out_dir.mkdir(parents=True)
    g1: list[str] = []
    g2: list[str] = []
    g4: list[dict[str, Any]] = []
    g5: list[dict[str, Any]] = []
    included: dict[str, Any] = {}
    excluded: list[dict[str, str]] = []
    for symbol in manifest_src["symbols"]:
        df = load_snapshot_symbol(snapshot_dir, symbol)
        errors = gate_g1_schema(df, symbol) + gate_g2_duplicates(df, symbol)
        gaps = gate_g4_gaps(df, symbol)
        cov = gate_g5_coverage(df, symbol)
        g1.extend(errors)
        g4.append(gaps)
        g5.append(cov)
        if errors:
            excluded.append({"symbol": symbol, "reason": "; ".join(errors)})
            continue
        if not gaps["passes"]:
            excluded.append({"symbol": symbol, "reason": f"gap gate failed: {gaps}"})
            continue
        if not cov["passes"]:
            excluded.append({"symbol": symbol, "reason": f"coverage gate failed: {cov}"})
            continue
        series = df.rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS].copy()
        series["timestamp"] = pd.to_datetime(series["timestamp"]).astype(str)
        path = out_dir / f"{symbol}_{TIMEFRAME}.csv"
        series.to_csv(path, index=False)
        included[symbol] = {"rows": int(len(series)), "first": series["timestamp"].iloc[0], "last": series["timestamp"].iloc[-1], "sha256": sha256_file(path)}
    status = "OK" if len(included) == len(manifest_src["symbols"]) else ("PARTIAL" if len(included) >= len(manifest_src["symbols"]) - 2 else "REJECTED")
    manifest = {
        "schema": "gen2_d3_series_v1",
        "artifact_id": series_version,
        "source_snapshot": manifest_src["artifact_id"],
        "source_snapshot_dir": str(snapshot_dir),
        "gates": {"g1_g2_errors": g1 + g2, "g4_gaps": g4, "g5_coverage": g5},
        "included_symbols": included,
        "excluded_symbols": excluded,
        "status": status,
    }
    write_manifest(out_dir, manifest, "series_manifest")
    return {"series_version": series_version, "directory": str(out_dir), "status": status, "included": len(included), "excluded": excluded}


def load_series_symbol(series_dir: Path, symbol: str) -> pd.DataFrame:
    df = read_csv_exact(series_dir / f"{symbol}_{TIMEFRAME}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------- dataset stage
def gen1_feature_contract() -> list[str]:
    header = read_csv_exact(GEN1_REFERENCE_DATASET, nrows=0)
    return [c for c in header.columns if c.startswith("feature.")]


def build_dataset(series_version: str, out_tag: str, enforce_contract: bool = True) -> dict[str, Any]:
    series_dir = SERIES_ROOT / series_version
    series_manifest = json.loads((series_dir / "series_manifest.json").read_text())
    symbols = [s for s in SYMBOLS if s in series_manifest["included_symbols"]]
    out_dir = DATASET_ROOT / f"d3-v1-{out_tag}"
    validate_gen2_path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"dataset dir exists (immutability): {out_dir}")
    out_dir.mkdir(parents=True)

    context_cache: dict[str, pd.DataFrame] = {}
    for ctx_symbol in ("BTCUSDT", "ETHUSDT"):
        ctx = load_series_symbol(series_dir, ctx_symbol) if ctx_symbol in series_manifest["included_symbols"] else pd.DataFrame()
        context_cache[ctx_symbol] = compute_causal_features(ctx) if not ctx.empty else pd.DataFrame()

    frames: list[pd.DataFrame] = []
    sampling: dict[str, Any] = {}
    for symbol in symbols:
        candles = load_series_symbol(series_dir, symbol)
        frame, manifest = build_rows_for_symbol(
            candles, symbol, TIMEFRAME, FROZEN_HORIZONS, FROZEN_MAX_ROWS_PER_COMBO, FROZEN_SEED, context_cache
        )
        sampling[symbol] = manifest
        if not frame.empty:
            frames.append(frame)
    dense = pd.concat(frames, axis=0).sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True)
    audit = audit_targets(dense, embargo_minutes=FROZEN_EMBARGO_MINUTES)
    dense = audit["dataframe"].sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True)
    strided = strided_view(dense)

    features = [c for c in dense.columns if c.startswith("feature.")]
    contract = gen1_feature_contract() if enforce_contract else features
    contract_ok = features == contract
    feature_hash = sha256_text("\n".join(features))
    contract_hash = sha256_text("\n".join(contract))

    dense_path = out_dir / "dense.csv"
    strided_path = out_dir / "strided.csv"
    dense.to_csv(dense_path, index=False)
    strided.to_csv(strided_path, index=False)
    labels = dense.groupby(["id.symbol", "id.horizon"]).agg(
        rows=("id.timestamp", "size"),
        tail_roe_030=(FROZEN_TARGET, "mean"),
        bad_entry=("target.bad_entry_v4", "mean"),
        clean_entry=("label.clean_entry_v4", "mean"),
    ).reset_index()
    labels_path = out_dir / "labels_summary.csv"
    labels.to_csv(labels_path, index=False)

    manifest = {
        "schema": "gen2_d3_dataset_v1",
        "artifact_id": f"d3-v1-{out_tag}",
        "series_version": series_version,
        "series_manifest_source_snapshot": series_manifest["source_snapshot"],
        "frozen_contract": {
            "horizons": FROZEN_HORIZONS,
            "max_rows_per_combo": FROZEN_MAX_ROWS_PER_COMBO,
            "seed": FROZEN_SEED,
            "warmup_index": FROZEN_WARMUP_INDEX,
            "embargo_minutes": FROZEN_EMBARGO_MINUTES,
            "frozen_target": FROZEN_TARGET,
        },
        "feature_columns": features,
        "feature_hash": feature_hash,
        "gen1_contract_hash": contract_hash,
        "feature_contract_ok": bool(contract_ok),
        "rows_dense": int(len(dense)),
        "rows_strided": int(len(strided)),
        "timestamp_min": str(dense["id.timestamp"].min()),
        "timestamp_max": str(dense["id.timestamp"].max()),
        "target_rates": {k: v.get("global_rate") for k, v in audit["target_stats"].items()},
        "audit_selected_target_informational": audit["selected_candidate"].get("selected_target"),
        "sampling": sampling,
        "content_sha256": {"dense.csv": sha256_file(dense_path), "strided.csv": sha256_file(strided_path), "labels_summary.csv": sha256_file(labels_path)},
    }
    write_manifest(out_dir, manifest, "dataset_manifest")
    if not contract_ok:
        missing = [c for c in contract if c not in features]
        extra = [c for c in features if c not in contract]
        return {"decision": "D3_FEATURE_CONTRACT_BROKEN", "directory": str(out_dir), "missing_vs_gen1": missing[:20], "extra_vs_gen1": extra[:20]}
    return {"decision": "DATASET_BUILT", "directory": str(out_dir), "rows_dense": int(len(dense)), "rows_strided": int(len(strided)), "feature_hash": feature_hash, "tail_rate": manifest["target_rates"].get(FROZEN_TARGET.replace("target.", "")) or manifest["target_rates"].get(FROZEN_TARGET)}


def double_check(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    ma = json.loads((dir_a / "dataset_manifest.json").read_text())
    mb = json.loads((dir_b / "dataset_manifest.json").read_text())
    files = ("dense.csv", "strided.csv", "labels_summary.csv")
    comparison = {
        f: {"a": ma["content_sha256"][f], "b": mb["content_sha256"][f], "identical": ma["content_sha256"][f] == mb["content_sha256"][f]}
        for f in files
    }
    identical = all(v["identical"] for v in comparison.values())
    checks = {
        "content_bit_identical": identical,
        "feature_hash_identical": ma["feature_hash"] == mb["feature_hash"],
        "row_counts_identical": ma["rows_dense"] == mb["rows_dense"] and ma["rows_strided"] == mb["rows_strided"],
        "target_rates_identical": ma["target_rates"] == mb["target_rates"],
    }
    return {"gate": "DOUBLE_BUILD_DETERMINISM", "passed": all(checks.values()), "checks": checks, "files": comparison}


# ----------------------------------------------------------------- replay (G7)
def replay_check(dataset_dir: Path, replay_snapshot_dir: Path) -> dict[str, Any]:
    from aegis_alpha.tools.build_risk_v4_long_horizon_dataset_d2 import build_rows_for_symbol as rows_fn

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    dense = read_csv_exact(dataset_dir / "dense.csv")
    dense["_ts"] = pd.to_datetime(dense["id.timestamp"], errors="coerce")
    snap_manifest = json.loads((replay_snapshot_dir / "snapshot_manifest.json").read_text())
    symbols = [s for s in snap_manifest["symbols"] if f"{s}" in set(dense["id.symbol"])]

    context_cache: dict[str, pd.DataFrame] = {}
    replay_candles: dict[str, pd.DataFrame] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        if symbol in snap_manifest["symbols"]:
            raw = load_snapshot_symbol(replay_snapshot_dir, symbol).rename(columns={"taker_buy_base_volume": "buy_volume"})
            raw = raw[SERIES_COLUMNS].copy()
            context_cache[symbol] = compute_causal_features(raw)
    total = 0
    matched = 0
    rows_expected = 0
    rows_rebuilt = 0
    per_feature_mismatch: dict[str, int] = {}
    compare_cols: list[str] = []
    for symbol in symbols:
        raw = load_snapshot_symbol(replay_snapshot_dir, symbol).rename(columns={"taker_buy_base_volume": "buy_volume"})
        candles = raw[SERIES_COLUMNS].copy().reset_index(drop=True)
        replay_candles[symbol] = candles
        ts_index = {ts: i for i, ts in enumerate(pd.to_datetime(candles["timestamp"]))}
        window_start = pd.to_datetime(candles["timestamp"].iloc[0]) + pd.Timedelta(days=REPLAY_WARMUP_DAYS)
        window_end = pd.to_datetime(candles["timestamp"].iloc[-1]) - pd.Timedelta(minutes=BAR_MINUTES * max(FROZEN_HORIZONS))
        sub = dense[(dense["id.symbol"] == symbol) & (dense["_ts"] >= window_start) & (dense["_ts"] <= window_end)]
        if sub.empty:
            continue
        # Rebuild rows at exactly the dataset's sampled indices using the frozen row
        # builder. build_rows_for_symbol calls uniform_steps(valid, max_rows, seed +
        # horizon); redirecting it to the wanted indices per horizon keeps every other
        # line of the frozen Gen1 logic untouched.
        import aegis_alpha.tools.build_risk_v4_long_horizon_dataset_d2 as d2mod

        original_uniform = d2mod.uniform_steps
        wanted_by_horizon = {
            int(h): np.array(sorted(ts_index[t] for t in g["_ts"] if t in ts_index), dtype=int)
            for h, g in sub.groupby("id.horizon")
        }
        try:
            d2mod.uniform_steps = lambda valid, max_rows, seed: np.intersect1d(
                valid, wanted_by_horizon.get(int(seed) - FROZEN_SEED, np.array([], dtype=int))
            )
            frame, _ = rows_fn(candles, symbol, TIMEFRAME, FROZEN_HORIZONS, FROZEN_MAX_ROWS_PER_COMBO, FROZEN_SEED, context_cache)
        finally:
            d2mod.uniform_steps = original_uniform
        rows_expected += int(len(sub))
        rows_rebuilt += int(len(frame))
        if frame.empty:
            continue
        if not compare_cols:
            compare_cols = [c for c in frame.columns if c.startswith(("feature.", "target.", "label.", "future_eval."))]
        merged = sub.merge(frame, on=["id.symbol", "id.timestamp", "id.horizon"], suffixes=("_d3", "_rp"))

        def _num(series: pd.Series) -> pd.Series:
            x = pd.to_numeric(series, errors="coerce")
            if x.isna().any():
                mapped = series.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
                x = x.where(~x.isna(), mapped)
            return x.astype("float64")

        for col in compare_cols:
            if f"{col}_d3" not in merged or f"{col}_rp" not in merged:
                continue
            a = _num(merged[f"{col}_d3"])
            b = _num(merged[f"{col}_rp"])
            both_nan = a.isna() & b.isna()
            ok = both_nan | ((a - b).abs() <= REPLAY_ATOL + REPLAY_RTOL * b.abs())
            total += int(len(merged))
            matched += int(ok.sum())
            bad = int((~ok).sum())
            if bad:
                per_feature_mismatch[col] = per_feature_mismatch.get(col, 0) + bad
    parity = matched / total if total else 0.0
    passed = bool(rows_expected > 0 and rows_rebuilt == rows_expected and parity >= REPLAY_PARITY_MIN)
    result = {
        "gate": "G7_INDEPENDENT_REPLAY",
        "replay_snapshot": snap_manifest["artifact_id"],
        "rows_expected": rows_expected,
        "rows_rebuilt": rows_rebuilt,
        "values_compared": total,
        "parity": parity,
        "parity_required": REPLAY_PARITY_MIN,
        "worst_features": sorted(per_feature_mismatch.items(), key=lambda kv: -kv[1])[:15],
        "passed": passed,
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report = REPORTS_ROOT / f"aegis_gen2_d3_g7_{utc_stamp()}.json"
    report.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    result["report"] = str(report)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-D3 series/dataset builder")
    p.add_argument("--mode", choices=["series", "dataset", "double-check", "replay"], required=True)
    p.add_argument("--snapshot-dir", default="")
    p.add_argument("--series-version", default="v1")
    p.add_argument("--out-tag", default="build_a")
    p.add_argument("--dir-a", default="")
    p.add_argument("--dir-b", default="")
    p.add_argument("--dataset-dir", default="")
    p.add_argument("--replay-snapshot-dir", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "series":
        result = build_series(Path(args.snapshot_dir), args.series_version)
    elif args.mode == "dataset":
        result = build_dataset(args.series_version, args.out_tag)
    elif args.mode == "double-check":
        result = double_check(Path(args.dir_a), Path(args.dir_b))
    else:
        result = replay_check(Path(args.dataset_dir), Path(args.replay_snapshot_dir))
    print(json.dumps(result, indent=2, default=json_default)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
